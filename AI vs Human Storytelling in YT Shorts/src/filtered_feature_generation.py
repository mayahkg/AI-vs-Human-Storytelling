"""
Turn data/transcripts.csv into data/features_filtered_by_ai.csv.

What it does, in one pass:
  1. Reads data/transcripts.csv (rows that have a transcript).
  2. Asks an LLM about each transcript: is it a story (scripted dialogue skit or narrated story) or not
     (explainer, news, vlog, lyrics, tutorial...)? The model never sees the AI/human label.
  3. Keeps the stories and computes the text features.
  4. Writes data/features_filtered_by_ai.csv.

Nothing is saved between runs, so every run judges every transcript again.

Usage (paths are anchored to the project root, one folder above src/):
    python src/filtered_feature_generation.py
    python src/filtered_feature_generation.py --min-conf 0.7   # also drop low-confidence story calls
    python src/filtered_feature_generation.py --allow-partial  # write the file even if some calls failed

Setup: put a key in the root .env
    GEMINI_API_KEY=...      free key from https://aistudio.google.com/apikey
    ANTHROPIC_API_KEY=...   (used instead if both are set)
Optional in .env: GEMINI_MODEL=... or ANTHROPIC_MODEL=... to choose a model yourself.
If GEMINI_MODEL is not set, the script asks Google which models your key can use and picks a Flash model.
Free Gemini keys are rate-limited, so about 475 transcripts take roughly half an hour.
"""
import argparse
import hashlib
import json
import os
import pathlib
import re
import time
from collections import Counter

import numpy as np
import pandas as pd
import requests

try:
    from dotenv import load_dotenv
except ImportError:                      # python-dotenv is optional here
    def load_dotenv(*args, **kwargs):
        return False

ROOT = pathlib.Path(__file__).resolve().parent.parent     # project root (src/ lives inside it)
DATA = ROOT / "data"
load_dotenv(ROOT / ".env")

ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-5-5")
STORY_CATEGORIES = {"scripted_story", "narrated_story"}
VALID_CATEGORIES = STORY_CATEGORIES | {"non_story"}
RATE_PER = 100            # rates are "per 100 words"
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"


class FatalAPIError(Exception):
    """A problem retrying will not fix (wrong key, unknown model...). Stops the run with a clear message."""


# ======================================================================================
# 1. LLM story filter
# ======================================================================================
SYSTEM_PROMPT = """You are labeling transcripts of short-form videos (under about 60 seconds).
Decide whether each transcript is STORYTELLING content.

Categories:
- scripted_story: scripted scene, skit, sketch or drama told mainly through dialogue between characters
  (including comedy skits, sitcom or movie scenes, and animated or fruit/character dramas).
- narrated_story: a story told by a narrator or in first person, with specific characters or events
  and a beginning, middle and end (personal anecdote, true-crime retelling, "AITA"-style story,
  a short fictional story).
- non_story: anything else, for example explainers, tutorials and how-tos, news or commentary,
  reactions, opinion or advice, listicles and fact videos, product demos, ads, promos or channel
  intros, lyrics or songs, pure stand-up or joke monologues without a narrative, step-by-step
  experiments or crafts.

Rules:
- Judge only from the text. Do NOT use whether the topic is AI, whether the text sounds AI-written,
  or the writing quality. A skit with a chatbot character is still a story. A video that merely
  talks about AI trends is not.
- A short ad or "like and subscribe" line at the end of an otherwise real story does not make it non_story.
- If the transcript is too short or too garbled to tell, use non_story with low confidence.
- confidence is a number from 0 to 1.

Respond with ONLY a JSON object, no other text:
{"category": "scripted_story" | "narrated_story" | "non_story", "confidence": 0.0, "reason": "<max 15 words>"}"""


def _user_prompt(transcript):
    return f'Transcript:\n"""\n{transcript}\n"""'


def list_gemini_models(key):
    """Names of the Gemini models this key can call with generateContent."""
    names, token = [], None
    while True:
        params = {"pageSize": 200}
        if token:
            params["pageToken"] = token
        r = requests.get(f"{GEMINI_BASE}/models", headers={"x-goog-api-key": key}, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()
        for m in data.get("models", []):
            if "generateContent" in m.get("supportedGenerationMethods", []):
                names.append(m["name"].split("/", 1)[-1])
        token = data.get("nextPageToken")
        if not token:
            return names


def pick_gemini_model(names):
    """Cheapest reasonable choice: newest stable Flash-Lite, then newest stable Flash, then previews/aliases."""
    def best(pattern):
        found = []
        for n in names:
            m = re.fullmatch(pattern, n)
            if m:
                found.append((float(m.group(1)), n))
        return max(found)[1] if found else None

    return (best(r"gemini-(\d+(?:\.\d+)?)-flash-lite")
            or best(r"gemini-(\d+(?:\.\d+)?)-flash")
            or best(r"gemini-(\d+(?:\.\d+)?)-flash-lite-preview[\w.-]*")
            or best(r"gemini-(\d+(?:\.\d+)?)-flash-preview[\w.-]*")
            or next((n for n in ("gemini-flash-lite-latest", "gemini-flash-latest") if n in names), None))


def call_anthropic(transcript, key, model):
    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
        json={"model": model, "max_tokens": 150, "temperature": 0, "system": SYSTEM_PROMPT,
              "messages": [{"role": "user", "content": _user_prompt(transcript)}]},
        timeout=60)
    r.raise_for_status()
    return r.json()["content"][0]["text"]


def call_gemini(transcript, key, model):
    r = requests.post(
        f"{GEMINI_BASE}/models/{model}:generateContent",
        headers={"x-goog-api-key": key, "content-type": "application/json"},
        json={"systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
              "contents": [{"role": "user", "parts": [{"text": _user_prompt(transcript)}]}],
              "generationConfig": {"temperature": 0, "maxOutputTokens": 1024,
                                   "responseMimeType": "application/json"}},
        timeout=60)
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]


def get_provider():
    """Pick an LLM from whichever key is in the environment. Returns (function, key, model, name) or None."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return call_anthropic, os.environ["ANTHROPIC_API_KEY"], ANTHROPIC_MODEL, "Anthropic"
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        return None
    model = os.environ.get("GEMINI_MODEL")
    if not model:
        try:
            names = list_gemini_models(key)
        except Exception as e:
            raise FatalAPIError(f"Could not ask Google which models your key can use: {e}\n"
                                "Check GEMINI_API_KEY in .env, or set GEMINI_MODEL yourself.")
        model = pick_gemini_model(names)
        if not model:
            raise FatalAPIError("No Flash model is available to this key. Models it can use: "
                                + ", ".join(names[:15]) + "\nSet GEMINI_MODEL in .env to one of them.")
    return call_gemini, key, model, "Gemini"


def http_status(err):
    resp = getattr(err, "response", None)
    if resp is not None and getattr(resp, "status_code", None):
        return int(resp.status_code)
    m = re.match(r"\s*(\d{3})\b", str(err))
    return int(m.group(1)) if m else None


def parse_json(text):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return json.loads(m.group(0) if m else text)


def judge(transcript, provider, retries=4):
    """Returns the model's decision, or None if it kept failing for a temporary reason (rate limit, bad JSON)."""
    fn, key, model, name = provider
    last = None
    for attempt in range(retries):
        try:
            out = parse_json(fn(transcript, key, model))
            if out.get("category") not in VALID_CATEGORIES:
                raise ValueError(f"bad category: {out}")
            return {"llm_category": out["category"], "llm_confidence": float(out.get("confidence", 0)),
                    "llm_reason": str(out.get("reason", ""))[:200]}
        except Exception as e:
            last = e
            status = http_status(e)
            if status in (400, 401, 403, 404):          # retrying will not help
                raise FatalAPIError(f"{name} refused the request ({e}).\n"
                                    + ("The model name was not found. Remove GEMINI_MODEL from .env to let the "
                                       "script pick one, or set it to a model your key can use."
                                       if status == 404 else "Check the API key in .env."))
            time.sleep(2 ** attempt * (3 if status == 429 else 1))
    print(f"  could not judge one transcript: {last}")
    return None


def row_key(df):
    """video_id if the file has one, otherwise a short hash of the transcript text."""
    if "video_id" in df.columns and df["video_id"].notna().all():
        return df["video_id"].astype(str)
    return df["transcript"].apply(lambda t: "h_" + hashlib.md5(t.encode()).hexdigest()[:12])


# ======================================================================================
# 2. Text features (same definitions as features_v2.csv)
# ======================================================================================
NEGATION_WORDS = {"no", "not", "never", "nothing", "nobody", "none", "neither", "nor", "cannot", "nope", "without"}
REACTION_WORDS = {"oh", "ooh", "ah", "aah", "wow", "whoa", "woah", "ha", "haha", "hah", "hmm", "hm", "ugh",
                  "eww", "ew", "yay", "yikes", "omg", "oops", "aww", "aw", "huh", "damn", "gosh", "geez"}
FILLER_WORDS = {"um", "umm", "uh", "uhh", "uhm", "er", "erm"}
INFORMAL = {"gonna", "wanna", "gotta"}
LOVE_FORMS = {"love", "loves", "loved", "loving"}
CONTRACTION_RE = re.compile(r"(n't|'re|'ve|'ll|'d|'m|'s)$")      # 's also matches possessives
FIRST = {"i", "me", "my", "mine", "myself", "i'm", "i'll", "i've", "i'd"}
SECOND = {"you", "your", "yours", "yourself", "you're", "you'll", "you've", "you'd"}
WE = {"we", "us", "our", "ours", "we're", "we'll", "we've"}
PROFANITY = {"fuck", "fucking", "shit", "bitch", "damn", "hell", "ass", "bastard", "motherfucker", "whore",
             "nigga", "crap"}
CTA_RE = re.compile(r"\b(subscribe|follow|comment|like and|part two|next part|link (below|in)|dm me)\b", re.I)


def tokenize(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)*", text.lower().replace("\u2019", "'"))


def ttr(t):
    return len(set(t)) / len(t) if t else np.nan


def mattr(t, w=30):
    if len(t) <= w:
        return ttr(t)
    return float(np.mean([len(set(t[i:i + w])) / w for i in range(len(t) - w + 1)]))


def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def v1_features(text):
    t = tokenize(text)
    n = len(t)
    cnt = lambda pred: sum(1 for w in t if pred(w))
    neg = cnt(lambda w: w in NEGATION_WORDS or w.endswith("n't"))
    rea = cnt(lambda w: w in REACTION_WORDS)
    con = cnt(lambda w: bool(CONTRACTION_RE.search(w)) or w in INFORMAL)
    fil = cnt(lambda w: w in FILLER_WORDS)
    rate = lambda c: c / n * RATE_PER
    ev, lv, wl = cnt(lambda w: w == "everything"), cnt(lambda w: w in LOVE_FORMS), cnt(lambda w: w == "well")
    return {"n_words_calc": n, "lex_var_30": ttr(t[:30]), "mattr_30": mattr(t),
            "negation_count": neg, "negation_rate": rate(neg),
            "reaction_count": rea, "reaction_rate": rate(rea),
            "contraction_count": con, "contraction_rate": rate(con),
            "everything_count": ev, "love_count": lv, "well_count": wl,
            "filler_count": fil, "filler_rate": rate(fil),
            "everything_rate": rate(ev), "love_rate": rate(lv), "well_rate": rate(wl)}


def v2_features(text):
    w = tokenize(text)
    n = len(w)
    S = sentences(text)
    ns = max(len(S), 1)
    sl = [len(tokenize(s)) for s in S] or [0]
    wc = Counter(w)
    bigrams = list(zip(w, w[1:]))
    bc = Counter(bigrams)
    caps = [m for s in S for m in re.findall(r"(?<!^)\b[A-Z][a-z]+", s)]   # capitalised, not sentence-initial
    per100 = lambda c: c / n * 100
    return {
        "n_sentences": len(S), "avg_sent_len": np.mean(sl), "sent_len_std": np.std(sl), "max_sent_len": max(sl),
        "short_sent_rate": np.mean([x <= 4 for x in sl]) * 100,
        "question_rate": text.count("?") / ns * 100, "exclaim_rate": text.count("!") / ns * 100,
        "ellipsis_rate": text.count("...") / ns * 100,
        "avg_word_len": np.mean([len(x) for x in w]), "long_word_rate": per100(sum(len(x) >= 7 for x in w)),
        "hapax_rate": per100(sum(v == 1 for v in wc.values())), "ttr_full": len(wc) / n,
        "top_word_share": max(wc.values()) / n * 100,
        "profanity_rate": per100(sum(x in PROFANITY for x in w)), "digit_rate": per100(len(re.findall(r"\d+", text))),
        "first_person_rate": per100(sum(x in FIRST for x in w)),
        "second_person_rate": per100(sum(x in SECOND for x in w)),
        "we_rate": per100(sum(x in WE for x in w)),
        "repeat_bigram_rate": sum(v - 1 for v in bc.values() if v > 1) / max(len(bigrams), 1) * 100,
        "repeat_sentence_rate": (len(S) - len(set(s.lower() for s in S))) / ns * 100,
        "proper_noun_rate": per100(len(caps)),
        "cta_flag": int(bool(CTA_RE.search(text))),
    }


V1_COLS = ["n_words_calc", "lex_var_30", "mattr_30", "negation_count", "negation_rate", "reaction_count",
           "reaction_rate", "contraction_count", "contraction_rate", "everything_count", "love_count",
           "well_count", "filler_count", "filler_rate", "everything_rate", "love_rate", "well_rate"]
V2_COLS = ["n_sentences", "avg_sent_len", "sent_len_std", "max_sent_len", "short_sent_rate", "question_rate",
           "exclaim_rate", "ellipsis_rate", "avg_word_len", "long_word_rate", "hapax_rate", "ttr_full",
           "top_word_share", "profanity_rate", "digit_rate", "first_person_rate", "second_person_rate",
           "we_rate", "repeat_bigram_rate", "repeat_sentence_rate", "proper_noun_rate", "cta_flag"]
INT_COLS = ["n_words_calc", "negation_count", "reaction_count", "contraction_count", "everything_count",
            "love_count", "well_count", "filler_count", "n_sentences", "max_sent_len", "cta_flag"]


def build_features(df):
    feats = pd.DataFrame([{**v1_features(t), **v2_features(t)} for t in df["transcript"]], index=df.index)
    out = pd.concat([df[["video_id", "label"]], feats[V1_COLS + V2_COLS]], axis=1)
    out["y"] = (out["label"] == "AI").astype(int)
    out[INT_COLS] = out[INT_COLS].astype(int)
    return out.round(4)


# ======================================================================================
# 3. Main
# ======================================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--transcripts", default=str(DATA / "transcripts.csv"))
    ap.add_argument("--out", default=str(DATA / "features_filtered_by_ai.csv"))
    ap.add_argument("--min-conf", type=float, default=0.0, help="drop story calls below this confidence")
    ap.add_argument("--sleep", type=float, default=None,
                    help="seconds between LLM calls (default: 4.5 for Gemini's free tier, 0.2 for Anthropic)")
    ap.add_argument("--allow-partial", action="store_true",
                    help="write the output even if some transcripts could not be judged")
    args = ap.parse_args()
    out_path = pathlib.Path(args.out)

    # ---- load transcripts ----
    df = pd.read_csv(args.transcripts)
    if "status" in df.columns:
        df = df[df["status"] == "ok"]
    df = df[df["transcript"].notna() & (df["transcript"].astype(str).str.strip() != "")].copy()
    df["transcript"] = df["transcript"].astype(str)
    df["video_id"] = row_key(df)
    n_dupes = int(df.duplicated("transcript").sum())
    df = df.drop_duplicates("transcript").reset_index(drop=True)
    print(f"{len(df)} transcripts to judge ({n_dupes} duplicate transcripts removed)")

    # ---- judge every transcript ----
    try:
        provider = get_provider()
        if provider is None:
            raise SystemExit("No API key found. Add GEMINI_API_KEY (free: https://aistudio.google.com/apikey) "
                             "or ANTHROPIC_API_KEY to the root .env and run again.")
        pause = args.sleep if args.sleep is not None else (4.5 if provider[3] == "Gemini" else 0.2)
        print(f"Judging with {provider[3]} model {provider[2]} (about {len(df) * pause / 60:.0f} min)")

        decisions, failed, fails_in_a_row = {}, [], 0
        for i, row in enumerate(df.itertuples(), 1):
            rec = judge(row.transcript, provider)
            if rec is None:
                failed.append(row.Index)
                fails_in_a_row += 1
                if fails_in_a_row >= 5:
                    failed += list(df.index[i:])            # everything after this was not tried
                    print("Stopping: 5 calls in a row failed (probably a rate or daily limit).")
                    break
            else:
                decisions[row.Index] = rec
                fails_in_a_row = 0
            if i % 25 == 0:
                print(f"  {i}/{len(df)} judged")
            time.sleep(pause)

        if failed and fails_in_a_row < 5:                    # one more try for the ones that failed
            print(f"Retrying {len(failed)} transcripts that failed...")
            still = []
            for idx in failed:
                time.sleep(max(pause, 5))
                rec = judge(df.at[idx, "transcript"], provider, retries=2)
                (decisions.__setitem__(idx, rec) if rec else still.append(idx))
            failed = still
    except FatalAPIError as e:
        raise SystemExit(f"\nStopped: {e}")

    if failed and not args.allow_partial:
        raise SystemExit(f"\n{len(failed)} of {len(df)} transcripts could not be judged, so nothing was written "
                         "(a partial file would silently drop videos). Wait a bit (free-tier limits reset) and "
                         "run again, or use --allow-partial.")

    # ---- keep stories ----
    judged = df.loc[list(decisions)].join(pd.DataFrame.from_dict(decisions, orient="index"))
    judged["keep"] = judged["llm_category"].isin(STORY_CATEGORIES) & (judged["llm_confidence"] >= args.min_conf)
    print("\nStory filter:")
    summary = judged.groupby("label")["keep"].agg(kept="sum", total="count")
    summary["keep_rate"] = (summary["kept"] / summary["total"]).round(2)
    print(summary.to_string())
    if "theme" in judged.columns:
        print("\nKept stories by theme:")
        print(pd.crosstab(judged[judged["keep"]]["theme"], judged[judged["keep"]]["label"]).to_string())

    kept = judged[judged["keep"]].reset_index(drop=True)
    if kept.empty:
        raise SystemExit("No stories left after filtering, so no features were written.")

    # ---- features ----
    feats = build_features(kept)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feats.to_csv(out_path, index=False)
    print(f"\nSaved {len(feats)} rows x {feats.shape[1]} columns to {out_path}")
    print(feats["label"].value_counts().to_string())


if __name__ == "__main__":
    main()