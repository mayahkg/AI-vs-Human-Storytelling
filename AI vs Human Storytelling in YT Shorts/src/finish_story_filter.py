"""Optional step 3b: finish the story filter when the free Gemini flash-lite quota (500 calls/model/day) runs out.
Usage: python src/finish_story_filter.py   (needs GEMINI_API_KEY, optionally GEMINI_API_KEY_2, in .env)
This is how data/features_filtered_by_ai.csv on this branch was built.

1. Gemma (gemma-4-26b-a4b-it, same prompt via the repo's SYSTEM_PROMPT/_user_prompt/judge) judges the transcripts no model
   has judged yet, plus 40 already judged by both flash-lite models as an agreement check.
2. Each transcript's decision = gemini-3.1-flash-lite, else gemini-3.5-flash-lite, else gemma.
3. Keep stories and write data/features_filtered_by_ai.csv with the repo's build_features (same as main())."""
import queue
import sys
import threading
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import dotenv_values

REPO = Path(__file__).resolve().parent.parent     # project root (src/ lives inside it)
sys.path.insert(0, str(REPO / "src"))
sys.stdout.reconfigure(encoding="utf-8")
import filtered_feature_generation as f  # noqa: E402

GEMMA = "gemma-4-26b-a4b-it"
ORDER = ["gemini-3.1-flash-lite", "gemini-3.5-flash-lite", GEMMA]
env = dotenv_values(REPO / ".env")
KEYS = [env[n] for n in ("GEMINI_API_KEY", "GEMINI_API_KEY_2") if env.get(n)]   # never printed
THREADS_PER_KEY, MIN_GAP = 3, 2.5


def call_gemma(transcript, key, model):
    """Same request as the repo's call_gemini, minus responseMimeType (Gemma rejects it)."""
    r = requests.post(
        f"{f.GEMINI_BASE}/models/{model}:generateContent",
        headers={"x-goog-api-key": key, "content-type": "application/json"},
        json={"systemInstruction": {"parts": [{"text": f.SYSTEM_PROMPT}]},
              "contents": [{"role": "user", "parts": [{"text": f._user_prompt(transcript)}]}],
              "generationConfig": {"temperature": 0, "maxOutputTokens": 1024}},
        timeout=120)
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][-1]["text"]


df = pd.read_csv(REPO / "data/transcripts.csv")
df = df[(df["status"] == "ok") & df["transcript"].notna() & (df["transcript"].astype(str).str.strip() != "")].copy()
df["transcript"] = df["transcript"].astype(str)
df["video_id"] = f.row_key(df)
df = df.drop_duplicates("transcript").reset_index(drop=True)

saved = {m: f.load_judgments(m) for m in ORDER}
have_flash = df["video_id"].isin(list(saved[ORDER[0]])) | df["video_id"].isin(list(saved[ORDER[1]]))
both = df[df["video_id"].isin(list(saved[ORDER[0]])) & df["video_id"].isin(list(saved[ORDER[1]]))]
check = both.sample(min(40, len(both)), random_state=0)
todo = pd.concat([df[~have_flash], check])
todo = todo[~todo["video_id"].isin(list(saved[GEMMA]))]
print(f"{len(df)} transcripts | judged by a flash-lite model: {int(have_flash.sum())} | "
      f"Gemma to judge: {len(todo)} ({int((~have_flash).sum())} unjudged + agreement sample)", flush=True)

q = queue.Queue()
for r in todo.itertuples():
    q.put((r.video_id, r.transcript))
lock = threading.Lock()
gaps = [{"lock": threading.Lock(), "last": 0.0} for _ in KEYS]
stats = {"done": 0, "failed": 0}


def worker(k):
    provider = (call_gemma, KEYS[k], GEMMA, "Gemini")
    while True:
        try:
            vid, text = q.get_nowait()
        except queue.Empty:
            return
        with gaps[k]["lock"]:
            wait = gaps[k]["last"] + MIN_GAP - time.time()
            if wait > 0:
                time.sleep(wait)
            gaps[k]["last"] = time.time()
        try:
            rec = f.judge(text, provider, retries=3)
        except f.FatalAPIError as e:
            print("refused:", str(e)[:150], flush=True)
            rec = None
        with lock:
            if rec:
                f.save_judgment(vid, GEMMA, rec)
                stats["done"] += 1
                if stats["done"] % 25 == 0:
                    print(f"  {stats['done']}/{len(todo)} judged by Gemma", flush=True)
            else:
                stats["failed"] += 1


threads = [threading.Thread(target=worker, args=(k,)) for k in range(len(KEYS)) for _ in range(THREADS_PER_KEY)]
[t.start() for t in threads]
[t.join() for t in threads]
print(f"Gemma finished: {stats['done']} judged, {stats['failed']} failed", flush=True)

# ---- agreement check
saved = {m: f.load_judgments(m) for m in ORDER}
story = lambda m, v: saved[m][v]["llm_category"] in f.STORY_CATEGORIES
pairs = [("gemini-3.1-flash-lite", "gemini-3.5-flash-lite"), (GEMMA, "gemini-3.1-flash-lite"), (GEMMA, "gemini-3.5-flash-lite")]
print("\nStory / not-story agreement between models:")
for a, b in pairs:
    ids = [v for v in df["video_id"] if v in saved[a] and v in saved[b]]
    if ids:
        print(f"  {a} vs {b}: {sum(story(a, v) == story(b, v) for v in ids) / len(ids):.1%} of {len(ids)}")

# ---- combine and build features exactly like main()
decisions, source = {}, {}
for idx, v in df["video_id"].items():
    for m in ORDER:
        if v in saved[m]:
            decisions[idx], source[idx] = saved[m][v], m
            break
missing = len(df) - len(decisions)
print(f"\nDecisions used: {pd.Series(source).value_counts().to_dict()} | still unjudged (dropped): {missing}")
judged = df.loc[list(decisions)].join(pd.DataFrame.from_dict(decisions, orient="index"))
judged["keep"] = judged["llm_category"].isin(f.STORY_CATEGORIES)
s = judged.groupby("label")["keep"].agg(kept="sum", total="count")
s["keep_rate"] = (s["kept"] / s["total"]).round(2)
print("\nStory filter:\n" + s.to_string())
print("\nKept stories by theme:\n" + pd.crosstab(judged[judged.keep]["theme"], judged[judged.keep]["label"]).to_string())
kept = judged[judged["keep"]].reset_index(drop=True)
feats = f.build_features(kept)
out = REPO / "data/features_filtered_by_ai.csv"
feats.to_csv(out, index=False)
print(f"\nSaved {len(feats)} rows x {feats.shape[1]} columns to {out}")
print(feats["label"].value_counts().to_string())
