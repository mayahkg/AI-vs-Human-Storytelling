import re
import sys
import time
import random
import pathlib
import subprocess

import pandas as pd
from faster_whisper import WhisperModel

# ---------------- settings you can edit ----------------
POOL_PER_THEME_LABEL = 40    # max usable transcripts to collect per (theme, label)
TARGET_PER_THEME_LABEL = None  # cap per (theme, label) in the balanced file; None = as many as balance allows
MODEL_SIZE = "small"         # "base" is faster; "medium" is more accurate but slower
MIN_WORDS = 30               # shorter transcripts are discarded
MIN_LANG_PROB = 0.7          # must be English with at least this confidence
MAX_LAUGH_SHARE = 0.12       # transcripts that are mostly "ha ha ha" are marked junk
MAX_REPEAT_SHARE = 0.4       # share of repeated 3-word phrases; above this is a transcription loop, not a story
DELAY_RANGE = (3, 8)         # random pause (seconds) between downloads
RETRY_FAILED = False         # True = retry videos whose download/transcription failed last run
APPLY_TITLE_FILTER = True    # drop obvious non-stories by title (same rule for AI and human)
# Candidate files to look for, first match wins (discover.py writes data/candidates.csv)
SRC_CANDIDATES = ["data/candidates.csv", "candidates.csv",
                  "data/candidates_v2.csv", "candidates_v2.csv"]
# --------------------------------------------------------

# Maps batch-1 source_queries to a theme. Batch-2 rows from discover.py already have a 'topic'
# column, which is used directly, so only older queries need to be listed here.
# Each theme needs BOTH AI and human rows, otherwise the balanced file skips it. Edit freely.
THEME_MAP = {
    # romance
    "ai romance short film": "romance",
    "ai love story episode english": "romance",
    # couple (infidelity / arguments)
    "ai cheating story drama": "couple",
    "ai husband wife argument": "couple",
    "cheating boyfriend skit acting": "couple",
    # family
    "ai mother in law drama": "family",
    "ai family drama story": "family",
    "family drama skit acting": "family",
    "family betrayal drama scene": "family",
    # baby / animal / fruit / work / horror
    "ai baby drama": "baby",
    "ai animal drama": "animal",
    "ai fruit drama": "fruit",
    "ai fruit story episode": "fruit",
    "animated food characters drama": "fruit",
    "ai office drama story": "work",
    "ai horror short story": "horror",
    # animated / toy / puppet (human-heavy; AI side is cartoons)
    "ai cartoon drama story": "animated_toy",
    "2d animated drama story voice acted": "animated_toy",
    "stop motion drama story": "animated_toy",
    "claymation drama skit": "animated_toy",
    "puppet drama skit": "animated_toy",
    "toy drama story skit": "animated_toy",
    "barbie doll drama skit": "animated_toy",
    "sylvanian families drama": "animated_toy",
    # general drama with no specific topic
    "ai revenge story": "general_drama",
    "short film drama emotional scene": "general_drama",
    "ai mini drama": "general_drama",
    "ai drama episode": "general_drama",
    "ai generated drama story": "general_drama",
    "ai generated story drama": "general_drama",
    "ai generated realistic drama people": "general_drama",
    "veo 3 drama scene": "general_drama",
    "sora short story drama": "general_drama",
    "drama skit": "general_drama",
    "mini drama skit original": "general_drama",
    "pov drama skit actress": "general_drama",
}

HERE = pathlib.Path(__file__).parent
SRC = next((HERE / p for p in SRC_CANDIDATES if (HERE / p).exists()), None)
WORK = HERE / "transcripts_work_v2.csv"   # progress file (every video tried, so reruns resume)
FINAL = HERE / "dataset_v2.csv"           # balanced per theme: the one to model on
FINAL_ALL = HERE / "dataset_v2_all.csv"   # every usable transcript, unbalanced
AUDIO_DIR = HERE / "audio"
AUDIO_DIR.mkdir(exist_ok=True)

WORK_COLS = ["video_id", "label", "theme", "url", "title", "source_query",
             "status", "language", "lang_prob", "n_words", "transcript", "note"]

# Same rule is applied to both labels so it can't bias the comparison.
JUNK_TITLE_RE = re.compile(
    r"\b(?:prank|recipe|tutorial|how to|hack|explained|ranking|review|reaction|"
    r"vibe coding|startup|movies|advice|compilation|dr julie|cheater buster|"
    r"job interviews?)\b", re.I)
LAUGH_RE = re.compile(r"\b(?:ha|haha|hahaha|hehe|heh|hee)\b", re.I)


def junk_reason(text):
    """Conservative checks for transcripts that aren't storytelling. Same rule for both labels.
    Only catches the obvious cases (laughter tracks, Whisper repeating itself); songs, commentary
    and other off-target content still need a manual or LLM pass."""
    text = "" if pd.isna(text) else str(text)
    words = re.findall(r"[a-z']+", text.lower())
    if not words:
        return "empty"
    if len(LAUGH_RE.findall(text)) / len(words) > MAX_LAUGH_SHARE:
        return "laughter"
    tri = [" ".join(words[i:i + 3]) for i in range(len(words) - 2)]
    if tri and 1 - len(set(tri)) / len(tri) > MAX_REPEAT_SHARE:
        return "repetition_loop"
    return ""


# ---- load source and progress ----
if SRC is None:
    raise SystemExit(f"Could not find a candidates file. Looked for: {SRC_CANDIDATES}")
print("Reading candidates from:", SRC.relative_to(HERE))

src = pd.read_csv(SRC)
if "theme" not in src.columns:
    # Batch-2 rows: use the 'topic' column from discover.py. Batch-1 rows: map source_query via THEME_MAP.
    theme = src["source_query"].astype(str).str.strip().str.lower().map(THEME_MAP)
    if "topic" in src.columns:
        theme = theme.fillna(src["topic"].where(src["topic"] != "batch1"))
    src["theme"] = theme.fillna("other")
    print("Derived 'theme'. Rows per theme and label:")
    print(pd.crosstab(src["theme"], src["label"]).to_string())
    if (src["theme"] == "other").any():
        print("Unmapped source_query values:",
              src.loc[src["theme"] == "other", "source_query"].unique().tolist())

src["video_id"] = src["url"].str.rstrip("/").str.split("/").str[-1].str.split("?").str[0]
src = src.drop_duplicates("video_id")
src_all = src.copy()   # keeps theme + channel/date/duration for every candidate, used when building the outputs

if APPLY_TITLE_FILTER:
    junk = src["title"].fillna("").str.contains(JUNK_TITLE_RE)
    print("Title filter removed:", src[junk].groupby("label").size().to_dict() or "nothing")
    src = src[~junk]

if WORK.exists():
    work = pd.read_csv(WORK)
    if RETRY_FAILED:
        work = work[work["status"] != "failed"]
else:
    work = pd.DataFrame(columns=WORK_COLS)

if len(work):
    # Re-derive themes so earlier transcripts use the same theme names as new ones
    work["theme"] = work["video_id"].map(src_all.set_index("video_id")["theme"]).fillna(work["theme"])
    # Apply the junk check to transcripts collected in earlier runs
    reasons = work["transcript"].where(work["status"] == "ok").map(
        lambda t: junk_reason(t) if isinstance(t, str) else "")
    flagged = reasons != ""
    if flagged.any():
        work["note"] = work["note"].astype("object")   # empty notes load as NaN floats
        work.loc[flagged, "status"] = "junk"
        work.loc[flagged, "note"] = reasons[flagged].to_numpy()
        print("Earlier transcripts re-marked as junk:", work[flagged].groupby("label").size().to_dict())
done_ids = set(work["video_id"])

# One queue per (theme, label), so both sides of each theme fill up together.
queues = {}
for (theme, label), grp in src.groupby(["theme", "label"]):
    rows = grp[~grp.video_id.isin(done_ids)]
    queues[(theme, label)] = rows.sample(frac=1, random_state=42).to_dict("records")
queue_keys = sorted(queues)


def n_ok(theme, label):
    return int(((work["theme"] == theme) & (work["label"] == label)
                & (work["status"] == "ok")).sum())


print("Videos to process:", sum(len(q) for q in queues.values()))
print(f"Loading Whisper model '{MODEL_SIZE}' (the first run downloads it)...")
model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")


def download_audio(video_id, url):
    for old in AUDIO_DIR.glob(f"{video_id}.*"):
        old.unlink()
    result = subprocess.run(
        [sys.executable, "-m", "yt_dlp", "-f", "bestaudio", "--no-playlist",
         "--socket-timeout", "20", "--retries", "2",
         "-o", str(AUDIO_DIR / f"{video_id}.%(ext)s"), url],
        capture_output=True, text=True)
    files = list(AUDIO_DIR.glob(f"{video_id}.*"))
    if result.returncode != 0 or not files:
        raise RuntimeError(result.stderr.strip().splitlines()[-1][:150]
                           if result.stderr.strip() else "download failed")
    return files[0]


def process(row):
    out = {"video_id": row["video_id"], "label": row["label"], "theme": row["theme"],
           "url": row["url"], "title": row["title"], "source_query": row["source_query"],
           "status": "", "language": "", "lang_prob": None, "n_words": None,
           "transcript": "", "note": ""}
    audio = None
    try:
        audio = download_audio(row["video_id"], row["url"])
        segments, info = model.transcribe(str(audio), language=None, vad_filter=True)
        text = " ".join(s.text.strip() for s in segments).strip()
        out["language"] = info.language
        out["lang_prob"] = round(info.language_probability, 3)
        out["n_words"] = len(text.split())
        if info.language != "en" or info.language_probability < MIN_LANG_PROB:
            out["status"] = "non_english"
        elif out["n_words"] < MIN_WORDS:
            out["status"] = "too_short"
        elif junk_reason(text):
            out["status"] = "junk"
            out["note"] = junk_reason(text)
            out["transcript"] = text          # kept in the work file so you can inspect it
        else:
            out["status"] = "ok"
            out["transcript"] = text
    except Exception as e:
        out["status"] = "failed"
        out["note"] = str(e)[:150]
    finally:
        if audio is not None and pathlib.Path(audio).exists():
            pathlib.Path(audio).unlink()      # keep the text, not the audio
    return out


# ---- main loop: round-robin over (theme, label) so everything fills up evenly ----
while True:
    active = [k for k in queue_keys if queues[k] and n_ok(*k) < POOL_PER_THEME_LABEL]
    if not active:
        break
    for key in active:
        theme, label = key
        row = queues[key].pop(0)
        result = process(row)
        work = pd.concat([work, pd.DataFrame([result])], ignore_index=True)
        work.to_csv(WORK, index=False)        # save after every video
        print(f"{theme:13} {label:5} {row['video_id']} -> {result['status']:11} "
              f"({result['n_words']} words)")
        time.sleep(random.uniform(*DELAY_RANGE))

# ---- build final CSVs ----
ok = work[work.status == "ok"].copy()
if ok.empty:
    raise SystemExit("No usable transcripts yet. Check the status counts in transcripts_work_v2.csv.")
ok["n_words"] = ok["n_words"].astype(float)

# Attach channel / publish date / duration from the candidates file (needed for channel-grouped CV)
meta_cols = [c for c in ["channel", "published", "duration"] if c in src_all.columns]
ok = ok.drop(columns=[c for c in meta_cols if c in ok.columns]).merge(
    src_all[["video_id"] + meta_cols], on="video_id", how="left")

# Drop copy-pasted transcripts (same script posted as several videos)
ok["_norm"] = (ok.transcript.str.lower()
               .str.replace(r"[^a-z0-9 ]", "", regex=True)
               .str.split().str.join(" "))
n_before = len(ok)
ok = ok.drop_duplicates("_norm")
print(f"\nDuplicate transcripts removed: {n_before - len(ok)}")

OUT_COLS = (["label", "theme", "video_id"] + meta_cols +
            ["n_words", "language", "title", "source_query", "transcript"])
ok[OUT_COLS].sort_values(["theme", "label"]).to_csv(FINAL_ALL, index=False)

# Balanced file: equal AI and human counts inside every theme, chosen at RANDOM
# (not by length, which would quietly bias the comparison).
parts = []
for theme, g in ok.groupby("theme"):
    counts = g.label.value_counts()
    if len(counts) < 2:
        print(f"Skipping theme '{theme}': only one label has usable transcripts")
        continue
    n = int(counts.min())
    if TARGET_PER_THEME_LABEL:
        n = min(n, TARGET_PER_THEME_LABEL)
    for _, gg in g.groupby("label"):
        parts.append(gg.sample(n=n, random_state=42))

if not parts:
    raise SystemExit("No theme has usable transcripts for both labels.")
final = pd.concat(parts).sort_values(["theme", "label"])
final[OUT_COLS].to_csv(FINAL, index=False)

# ---- summary ----
print("\nStatus counts by label:")
print(pd.crosstab(work.label, work.status).to_string())

print("\nStatus counts by theme and label:")
print(pd.crosstab([work.theme, work.label], work.status).to_string())

rate = (work.assign(ok=work.status == "ok").groupby("label")["ok"].mean() * 100).round(1)
print("\nShare of tried videos with a usable English transcript (%):")
print(rate.to_string())
print("If these differ a lot between AI and human, that gap is a finding "
      "and also a source of bias.")

print(f"\nSaved {len(ok)} usable rows to {FINAL_ALL}")
print(f"Saved {len(final)} balanced rows to {FINAL}")
print("\nBalanced rows per theme and label (under ~10 per side is thin):")
print(pd.crosstab(final.theme, final.label).to_string())

if "channel" in final.columns:
    print("\nUnique channels in balanced set:", final.channel.nunique())

print("\nMedian transcript length (words) in balanced set:")
print(final.groupby("label")["n_words"].median().to_string())
print("\nNote: drop 'title', 'source_query', 'channel' and 'theme' before modeling (titles contain 'AI' words).")
print("Use 'channel' as the groups argument in GroupKFold, and 'theme' to check topic balance.")