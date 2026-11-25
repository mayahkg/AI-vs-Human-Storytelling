import re
import sys
import time
import random
import pathlib
import subprocess

import pandas as pd
from faster_whisper import WhisperModel

# Settings
POOL_PER_THEME_LABEL = 40    # Preventing any particular themes from dominating
MODEL_SIZE = "small"        # Chosen Whisper Model
MIN_WORDS = 30             
MIN_LANG_PROB = 0.7       
MAX_LAUGH_SHARE = 0.12      
MAX_REPEAT_SHARE = 0.4      
DELAY_RANGE = (3, 8)    
RETRY_FAILED = False         
APPLY_TITLE_FILTER = True 

# Set themes to compare 
THEMES_TO_FILL = {"animal", "baby", "couple", "family", "fruit", "horror", "school", "work"}

# Mapping queries to the themes
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
    "ai boss employee story": "work",
    "ai workplace drama": "work",
    "ai waitress drama": "work",
    "ai ceo revenge story": "work",
    "ai horror story": "horror",
    "ai scary story animation": "horror",
    "ai ghost story": "horror",
    "horror short film": "horror",
    "scary short film acting": "horror",
    "creepy skit": "horror",
    "ai cat story": "animal",
    "ai dog story": "animal",
    "ai animal rescue story": "animal",
    "dog skit acting": "animal",
    "talking animals skit": "animal",
    "talking fruit skit": "fruit",
    "ai school drama": "school",
    "ai teacher student story": "school",
    "ai bully story": "school",
    "ai baby story": "baby",
    "ai pregnant wife story": "baby",
}

ROOT = pathlib.Path(__file__).resolve().parents[1]   # project root (the folder above src/)
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
SRC = DATA / "video_urls.csv"             # input, written by collect_videos.py
WORK = DATA / "transcripts.csv"           # output + progress file (every video tried, so reruns resume)
AUDIO_DIR = ROOT / "audio"
AUDIO_DIR.mkdir(exist_ok=True)

WORK_COLS = ["video_id", "label", "theme", "url", "title", "source_query",
             "status", "language", "lang_prob", "n_words", "transcript", "note"]

JUNK_TITLE_RE = re.compile(
    r"\b(?:prank|recipe|tutorial|how to|hack|explained|ranking|review|reaction|"
    r"vibe coding|startup|movies|advice|compilation|dr julie|cheater buster|"
    r"job interviews?)\b", re.I)
LAUGH_RE = re.compile(r"\b(?:ha|haha|hahaha|hehe|heh|hee)\b", re.I)

# Check for non-stories
def junk_reason(text):
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


# Load Progress 
if not SRC.exists():
    raise SystemExit(f"Could not find {SRC}. Run src/collect_videos.py first.")
print("Reading candidates from:", SRC.relative_to(ROOT))

src = pd.read_csv(SRC)
if "theme" not in src.columns:
    # Batch-2 rows: use the 'topic' column from collect_videos.py. Batch-1 rows: map source_query via THEME_MAP.
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
src_all = src.copy()   # every candidate with its theme, used to re-derive themes for earlier transcripts

if APPLY_TITLE_FILTER:
    junk = src["title"].fillna("").str.contains(JUNK_TITLE_RE)
    print("Title filter removed:", src[junk].groupby("label").size().to_dict() or "nothing")
    src = src[~junk]

if THEMES_TO_FILL:
    skipped = src[~src["theme"].isin(THEMES_TO_FILL)]
    print("Skipped (theme not in THEMES_TO_FILL):", skipped.groupby("theme").size().to_dict() or "nothing")
    src = src[src["theme"].isin(THEMES_TO_FILL)]

if WORK.exists():
    work = pd.read_csv(WORK)
    if RETRY_FAILED:
        work = work[work["status"] != "failed"]
else:
    work = pd.DataFrame(columns=WORK_COLS)

if len(work):
    derived = work["video_id"].map(src_all.set_index("video_id")["theme"]).fillna(work["theme"])
    keep = work["theme"].isin(THEMES_TO_FILL) if THEMES_TO_FILL else pd.Series(False, index=work.index)
    work["theme"] = work["theme"].where(keep, derived)
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

# Build queues of videos to process, grouped by theme and label, and shuffle them
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


# Filling Up Every Theme/Label Queue
while True:
    active = [k for k in queue_keys if queues[k] and n_ok(*k) < POOL_PER_THEME_LABEL]
    if not active:
        break
    for key in active:
        theme, label = key
        row = queues[key].pop(0)
        result = process(row)
        work = pd.concat([work, pd.DataFrame([result])], ignore_index=True)
        work["n_words"] = work["n_words"].astype("Int64")   # keep whole numbers (failed rows are blank)
        work.to_csv(WORK, index=False, lineterminator="\n")  # save after every video; \n keeps git diffs clean on Windows
        print(f"{theme:13} {label:5} {row['video_id']} -> {result['status']:11} "
              f"({result['n_words']} words)")
        time.sleep(random.uniform(*DELAY_RANGE))

# Print Summary
ok = work[work.status == "ok"]
if ok.empty:
    raise SystemExit("No usable transcripts yet. Check the status counts in data/transcripts.csv.")

print("\nStatus counts by label:")
print(pd.crosstab(work.label, work.status).to_string())

print("\nStatus counts by theme and label:")
print(pd.crosstab([work.theme, work.label], work.status).to_string())

rate = (work.assign(ok=work.status == "ok").groupby("label")["ok"].mean() * 100).round(1)
print("\nShare of tried videos with a usable English transcript (%):")
print(rate.to_string())
print("If these differ a lot between AI and human, that gap is a finding "
      "and also a source of bias.")

print(f"\nUsable transcripts in {WORK.name}: {len(ok)}")
print("\nUsable transcripts per theme and label (under ~10 per side is thin):")
print(pd.crosstab(ok.theme, ok.label).to_string())

print("\nMedian transcript length (words) of usable transcripts:")
print(ok.assign(n_words=ok.n_words.astype(float)).groupby("label")["n_words"].median().to_string())
print("\nNote: drop 'title', 'source_query' and 'theme' before modeling (titles contain 'AI' words).")