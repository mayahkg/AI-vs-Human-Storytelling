import os
import re
import json
import shutil
import hashlib
import pathlib
import isodate
import numpy as np
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Loading in Environment Variables
from dotenv import load_dotenv
load_dotenv()

# Creating API Client
yt = build("youtube", "v3", developerKey=os.environ["YT_API_KEY"])

# File paths (anchored to this script's folder)
OUT = pathlib.Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)
CSV = OUT / "candidates.csv"
COLS = ["label", "url", "title", "source_query",
        "video_id", "channel", "topic", "published", "duration"]

# ---------------------------------------------------------------------------
# Batch 2 search queries, written as topic-matched pairs: (topic, AI query, human query).
# Matching topics across classes keeps the classifier from learning "fruit = AI".
# The script alternates AI/human queries, so if a run stops early both classes stay balanced.
# ---------------------------------------------------------------------------
PAIRS = [
    ("fruit",   "ai fruit love story",            "animated fruit characters story"),
    ("fruit",   "ai fruit betrayal drama",        "fruit cartoon drama animation"),
    ("animal",  "ai cat drama story",             "animated cat drama story"),
    ("animal",  "ai dog rescue story",            "dog rescue short film"),
    ("baby",    "ai toddler drama",               "toddler drama skit acting"),
    ("baby",    "ai baby story episode",          "baby drama skit"),
    ("family",  "ai stepmom drama story",         "stepmom drama skit"),
    ("family",  "ai grandma drama story",         "grandma drama skit acting"),
    ("couple",  "ai girlfriend cheating drama",   "girlfriend cheating skit acting"),
    ("couple",  "ai wedding drama story",         "wedding drama skit"),
    ("work",    "ai boss employee drama",         "boss employee drama skit"),
    ("work",    "ai coworker revenge story",      "coworker drama skit"),
    ("school",  "ai school bully story",          "school bully skit acting"),
    ("school",  "ai teacher student drama",       "teacher student drama skit"),
    ("horror",  "ai scary story short",           "scary story short film acting"),
    ("romance", "ai romance love story",          "romantic short film acting"),
]

# Batch 1 queries, kept for the record. Already collected, so they are not re-run.
BATCH1_HUMAN_QUERIES = [
    "drama skit", "family betrayal drama scene", "pov drama skit actress",
    "cheating boyfriend skit acting", "family drama skit acting", "mini drama skit original",
    "short film drama emotional scene", "stop motion drama story", "claymation drama skit",
    "puppet drama skit", "barbie doll drama skit", "toy drama story skit",
    "sylvanian families drama", "animated food characters drama", "2d animated drama story voice acted",
]
BATCH1_AI_QUERIES = [
    "ai fruit drama", "ai mini drama", "ai generated drama story", "ai fruit story episode",
    "ai animal drama", "ai baby drama", "ai love story episode english", "ai generated story drama",
    "ai drama episode", "veo 3 drama scene", "ai generated realistic drama people",
    "ai cheating story drama", "ai revenge story", "ai mother in law drama", "ai family drama story",
    "ai office drama story", "ai husband wife argument", "ai romance short film",
    "ai cartoon drama story", "sora short story drama", "ai horror short story",
]

# API Search Settings
PAGES = 4                              # pages of 50 results per query (5-10 pages is the max useful depth)
MAX_SEARCH_CALLS = 90                  # live search calls per run; each costs 100 of the 10,000 daily quota units
HUMAN_BEFORE = "2023-01-10T00:00:00Z"  # Only Shorts from before AI was widely available
AI_AFTER = "2025-01-01T00:00:00Z"
MIN_SEC, MAX_SEC = 15, 90              # Shorts length
MAX_PER_CHANNEL = 4                    # Avoid over-representing one creator (applied to the merged file)

# Terms that confirm a video is AI-generated
AI_TERMS = [
    "ai generated", "ai-generated", "made with ai", "aivideo", "aigenerated",
    "veo", "sora", "kling", "aicrashout", "aifruitdrama",
    "aicinema", "aidrama", "aidorama", "aistory",
    "ai drama", "ai story", "ai fruit", "ai video",
]
# Words that mean a human-query result is off-target
HUMAN_BLOCK = [
    "kdrama", "k-drama", "cdrama", "c-drama", "kmovie", "kmovies", "korean drama",
    "chinese drama", "dorama", "manhwa", "minecraft", "reddit", "askreddit", "tts",
    "podcast", "true crime", "stream", "musical", "trailer", "movie", "netflix",
    "sopranos", "breaking bad", "game of thrones", "spongebob", "joke", "jokes", "status",
    "tutorial", "how to", "asmr", "review", "unboxing", "speedrun", "walkthrough",
    "gameplay", "craft", "diy", "slime",
]
# Words that mean an AI-query result is off-target
AI_BLOCK = [
    "art generator", "image generator", "tutorial", "how to", "prompt",
    "free ai", "app", "tool", "review",
]

# Turning Blocked/Key terms into regexes for filtering
def make_regex(terms):
    return re.compile(r"\b(" + "|".join(re.escape(t) for t in terms) + r")\b")

# Filtering out videos that contain any of the blocked terms, or that don't contain any of the AI terms
AI_RE, HUMAN_BLOCK_RE, AI_BLOCK_RE = (
    make_regex(AI_TERMS), make_regex(HUMAN_BLOCK), make_regex(AI_BLOCK))
AI_LOOSE_RE = re.compile(r"\bai\b")


class BudgetReached(Exception):
    """Raised when this run has used its search-call budget."""


LIVE_CALLS = 0  # search.list calls actually sent to the API this run (cached searches are free)


# Saving searches to files (PAGES is part of the key so changing depth never reuses a shallower cache)
def cache_path(q, filters):
    key = json.dumps([q, filters, PAGES], sort_keys=True)
    return OUT / f"cache_{hashlib.md5(key.encode()).hexdigest()[:10]}.json"


# Retrieve ids from YouTube API, caching results to avoid quota issues
def search_ids(q, **filters):
    global LIVE_CALLS
    path = cache_path(q, filters)
    if path.exists():
        return json.loads(path.read_text())
    if LIVE_CALLS + PAGES > MAX_SEARCH_CALLS:
        raise BudgetReached
    ids, token = [], None
    for _ in range(PAGES):
        resp = yt.search().list(
            q=q, part="id", type="video", videoDuration="short",
            maxResults=50, pageToken=token, relevanceLanguage="en", **filters,
        ).execute()
        LIVE_CALLS += 1
        ids += [it["id"]["videoId"] for it in resp["items"]]
        token = resp.get("nextPageToken")
        if not token:
            break
    path.write_text(json.dumps(ids))
    return ids


# Requesting necessary video details based off youtube ids (1 quota unit per 50 videos)
def get_details(ids):
    rows = []
    for k in range(0, len(ids), 50):
        resp = yt.videos().list(
            part="snippet,contentDetails,status",
            id=",".join(ids[k:k + 50]),
        ).execute()
        for v in resp["items"]:
            s = v["snippet"]
            rows.append({
                "video_id": v["id"],
                "url": f"https://www.youtube.com/shorts/{v['id']}",
                "channel": s["channelId"],
                "title": s["title"],
                "description": s.get("description", ""),
                "tags": " ".join(s.get("tags", [])),
                "published": s["publishedAt"],
                "audio_lang": s.get("defaultAudioLanguage") or s.get("defaultLanguage"),
                "duration": isodate.parse_duration(
                    v["contentDetails"]["duration"]).total_seconds(),
                "synthetic_flag": v.get("status", {}).get("containsSyntheticMedia"),
            })
    return rows


# Collecting one query: ids, then details. Returns (rows, stop_reason); stop_reason is None, "quota" or "budget"
def collect(query, query_type, topic, **filters):
    try:
        ids = search_ids(query, **filters)
        got = get_details(ids)
    except BudgetReached:
        return [], "budget"
    except HttpError as e:
        print(f"[{query_type}] {query!r} FAILED: {e}")
        return [], ("quota" if "quota" in str(e).lower() else None)
    for r in got:
        r["source_query"], r["query_type"], r["topic"] = query, query_type, topic
    print(f"[{query_type}/{topic}] {query!r}: {len(ids)} ids -> {len(got)} details")
    return got, None


# Old rows only have label/url/title/source_query. Fetch channel, date and duration for them
# (about 1 quota unit per 50 videos) so the channel cap and group-aware CV work on the whole dataset.
def backfill_metadata(old):
    for col in ["video_id", "channel", "topic", "published", "duration"]:
        if col not in old.columns:
            old[col] = np.nan
    old["video_id"] = old.video_id.fillna(old.url.str.extract(r"(?:shorts/|v=)([\w-]{11})")[0])
    missing = old.loc[old.channel.isna() & old.video_id.notna(), "video_id"].tolist()
    if missing:
        try:
            info_rows = get_details(missing)
        except HttpError as e:
            print(f"Backfill failed, old rows keep missing metadata: {e}")
            info_rows = []
        if info_rows:
            info = pd.DataFrame(info_rows).drop_duplicates("video_id").set_index("video_id")
            for col in ["channel", "published", "duration"]:
                old[col] = old[col].fillna(old.video_id.map(info[col]))
        print(f"Backfilled metadata for {old.channel.notna().sum()} of {len(old)} existing rows")
    old["topic"] = old.topic.fillna("batch1")
    return old


# Actual collection, alternating AI and human queries within each topic pair
all_rows, stop = [], None
for topic, ai_q, human_q in PAIRS:
    for q, qtype, flt in (
        (ai_q, "ai_q", {"publishedAfter": AI_AFTER}),
        (human_q, "human_q", {"publishedBefore": HUMAN_BEFORE}),
    ):
        rows, stop = collect(q, qtype, topic, **flt)
        all_rows += rows
        if stop:
            break
    if stop:
        break

if stop == "quota":
    print("\nQuota exceeded. Cached searches are kept; rerun after the reset (midnight Pacific).")
elif stop == "budget":
    print(f"\nReached this run's budget of {MAX_SEARCH_CALLS} search calls. "
          "Rerun tomorrow (after the midnight Pacific reset) to finish the remaining queries.")
print(f"Live search calls this run: {LIVE_CALLS} (~{LIVE_CALLS * 100} quota units)")

if not all_rows:
    raise SystemExit("No results collected. Check the errors above (key, API enabled, quota).")

# Cleaning and filtering by duration, language, and blocked terms
df = pd.DataFrame(all_rows).drop_duplicates("video_id")
print("\nAfter dedupe, by query type:")
print(df.query_type.value_counts())

df = df[(df.duration >= MIN_SEC) & (df.duration <= MAX_SEC)].copy()
print("After duration filter:", df.query_type.value_counts().to_dict())

df = df[df.audio_lang.isna() | df.audio_lang.astype(str).str.lower().str.startswith("en")].copy()
print("After language filter:", df.query_type.value_counts().to_dict())

df["text"] = (df.title + " " + df.description + " " + df.tags).str.lower()

# Labelling Data
def label(r):
    flagged = pd.notna(r.synthetic_flag) and bool(r.synthetic_flag)
    is_ai = bool(AI_RE.search(r.text)) or flagged or (
        r.query_type == "ai_q" and bool(AI_LOOSE_RE.search(r.text)))
    if is_ai and r.published >= AI_AFTER:
        return "uncertain" if AI_BLOCK_RE.search(r.text) else "ai_candidate"
    if r.query_type == "human_q" and r.published < HUMAN_BEFORE:
        return "uncertain" if HUMAN_BLOCK_RE.search(r.text) else "human_candidate"
    return "uncertain"

# Runs label on every video and drops the "uncertain" ones
df["group"] = df.apply(label, axis=1)
print("\nQuery type vs label:")
print(pd.crosstab(df.query_type, df.group))

df = df[df.group != "uncertain"]
if df.empty:
    raise SystemExit("Nothing left after filtering. Try more queries or looser settings.")

# Shuffle so the channel cap below doesn't favor whichever query ran first
df = df.sample(frac=1, random_state=42)
df["label"] = df.group.map({"ai_candidate": "AI", "human_candidate": "human"})
new = df[COLS]

# Merge with earlier results into one clean csv
if CSV.exists():
    shutil.copy(CSV, OUT / "candidates_backup.csv")
    old = pd.read_csv(CSV)
    if "label" not in old.columns:
        old["label"] = np.where(old.group.str.startswith("ai"), "AI", "human")
    old = backfill_metadata(old)
    fresh = new[~new.url.isin(old.url)]
    print(f"\nExisting rows kept: {len(old)} | new rows added: {len(fresh)}")
    out = pd.concat([old, fresh], ignore_index=True)
else:
    out = new.reset_index(drop=True)

# Channel cap on the merged data (old rows come first, so they're kept over new ones)
out = out.drop_duplicates("url")
out["_ch"] = out.channel.fillna(out.url)
n_before = len(out)
out = out.groupby(["label", "_ch"], sort=False).head(MAX_PER_CHANNEL).drop(columns="_ch")
print(f"Channel cap ({MAX_PER_CHANNEL}) removed {n_before - len(out)} rows")

out = out.reindex(columns=COLS)
out.to_csv(CSV, index=False)

# Summary
print(f"\nSaved {len(out)} rows to {CSV}")
print(out.label.value_counts().to_string())
print("\nTopic balance:")
print(pd.crosstab(out.topic, out.label).to_string())
print(f"\nUnique channels: {out.channel.nunique()}")
print("\nSample AI videos:")
print(out[out.label == "AI"].head(5)[["label", "url", "title", "source_query"]].to_string(index=False))
print("\nSample human videos:")
print(out[out.label == "human"].head(5)[["label", "url", "title", "source_query"]].to_string(index=False))