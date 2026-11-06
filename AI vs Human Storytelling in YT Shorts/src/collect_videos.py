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

# Search Quries
HUMAN_QUERIES = [
    "drama skit",
    "family betrayal drama scene",
    "pov drama skit actress",
    "cheating boyfriend skit acting",
    "family drama skit acting",
    "mini drama skit original",
    "short film drama emotional scene",
    "stop motion drama story",
    "claymation drama skit",
    "puppet drama skit",
    "barbie doll drama skit",
    "toy drama story skit",
    "sylvanian families drama",
    "animated food characters drama",
    "2d animated drama story voice acted",
]
AI_QUERIES = [
    "ai fruit drama",
    "ai mini drama",
    "ai generated drama story",
    "ai fruit story episode",
    "ai animal drama",
    "ai baby drama",
    "ai love story episode english",
    "ai generated story drama",
    "ai drama episode",
    "veo 3 drama scene",
    "ai generated realistic drama people",
    "ai cheating story drama",
    "ai revenge story",
    "ai mother in law drama",
    "ai family drama story",
    "ai office drama story",
    "ai husband wife argument",
    "ai romance short film",
    "ai cartoon drama story",
    "sora short story drama",
    "ai horror short story",
]

# API Search Settings
PAGES = 3
HUMAN_BEFORE = "2023-01-10T00:00:00Z"  # Only Shorts from before AI was widely available
AI_AFTER = "2025-01-01T00:00:00Z"
MIN_SEC, MAX_SEC = 15, 90              # Shorts length
MAX_PER_CHANNEL = 4                    # Avoid over-representing one creator

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

# Saving Searches to files
def cache_path(q, filters):
    key = json.dumps([q, filters], sort_keys=True)
    return OUT / f"cache_{hashlib.md5(key.encode()).hexdigest()[:10]}.json"

# Retreive ids from YouTube API, caching results to avoid quota issues
def search_ids(q, **filters):
    path = cache_path(q, filters)
    if path.exists():
        return json.loads(path.read_text())
    ids, token = [], None
    for _ in range(PAGES):
        resp = yt.search().list(
            q=q, part="id", type="video", videoDuration="short",
            maxResults=50, pageToken=token, relevanceLanguage="en", **filters,
        ).execute()
        ids += [it["id"]["videoId"] for it in resp["items"]]
        token = resp.get("nextPageToken")
        if not token:
            break
    path.write_text(json.dumps(ids))
    return ids

# Requesting necessary video details based off youtube ids
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

# Collecting all the results for a set of queries, collecting the ids and then the details, which will be stored in jsons
def collect(queries, query_type, **filters):
    rows, quota_hit = [], False
    for q in queries:
        try:
            ids = search_ids(q, **filters)
            got = get_details(ids)
            for r in got:
                r["source_query"], r["query_type"] = q, query_type
            rows += got
            print(f"[{query_type}] {q!r}: {len(ids)} ids -> {len(got)} details")
        except HttpError as e:
            print(f"[{query_type}] {q!r} FAILED: {e}")
            if "quota" in str(e).lower():
                print("Quota exceeded. Cached searches are kept; rerun after the reset (midnight Pacific).")
                quota_hit = True
                break
    return rows, quota_hit

# Actual AI Collection
all_rows = []
ai_rows, quota_hit = collect(AI_QUERIES, "ai_q", publishedAfter=AI_AFTER)
all_rows += ai_rows
if not quota_hit:
    human_rows, quota_hit = collect(HUMAN_QUERIES, "human_q", publishedBefore=HUMAN_BEFORE)
    all_rows += human_rows
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

# Runs label on every video, droping the "uncertain" ones, preventing certain channels from dominating
df["group"] = df.apply(label, axis=1)
print("\nQuery type vs label:")
print(pd.crosstab(df.query_type, df.group))

df = df[df.group != "uncertain"]
if df.empty:
    raise SystemExit("Nothing left after filtering. Try more queries or looser settings.")

df = (df.sample(frac=1, random_state=42)
        .groupby(["group", "channel"]).head(MAX_PER_CHANNEL))

# Keep label, link, title, and the query that found it
df["label"] = df.group.map({"ai_candidate": "AI", "human_candidate": "human"})
new = df[["label", "url", "title", "source_query"]]

# Merge with earlier results into one clean csv
if CSV.exists():
    shutil.copy(CSV, OUT / "candidates_backup.csv")
    old = pd.read_csv(CSV)
    if "label" not in old.columns:
        old["label"] = np.where(old.group.str.startswith("ai"), "AI", "human")
    fresh = new[~new.url.isin(old.url)]
    print(f"\nExisting rows kept: {len(old)} | new rows added: {len(fresh)}")
    out = pd.concat([old, fresh], ignore_index=True)
else:
    out = new.reset_index(drop=True)

out = out.drop_duplicates("url")
out = out[["label", "url", "title", "source_query"]]
out.to_csv(CSV, index=False)

# Summary
print(f"\nSaved {len(out)} rows to {CSV}")
print(out.label.value_counts().to_string())
print("\nSample AI videos:")
print(out[out.label == "AI"].head(5).to_string(index=False))
print("\nSample human videos:")
print(out[out.label == "human"].head(5).to_string(index=False))