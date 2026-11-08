"""
Adds ~80 handcrafted features + GPT-2 detector signals and embeddings
on top of the basic features from filtered_feature_generation.py.

Reads:
  data/features_filtered_by_ai.csv   (basic features + video_id + y)
  data/transcripts.csv               (or transcripts_filtered.csv) for the raw text

Writes:
  data/features_and_embeddings.csv   transcript + all handcrafted features + y
  data/gpt2_embeddings.csv           video_id + 768 GPT-2 embedding dimensions

Run from project root:
    python src/embeddings_generation.py

Needs:
    pip install vaderSentiment torch transformers
    (downloads GPT-2 ~500 MB the first time)
"""
import math
import os
import re
import pathlib
from collections import Counter

import numpy as np
import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------------------------------------------------------------------------
# Paths (script lives in src/, data/ is one level up)
# ---------------------------------------------------------------------------
DATA = pathlib.Path(__file__).resolve().parent.parent / "data"
FEATURES_IN = DATA / "features_filtered_by_ai.csv"
FEATURES_OUT = DATA / "features_and_embeddings.csv"
EMBEDDINGS_OUT = DATA / "gpt2_embeddings.csv"

# Prefer the filtered transcripts file if it exists, otherwise the full one
if (DATA / "transcripts_filtered.csv").exists():
    TRANSCRIPTS = DATA / "transcripts_filtered.csv"
else:
    TRANSCRIPTS = DATA / "transcripts.csv"

vader = SentimentIntensityAnalyzer()
LEXICON = vader.lexicon
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
gpt2 = GPT2LMHeadModel.from_pretrained("gpt2", output_hidden_states=True).eval()


def words(*items):
    """Regex that matches any of the given words/phrases as whole words."""
    return re.compile(r"\b(?:" + "|".join(items) + r")\b", re.I)


def starts(pattern):
    """Regex that matches at the start of a word (so stems like 'jealous' also match 'jealousy')."""
    return re.compile(r"\b(?:" + pattern + r")", re.I)


# ---------------------------------------------------------------------------
# 11 word families (exact lists from cole_features_simple.py)
# ---------------------------------------------------------------------------
SIMPLE_FAMILIES = {
    # --- spoken vs. written style ---
    "discourse_markers": words(
        "oh", "yeah", "well", "okay", "ok", "so", "hey", "hi", "right", "alright", "all right",
        "you know", "i mean", "actually", "anyway", "um", "uh", "i guess", "like",
    ),
    "casual_verbs": words(
        "got", "gotta", "gonna", "wanna", "get", "gets", "getting", "trying", "tryna", "come on",
        "kinda", "sorta", "ain't", "y'all", "lemme", "gimme", "dunno",
    ),
    "slang": words(
        "gonna", "wanna", "gotta", "kinda", "sorta", "ain't", "y'all", "bro", "bruh", "dude", "guys",
        "lol", "nah", "yo", "dawg", "sus", "lowkey", "deadass", "cuz", "ya", "yup", "lemme", "gimme",
        "dunno", "whatcha", "imma",
    ),
    "subordinators": words(
        "who", "whom", "whose", "which", "while", "before", "after", "until", "because",
        "although", "though", "since", "unless", "whenever", "wherever", "whereas", "once",
    ),
    "time_connectives": words(
        "then", "before", "after", "later", "finally", "until", "while", "once", "suddenly",
        "years", "that night", "the next day", "one day", "since",
    ),
    # --- story content ---
    "reveal_words": words(
        "truth", "secret", "secrets", "hiding", "hide", "hid", "hidden", "kept", "explain",
        "lie", "lied", "lying", "found out", "find out", "discover", "discovered", "knew",
        "realized", "realize", "confess", "admit",
    ),
    "absolutist": words(
        "absolutely", "all", "always", "complete", "completely", "constant", "constantly",
        "definitely", "entire", "entirely", "ever", "every", "everyone", "everybody", "everything",
        "full", "fully", "must", "never", "nothing", "nobody", "no one", "totally", "whole", "forever",
    ),
    "kinship": words(
        "mother", "father", "mom", "dad", "mommy", "daddy", "son", "daughter", "sister", "brother",
        "husband", "wife", "grandma", "grandpa", "grandmother", "grandfather", "family", "parents",
        "aunt", "uncle",
    ),
    "future_modals": words("will", "won't", "shall", "going to", "must", "promise"),
    "love_talk": words(
        "i love you", "love you", "my love", "darling", "sweetheart", "honey", "babe", "kiss",
        "marry me", "soulmate", "i miss you", "beautiful", "handsome",
    ),
    "apology": words(
        "i'm sorry", "im sorry", "i am sorry", "sorry", "forgive me", "my fault", "i didn't mean",
        "i shouldn't have", "i messed up",
    ),
}

# ---------------------------------------------------------------------------
# Word lists used by the extra features
# ---------------------------------------------------------------------------
URGENCY = {"tonight", "today", "tomorrow", "now", "forever", "yesterday", "soon", "again", "finally"}
IMPERATIVE_STARTS = {
    "go", "come", "look", "listen", "wait", "stop", "give", "get", "tell", "let",
    "take", "don't", "run", "help", "please", "leave", "relax", "hold", "watch",
    "shut", "move", "stay",
}
THIRD_PERSON = {
    "he", "she", "they", "him", "her", "them", "his", "hers", "their",
    "he's", "she's", "they're", "himself", "herself",
}
FIRST_SECOND = {
    "i", "me", "my", "you", "your", "i'm", "you're", "mine", "yours",
    "myself", "yourself",
}
PAST_TENSE_VERBS = {
    "was", "were", "had", "said", "told", "went", "came", "saw", "knew", "took",
    "felt", "thought", "found", "left", "got", "made",
}
STOPWORDS = set("""
a an the and or but so to of in on at for with is are was were be been am i you he she it we they me
him her us them my your his its our their this that these those do does did not no yes oh what why how who where
when just can will would could should have has had there here then than if as up out down off all i'm you're it's
don't can't that's okay ok yeah hey well now go get got know like want come one let's gonna really
""".split())
NOT_NAMES = {
    "The", "And", "But", "You", "What", "Why", "How", "Oh", "Okay", "God", "Mom", "Dad",
    "Yes", "Hey", "This", "That", "Come", "Wait", "Please", "Mr", "Mrs", "Miss", "Sir",
    "Well", "So", "No", "I'm", "It's",
}

BLAME = starts(
    r"how could you|it was you|it's your fault|your fault|you|you did this|you lied|you cheated|"
    r"you betrayed|you ruined|you knew|why did you|you never|you always|because of you|"
    r"you stole|you broke|blame"
)
SOAP = starts(
    r"jealous\w*|envy|envious|who is (she|he)|who's (she|he)|is that your|"
    r"stay away from (him|her|my)|embarrass\w*|humiliat\w*|ashamed|shame|"
    r"everyone (is )?laughing|disgrace|secret|the truth|i know what you did|"
    r"i saw (you|it|them)|you knew|found out|caught|all along|turns out|i have to tell you"
)
CLICHE = starts(
    r"i can explain|it's not what it looks like|we need to talk|this isn't over|"
    r"you'll regret this|how could you|i trusted you|after everything|you mean nothing|"
    r"i never want to see you|get out of my (house|life)|you don't deserve|"
    r"i should have known|it was all a lie|you're nothing|i've always loved you|"
    r"i can't do this anymore|let me explain|this changes everything|what have you done"
)
CONFLICT = starts(
    r"cheat\w*|li(e|ed|ar|ying)\b|secret|betray\w*|kill\w*|dead\b|die\b|steal|stole|"
    r"how dare|how could you|it was you|why did you|you knew|caught|pregnant|divorce|"
    r"affair|truth|hate|leave me|get out|stop|no way"
)
NARRATOR_VOCAB = starts(
    r"whisper\w*|realiz\w*|smil\w*|tears?|heart\w*|eyes|suddenly|little did|"
    r"from that day|never forget|couldn't believe|took a deep breath|froze|"
    r"gasp\w*|trembl\w*|glanc\w*|stared|murmur\w*|nodded|sigh\w*"
)
VIRTUE = starts(
    r"kindness|kind|courage|strength|brave\w*|hope|dream\w*|journey|lesson|"
    r"believe|grateful|gratitude|compassion|respect|forgive\w*|"
    r"true (friend|love|strength)|matter\w*|worth|deserve\w*|proud"
)
HOOK = starts(
    r"you need to watch|watch (this|till|until) the end|wait for it|"
    r"this is the story|this is (\w+)'s story|imagine|what would you do|pov|"
    r"story time|let me tell you|you won't believe|nobody expected|"
    r"if you've ever|have you ever"
)
MORAL = starts(
    r"lesson|remember|never judge|be kind|always be|the moral|kindness|"
    r"true (strength|love|friend)|matters? most|what really matters|that's why|"
    r"in the end|from that day|never forget|you never know|be grateful|"
    r"treat (people|others)"
)

PLOT_BOMBS = [
    starts(p) for p in [
        r"pregnant|baby|the child|my child|your child",
        r"marry|married|wedding|engaged|propose|husband|wife",
        r"break up|broke up|divorce|leave me|i'm leaving|we're done|it's over",
        r"cheat\w*|affair|someone else|another (man|woman)",
        r"\bdie\b|died|dead|death|kill\w*|funeral|murder\w*",
        r"rich|poor|money|billion\w*|million\w*|fired|job|ceo|inherit\w*",
        r"arrest\w*|police|steal|stole|prison|jail",
        r"hospital|sick|accident|doctor|surgery|cancer",
        r"real (father|mother|dad|mom)|adopted|twin|disguise|who you really|i'm actually|you're actually",
        r"secret|the truth|found out|i know what you did",
    ]
]

MORE_FAMILIES = {
    "absolutist": SIMPLE_FAMILIES["absolutist"],
    "audience_address": words(
        "guys", "you guys", "everyone", "hello", "hi", "welcome", "subscribe",
        "comment", "follow", "today we", "let's go", "ready", "check this out", "watch this",
    ),
    "cognitive": words(
        "think", "thought", "know", "knew", "realize", "realized", "understand",
        "understood", "believe", "remember", "wonder", "mean", "meant", "decide", "decided", "figure",
    ),
    "perception": words(
        "see", "saw", "seen", "look", "looked", "looking", "hear", "heard", "feel",
        "felt", "watch", "watched", "listen", "smell", "touch",
    ),
    "certainty": words(
        "sure", "definitely", "certainly", "of course", "obviously", "clearly",
        "absolutely", "exactly",
    ),
    "hedges": words(
        "maybe", "probably", "perhaps", "kind of", "sort of", "i guess", "i think",
        "might", "possibly",
    ),
    "body": words(
        "hand", "hands", "eyes", "face", "head", "heart", "arms", "arm", "hair",
        "lips", "voice", "chest", "tears", "fingers", "shoulder",
    ),
    "number_words": words(
        r"\d+", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "hundred", "thousand", "million", "first", "second", "third", "dollars",
    ),
    "politeness": words("please", "thank you", "thanks", "excuse me", "pardon"),
    "intensifiers": words(
        "very", "really", "so", "too", "totally", "super", "extremely", "literally", "such",
    ),
    "negative_contractions": words(
        "isn't", "aren't", "wasn't", "weren't", "don't", "doesn't", "didn't",
        "won't", "can't", "couldn't", "shouldn't", "wouldn't", "haven't", "hasn't",
    ),
    "conjunction_starts": re.compile(r"(?:^|[.!?]\s+)(?:and|but|so|or|because)\b", re.I),
    "wh_questions": re.compile(r"\b(?:what|why|how|who|where|when|which)\b[^.!?]*\?", re.I),
}

PRONOUNS = {
    "i_rate": {"i", "me", "my", "mine", "myself", "i'm", "i've", "i'll", "i'd"},
    "you_rate": {"you", "your", "yours", "yourself", "you're", "you've", "you'll", "you'd"},
    "he_rate": {"he", "him", "his", "himself", "he's"},
    "she_rate": {"she", "her", "hers", "herself", "she's"},
    "they_rate": {"they", "them", "their", "theirs", "themselves", "they're"},
}


def tokenize(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)*", text.lower())


def split_sentences(text):
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+", text.strip()) if tokenize(p)] or [text]


def syllables(word):
    groups = re.findall(r"[aeiouy]+", word)
    return max(1, len(groups) - (1 if word.endswith("e") and len(groups) > 1 and not word.endswith("le") else 0))


def gpt2_features(text):
    """GPT-2 reads the transcript; returns detector signals and the mean final-layer embedding."""
    ids = tokenizer(text, return_tensors="pt").input_ids[:, :1023]
    ids = torch.cat([torch.tensor([[tokenizer.bos_token_id]]), ids], dim=1)
    with torch.no_grad():
        out = gpt2(ids)
    logp = torch.log_softmax(out.logits[0, :-1], -1)
    target = ids[0, 1:]
    target_logp = logp[torch.arange(len(target)), target]
    surprisal = (-target_logp / math.log(2)).numpy()                 # bits per token
    rank = (logp > target_logp[:, None]).sum(1).numpy() + 1          # 1 = GPT-2's top guess
    entropy = (-(logp.exp() * logp).sum(1) / math.log(2)).numpy()
    feats = {
        "gpt2_surprisal": surprisal.mean(),
        "gpt2_low_surprise_share": (surprisal < 2).mean(),
        "gpt2_surprisal_sd": surprisal.std(),
        "gltr_top1": np.mean(rank == 1),
        "gltr_top10": np.mean(rank <= 10),
        "gltr_top100": np.mean(rank <= 100),
        "gltr_over1000": np.mean(rank > 1000),
        "gpt2_log_rank": np.mean(np.log(rank)),
        "gpt2_entropy": entropy.mean(),
        "gpt2_surprise_minus_entropy": surprisal.mean() - entropy.mean(),
    }
    return feats, out.hidden_states[-1][0, 1:].mean(0).numpy()


def extract(transcript):
    text = transcript.replace("’", "'")
    tokens = tokenize(text)
    n = len(tokens) or 1
    sents = split_sentences(text)
    k = len(sents)
    lengths = np.array([len(tokenize(s)) for s in sents], float)
    pos = (np.cumsum(lengths) - lengths / 2) / lengths.sum()          # middle of each sentence, 0-1
    first, last = pos < 1 / 3, pos >= 2 / 3
    mood = np.array([vader.polarity_scores(s)["compound"] for s in sents])
    intensity = np.abs(mood)
    per100 = lambda pattern: 100 * len(pattern.findall(text)) / n

    # escalations: times emotional intensity sets a new high
    running, raises = 0.0, 0
    for v in intensity:
        if v > running + 0.1:
            raises, running = raises + 1, v

    # topic jumps: gaps where the 3 sentences before and after share no content words
    content = [set(w for w in tokenize(s) if w not in STOPWORDS) for s in sents]
    overlaps = []
    for i in range(1, k):
        before = set().union(*content[max(0, i - 3):i])
        after = set().union(*content[i:i + 3])
        if before and after:
            overlaps.append(len(before & after) / len(before | after))

    names = {m.group(1) for m in re.finditer(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-z]{2,})\b", text)} - NOT_NAMES
    capitalized = Counter(m.group(0) for m in re.finditer(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-z]{2,}\b", text))
    syl = [syllables(w) for w in tokens] or [1]
    valences = [LEXICON[w] for w in tokens if w in LEXICON]
    openers = [tokenize(s)[0] for s in sents if tokenize(s)]
    ending = mood[last].mean() if last.any() else mood[-1]

    f = {
        # 11 word families
        **{f"{name}_per100": 100 * len(p.findall(text)) / n for name, p in SIMPLE_FAMILIES.items()},
        # story structure
        "climax_position": pos[int(intensity.argmax())],
        "conflict_position": next((pos[i] for i, s in enumerate(sents) if CONFLICT.search(s)), 1.0),
        "plot_bombs": sum(bool(p.search(text)) for p in PLOT_BOMBS),
        "topic_jump_share": np.mean(np.array(overlaps) == 0) if overlaps else 0.5,
        "stakes_raises": raises / k,
        "tempo_change": lengths[last].mean() / lengths[first].mean() if last.any() and first.any() else 1.0,
        "monologue_share": lengths.max() / lengths.sum(),
        "imperative_share": sum(tokenize(s)[0] in IMPERATIVE_STARTS for s in sents) / k,
        "n_named_characters": len(names),
        "top_name_repeats": max(capitalized.values()) / k if capitalized else 0,
        "distinct_capitalized": len(capitalized),
        # dialogue moves
        "blame_per100": per100(BLAME),
        "soap_emotion_per100": per100(SOAP),
        "cliche_per100": per100(CLICHE),
        "urgency_rate": sum(w in URGENCY for w in tokens) / n,
        "word_runs_per100": 100 * (
            sum(a == b for a, b in zip(tokens, tokens[1:]))
            + len(re.findall(r"\b\w{1,3}-(\s|$)|\w—|\.\.\.", text))
        ) / n,
        # emotional arc
        "sent_mean": mood.mean(),
        "sent_intensity": intensity.mean(),
        "sent_swing": np.abs(np.diff(mood)).mean() if k > 1 else 0.0,
        "ending_sentiment": ending,
        "arc_rise": ending - (mood[first].mean() if first.any() else mood[0]),
        "vader_pos_word_rate": 100 * sum(v > 0 for v in valences) / n,
        "vader_neg_word_rate": 100 * sum(v < 0 for v in valences) / n,
        "vader_extreme_word_rate": 100 * sum(abs(v) >= 2.5 for v in valences) / n,
        "vader_mean_valence": np.mean(valences) if valences else 0,
        # narration style
        "narr_vocab_per100": per100(NARRATOR_VOCAB),
        "virtue_per100": per100(VIRTUE),
        "hook_opening": int(bool(HOOK.search(" ".join(sents[:2])))),
        "moral_ending": int(bool(MORAL.search(" ".join(sents[-2:])))),
        "third_person_ratio": sum(w in THIRD_PERSON for w in tokens) / max(1, sum(w in FIRST_SECOND for w in tokens)),
        "past_tense_rate": (
            sum(w.endswith("ed") and len(w) > 4 for w in tokens) / n
            + sum(w in PAST_TENSE_VERBS for w in tokens) / n
        ),
        # readability and punctuation
        "syllables_per_word": np.mean(syl),
        "polysyllable_rate": 100 * np.mean([s >= 3 for s in syl]),
        "flesch_reading_ease": 206.835 - 1.015 * (n / k) - 84.6 * np.mean(syl),
        "fk_grade": 0.39 * (n / k) + 11.8 * np.mean(syl) - 15.59,
        "comma_per_sentence": text.count(",") / k,
        "comma_per100": 100 * text.count(",") / n,
        "apostrophe_per100": 100 * text.count("'") / n,
        "short_word_share": np.mean([len(w) <= 3 for w in tokens]) if tokens else 0,
        # sentence openings
        "first_word_diversity": len(set(openers)) / max(1, len(openers)),
        "starts_with_i_share": np.mean([w in {"i", "i'm", "i've", "i'll"} for w in openers]) if openers else 0,
        "starts_with_you_share": np.mean([w in {"you", "you're"} for w in openers]) if openers else 0,
        # more word families and pronouns
        **{f"{name}_per100": 100 * len(p.findall(text)) / n for name, p in MORE_FAMILIES.items()},
        **{name: 100 * sum(w in group for w in tokens) / n for name, group in PRONOUNS.items()},
    }
    g, embedding = gpt2_features(text)
    f.update(g)
    return f, embedding


def main():
    if not FEATURES_IN.exists():
        raise SystemExit(
            f"Could not find {FEATURES_IN}.\n"
            "Run filtered_feature_generation.py first to create features_filtered_by_ai.csv."
        )
    if not TRANSCRIPTS.exists():
        raise SystemExit(
            f"Could not find {TRANSCRIPTS}.\n"
            "Need a transcripts file with at least video_id and transcript columns."
        )

    features = pd.read_csv(FEATURES_IN)
    transcripts = pd.read_csv(TRANSCRIPTS)

    if "video_id" not in features.columns:
        raise SystemExit(f"{FEATURES_IN} is missing a video_id column.")
    if "video_id" not in transcripts.columns or "transcript" not in transcripts.columns:
        raise SystemExit(f"{TRANSCRIPTS} must contain video_id and transcript columns.")

    # Keep only rows that still have a usable transcript
    transcripts = transcripts.dropna(subset=["transcript"])
    transcripts["transcript"] = transcripts["transcript"].astype(str)
    transcripts = transcripts[transcripts["transcript"].str.strip() != ""]

    # Join on video_id (order-independent)
    merged = features.merge(
        transcripts[["video_id", "transcript"]],
        on="video_id",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) == 0:
        raise SystemExit("No overlapping video_ids between features and transcripts.")
    if len(merged) < len(features):
        print(f"Warning: dropped {len(features) - len(merged)} feature rows with no matching transcript.")

    print(f"Computing extra features + GPT-2 embeddings for {len(merged)} transcripts...")
    rows, embeddings = zip(*[extract(t) for t in merged["transcript"]])
    extra = pd.DataFrame(list(rows))

    # Build output in the same style as the existing features_and_embeddings.csv:
    # transcript first, then the basic features (drop video_id / label), then the new ones, then y
    base_cols = [c for c in features.columns if c not in ("video_id", "label", "y")]
    out = pd.concat(
        [
            merged[["transcript"]].reset_index(drop=True),
            merged[base_cols].reset_index(drop=True),
            extra.reset_index(drop=True),
            merged[["y"]].reset_index(drop=True),
        ],
        axis=1,
    )
    out.to_csv(FEATURES_OUT, index=False)

    emb = pd.DataFrame(
        np.array(embeddings).round(6),
        columns=[f"gpt2_emb_{i}" for i in range(768)],
    )
    emb.insert(0, "video_id", merged["video_id"].values)
    emb.to_csv(EMBEDDINGS_OUT, index=False)

    print(f"Saved {len(out)} rows x {out.shape[1]} columns -> {FEATURES_OUT}")
    print(f"Saved {emb.shape[0]} x 768 GPT-2 embeddings -> {EMBEDDINGS_OUT}")


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    main()