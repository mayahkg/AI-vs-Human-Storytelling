"""
Max version (Cole): adds 80 handcrafted features to data/features.csv and saves GPT-2 embeddings.

Writes (data/features.csv itself is not changed):
  data/features_cole_max.csv      original 36 features + 80 new ones (116 handcrafted features)
  data/gpt2_embeddings_cole.csv   768 GPT-2 embedding numbers per transcript

Run from anywhere:  python src/cole_features_max.py
Needs: pip install vaderSentiment torch transformers   (downloads GPT-2, ~500 MB, the first time)
Both output files are committed, so notebooks/model_cole_max.ipynb runs without any of this.

The 80 new features, all computed on the original transcripts in data/transcripts_filtered.csv:
  - the 11 word families from the simple version (src/cole_features_simple.py)
  - story structure: climax position, conflict position, plot bombs, topic jumps, stakes raises, tempo change, ...
  - emotional arc (VADER sentiment per sentence): average mood, swings, ending mood, rise
  - narration style: past tense, third person, narrator vocabulary, hook openings, moral endings
  - readability, punctuation, pronouns, sentence openings, more word families
  - GPT-2 "AI detector" signals: how predictable each word is to GPT-2 (surprisal, rank, entropy)
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

from cole_features_simple import FAMILIES as SIMPLE_FAMILIES, words

# File paths (anchored to this script's folder)
DATA = pathlib.Path(__file__).parent.parent / "data"
TRANSCRIPTS = DATA / "transcripts_filtered.csv"
FEATURES_IN = DATA / "features.csv"
FEATURES_OUT = DATA / "features_cole_max.csv"
EMBEDDINGS_OUT = DATA / "gpt2_embeddings_cole.csv"

vader = SentimentIntensityAnalyzer()
LEXICON = vader.lexicon
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
gpt2 = GPT2LMHeadModel.from_pretrained("gpt2", output_hidden_states=True).eval()


def starts(pattern):
    """Regex that matches at the start of a word (so stems like 'jealous' also match 'jealousy')."""
    return re.compile(r"\b(?:" + pattern + r")", re.I)


# ---------------------------------------------------------------------------
# Word lists (edit freely)
# ---------------------------------------------------------------------------
URGENCY = {"tonight", "today", "tomorrow", "now", "forever", "yesterday", "soon", "again", "finally"}
IMPERATIVE_STARTS = {"go", "come", "look", "listen", "wait", "stop", "give", "get", "tell", "let", "take", "don't",
                     "run", "help", "please", "leave", "relax", "hold", "watch", "shut", "move", "stay"}
THIRD_PERSON = {"he", "she", "they", "him", "her", "them", "his", "hers", "their", "he's", "she's", "they're",
                "himself", "herself"}
FIRST_SECOND = {"i", "me", "my", "you", "your", "i'm", "you're", "mine", "yours", "myself", "yourself"}
PAST_TENSE_VERBS = {"was", "were", "had", "said", "told", "went", "came", "saw", "knew", "took", "felt", "thought",
                    "found", "left", "got", "made"}
STOPWORDS = set("""a an the and or but so to of in on at for with is are was were be been am i you he she it we they me
him her us them my your his its our their this that these those do does did not no yes oh what why how who where
when just can will would could should have has had there here then than if as up out down off all i'm you're it's
don't can't that's okay ok yeah hey well now go get got know like want come one let's gonna really""".split())
NOT_NAMES = {"The", "And", "But", "You", "What", "Why", "How", "Oh", "Okay", "God", "Mom", "Dad", "Yes", "Hey", "This",
             "That", "Come", "Wait", "Please", "Mr", "Mrs", "Miss", "Sir", "Well", "So", "No", "I'm", "It's"}

BLAME = starts(r"how could you|it was you|it's your fault|your fault|you did this|you lied|you cheated|you betrayed|"
               r"you ruined|you knew|why did you|you never|you always|because of you|you stole|you broke|blame")
SOAP = starts(r"jealous\w*|envy|envious|who is (she|he)|who's (she|he)|is that your|stay away from (him|her|my)|"
              r"embarrass\w*|humiliat\w*|ashamed|shame|everyone (is )?laughing|disgrace|"
              r"secret|the truth|i know what you did|i saw (you|it|them)|you knew|found out|caught|all along|"
              r"turns out|i have to tell you")
CLICHE = starts(r"i can explain|it's not what it looks like|we need to talk|this isn't over|you'll regret this|"
                r"how could you|i trusted you|after everything|you mean nothing|i never want to see you|"
                r"get out of my (house|life)|you don't deserve|i should have known|it was all a lie|you're nothing|"
                r"i've always loved you|i can't do this anymore|let me explain|this changes everything|what have you done")
CONFLICT = starts(r"cheat\w*|li(e|ed|ar|ying)\b|secret|betray\w*|kill\w*|dead\b|die\b|steal|stole|how dare|"
                  r"how could you|it was you|why did you|you knew|caught|pregnant|divorce|affair|truth|hate|leave me|"
                  r"get out|stop|no way")
NARRATOR_VOCAB = starts(r"whisper\w*|realiz\w*|smil\w*|tears?|heart\w*|eyes|suddenly|little did|from that day|"
                        r"never forget|couldn't believe|took a deep breath|froze|gasp\w*|trembl\w*|glanc\w*|stared|"
                        r"murmur\w*|nodded|sigh\w*")
VIRTUE = starts(r"kindness|kind|courage|strength|brave\w*|hope|dream\w*|journey|lesson|believe|grateful|gratitude|"
                r"compassion|respect|forgive\w*|true (friend|love|strength)|matter\w*|worth|deserve\w*|proud")
HOOK = starts(r"you need to watch|watch (this|till|until) the end|wait for it|this is the story|this is (\w+)'s story|"
              r"imagine|what would you do|pov|story time|let me tell you|you won't believe|nobody expected|"
              r"if you've ever|have you ever")
MORAL = starts(r"lesson|remember|never judge|be kind|always be|the moral|kindness|true (strength|love|friend)|"
               r"matters? most|what really matters|that's why|in the end|from that day|never forget|"
               r"you never know|be grateful|treat (people|others)")
# big life events a story can pivot on ("plot bombs")
PLOT_BOMBS = [starts(p) for p in [
    r"pregnant|baby|the child|my child|your child", r"marry|married|wedding|engaged|propose|husband|wife",
    r"break up|broke up|divorce|leave me|i'm leaving|we're done|it's over", r"cheat\w*|affair|someone else|another (man|woman)",
    r"\bdie\b|died|dead|death|kill\w*|funeral|murder\w*", r"rich|poor|money|billion\w*|million\w*|fired|job|ceo|inherit\w*",
    r"arrest\w*|police|steal|stole|prison|jail", r"hospital|sick|accident|doctor|surgery|cancer",
    r"real (father|mother|dad|mom)|adopted|twin|disguise|who you really|i'm actually|you're actually",
    r"secret|the truth|found out|i know what you did"]]
MORE_FAMILIES = {
    "absolutist": SIMPLE_FAMILIES["absolutist"],
    "audience_address": words("guys", "you guys", "everyone", "hello", "hi", "welcome", "subscribe", "comment",
                              "follow", "today we", "let's go", "ready", "check this out", "watch this"),
    "cognitive": words("think", "thought", "know", "knew", "realize", "realized", "understand", "understood", "believe",
                       "remember", "wonder", "mean", "meant", "decide", "decided", "figure"),
    "perception": words("see", "saw", "seen", "look", "looked", "looking", "hear", "heard", "feel", "felt", "watch",
                        "watched", "listen", "smell", "touch"),
    "certainty": words("sure", "definitely", "certainly", "of course", "obviously", "clearly", "absolutely", "exactly"),
    "hedges": words("maybe", "probably", "perhaps", "kind of", "sort of", "i guess", "i think", "might", "possibly"),
    "body": words("hand", "hands", "eyes", "face", "head", "heart", "arms", "arm", "hair", "lips", "voice", "chest",
                  "tears", "fingers", "shoulder"),
    "number_words": words(r"\d+", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                          "hundred", "thousand", "million", "first", "second", "third", "dollars"),
    "politeness": words("please", "thank you", "thanks", "excuse me", "pardon"),
    "intensifiers": words("very", "really", "so", "too", "totally", "super", "extremely", "literally", "such"),
    "negative_contractions": words("isn't", "aren't", "wasn't", "weren't", "don't", "doesn't", "didn't", "won't",
                                   "can't", "couldn't", "shouldn't", "wouldn't", "haven't", "hasn't"),
    "conjunction_starts": re.compile(r"(?:^|[.!?]\s+)(?:and|but|so|or|because)\b", re.I),
    "wh_questions": re.compile(r"\b(?:what|why|how|who|where|when|which)\b[^.!?]*\?", re.I),
}
PRONOUNS = {"i_rate": {"i", "me", "my", "mine", "myself", "i'm", "i've", "i'll", "i'd"},
            "you_rate": {"you", "your", "yours", "yourself", "you're", "you've", "you'll", "you'd"},
            "he_rate": {"he", "him", "his", "himself", "he's"},
            "she_rate": {"she", "her", "hers", "herself", "she's"},
            "they_rate": {"they", "them", "their", "theirs", "themselves", "they're"}}


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
    pos = (np.cumsum(lengths) - lengths / 2) / lengths.sum()          # middle of each sentence, 0-1 through the clip
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
        before, after = set().union(*content[max(0, i - 3):i]), set().union(*content[i:i + 3])
        if before and after:
            overlaps.append(len(before & after) / len(before | after))

    names = {m.group(1) for m in re.finditer(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-z]{2,})\b", text)} - NOT_NAMES
    capitalized = Counter(m.group(0) for m in re.finditer(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-z]{2,}\b", text))
    syl = [syllables(w) for w in tokens] or [1]
    valences = [LEXICON[w] for w in tokens if w in LEXICON]
    openers = [tokenize(s)[0] for s in sents if tokenize(s)]
    ending = mood[last].mean() if last.any() else mood[-1]

    f = {
        # simple-version word families
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
        "word_runs_per100": 100 * (sum(a == b for a, b in zip(tokens, tokens[1:]))
                                   + len(re.findall(r"\b\w{1,3}-(\s|$)|\w—|\.\.\.", text))) / n,
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
        "past_tense_rate": sum(w.endswith("ed") and len(w) > 4 for w in tokens) / n
                           + sum(w in PAST_TENSE_VERBS for w in tokens) / n,
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
    features = pd.read_csv(FEATURES_IN, float_precision="round_trip")  # keep the original values digit-for-digit
    transcripts = pd.read_csv(TRANSCRIPTS)

    # features.csv rows are in the same order as transcripts_filtered.csv; make sure before joining
    norm = lambda s: s.str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
    assert len(features) == len(transcripts) and (norm(features.transcript).values == norm(transcripts.transcript).values).all(), \
        "features.csv and transcripts_filtered.csv are not in the same row order"

    rows, embeddings = zip(*[extract(t) for t in transcripts.transcript])
    extra = pd.DataFrame(list(rows))
    out = pd.concat([features.drop(columns="y"), extra, features[["y"]]], axis=1)
    out.to_csv(FEATURES_OUT, index=False)

    emb = pd.DataFrame(np.array(embeddings).round(6), columns=[f"gpt2_emb_{i}" for i in range(768)])
    emb.insert(0, "video_id", transcripts.video_id.values)
    emb.to_csv(EMBEDDINGS_OUT, index=False)

    print(f"Saved {len(out)} rows, {extra.shape[1]} new features ({out.shape[1] - 2} handcrafted total) -> {FEATURES_OUT}")
    print(f"Saved {emb.shape[0]} x 768 GPT-2 embeddings -> {EMBEDDINGS_OUT}")


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    main()
