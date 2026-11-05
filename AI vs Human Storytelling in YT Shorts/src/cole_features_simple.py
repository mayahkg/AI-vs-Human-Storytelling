"""
Simple version (Cole): adds 11 word-family features to data/features.csv
and writes the result to data/features_cole_simple.csv. data/features.csv itself is not changed.

Run from anywhere:  python src/cole_features_simple.py

Each feature is a rate per 100 words: how often the transcript uses words from one family.
Families 1-5 measure spoken vs. written style, families 6-11 measure story content.
Words are matched as whole words/phrases, case-insensitive, on the original transcripts
in data/transcripts_filtered.csv (features.csv only has lowercased text).
"""
import re
import pathlib
import pandas as pd

# File paths (anchored to this script's folder)
DATA = pathlib.Path(__file__).parent.parent / "data"
TRANSCRIPTS = DATA / "transcripts_filtered.csv"
FEATURES_IN = DATA / "features.csv"
FEATURES_OUT = DATA / "features_cole_simple.csv"


def words(*items):
    """Regex that matches any of the given words/phrases as whole words."""
    return re.compile(r"\b(?:" + "|".join(items) + r")\b", re.I)


# ---------------------------------------------------------------------------
# Word families (edit freely)
# ---------------------------------------------------------------------------
FAMILIES = {
    # --- spoken vs. written style ---
    # conversational discourse markers: human dialogue is full of them
    "discourse_markers": words("oh", "yeah", "well", "okay", "ok", "so", "hey", "hi", "right", "alright", "all right",
                               "you know", "i mean", "actually", "anyway", "um", "uh", "i guess", "like"),
    # informal verb forms
    "casual_verbs": words("got", "gotta", "gonna", "wanna", "get", "gets", "getting", "trying", "tryna", "come on",
                          "kinda", "sorta", "ain't", "y'all", "lemme", "gimme", "dunno"),
    "slang": words("gonna", "wanna", "gotta", "kinda", "sorta", "ain't", "y'all", "bro", "bruh", "dude", "guys",
                   "lol", "nah", "yo", "dawg", "sus", "lowkey", "deadass", "cuz", "ya", "yup", "lemme", "gimme",
                   "dunno", "whatcha", "imma"),
    # subordinating conjunctions + relative pronouns: clause-heavy, written-style narration
    "subordinators": words("who", "whom", "whose", "which", "while", "before", "after", "until", "because",
                           "although", "though", "since", "unless", "whenever", "wherever", "whereas", "once"),
    # sequence words a narrator uses to move the plot along
    "time_connectives": words("then", "before", "after", "later", "finally", "until", "while", "once", "suddenly",
                              "years", "that night", "the next day", "one day", "since"),
    # --- story content ---
    # secrets, hiding and explaining: reveal-driven plots
    "reveal_words": words("truth", "secret", "secrets", "hiding", "hide", "hid", "hidden", "kept", "explain",
                          "lie", "lied", "lying", "found out", "find out", "discover", "discovered", "knew",
                          "realized", "realize", "confess", "admit"),
    # absolutist words (all-or-nothing language)
    "absolutist": words("absolutely", "all", "always", "complete", "completely", "constant", "constantly",
                        "definitely", "entire", "entirely", "ever", "every", "everyone", "everybody", "everything",
                        "full", "fully", "must", "never", "nothing", "nobody", "no one", "totally", "whole", "forever"),
    "kinship": words("mother", "father", "mom", "dad", "mommy", "daddy", "son", "daughter", "sister", "brother",
                     "husband", "wife", "grandma", "grandpa", "grandmother", "grandfather", "family", "parents",
                     "aunt", "uncle"),
    # promises and threats about the future
    "future_modals": words("will", "won't", "shall", "going to", "must", "promise"),
    "love_talk": words("i love you", "love you", "my love", "darling", "sweetheart", "honey", "babe", "kiss",
                       "marry me", "soulmate", "i miss you", "beautiful", "handsome"),
    "apology": words("i'm sorry", "im sorry", "i am sorry", "sorry", "forgive me", "my fault", "i didn't mean",
                     "i shouldn't have", "i messed up"),
}


def tokenize(text: str):
    return re.findall(r"[a-z]+(?:'[a-z]+)*", text.lower())


def extra_features(transcript: str) -> dict:
    text = transcript.replace("’", "'")
    n = max(len(tokenize(text)), 1)
    return {f"{name}_per100": round(100 * len(pattern.findall(text)) / n, 4) for name, pattern in FAMILIES.items()}


def main():
    features = pd.read_csv(FEATURES_IN, float_precision="round_trip")  # keep the original values digit-for-digit
    transcripts = pd.read_csv(TRANSCRIPTS)

    # features.csv rows are in the same order as transcripts_filtered.csv; make sure before joining
    norm = lambda s: s.str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
    assert len(features) == len(transcripts) and (norm(features.transcript).values == norm(transcripts.transcript).values).all(), \
        "features.csv and transcripts_filtered.csv are not in the same row order"

    extra = pd.DataFrame([extra_features(t) for t in transcripts.transcript])
    out = pd.concat([features.drop(columns="y"), extra, features[["y"]]], axis=1)
    out.to_csv(FEATURES_OUT, index=False)

    print(out.groupby("y")[list(extra.columns)].mean().T.rename(columns={0: "human", 1: "AI"}).round(3))
    print(f"\nSaved {len(out)} rows, {out.shape[1]} columns -> {FEATURES_OUT}")


if __name__ == "__main__":
    main()
