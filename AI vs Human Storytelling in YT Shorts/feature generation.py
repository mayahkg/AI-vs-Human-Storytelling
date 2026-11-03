"""
AI vs Human Transcript Analysis
- Feature engineering for stylistic/linguistic signals
- Visualisations comparing most promising features
- Logistic Regression baseline (implemented from scratch)
"""

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from scipy.special import expit  # sigmoid

# -------------------------------------------------
# 1. Synthetic data generation
#    (real transcripts_filtered.csv is not present)
# -------------------------------------------------
np.random.seed(42)

def generate_synthetic_transcripts(n_human=160, n_ai=150):
    """Create plausible short transcripts with stylistic differences."""
    human_templates = [
        "um so i was thinking about this the other day and like you know it's kind of crazy how everything just changes so fast well i don't know maybe i'm overthinking it but yeah",
        "oh my god you guys i literally can't believe what just happened i was like wait what and then she said no way so of course i had to tell you all about it right away",
        "well honestly i've been feeling a bit off lately not sure why but maybe it's just the weather or something you know how it is sometimes you just need a break",
        "hey everyone so today i wanted to talk about something personal i've been struggling with and i hope this helps someone out there who feels the same way",
        "like i said before it's not that simple you have to really think about what you want and then go for it even if it's scary at first don't give up",
        "so basically what happened was i tried this new thing and it didn't work at all then i tried again and boom it finally clicked can you believe that",
        "i mean come on we all know that's not true right but people keep saying it anyway and it drives me crazy every single time i hear it",
        "um yeah so the main point is that you should always check the details first because otherwise you might miss something important and then it's too late"
    ]

    ai_templates = [
        "in this discussion we will examine the key factors that contribute to effective decision making in complex environments the process involves careful analysis",
        "it is important to consider multiple perspectives when evaluating potential outcomes furthermore one must weigh both short term and long term implications carefully",
        "the primary objective of this analysis is to identify patterns that emerge across different scenarios and to propose structured approaches for improvement",
        "research indicates that consistent practice combined with deliberate reflection leads to measurable progress over time therefore dedication remains essential",
        "several strategies can be employed to address the challenges outlined previously each approach offers distinct advantages depending on the specific context",
        "overall the evidence supports the conclusion that systematic planning yields superior results compared to ad hoc methods in most professional settings",
        "to summarize the main recommendations include establishing clear priorities allocating resources efficiently and monitoring progress at regular intervals",
        "this framework provides a comprehensive foundation for understanding the underlying mechanisms and for developing practical solutions moving forward"
    ]

    rows = []
    for i in range(n_human):
        base = np.random.choice(human_templates)
        # add some variation / repetition
        extra = " " + np.random.choice(["yeah", "you know", "right", "i guess", "honestly", ""])
        transcript = (base + extra).strip().lower()
        n_words = len(transcript.split())
        rows.append({"label": "human", "transcript": transcript, "n_words": n_words})

    for i in range(n_ai):
        base = np.random.choice(ai_templates)
        transcript = base.strip().lower()
        n_words = len(transcript.split())
        rows.append({"label": "AI", "transcript": transcript, "n_words": n_words})

    df = pd.DataFrame(rows)
    # shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


# -------------------------------------------------
# 2. Feature engineering functions
# -------------------------------------------------
FILLERS = {
    "um", "uh", "uhm", "er", "ah", "like", "you know", "i mean",
    "basically", "literally", "actually", "so", "well", "right",
    "okay", "ok", "yeah", "yep", "nah"
}
NEGATIONS = {
    "not", "no", "never", "n't", "don't", "doesn't", "didn't",
    "won't", "can't", "cannot", "isn't", "aren't", "wasn't",
    "weren't", "haven't", "hasn't", "hadn't", "neither", "nor"
}
CONTRACTIONS = {
    "don't", "doesn't", "didn't", "won't", "can't", "isn't", "aren't",
    "wasn't", "weren't", "i'm", "you're", "he's", "she's", "it's",
    "we're", "they're", "i've", "you've", "we've", "they've",
    "i'll", "you'll", "he'll", "she'll", "we'll", "they'll",
    "i'd", "you'd", "he'd", "she'd", "we'd", "they'd", "that's",
    "what's", "who's", "where's", "how's", "let's"
}
FIRST_PERSON = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"}
SECOND_PERSON = {"you", "your", "yours", "yourself", "yourselves"}
WE_WORDS = {"we", "us", "our", "ours", "ourselves"}
REACTION = {"oh", "wow", "ugh", "whoa", "ooh", "ahh", "yay", "aww", "hmm", "huh"}


def tokenize(text: str):
    """Simple whitespace + punctuation-aware tokenization."""
    text = re.sub(r"[^\w\s'\-]", " ", text)  # keep apostrophes for contractions
    return [w for w in text.lower().split() if w]


def sentence_split(text: str):
    """Very simple sentence splitter."""
    # split on . ! ? while keeping the punctuation attached for rate calc
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def compute_features(text: str) -> dict:
    """Compute a focused set of the most promising stylistic features."""
    tokens = tokenize(text)
    n_words = max(len(tokens), 1)
    unique = set(tokens)
    counts = Counter(tokens)

    # sentences
    sents = sentence_split(text)
    n_sents = max(len(sents), 1)
    sent_lens = [len(tokenize(s)) for s in sents]
    avg_sent_len = np.mean(sent_lens) if sent_lens else 0
    sent_len_std = np.std(sent_lens) if len(sent_lens) > 1 else 0
    short_sent_rate = 100 * sum(1 for L in sent_lens if L <= 5) / n_sents

    # question / exclaim / ellipsis
    question_rate = 100 * text.count("?") / n_sents
    exclaim_rate = 100 * text.count("!") / n_sents
    ellipsis_rate = 100 * text.count("...") / n_sents

    # lexical diversity
    ttr_full = len(unique) / n_words
    hapax = sum(1 for w, c in counts.items() if c == 1)
    hapax_rate = 100 * hapax / max(len(unique), 1)

    # rates per 100 words
    def rate(word_set):
        return 100 * sum(counts.get(w, 0) for w in word_set) / n_words

    filler_rate = rate(FILLERS)
    # multi-word fillers approx
    filler_rate += 100 * text.count("you know") / n_words
    filler_rate += 100 * text.count("i mean") / n_words

    negation_rate = rate(NEGATIONS)
    contraction_rate = 100 * sum(1 for t in tokens if t in CONTRACTIONS) / n_words
    first_person_rate = rate(FIRST_PERSON)
    second_person_rate = rate(SECOND_PERSON)
    we_rate = rate(WE_WORDS)
    reaction_rate = rate(REACTION)

    # specific word rates
    everything_rate = 100 * counts.get("everything", 0) / n_words
    love_rate = 100 * counts.get("love", 0) / n_words
    well_rate = 100 * counts.get("well", 0) / n_words

    # word length
    avg_word_len = np.mean([len(t) for t in tokens]) if tokens else 0
    long_word_rate = 100 * sum(1 for t in tokens if len(t) >= 7) / n_words

    # top word share
    top_count = counts.most_common(1)[0][1] if counts else 0
    top_word_share = 100 * top_count / n_words

    # simple CTA heuristic
    cta_phrases = ["subscribe", "like and", "comment below", "follow me", "hit the bell", "link in bio"]
    cta_flag = int(any(p in text for p in cta_phrases))

    return {
        "n_words_calc": n_words,
        "avg_word_len": round(avg_word_len, 3),
        "long_word_rate": round(long_word_rate, 3),
        "ttr_full": round(ttr_full, 4),
        "hapax_rate": round(hapax_rate, 3),
        "filler_rate": round(filler_rate, 3),
        "negation_rate": round(negation_rate, 3),
        "contraction_rate": round(contraction_rate, 3),
        "first_person_rate": round(first_person_rate, 3),
        "second_person_rate": round(second_person_rate, 3),
        "we_rate": round(we_rate, 3),
        "reaction_rate": round(reaction_rate, 3),
        "everything_rate": round(everything_rate, 3),
        "love_rate": round(love_rate, 3),
        "well_rate": round(well_rate, 3),
        "n_sentences": n_sents,
        "avg_sent_len": round(avg_sent_len, 3),
        "sent_len_std": round(sent_len_std, 3),
        "short_sent_rate": round(short_sent_rate, 3),
        "question_rate": round(question_rate, 3),
        "exclaim_rate": round(exclaim_rate, 3),
        "ellipsis_rate": round(ellipsis_rate, 3),
        "top_word_share": round(top_word_share, 3),
        "cta_flag": cta_flag,
    }


# -------------------------------------------------
# 3. Build the full feature matrix
# -------------------------------------------------
print("Generating synthetic data ...")
df = generate_synthetic_transcripts()
print(f"Dataset shape: {df.shape}")
print(df["label"].value_counts())

print("\nComputing features (this may take a moment) ...")
feature_rows = [compute_features(t) for t in df["transcript"]]
feat_df = pd.DataFrame(feature_rows)
full = pd.concat([df[["label", "transcript", "n_words"]], feat_df], axis=1)

# encode label
full["y"] = (full["label"] == "AI").astype(int)  # 1 = AI, 0 = human

print("Feature matrix ready.")
print(full.columns.tolist())

# -------------------------------------------------
# 4. Visualisations – most promising features
# -------------------------------------------------
# Select features that typically discriminate AI vs human speech
promising = [
    "filler_rate",
    "contraction_rate",
    "first_person_rate",
    "second_person_rate",
    "avg_sent_len",
    "ttr_full",
    "hapax_rate",
    "avg_word_len",
    "long_word_rate",
    "negation_rate",
    "well_rate",
    "question_rate",
    "short_sent_rate",
    "n_words_calc",
]

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120

# --- Boxplots ---
n_feats = len(promising)
n_cols = 4
n_rows = (n_feats + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.5 * n_rows))
axes = axes.flatten()

for i, feat in enumerate(promising):
    ax = axes[i]
    sns.boxplot(data=full, x="label", y=feat, ax=ax, hue="label", legend=False)
    ax.set_title(feat, fontsize=11)
    ax.set_xlabel("")
    ax.set_ylabel("")

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Feature Distributions: AI vs Human Transcripts", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig("/home/workdir/artifacts/feature_boxplots.png", bbox_inches="tight")
print("Saved: feature_boxplots.png")
plt.close()

# --- Violin plots for top 6 ---
top6 = ["filler_rate", "contraction_rate", "first_person_rate",
        "avg_sent_len", "ttr_full", "avg_word_len"]

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for i, feat in enumerate(top6):
    sns.violinplot(data=full, x="label", y=feat, ax=axes[i], hue="label", legend=False, inner="quartile")
    axes[i].set_title(feat)
    axes[i].set_xlabel("")

plt.suptitle("Top Discriminative Features (Violin Plots)", fontsize=14)
plt.tight_layout()
plt.savefig("/home/workdir/artifacts/feature_violins.png", bbox_inches="tight")
print("Saved: feature_violins.png")
plt.close()

# --- Correlation with label ---
corr = full[promising + ["y"]].corr()["y"].drop("y").sort_values()
fig, ax = plt.subplots(figsize=(8, 6))
colors = ["#d62728" if v < 0 else "#2ca02c" for v in corr.values]
corr.plot(kind="barh", color=colors, ax=ax)
ax.set_title("Pearson Correlation of Features with Label (AI = 1)")
ax.axvline(0, color="black", linewidth=0.8)
plt.tight_layout()
plt.savefig("/home/workdir/artifacts/feature_correlations.png", bbox_inches="tight")
print("Saved: feature_correlations.png")
plt.close()

# -------------------------------------------------
# 5. Logistic Regression baseline (from scratch)
# -------------------------------------------------
print("\n=== Logistic Regression Baseline ===")

feature_cols = promising  # use the same set
X = full[feature_cols].values.astype(float)
y = full["y"].values.astype(float)

# standardise
mu = X.mean(axis=0)
sigma = X.std(axis=0) + 1e-8
X_std = (X - mu) / sigma

# train / test split (80/20)
n = len(y)
idx = np.random.permutation(n)
split = int(0.8 * n)
train_idx, test_idx = idx[:split], idx[split:]

X_train, X_test = X_std[train_idx], X_std[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# add bias term
X_train_b = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
X_test_b  = np.hstack([np.ones((X_test.shape[0], 1)),  X_test])

# gradient descent
def logistic_regression(X, y, lr=0.1, n_iter=2000, l2=0.01):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    losses = []
    for i in range(n_iter):
        z = X @ w
        p = expit(z)
        # binary cross-entropy + L2
        loss = -np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))
        loss += 0.5 * l2 * np.sum(w[1:] ** 2)  # no penalty on bias
        losses.append(loss)
        grad = (X.T @ (p - y)) / n_samples
        grad[1:] += l2 * w[1:]
        w -= lr * grad
        if i % 500 == 0:
            print(f"  iter {i:4d}  loss={loss:.4f}")
    return w, losses

print("Training ...")
weights, loss_hist = logistic_regression(X_train_b, y_train, lr=0.15, n_iter=2500, l2=0.05)

# predictions
def predict_proba(X, w):
    return expit(X @ w)

def predict(X, w, thresh=0.5):
    return (predict_proba(X, w) >= thresh).astype(int)

y_pred_train = predict(X_train_b, weights)
y_pred_test  = predict(X_test_b,  weights)

def metrics(y_true, y_pred):
    acc = np.mean(y_true == y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    prec = tp / (tp + fp + 1e-12)
    rec  = tp / (tp + fn + 1e-12)
    f1   = 2 * prec * rec / (prec + rec + 1e-12)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}

train_m = metrics(y_train, y_pred_train)
test_m  = metrics(y_test,  y_pred_test)

print("\nTrain metrics:")
for k, v in train_m.items():
    if k in ("tp", "fp", "fn", "tn"):
        print(f"  {k}: {int(v)}")
    else:
        print(f"  {k}: {v:.4f}")

print("\nTest metrics:")
for k, v in test_m.items():
    if k in ("tp", "fp", "fn", "tn"):
        print(f"  {k}: {int(v)}")
    else:
        print(f"  {k}: {v:.4f}")

# confusion matrix visualisation
cm = np.array([[test_m["tn"], test_m["fp"]],
               [test_m["fn"], test_m["tp"]]])
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Pred Human", "Pred AI"],
            yticklabels=["True Human", "True AI"], ax=ax)
ax.set_title("Confusion Matrix (Test Set)")
plt.tight_layout()
plt.savefig("/home/workdir/artifacts/confusion_matrix.png", bbox_inches="tight")
print("Saved: confusion_matrix.png")
plt.close()

# feature coefficients (importance)
coef = weights[1:]  # skip bias
coef_df = pd.DataFrame({
    "feature": feature_cols,
    "coefficient": coef
}).sort_values("coefficient")

fig, ax = plt.subplots(figsize=(8, 6))
colors = ["#d62728" if c < 0 else "#2ca02c" for c in coef_df["coefficient"]]
ax.barh(coef_df["feature"], coef_df["coefficient"], color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Logistic Regression Coefficients\n(positive → more AI-like)")
plt.tight_layout()
plt.savefig("/home/workdir/artifacts/lr_coefficients.png", bbox_inches="tight")
print("Saved: lr_coefficients.png")
plt.close()

# loss curve
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(loss_hist)
ax.set_xlabel("Iteration")
ax.set_ylabel("Loss (BCE + L2)")
ax.set_title("Training Loss Curve")
plt.tight_layout()
plt.savefig("/home/workdir/artifacts/loss_curve.png", bbox_inches="tight")
print("Saved: loss_curve.png")
plt.close()

# -------------------------------------------------
# 6. Summary table of feature means by class
# -------------------------------------------------
summary = full.groupby("label")[promising].mean().T
summary["diff (AI - human)"] = summary["AI"] - summary["human"]
summary = summary.sort_values("diff (AI - human)")
print("\n=== Mean feature values by class ===")
print(summary.round(3).to_string())

# save the engineered dataset for later use
full.to_csv("/home/workdir/artifacts/engineered_features.csv", index=False)
print("\nSaved engineered dataset → engineered_features.csv")

print("\nDone.")
