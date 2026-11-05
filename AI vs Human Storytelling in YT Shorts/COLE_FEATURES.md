# Extra features (Cole): two versions

Two ways to improve the model in `notebooks/model.ipynb`. Both start from the 36 features in `data/features.csv`
and only **add** new files; no existing file is changed.

|  | Original | **Simple version** | **Max version** |
|---|---|---|---|
| Accuracy | 75.9% | **81.2%** | **84.6%** |
| AUC | 0.825 | 0.873 | 0.910 |
| Accuracy on a theme the model never saw | 76.2% | 80.1% | 83.6% |
| Beats the original on (of 20 splits) | – | 90% | 95% |
| What it adds | – | 11 word-family features | 80 handcrafted features + GPT-2 embedding + word/character patterns |
| Model | logistic regression | logistic regression | 4 logistic regressions combined by stacking |
| Explains *why* | yes | **yes, every feature is a readable word list** | partly (embedding and patterns are black boxes) |
| Needs to run the notebook | pandas, scikit-learn | pandas, scikit-learn | pandas, scikit-learn |
| Needs to regenerate features | – | nothing extra | `torch`, `transformers`, `vaderSentiment` + GPT-2 download (~500 MB) |

Suggested use: the **max version** shows *how well* AI-made stories can be detected; the **simple version** shows
*which storytelling differences* explain it.

## How the numbers were measured
One 80/20 split of 311 clips is only 63 test clips, so a single score moves several points from split to split.
Every number above is the average over the **same 20 train/test splits** (5-fold CV repeated 4 times).
Settings such as the regularization strength `C` are tuned on the training part of each split only, so the
test clips never influence any choice. The "unseen theme" row trains on 7 themes and tests on the 8th, which checks
that the model is not just learning topic words (e.g. "fruit" or "class"). Theme alone predicts the label about 58%
of the time.

## Simple version: 11 word-family features
Each feature is how often the transcript uses words from one family, per 100 words.

| Feature | Example words | Leans |
|---|---|---|
| `discourse_markers_per100` | oh, yeah, well, okay, I mean | human |
| `casual_verbs_per100` | got, gonna, trying, come on | human |
| `slang_per100` | bro, dude, guys, nah | human |
| `apology_per100` | sorry, my fault, forgive me | human |
| `subordinators_per100` | who, while, before, because | AI |
| `time_connectives_per100` | then, later, finally, one day | AI |
| `reveal_words_per100` | truth, secret, hiding, explain, knew | AI |
| `absolutist_per100` | never, everything, forever, nothing | AI |
| `kinship_per100` | mother, father, sister, husband | AI |
| `future_modals_per100` | will, won't, going to, promise | AI |
| `love_talk_per100` | I love you, my love, honey | AI |

In one sentence: AI scripts read like written narration about secrets and promises; human clips sound like people talking.

Files:
- `src/cole_features_simple.py`: makes `data/features_cole_simple.csv` (original 36 + these 11)
- `notebooks/model_cole_simple.ipynb`: `model.ipynb` run on the new file, plus the fair comparison

## Max version: stacked model
Three kinds of input, each with its own logistic regression; a final logistic regression learns how much to trust each:
1. **116 handcrafted features**: the original 36, the 11 above, and 69 more, grouped as:
   - story structure (where the climax and first conflict fall, "plot bombs" like weddings/deaths/secrets, topic jumps)
   - dialogue moves (blame, soap-opera emotions, clichés, urgency, repeated words)
   - emotional arc (VADER sentiment per sentence: average mood, swings, ending mood)
   - narration style (past tense, third person, narrator words like "whispered", hook openings, moral endings)
   - readability, punctuation, pronouns, sentence openings, more word families
   - GPT-2 "AI detector" signals (how predictable each word is to GPT-2)
2. **GPT-2 embedding**: 768 numbers summarizing each transcript.
3. **Word and character patterns** (TF-IDF), learned from the text.

Files:
- `src/cole_features_max.py`: makes `data/features_cole_max.csv` and `data/gpt2_embeddings_cole.csv`
- `notebooks/model_cole_max.ipynb`: the stacked model, the fair comparison, and the unseen-theme check (5-10 min)

## Notes
- `notebooks/model.ipynb` drops a `label` column that the committed `data/features.csv` does not have, so it errors
  on that file. `model_cole_simple.ipynb` uses `errors="ignore"` for that drop.
- `src/feature_generation.py` builds its features from generated example transcripts, not from the real data.
- The word lists were written after looking at this data, so the real-world gain may be slightly smaller than measured.
