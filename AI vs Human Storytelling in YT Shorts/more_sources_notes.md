# More sources: what changed and what we found (Cole)

Goal: run every step of the pipeline ourselves with more video sources, and keep accuracy the same or better.

## What was added

- **21 new search phrases** in `src/collect_videos.py`, aimed at the theme/label groups with the fewest clips
  (AI work, horror, animal, school and baby; human horror, animal and fruit). They brought in 738 new candidate URLs.
- **644 usable English transcripts**, up from about 475.
- **483 stories kept** by the story filter, up from 364 on `main`:

| | Clips |
|---|---|
| Original clips, still kept | 352 of 364 |
| Old transcripts the new filter kept that `main` had dropped | 11 |
| **Brand-new videos** | **120** (96 AI, 24 human) |

The new videos lean AI. The human searches (limited to videos posted before 2023) found fewer usable clips,
and the filter kept 85% of AI transcripts but only 68% of human ones.

**No features were added.** The 119 handcrafted columns and 768 GPT-2 embedding columns are identical to `main`.
Only the set of clips changed.

## Story filter (step 3) and the free Gemini limit

- A free Gemini key allows **500 requests per model per day, per Google Cloud project**. A new key made inside an
  existing project shares that project's 500. Requests rejected for going too fast still count toward the 500.
- Judging all 644 transcripts doesn't fit in one free run, so `filtered_feature_generation.py` now saves every
  decision to `data/llm_judgments.csv` and picks up where it stopped. `--no-cache` restores the old behavior.
- The daily limits forced the filter onto three models. Each transcript's decision comes from the first model
  on this list that judged it:

| Model | Decisions used |
|---|---|
| gemini-3.1-flash-lite | 346 |
| gemini-3.5-flash-lite | 96 |
| gemma-4-26b-a4b-it (via `src/finish_story_filter.py`) | 191 |
| Could not be judged (Gemma kept returning server errors), dropped | 11 |

- How often the models agree on story vs. not-story:
  - 3.1 vs 3.5: 94.1% (305 clips)
  - Gemma vs 3.1: 94.9% (39 clips)
  - Gemma vs 3.5: 92.3% (39 clips)

  Agreement is high enough that mixing the three models shouldn't change the results much.

## Accuracy

The model code is `notebooks/model.ipynb` (functions run as-is), on the same 20 RepeatedStratifiedKFold splits:

| Model | `main` (364 clips) | This branch (483 clips) |
|---|---|---|
| Original features | 0.742 | 0.757, AUC 0.825 |
| Simple (+11 feature families) | 0.777 | 0.775, AUC 0.845 |
| Max (stacked) | 0.836, AUC 0.913 | **0.837, AUC 0.921** |

Accuracy is the same or slightly better with the extra sources.

**Same-test-clips check:** the 352 original clips kept by both filters, 20 splits, stacked model. Trained on
`main`'s data the model scores 0.847 (AUC 0.919); trained on this branch's data it scores 0.823 (AUC 0.911).
AUC is close, so the accuracy gap most likely comes from the shift toward AI clips (56% vs 47%), which moves the
0.5 cutoff on the more human-heavy original set.

## Are we overfitting?

Partly, yes. The 120 brand-new videos are a fair test because nobody had seen them when the features and word
lists were designed. Trained on the original clips only, tested on the new videos:

| Model | Accuracy on its own training clips | Accuracy on new videos | AUC on new videos |
|---|---|---|---|
| Original features | 0.858 | 0.758 | 0.789 |
| Max (stacked) | 0.991 | **0.767** | **0.924** |

- The stacked model memorizes its training clips (99%). That's expected; the scores that matter are on unseen clips.
- **Cross-validated accuracy (about 84%) overstates real-world accuracy.** On videos the model has never seen it's
  about 77%. An earlier test on 206 fresh clips from an older dataset gave about 75%. Some of the gap comes from
  features tuned on the original clips, and some from the new videos coming from different searches.
- **Ranking holds up:** AUC stays around 0.92, so the model still scores AI clips above human ones. The 0.5 cutoff
  just lands poorly on new data: it caught 88% of human clips but only 74% of AI clips.
- Only 24 human clips are among the new videos, so treat these numbers as rough.
- An earlier sanity check, retraining the full stacked pipeline on shuffled labels, scored about 50%, so there's
  no label leakage.

**A fair claim for the writeup:** about 84% in cross-validation, about 75–77% on videos the model has never seen,
AUC about 0.92 on both.

## Not done

- `notebooks/model.ipynb` was not re-run, so its saved outputs still show `main`'s numbers. The numbers above come
  from the same functions run on the new data.
- The 11 unjudged transcripts were dropped rather than judged on a later day.
