# AI vs Human Storytelling in YouTube Shorts

Can AI-generated and human-made short-form stories be told apart from their transcripts alone?
This project collects YouTube Shorts, transcribes them, keeps only the story videos, and trains a classifier to predict whether each story is AI-made or human-made.

## How it works
1. **Collect**: find videos with the YouTube API.
2. **Transcribe**: convert audio to text with Whisper.
3. **Filter**: an LLM (Gemini) keeps only stories (scripted or narrated) and drops explainers, news, lyrics, etc. The LLM never sees the AI/human label.
4. **Features**: compute 119 handcrafted text features (sentence length, contractions, reaction words, etc.) plus 768 GPT-2 embedding columns.
5. **Model**: train and cross-validate classifiers in the notebooks.

## Setup
```bash
pip install -r requirements.txt
cp .env.example .env    # add your YouTube API key and Gemini API key
```

## Run the pipeline
```bash
python src/collect_videos.py                  # collect YouTube videos
python src/transcribe_videos.py               # transcribe them
python src/filtered_feature_generation.py     # filter stories and generate features
python src/embeddings_generation.py           # generate embeddings
```

Then run the notebooks in this order:
1. `data_cleaning`
2. `visualisations`
3. `model`

## Results
| Test | Accuracy | AUC |
|---|---|---|
| Cross-validation (483 clips) | about 0.84 | 0.92 |
| Videos of a theme the model has never seen | about 0.75 to 0.77 | about 0.92 |
Cross-validated accuracy overstates real-world accuracy. The ranking (AUC) holds up on new videos, but the 0.5 cutoff works less well on them.