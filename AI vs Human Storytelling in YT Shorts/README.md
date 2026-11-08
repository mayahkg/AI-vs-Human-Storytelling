# AI vs Human Storytelling in YT Shorts
This model asks whether AI-generated and human-made short-form stories can be
told apart from their transcripts alone.

## Data Collection
Data is retreived from Youtube API and trascribed with whisper then filtered with GEMINI LLM

## How to run
```bash
pip install -r requirements.txt
cp .env.example .env            # add your YouTube API key and Gemini API Key
python src/collect_videos.py     # collect your Youtube videos
python src/transcribe_videos.py   # transcribe the youtube videos
python src/filtered_feature_generation.py   # filter transcripts and generate features
python src/embeddings_generation.py   # generate embeddings
run notebooks: data_cleaning,  visualisations, model # to run the model