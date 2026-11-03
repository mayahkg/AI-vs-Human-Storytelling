# AI vs Human Storytelling in YT Shorts
This model asks whether AI-generated and human-made short-form stories can be
told apart from their transcripts alone.

## Overview

## Data Collection

### Files

## Method

## Results

## Limitations

## How to run
```bash
pip install -r requirements.txt
cp .env.example .env            # add your YouTube API key
python pipeline/discover.py     # collect candidate videos
python pipeline/transcribe.py   # transcribe them
# then open the notebook for features, visualizations, and modeling
```