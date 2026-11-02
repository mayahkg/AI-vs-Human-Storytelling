# AI vs. Human Storytelling
How does AI-generated storytelling differ from human storytelling, especially in
short-form content? This repository explores that question through two models and a
data collection pipeline. It was completed as a course project for **Berkeley Data 189**.

## Overview

## Getting Started

## Data Collection Pipeline
1. **Discover** — `python pipeline/discover.py` collects YouTube Shorts using
   search queries and labels each as AI-generated or human-made.
2. **Transcribe** — `python pipeline/transcribe.py` produces transcripts for each video.
3. **Model** — see each model folder for training and evaluation.

## Models
- [`text_baseline/`](text_baseline/) — [one-line summary: model type, features, dataset]
- [`shorts_transcripts/`](shorts_transcripts/) — [one-line summary: model type, features, dataset size]

## Results
| Model | Accuracy | F1 |
|-------|----------|----|
| Baseline (text) | [ ] | [ ] |
| Shorts transcripts | [ ] | [ ] |

## Limitations
- [Dataset size and how videos were labeled AI vs. human]
- [Topic imbalance, e.g. AI Shorts clustering around certain themes, which models may pick up on instead of style]
- [Transcription errors from automatic speech-to-text]

## Authors
Maya Yan — UC Berkeley, Data 189, Freshmen 

## Acknowledgments
Data 189 Fall 2026 Course Staff, UC Berkeley