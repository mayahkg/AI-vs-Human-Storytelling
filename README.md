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

## Limitations
- Human videos are all from before January 2023 and AI videos from 2025 on. Era differences, like trends and editing style, could leak into the transcripts. The date cutoff is also what makes your human labels trustworthy, so it’s a reasonable trade-off.
- Both labels come from keywords and dates, not manual verification. Open about 20 of the new links to confirm they’re correct before transcribing.

## Authors
Maya Yan — UC Berkeley, Data 189, Freshmen Year

## Acknowledgments
Data 189 TAs, Course Staff, UC Berkeley
