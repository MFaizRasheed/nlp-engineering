# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an NLP engineering project focused on sentiment analysis. It uses NLTK for text preprocessing (tokenization, stemming, stopwords) and scikit-learn for machine learning models. The project analyzes course reviews to determine sentiment (positive/negative).

## Running Code

Use the virtual environment's Python interpreter:
```bash
.venv/Scripts/python.exe <script>.py
```

For Jupyter notebooks in the `notebook/` directory, use the IDE's notebook execution (mcp__ide__executeCode).

## Project Structure

- **data/**: CSV datasets containing course reviews with `review_text` and `sentiment` columns (1=positive, 0=negative)
- **labs/**: Course lab materials with utilities for tweet preprocessing
- **notebook/**: Jupyter notebooks for experimentation and testing
- **labs/utils.py**: Core NLP utilities including `process_tweet()` (tokenization, stemming, stopword removal) and `build_freqs()` (word frequency counting)

## Key Dependencies

- nltk (NLP processing)
- scikit-learn (ML models)
- pandas (data handling)
- spacy, nltk, matplotlib, seaborn, streamlit (visualization and UI)

## Notes

- The `model.joblib` file in the root is a saved scikit-learn model
- The project uses Jupyter notebooks for iterative development and exploration