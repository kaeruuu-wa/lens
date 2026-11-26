"""Sentiment analysis with a Hugging Face model."""

import streamlit as st
from transformers import pipeline

from utils.summarizer import get_hf_token
from utils.text_analysis import split_into_chunks

SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
MAX_CHUNKS = 8


class SentimentError(Exception):
    """Friendly, user-facing sentiment problem."""


@st.cache_resource(show_spinner=False)
def load_sentiment_pipeline():
    return pipeline("sentiment-analysis", model=SENTIMENT_MODEL, token=get_hf_token())


def analyze_sentiment(text):
    """Score evenly spaced chunks and average them into one overall tone."""
    chunks = split_into_chunks(text, 200)
    if not chunks:
        raise SentimentError("There isn't enough text to check the tone.")
    if len(chunks) > MAX_CHUNKS:
        step = (len(chunks) - 1) / (MAX_CHUNKS - 1)
        chunks = [chunks[round(i * step)] for i in range(MAX_CHUNKS)]

    pipe = load_sentiment_pipeline()
    results = pipe(chunks, truncation=True, max_length=512, batch_size=4)

    # Positive chunks count as +score, negative chunks as -score
    signed = [r["score"] if r["label"] == "POSITIVE" else -r["score"] for r in results]
    avg = sum(signed) / len(signed)

    if avg > 0.35:
        label = "Positive"
        confidence = sum(r["score"] for r in results if r["label"] == "POSITIVE") / sum(
            1 for r in results if r["label"] == "POSITIVE"
        )
    elif avg < -0.35:
        label = "Negative"
        confidence = sum(r["score"] for r in results if r["label"] == "NEGATIVE") / sum(
            1 for r in results if r["label"] == "NEGATIVE"
        )
    else:
        label = "Neutral"
        confidence = 1 - abs(avg)  # the closer to balanced, the more "neutral"

    pos = sum(1 for r in results if r["label"] == "POSITIVE")
    note = (f"Based on {len(results)} section(s): {pos} leaned positive and {len(results) - pos} leaned negative. "
            "The model only knows positive and negative, so \"Neutral\" means the sections balanced out.")
    return {"label": label, "confidence": float(confidence), "note": note}
