"""AI summarization with Hugging Face, with chunking for long documents."""

import os

import streamlit as st
from transformers import pipeline

from utils.text_analysis import split_into_chunks

SUMMARY_MODELS = {
    "Best quality: BART Large (slower)": "facebook/bart-large-cnn",
    "Fast: DistilBART (lighter)": "sshleifer/distilbart-cnn-12-6",
}
# If Streamlit Cloud runs out of memory, change this to the "Fast" label.
DEFAULT_MODEL_LABEL = "Best quality: BART Large (slower)"

# Token limits (not words) for the summary of each piece
LENGTH_SETTINGS = {
    "Short": {"chunk_max": 50, "chunk_min": 20, "final_max": 70, "final_min": 30},
    "Medium": {"chunk_max": 80, "chunk_min": 30, "final_max": 130, "final_min": 60},
    "Detailed": {"chunk_max": 120, "chunk_min": 50, "final_max": 220, "final_min": 100},
}

CHUNK_WORDS = 450   # keeps each piece safely under the model's input limit
MAX_CHUNKS = 8      # keeps long documents fast enough for a live demo


class SummaryError(Exception):
    """Friendly, user-facing summary problem."""


def get_hf_token():
    """Optional Hugging Face token from env var or Streamlit secrets. Never hardcoded."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        return st.secrets.get("HF_TOKEN")
    except Exception:
        return None  # no secrets file: totally fine for public models


@st.cache_resource(show_spinner=False)
def load_summarizer(model_name):
    """Load the model once and reuse it."""
    return pipeline("summarization", model=model_name, token=get_hf_token())


def _summarize_piece(pipe, text, max_len, min_len):
    n_words = len(text.split())
    max_len = min(max_len, max(15, int(n_words * 0.7)))
    min_len = min(min_len, max(5, max_len - 10))
    out = pipe(text, max_length=max_len, min_length=min_len, do_sample=False, truncation=True)
    return out[0]["summary_text"].strip()


def _pick_evenly(items, limit):
    if len(items) <= limit:
        return items
    step = (len(items) - 1) / (limit - 1)
    return [items[round(i * step)] for i in range(limit)]


def summarize_document(text, length, model_name, on_progress=None):
    """Return (summary, note). Long text is summarized in chunks, then combined."""
    if len(text.split()) < 40:
        raise SummaryError("This text is too short to summarize. Try something with a few paragraphs.")

    cfg = LENGTH_SETTINGS[length]
    pipe = load_summarizer(model_name)
    all_chunks = split_into_chunks(text, CHUNK_WORDS)
    chunks = _pick_evenly(all_chunks, MAX_CHUNKS)
    note = None
    if len(chunks) < len(all_chunks):
        note = (f"This document is long, so Lens summarized {len(chunks)} sections picked evenly "
                f"from all {len(all_chunks)} to keep things fast.")

    def tick(done, total):
        if on_progress:
            on_progress(done, total)

    if len(chunks) == 1:
        summary = _summarize_piece(pipe, chunks[0], cfg["final_max"], cfg["final_min"])
        tick(1, 1)
        return summary, note

    total_steps = len(chunks) + 1
    partials = []
    for i, chunk in enumerate(chunks, start=1):
        partials.append(_summarize_piece(pipe, chunk, cfg["chunk_max"], cfg["chunk_min"]))
        tick(i, total_steps)

    combined = " ".join(partials)
    # If the partial summaries still fit in one chunk, condense them once more
    if len(combined.split()) <= CHUNK_WORDS:
        combined = _summarize_piece(pipe, combined, cfg["final_max"], cfg["final_min"])
    tick(total_steps, total_steps)
    return combined, note
