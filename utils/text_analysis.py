"""Basic text statistics, frequent words, keywords, and chunking."""

import math
import re
from collections import Counter

STOP_WORDS = set("""
a about above after again against all also am an and any are aren't as at be because been before being
below between both but by can can't cannot could couldn't did didn't do does doesn't doing don't down
during each even few for from further get gets got had hadn't has hasn't have haven't having he he'd he'll
he's her here here's hers herself him himself his how how's however i i'd i'll i'm i've if in into is isn't
it it's its itself just let's like made make many may me might more most much must mustn't my myself no nor
not now of off on once one only or other ought our ours ourselves out over own per said same say shan't she
she'd she'll she's should shouldn't so some still such than that that's the their theirs them themselves
then there there's these they they'd they'll they're they've this those through thus to too under until up
upon us use used using very was wasn't we we'd we'll we're we've were weren't what what's when when's where
where's whether which while who who's whom whose why why's will with won't would wouldn't yet you you'd
you'll you're you've your yours yourself yourselves
""".split())

WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*[A-Za-z]|[A-Za-z]")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def clean_text(text):
    """Tidy up whitespace without changing the content."""
    text = text.replace("\r", "\n").replace("’", "'")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def tokenize(text):
    """Lowercase words only (no numbers or punctuation)."""
    return [w.lower().strip("'-") for w in WORD_RE.findall(text.replace("’", "'"))]


def split_sentences(text):
    flat = re.sub(r"\s+", " ", text)
    return [s.strip() for s in SENTENCE_SPLIT_RE.split(flat) if s.strip()]


def _meaningful(word):
    return word not in STOP_WORDS and len(word) > 2


def compute_stats(text):
    words = tokenize(text)
    sentences = split_sentences(text)
    n_words, n_sent = len(words), max(len(sentences), 1)
    return {
        "words": n_words,
        "sentences": len(sentences),
        "characters": len(text),
        "unique_words": len(set(words)),
        "avg_words_per_sentence": n_words / n_sent,
        "reading_minutes": max(1, math.ceil(n_words / 200)),  # ~200 words per minute
    }


def top_words(text, n=15):
    """Most common words after removing stop words. Returns [(word, count)]."""
    return Counter(w for w in tokenize(text) if _meaningful(w)).most_common(n)


def extract_keywords(text, n=10):
    """Repeated two-word phrases first, then the strongest single words."""
    unigrams = Counter(w for w in tokenize(text) if _meaningful(w))
    bigrams = Counter()
    for sentence in split_sentences(text):
        toks = tokenize(sentence)
        for a, b in zip(toks, toks[1:]):
            if _meaningful(a) and _meaningful(b):
                bigrams[(a, b)] += 1

    phrases = [(f"{a} {b}", c) for (a, b), c in bigrams.most_common() if c >= 2][:5]
    used = {w for phrase, _ in phrases for w in phrase.split()}
    singles = [w for w, _ in unigrams.most_common(n * 2) if w not in used]
    return [p for p, _ in phrases] + singles[: n - len(phrases)]


def split_into_chunks(text, max_words):
    """Split text into chunks of roughly max_words, breaking at sentence ends."""
    chunks, current, count = [], [], 0
    for sentence in split_sentences(text):
        words = sentence.split()
        if len(words) > max_words:  # one giant "sentence" (e.g. messy PDF text)
            if current:
                chunks.append(" ".join(current))
                current, count = [], 0
            for i in range(0, len(words), max_words):
                chunks.append(" ".join(words[i : i + max_words]))
            continue
        if count + len(words) > max_words and current:
            chunks.append(" ".join(current))
            current, count = [], 0
        current.append(sentence)
        count += len(words)
    if current:
        chunks.append(" ".join(current))
    return chunks
