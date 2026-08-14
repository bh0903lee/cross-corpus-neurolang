"""
feature_extractor.py — clinical linguistic feature extraction.
Extracts a 12-dim linguistic biomarker vector from transcript text (token/sentence
counts, lexical diversity, fillers, repetition, content density, pauses, word length).
"""
import re
import numpy as np
from typing import List

# ─── simple word lists (English) ─────────────────────────────────────────────
_PRONOUNS  = {"i","me","my","we","us","our","you","your","he","him","his",
              "she","her","they","them","their","it","its","this","that","these","those"}
_FILLERS   = {"um","uh","well","like","you","know","so","right","okay","oh",
              "hmm","er","ah","mhm","yeah","yep","hm"}
_FUNCTION  = {"the","a","an","is","are","was","were","be","been","being",
              "to","of","and","or","but","in","on","at","by","for","with",
              "from","as","it","that","this","have","has","had","do","does","did"}


def _tokenize(text: str) -> List[str]:
    """Lowercase and keep only letters and apostrophes."""
    text = text.lower()
    text = re.sub(r"[^a-z\s']", " ", text)
    return [w.strip("'") for w in text.split() if len(w.strip("'")) > 0]


def _sentences(text: str) -> List[str]:
    """Split text into sentences at . ? ! boundaries."""
    sents = re.split(r"[.?!]+", text)
    return [s.strip() for s in sents if len(s.strip().split()) >= 1]


def mattr(tokens: List[str], window: int = 10) -> float:
    """Moving Average Type-Token Ratio."""
    if len(tokens) < window:
        return len(set(tokens)) / max(len(tokens), 1)
    ttrs = []
    for i in range(len(tokens) - window + 1):
        w = tokens[i:i+window]
        ttrs.append(len(set(w)) / window)
    return float(np.mean(ttrs))


def extract_features(text: str) -> np.ndarray:
    """
    Return a 12-dim linguistic feature vector for one text (zeros if empty).
    """
    if not text or not text.strip():
        return np.zeros(12, dtype=np.float32)

    tokens  = _tokenize(text)
    sents   = _sentences(text)
    n_tok   = len(tokens)
    n_sent  = max(len(sents), 1)

    if n_tok == 0:
        return np.zeros(12, dtype=np.float32)

    # 0. total token count (log-normalized)
    f0 = np.log1p(n_tok)

    # 1. sentence count (log)
    f1 = np.log1p(n_sent)

    # 2. mean utterance length
    f2 = n_tok / n_sent

    # 3. TTR
    f3 = len(set(tokens)) / n_tok

    # 4. MATTR
    f4 = mattr(tokens, window=10)

    # 5. long-word ratio (> 6 chars)
    f5 = sum(1 for w in tokens if len(w) > 6) / n_tok

    # 6. pronoun ratio
    f6 = sum(1 for w in tokens if w in _PRONOUNS) / n_tok

    # 7. filler ratio (disfluency indicator)
    f7 = sum(1 for w in tokens if w in _FILLERS) / n_tok

    # 8. repetition ratio (adjacent identical tokens)
    repeats = sum(1 for i in range(1, n_tok) if tokens[i] == tokens[i-1])
    f8 = repeats / n_tok

    # 9. content-word density (non-function-word ratio)
    f9 = sum(1 for w in tokens if w not in _FUNCTION and w not in _FILLERS) / n_tok

    # 10. pause density (commas, ellipses per token)
    pause_chars = len(re.findall(r"[,;]|\.\.\.", text))
    f10 = pause_chars / n_tok

    # 11. mean word length
    f11 = np.mean([len(w) for w in tokens]) if tokens else 0.0

    feats = np.array([f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11], dtype=np.float32)
    return feats


def normalize_features(feat_matrix: np.ndarray) -> np.ndarray:
    """
    Z-score normalize the (N, 12) feature matrix; returns (normed, mean, std).
    """
    mean = feat_matrix.mean(axis=0)
    std  = feat_matrix.std(axis=0) + 1e-8
    return (feat_matrix - mean) / std, mean, std
