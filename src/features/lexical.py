"""Lexical features from the domain string.

These are the hand-crafted counterpart to the CNN-BiLSTM character branch.
Keeping both lets the ablation answer whether the learned representation adds
anything over classical lexical statistics - a question reviewers will ask.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import pandas as pd

VOWELS = set("aeiou")
CONSONANTS = set("bcdfghjklmnpqrstvwxyz")

# Populated from a benign corpus by fit_ngram_model(); never from the test split.
_NGRAM_MODEL: dict = {"bigram": {}, "trigram": {}, "fitted": False}


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def max_consecutive_consonants(s: str) -> int:
    runs = re.findall(r"[bcdfghjklmnpqrstvwxyz]+", s.lower())
    return max((len(r) for r in runs), default=0)


def fit_ngram_model(benign_domains, n_values=(2, 3)) -> dict:
    """Fit n-gram frequencies on TRAINING benign domains only.

    Fitting on the full corpus leaks test-set distribution into the features;
    this function must be called with the training split alone.
    """
    global _NGRAM_MODEL
    for n in n_values:
        counts = Counter()
        for d in benign_domains:
            s = _core(d)
            counts.update(s[i:i + n] for i in range(len(s) - n + 1))
        total = sum(counts.values()) or 1
        key = "bigram" if n == 2 else "trigram"
        _NGRAM_MODEL[key] = {g: c / total for g, c in counts.items()}
    _NGRAM_MODEL["fitted"] = True
    return _NGRAM_MODEL


def _ngram_score(s: str, n: int) -> float:
    key = "bigram" if n == 2 else "trigram"
    model = _NGRAM_MODEL.get(key, {})
    if not model or len(s) < n:
        return 0.0
    grams = [s[i:i + n] for i in range(len(s) - n + 1)]
    floor = 1e-8
    return sum(math.log(model.get(g, floor)) for g in grams) / len(grams)


def _core(domain: str) -> str:
    """Registrable label, lowercased - the part a DGA actually generates."""
    parts = domain.lower().strip(".").split(".")
    return parts[-2] if len(parts) >= 2 else parts[0]


def extract(domain: str) -> dict:
    d = domain.lower().strip(".")
    core = _core(d)
    labels = d.split(".")
    alpha = [c for c in core if c.isalpha()]
    return {
        "domain": domain,
        "length": len(d),
        "core_length": len(core),
        "n_labels": len(labels),
        "tld": labels[-1] if len(labels) > 1 else "",
        "shannon_entropy": shannon_entropy(core),
        "vowel_ratio": sum(c in VOWELS for c in core) / max(len(alpha), 1),
        "digit_ratio": sum(c.isdigit() for c in core) / max(len(core), 1),
        "hyphen_count": core.count("-"),
        "max_consec_consonants": max_consecutive_consonants(core),
        "bigram_score": _ngram_score(core, 2),
        "trigram_score": _ngram_score(core, 3),
        "unique_char_ratio": len(set(core)) / max(len(core), 1),
        "is_idn": d.startswith("xn--") or "xn--" in d,
        "has_digit": any(c.isdigit() for c in core),
        "starts_with_digit": core[:1].isdigit() if core else False,
    }


def extract_frame(domains) -> pd.DataFrame:
    return pd.DataFrame([extract(d) for d in domains])
