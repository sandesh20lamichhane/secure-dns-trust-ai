"""Turning a classifier output into a trust score.

A raw probability is not a trust score. Calibration on a held-out set - disjoint
from both training and test - is what licenses interpreting the number as a
probability, and therefore what licenses setting thresholds by operational cost.
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class Calibrator:
    def __init__(self, method: str = "isotonic"):
        if method not in ("isotonic", "platt", "none"):
            raise ValueError(f"unknown calibration method: {method}")
        self.method = method
        self.model = None

    def fit(self, scores, y):
        scores, y = np.asarray(scores).reshape(-1), np.asarray(y)
        if self.method == "isotonic":
            self.model = IsotonicRegression(out_of_bounds="clip").fit(scores, y)
        elif self.method == "platt":
            self.model = LogisticRegression().fit(scores.reshape(-1, 1), y)
        return self

    def transform(self, scores):
        scores = np.asarray(scores).reshape(-1)
        if self.method == "none" or self.model is None:
            return scores
        if self.method == "isotonic":
            return self.model.predict(scores)
        return self.model.predict_proba(scores.reshape(-1, 1))[:, 1]


def reliability_curve(y_true, y_prob, n_bins: int = 15):
    """Points for the reliability diagram - a required figure for the paper."""
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (y_prob > lo) & (y_prob <= hi)
        if mask.any():
            rows.append({"bin_lower": lo, "bin_upper": hi, "n": int(mask.sum()),
                         "mean_predicted": float(y_prob[mask].mean()),
                         "observed_frequency": float(y_true[mask].mean())})
    return rows
