"""Evaluation metrics.

Accuracy is deliberately absent from the headline set: at ~30:1 imbalance it is
uninformative. The operationally meaningful numbers are PR-AUC and the
false-positive rate at a fixed recall, because a false positive here blocks a
legitimate business domain.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_recall_curve, roc_auc_score, roc_curve)


def fpr_at_tpr(y_true, y_score, target_tpr: float = 0.95) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = np.searchsorted(tpr, target_tpr, side="left")
    idx = min(idx, len(fpr) - 1)
    return float(fpr[idx])


def threshold_at_tpr(y_true, y_score, target_tpr: float = 0.95) -> float:
    fpr, tpr, thr = roc_curve(y_true, y_score)
    idx = min(np.searchsorted(tpr, target_tpr, side="left"), len(thr) - 1)
    return float(thr[idx])


def expected_calibration_error(y_true, y_prob, n_bins: int = 15) -> float:
    """ECE. Without calibration a 'trust score' is just a renamed softmax
    output - this is the number that justifies the word 'trust'."""
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (y_prob > lo) & (y_prob <= hi)
        if not mask.any():
            continue
        ece += (mask.mean()) * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def max_calibration_error(y_true, y_prob, n_bins: int = 15) -> float:
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    gaps = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (y_prob > lo) & (y_prob <= hi)
        if mask.any():
            gaps.append(abs(y_true[mask].mean() - y_prob[mask].mean()))
    return float(max(gaps)) if gaps else 0.0


def best_f1_threshold(y_true, y_score) -> tuple:
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    i = int(np.nanargmax(f1[:-1])) if len(thr) else 0
    return float(thr[i]) if len(thr) else 0.5, float(f1[i])


def evaluate(y_true, y_score, threshold: float | None = None, n_bins: int = 15) -> dict:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if threshold is None:
        threshold, _ = best_f1_threshold(y_true, y_score)
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "fpr_at_95_tpr": fpr_at_tpr(y_true, y_score, 0.95),
        "fpr_at_99_tpr": fpr_at_tpr(y_true, y_score, 0.99),
        "brier_score": float(brier_score_loss(y_true, np.clip(y_score, 0, 1))),
        "ece": expected_calibration_error(y_true, np.clip(y_score, 0, 1), n_bins),
        "mce": max_calibration_error(y_true, np.clip(y_score, 0, 1), n_bins),
        "threshold": float(threshold),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "n": int(len(y_true)), "positive_rate": float(y_true.mean()),
    }
