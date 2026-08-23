"""Explanation QUALITY metrics.

Most XAI-in-security papers stop at a beeswarm plot. Measuring whether the
explanations are faithful, stable, and sparse is what separates a contribution
from a decoration.
"""
from __future__ import annotations

import numpy as np


def fidelity(model, X, shap_values, k: int = 5, baseline=None) -> dict:
    """Deletion fidelity: masking the top-k features by |SHAP| should degrade
    the prediction substantially more than masking k random features."""
    X_arr = X.values.astype(float).copy()
    base = baseline if baseline is not None else np.nanmedian(X_arr, axis=0)
    original = model.predict_proba(X)[:, 1]

    top_idx = np.argsort(-np.abs(shap_values), axis=1)[:, :k]
    X_top = X_arr.copy()
    rng = np.random.default_rng(42)
    X_rand = X_arr.copy()
    rand_idx = rng.integers(0, X_arr.shape[1], size=(X_arr.shape[0], k))
    for row in range(X_arr.shape[0]):
        X_top[row, top_idx[row]] = base[top_idx[row]]
        X_rand[row, rand_idx[row]] = base[rand_idx[row]]

    import pandas as pd
    p_top = model.predict_proba(pd.DataFrame(X_top, columns=X.columns))[:, 1]
    p_rand = model.predict_proba(pd.DataFrame(X_rand, columns=X.columns))[:, 1]
    d_top = float(np.mean(np.abs(original - p_top)))
    d_rand = float(np.mean(np.abs(original - p_rand)))
    return {"k": k, "delta_topk": d_top, "delta_random": d_rand,
            "fidelity_ratio": d_top / max(d_rand, 1e-9)}


def stability(shap_values, X, n_pairs: int = 1_000, seed: int = 42) -> dict:
    """Near-identical inputs should receive near-identical attributions.
    An unstable explainer cannot support an analyst's decision."""
    rng = np.random.default_rng(seed)
    Xn = X.values.astype(float)
    Xn = (Xn - np.nanmean(Xn, axis=0)) / (np.nanstd(Xn, axis=0) + 1e-9)
    Xn = np.nan_to_num(Xn)
    sims = []
    for _ in range(n_pairs):
        i, j = rng.integers(0, len(Xn), 2)
        input_d = np.linalg.norm(Xn[i] - Xn[j])
        if input_d > 1.0:
            continue
        a, b = shap_values[i], shap_values[j]
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
        sims.append(float(np.dot(a, b) / denom))
    return {"n_close_pairs": len(sims),
            "mean_cosine_similarity": float(np.mean(sims)) if sims else None,
            "std_cosine_similarity": float(np.std(sims)) if sims else None}


def sparsity(shap_values, threshold: float = 0.90) -> dict:
    """How many features carry most of the explanation? A 40-feature
    explanation is not an explanation an analyst can act on."""
    abs_v = np.abs(shap_values)
    sorted_v = -np.sort(-abs_v, axis=1)
    cum = np.cumsum(sorted_v, axis=1) / np.clip(sorted_v.sum(axis=1, keepdims=True), 1e-12, None)
    n_needed = (cum < threshold).sum(axis=1) + 1
    return {"threshold": threshold,
            "mean_features_for_threshold": float(n_needed.mean()),
            "median_features_for_threshold": float(np.median(n_needed))}
