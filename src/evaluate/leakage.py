"""Pre-modelling leakage audit.

Run BEFORE any model is trained. Every flagged feature gets an entry in
_LEAKAGE_NOTES.md with an explicit keep/drop decision. Stating in the paper
that a leakage audit preceded model development is worth considerably more
than a headline accuracy figure.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def single_feature_auc(df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    """A single feature that alone separates the classes almost perfectly is
    either the label in disguise or a downstream artifact of it."""
    y = df[label_col].values
    rows = []
    for col in df.columns:
        if col in (label_col, "domain") or df[col].dtype == "object":
            continue
        x = pd.to_numeric(df[col], errors="coerce")
        if x.notna().sum() < 100 or x.nunique() < 2:
            continue
        filled = x.fillna(x.median()).values
        try:
            auc = roc_auc_score(y, filled)
        except ValueError:
            continue
        rows.append({
            "feature": col,
            "auc": max(auc, 1 - auc),
            "missing_rate": float(x.isna().mean()),
            "n_unique": int(x.nunique()),
        })
    out = pd.DataFrame(rows, columns=["feature", "auc", "missing_rate", "n_unique"])
    out = out.sort_values("auc", ascending=False)
    out["suspicion"] = pd.cut(out["auc"], bins=[0, 0.75, 0.90, 0.97, 1.0],
                              labels=["low", "moderate", "high", "critical"])
    return out


def missingness_by_class(df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    """Class-dependent missingness leaks the label through NaN patterns alone -
    e.g. if certificate fields are absent mostly for malicious domains, the
    model can learn 'missing => malicious' without any certificate content."""
    rows = []
    for col in df.columns:
        if col in (label_col, "domain"):
            continue
        miss = df[col].isna()
        if not miss.any():
            continue
        m0, m1 = miss[df[label_col] == 0].mean(), miss[df[label_col] == 1].mean()
        rows.append({"feature": col, "missing_benign": float(m0),
                     "missing_malicious": float(m1), "gap": float(abs(m1 - m0))})
    out = pd.DataFrame(rows, columns=["feature", "missing_benign",
                                     "missing_malicious", "gap"])
    return out.sort_values("gap", ascending=False)


def duplicate_audit(df: pd.DataFrame, key: str = "domain",
                    label_col: str = "label") -> dict:
    """Duplicate domains across splits are the most common silent leak."""
    dup = df[df.duplicated(key, keep=False)]
    conflicting = (dup.groupby(key)[label_col].nunique() > 1).sum() if len(dup) else 0
    return {"n_rows": len(df), "n_unique_keys": int(df[key].nunique()),
            "n_duplicate_rows": int(len(dup)),
            "n_keys_with_conflicting_labels": int(conflicting)}


def report(df: pd.DataFrame, label_col: str = "label", auc_threshold: float = 0.97) -> dict:
    auc = single_feature_auc(df, label_col)
    return {
        "single_feature_auc": auc,
        "critical_features": auc[auc["auc"] >= auc_threshold]["feature"].tolist(),
        "missingness": missingness_by_class(df, label_col),
        "duplicates": duplicate_audit(df, "domain", label_col),
    }
