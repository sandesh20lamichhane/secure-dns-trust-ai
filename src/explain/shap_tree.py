"""SHAP for the tabular/fusion model.

TreeExplainer is used deliberately: on tree ensembles it computes EXACT Shapley
values in polynomial time, whereas KernelExplainer only approximates them. That
distinction matters when the explanations are themselves a claimed contribution.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def compute(model, X: pd.DataFrame, run_id: str, shap_dir,
            max_rows: int | None = 20_000, seed: int = 42):
    import shap

    if max_rows and len(X) > max_rows:
        X = X.sample(max_rows, random_state=seed)
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X)
    if isinstance(values, list):
        values = values[1]

    path = Path(shap_dir) / f"{run_id}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(values, columns=X.columns, index=X.index)
    out["_base_value"] = float(np.ravel(explainer.expected_value)[-1])
    out.to_parquet(path, compression="zstd")
    return values, X, explainer, path


def global_importance(values, feature_names) -> pd.DataFrame:
    mean_abs = np.abs(values).mean(axis=0)
    return (pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True))
