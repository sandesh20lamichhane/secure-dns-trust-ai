"""Per-domain prediction storage.

Saving only summary metrics is the mistake this module prevents. With raw
scores on disk, a reviewer asking for specificity, MCC, or a different
operating point costs one minute instead of a retraining run.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def save(run_id: str, predictions_dir, domains, y_true, raw_score,
         calibrated_score=None, split_part="test", extra: dict | None = None) -> Path:
    df = pd.DataFrame({
        "run_id": run_id,
        "domain": list(domains),
        "true_label": list(y_true),
        "raw_score": list(raw_score),
        "split_part": split_part,
    })
    df["calibrated_score"] = list(calibrated_score) if calibrated_score is not None else None
    for key, values in (extra or {}).items():
        df[key] = list(values)
    path = Path(predictions_dir) / f"{run_id}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression="zstd")
    return path


def load(run_id: str, predictions_dir) -> pd.DataFrame:
    return pd.read_parquet(Path(predictions_dir) / f"{run_id}.parquet")
