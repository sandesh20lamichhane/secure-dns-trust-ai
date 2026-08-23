"""Split creation. Splits are FILES, not functions.

Regenerating a split from a seed is fragile: a library upgrade or a change in
row ordering silently shifts membership, and results stop being comparable
across notebooks. Here each split is written once, hashed, and consumed by
filename thereafter.
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.config import file_hash


def _write(split_dir: Path, name: str, part: str, domains) -> Path:
    path = split_dir / f"{name}_{part}.txt"
    path.write_text("\n".join(map(str, domains)) + "\n")
    return path


def _metadata(split_dir: Path, name: str, meta: dict) -> Path:
    path = split_dir / f"{name}_metadata.json"
    path.write_text(json.dumps(meta, indent=2, default=str))
    return path


def random_split(df: pd.DataFrame, split_dir, name="random_v1", seed=42,
                 fractions=(0.70, 0.15, 0.15)) -> dict:
    split_dir = Path(split_dir); split_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    domains = np.asarray(df["domain"].unique(), dtype=object)
    rng.shuffle(domains)
    n = len(domains)
    n_tr, n_va = int(fractions[0] * n), int(fractions[1] * n)
    parts = {"train": domains[:n_tr], "val": domains[n_tr:n_tr + n_va],
             "test": domains[n_tr + n_va:]}
    files = {p: str(_write(split_dir, name, p, d)) for p, d in parts.items()}
    meta = {"split_name": name, "type": "random", "created_at": _dt.datetime.now().isoformat(),
            "seed": seed, "fractions": fractions,
            "n": {p: len(d) for p, d in parts.items()},
            "files": files, "file_hashes": {p: file_hash(f) for p, f in files.items()},
            "warning": "Optimistic. Report alongside family_disjoint and temporal."}
    _metadata(split_dir, name, meta)
    return meta


def family_disjoint_split(df: pd.DataFrame, split_dir, name="family_disjoint_v1",
                          seed=42, family_col="family", test_fraction=0.30) -> dict:
    """Whole DGA families are held out. Measures generalisation to unseen
    families - the number that actually matters operationally, and typically
    10-20 points below the random-split figure."""
    split_dir = Path(split_dir); split_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    mal = df[df["label"] == 1]
    families = np.asarray(mal[family_col].dropna().unique(), dtype=object)
    rng.shuffle(families)
    n_test = max(1, int(len(families) * test_fraction))
    test_fams = set(families[:n_test])
    val_fams = set(families[n_test:n_test + max(1, n_test // 2)])
    train_fams = set(families) - test_fams - val_fams

    benign = np.asarray(df[df["label"] == 0]["domain"].unique(), dtype=object)
    rng.shuffle(benign)
    b_tr = int(0.70 * len(benign)); b_va = int(0.15 * len(benign))

    def _mal(fams):
        return np.asarray(mal[mal[family_col].isin(fams)]["domain"].unique(), dtype=object)

    parts = {
        "train": np.concatenate([_mal(train_fams), benign[:b_tr]]),
        "val": np.concatenate([_mal(val_fams), benign[b_tr:b_tr + b_va]]),
        "test": np.concatenate([_mal(test_fams), benign[b_tr + b_va:]]),
    }
    files = {p: str(_write(split_dir, name, p, d)) for p, d in parts.items()}
    meta = {"split_name": name, "type": "family_disjoint",
            "created_at": _dt.datetime.now().isoformat(), "seed": seed,
            "n": {p: len(d) for p, d in parts.items()},
            "train_families": sorted(train_fams), "val_families": sorted(val_fams),
            "test_families": sorted(test_fams),
            "files": files, "file_hashes": {p: file_hash(f) for p, f in files.items()}}
    _metadata(split_dir, name, meta)
    return meta


def temporal_split(df: pd.DataFrame, split_dir, name="temporal_v1",
                   date_col="first_seen", train_end=None, val_end=None) -> dict:
    """Train on the past, test on the future. Detects concept drift and stops
    the model from being credited with hindsight."""
    split_dir = Path(split_dir); split_dir.mkdir(parents=True, exist_ok=True)
    d = df.dropna(subset=[date_col]).copy()
    d[date_col] = pd.to_datetime(d[date_col], utc=True, errors="coerce")
    d = d.dropna(subset=[date_col]).sort_values(date_col)

    train_end = pd.Timestamp(train_end, tz="UTC") if train_end else d[date_col].quantile(0.70)
    val_end = pd.Timestamp(val_end, tz="UTC") if val_end else d[date_col].quantile(0.85)

    parts = {
        "train": d[d[date_col] <= train_end]["domain"].unique(),
        "val": d[(d[date_col] > train_end) & (d[date_col] <= val_end)]["domain"].unique(),
        "test": d[d[date_col] > val_end]["domain"].unique(),
    }
    files = {p: str(_write(split_dir, name, p, x)) for p, x in parts.items()}
    meta = {"split_name": name, "type": "temporal",
            "created_at": _dt.datetime.now().isoformat(),
            "date_column": date_col, "train_end": str(train_end), "val_end": str(val_end),
            "n": {p: len(x) for p, x in parts.items()},
            "files": files, "file_hashes": {p: file_hash(f) for p, f in files.items()}}
    _metadata(split_dir, name, meta)
    return meta


def load_split(split_dir, name: str) -> dict:
    split_dir = Path(split_dir)
    meta = json.loads((split_dir / f"{name}_metadata.json").read_text())
    parts = {}
    for part in ("train", "val", "test"):
        path = split_dir / f"{name}_{part}.txt"
        parts[part] = set(path.read_text().split())
        current = file_hash(path)
        expected = meta.get("file_hashes", {}).get(part)
        if expected and current != expected:
            raise RuntimeError(
                f"Split file {path.name} changed since creation "
                f"(hash {current} != {expected}). Results would not be comparable."
            )
    return {"domains": parts, "metadata": meta,
            "split_file": str(split_dir / f"{name}_train.txt")}


def apply_split(df: pd.DataFrame, split: dict):
    d = split["domains"]
    return (df[df["domain"].isin(d["train"])],
            df[df["domain"].isin(d["val"])],
            df[df["domain"].isin(d["test"])])
