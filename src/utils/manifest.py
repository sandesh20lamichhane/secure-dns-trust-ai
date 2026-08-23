"""Run identity. One JSONL line per completed run; run_id indexes every artifact."""
from __future__ import annotations

import datetime as _dt
import importlib
import json
import subprocess
from pathlib import Path

from .config import config_hash, file_hash

_TRACKED_LIBS = ("xgboost", "torch", "sklearn", "shap", "numpy", "pandas", "lightgbm")


def git_sha(repo_root: str | Path = ".") -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        sha = out.stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        return f"{sha}-dirty" if dirty else sha or "unknown"
    except Exception:
        return "unknown"


def library_versions() -> dict:
    versions = {}
    for lib in _TRACKED_LIBS:
        try:
            versions[lib] = importlib.import_module(lib).__version__
        except Exception:
            pass
    return versions


def make_run_id(run_family: str, split_name: str, seed: int, counter: int) -> str:
    """e.g. xgb_family_disjoint_v1_s42_0042 - the primary key for all outputs."""
    return f"{run_family}_{split_name}_s{seed}_{counter:04d}"


def next_counter(manifest_path: str | Path) -> int:
    path = Path(manifest_path)
    if not path.exists():
        return 1
    return sum(1 for line in path.open() if line.strip()) + 1


def record(
    manifest_path: str | Path,
    run_id: str,
    run_family: str,
    cfg: dict,
    split_name: str,
    split_file: str | Path | None,
    metrics: dict,
    seed: int,
    dataset_version: str = "v1",
    repo_root: str | Path = ".",
    notes: str = "",
) -> dict:
    """Append one run to the manifest and return the entry."""
    entry = {
        "run_id": run_id,
        "run_family": run_family,
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "git_sha": git_sha(repo_root),
        "config_hash": config_hash(cfg),
        "split_name": split_name,
        "split_file_hash": file_hash(split_file) if split_file else None,
        "seed": seed,
        "dataset_version": dataset_version,
        "libraries": library_versions(),
        "metrics": metrics,
        "notes": notes,
    }
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(entry) + "\n")
    return entry


def load_manifest(manifest_path: str | Path):
    """Read the manifest as a DataFrame for the results notebook."""
    import pandas as pd

    path = Path(manifest_path)
    if not path.exists():
        return pd.DataFrame()
    rows = [json.loads(line) for line in path.open() if line.strip()]
    df = pd.json_normalize(rows)
    return df
