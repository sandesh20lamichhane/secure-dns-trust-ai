"""Configuration loading. Every path and hyperparameter enters the project here."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import yaml

_VAR = re.compile(r"\$\{([^}]+)\}")


def _resolve(value: Any, root: dict) -> Any:
    """Expand ${a.b} references against the top-level document."""
    if isinstance(value, str):
        for _ in range(10):                       # bounded, so cycles cannot hang
            match = _VAR.search(value)
            if not match:
                break
            key = match.group(1)
            node: Any = root
            for part in key.split("."):
                node = node[part]
            value = value.replace(match.group(0), str(node))
        return value
    if isinstance(value, dict):
        return {k: _resolve(v, root) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve(v, root) for v in value]
    return value


def load(name: str, config_dir: str | Path | None = None) -> dict:
    """Load configs/<name>.yaml with interpolation applied."""
    config_dir = Path(config_dir or os.environ.get("DNSTRUST_CONFIG_DIR", "configs"))
    path = config_dir / (name if name.endswith(".yaml") else f"{name}.yaml")
    doc = yaml.safe_load(path.read_text())
    return _resolve(doc, doc)


def paths(config_dir: str | Path | None = None) -> dict:
    return load("paths", config_dir)


def config_hash(cfg: dict) -> str:
    """Stable hash of a config dict, recorded in the run manifest."""
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def file_hash(path: str | Path) -> str:
    """Hash of a file on disk - used to pin the exact split file a run consumed."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def ensure_tree(p: dict) -> None:
    """Create every directory named in paths.yaml. Idempotent."""
    for section in ("data", "artifacts", "results", "local"):
        for value in p.get(section, {}).values():
            Path(value).mkdir(parents=True, exist_ok=True)
    Path(p["manifest"]).parent.mkdir(parents=True, exist_ok=True)
