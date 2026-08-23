"""Drive/local staging and sharded Parquet writing.

Two rules this module exists to enforce:
  1. Never read training batches directly off the Drive FUSE mount.
  2. Never write thousands of small files to Drive - shard into large Parquet.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable

import pandas as pd


def stage_local(src: str | Path, local_dir: str | Path, force: bool = False) -> Path:
    """Copy a file from Drive to local SSD once; return the local path."""
    src, local_dir = Path(src), Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    dst = local_dir / src.name
    if force or not dst.exists() or dst.stat().st_size != src.stat().st_size:
        shutil.copy2(src, dst)
    return dst


def sync_to_drive(local: str | Path, drive_dir: str | Path) -> Path:
    local, drive_dir = Path(local), Path(drive_dir)
    drive_dir.mkdir(parents=True, exist_ok=True)
    dst = drive_dir / local.name
    shutil.copy2(local, dst)
    return dst


class ShardWriter:
    """Buffer rows and flush to immutable Parquet shards.

    A shard is written once and never reopened - that is what makes
    data/01_collected/ safe to treat as a write-once observation record.
    """

    def __init__(self, out_dir: str | Path, prefix: str, rows_per_shard: int = 50_000,
                 compression: str = "zstd"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.rows_per_shard = rows_per_shard
        self.compression = compression
        self._buffer: list[dict] = []
        self._shard_id = self._next_shard_id()
        self.written = 0

    def _next_shard_id(self) -> int:
        existing = sorted(self.out_dir.glob(f"{self.prefix}_*.parquet"))
        if not existing:
            return 0
        return int(existing[-1].stem.split("_")[-1]) + 1

    def add(self, row: dict) -> None:
        self._buffer.append(row)
        if len(self._buffer) >= self.rows_per_shard:
            self.flush()

    def extend(self, rows: Iterable[dict]) -> None:
        for row in rows:
            self.add(row)

    def flush(self) -> Path | None:
        if not self._buffer:
            return None
        path = self.out_dir / f"{self.prefix}_{self._shard_id:05d}.parquet"
        pd.DataFrame(self._buffer).to_parquet(path, index=False,
                                              compression=self.compression)
        self.written += len(self._buffer)
        self._buffer.clear()
        self._shard_id += 1
        return path

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.flush()


def read_shards(shard_dir: str | Path, prefix: str, columns=None) -> pd.DataFrame:
    files = sorted(Path(shard_dir).glob(f"{prefix}_*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat((pd.read_parquet(f, columns=columns) for f in files),
                     ignore_index=True)
