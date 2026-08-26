"""Resumable Certificate Transparency collection.

The naive version of this - a serial loop with a sleep, buffering into a single
large shard - is unusable here: crt.sh responses take tens of seconds, so 2,000
domains exceed a Colab session, and nothing reaches disk until the very end. A
dropped runtime then loses every lookup.

This driver fixes both halves:

  * a SQLite ledger records per-domain state, so a re-run resumes;
  * a modest thread pool overlaps the waiting, since crt.sh latency is dominated
    by server-side query time rather than by our bandwidth;
  * shards are small and flushed often, so an abrupt kill loses one shard at
    most rather than the whole run.

Concurrency is kept deliberately low. crt.sh is a free community service and
rate-limits aggressively; eight in flight is polite and already an order of
magnitude faster than serial.
"""
from __future__ import annotations

import concurrent.futures
from pathlib import Path

from . import crtsh_client
from .ledger import Ledger
from ..utils.io import ShardWriter

DEFAULT_CONCURRENCY = 8
ROWS_PER_SHARD = 250          # small on purpose: bounds loss on an abrupt kill


def open_ledger(paths: dict, name: str = "ct_ledger.db") -> Ledger:
    """A CT ledger separate from the TLS one - the two runs progress
    independently and must not overwrite each other's state."""
    local_dir = Path(paths["local"]["ledger"]).parent
    return Ledger(local_dir / name,
                  drive_backup=f"{paths['artifacts']['logs']}/{name}")


def _one(domain: str) -> tuple:
    """Return (domain, status, row_or_None, error_or_None)."""
    status, payload = crtsh_client.fetch(domain)
    if status != "success":
        return domain, "connection_error", None, str(payload)[:200]
    row = crtsh_client.summarise(domain, payload)
    # Zero CT records is a finding (the domain never had a public cert issued),
    # not a failure - it must be recorded so the domain is not re-probed.
    return domain, "success", row, None


def run(ledger: Ledger, out_dir, prefix: str = "crtsh",
        batch_size: int = 400, concurrency: int = DEFAULT_CONCURRENCY,
        max_batches: int | None = None, logger=None) -> dict:
    """Resumable driver. Safe to interrupt; the ledger is the state."""
    writer = ShardWriter(out_dir, prefix, rows_per_shard=ROWS_PER_SHARD)
    batches = 0
    try:
        while True:
            todo = ledger.pending(limit=batch_size)
            if not todo:
                break
            if max_batches is not None and batches >= max_batches:
                break

            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                for domain, status, row, error in pool.map(_one, todo):
                    if row:
                        writer.add(row)
                    ledger.mark(domain, status, shard_id=writer._shard_id, error=error)

            writer.flush()               # persist before the ledger is backed up
            ledger.commit(backup=True)
            batches += 1
            if logger:
                logger.info("CT batch %d | %s", batches, ledger.summary())
    finally:
        writer.flush()
        ledger.commit(backup=True)
    return ledger.summary()
