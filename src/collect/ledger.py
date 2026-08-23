"""Resume ledger for long-running certificate collection.

Lives on LOCAL disk (/content), never on the Drive FUSE mount: Drive does not
implement POSIX advisory locking correctly and SQLite depends on it, so a
long-running writer on a mounted .db can corrupt on disconnect - which is
exactly when a Colab session dies. A periodic backup copy goes to Drive.

The ledger stores STATE ONLY. Certificate observations go to Parquet shards.
"""
from __future__ import annotations

import datetime as _dt
import shutil
import sqlite3
from pathlib import Path
from typing import Iterable

# Terminal: never retried.
TERMINAL = ("success", "no_certificate", "nxdomain", "abandoned")
# Transient: retried with backoff up to MAX_RETRIES.
TRANSIENT = ("timeout", "connection_error", "dns_error", "tls_error", "rate_limited")
MAX_RETRIES = 3

_SCHEMA = """
CREATE TABLE IF NOT EXISTS domains (
    domain        TEXT PRIMARY KEY,
    source        TEXT,
    label         TEXT,
    status        TEXT NOT NULL DEFAULT 'pending',
    attempted_at  TEXT,
    completed_at  TEXT,
    error_message TEXT,
    shard_id      INTEGER,
    retry_count   INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_status ON domains(status);
CREATE INDEX IF NOT EXISTS idx_retry ON domains(status, retry_count);
"""


class Ledger:
    def __init__(self, db_path: str | Path, drive_backup: str | Path | None = None,
                 backup_every: int = 5_000):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.drive_backup = Path(drive_backup) if drive_backup else None
        self.backup_every = backup_every
        self._since_backup = 0
        self.conn = sqlite3.connect(self.db_path, timeout=30)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    # ---------- population ----------

    def enqueue(self, rows: Iterable[dict]) -> int:
        """rows: {'domain', 'source', 'label'}. Idempotent - existing rows kept."""
        cur = self.conn.executemany(
            "INSERT OR IGNORE INTO domains(domain, source, label) VALUES (?,?,?)",
            [(r["domain"], r.get("source"), r.get("label")) for r in rows],
        )
        self.conn.commit()
        return cur.rowcount

    # ---------- work selection ----------

    def pending(self, limit: int | None = None) -> list[str]:
        """Domains still needing a probe: never attempted, or transient failure
        under the retry cap. Terminal statuses are never returned."""
        sql = (
            "SELECT domain FROM domains "
            "WHERE status = 'pending' "
            "   OR (status IN (%s) AND retry_count < ?) "
            "ORDER BY retry_count ASC, domain ASC"
        ) % ",".join("?" * len(TRANSIENT))
        params: list = [*TRANSIENT, MAX_RETRIES]
        if limit:
            sql += " LIMIT ?"
            params.append(limit)
        return [r[0] for r in self.conn.execute(sql, params)]

    # ---------- result recording ----------

    def mark(self, domain: str, status: str, shard_id: int | None = None,
             error: str | None = None) -> None:
        now = _dt.datetime.now(_dt.timezone.utc).isoformat()
        completed = now if status in TERMINAL else None
        self.conn.execute(
            "UPDATE domains SET status=?, attempted_at=?, completed_at=?, "
            "error_message=?, shard_id=?, "
            "retry_count = retry_count + CASE WHEN ? IN (%s) THEN 1 ELSE 0 END "
            "WHERE domain=?" % ",".join("?" * len(TRANSIENT)),
            (status, now, completed, error, shard_id, status, *TRANSIENT, domain),
        )
        self._since_backup += 1
        if self._since_backup >= self.backup_every:
            self.commit(backup=True)

    def mark_many(self, results: Iterable[tuple]) -> None:
        for domain, status, shard_id, error in results:
            self.mark(domain, status, shard_id, error)
        self.commit()

    def retire_exhausted(self) -> int:
        """Transient failures past the retry cap become 'abandoned' - stops the
        collector re-probing permanently dead hosts forever."""
        cur = self.conn.execute(
            "UPDATE domains SET status='abandoned' "
            "WHERE status IN (%s) AND retry_count >= ?" % ",".join("?" * len(TRANSIENT)),
            (*TRANSIENT, MAX_RETRIES),
        )
        self.conn.commit()
        return cur.rowcount

    # ---------- persistence ----------

    def commit(self, backup: bool = False) -> None:
        self.conn.commit()
        if backup and self.drive_backup:
            self.backup()
            self._since_backup = 0

    def backup(self) -> Path | None:
        """Consistent copy via SQLite's own backup API - safe while WAL is active."""
        if not self.drive_backup:
            return None
        self.drive_backup.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.drive_backup.with_suffix(".tmp")
        dest = sqlite3.connect(tmp)
        with dest:
            self.conn.backup(dest)
        dest.close()
        shutil.move(str(tmp), str(self.drive_backup))
        return self.drive_backup

    def restore_from_backup(self) -> bool:
        """Rebuild a lost local ledger from the Drive copy after a fresh runtime."""
        if self.drive_backup and self.drive_backup.exists() and not self.db_path.exists():
            shutil.copy2(self.drive_backup, self.db_path)
            return True
        return False

    # ---------- reporting ----------

    def summary(self) -> dict:
        rows = self.conn.execute(
            "SELECT status, COUNT(*) FROM domains GROUP BY status"
        ).fetchall()
        out = dict(rows)
        out["_total"] = sum(out.values())
        done = sum(out.get(s, 0) for s in TERMINAL)
        out["_complete_pct"] = round(100 * done / max(out["_total"], 1), 2)
        return out

    def close(self) -> None:
        self.commit(backup=True)
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
