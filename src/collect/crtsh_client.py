"""Certificate Transparency lookups via crt.sh.

Gives historical certificate issuance per domain - notably first-seen date and
issuance count, which are strong signals for freshly-registered malicious
domains. crt.sh is rate-sensitive and often slow; this client is deliberately
polite and fully resumable through the same ledger.
"""
from __future__ import annotations

import datetime as _dt
import json
import time
import urllib.parse
import urllib.request

BASE = "https://crt.sh/?q={}&output=json"
UA = "academic-research-crawler/1.0 (DNS trust scoring study)"


def fetch(domain: str, timeout: float = 30.0, retries: int = 2) -> tuple:
    """Return (status, rows). Never raises."""
    url = BASE.format(urllib.parse.quote(domain))
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status == 429:
                    time.sleep(30 * (attempt + 1))
                    continue
                body = resp.read().decode("utf-8", errors="replace")
            return "success", json.loads(body) if body.strip() else []
        except json.JSONDecodeError:
            return "success", []
        except Exception as exc:
            if attempt == retries:
                return "connection_error", str(exc)[:200]
            time.sleep(5 * (attempt + 1))
    return "connection_error", "exhausted"


def summarise(domain: str, rows: list) -> dict:
    """Collapse a CT history into per-domain features."""
    observed_at = _dt.datetime.now(_dt.timezone.utc).isoformat()
    if not rows:
        return {"domain": domain, "observed_at": observed_at, "ct_log_count": 0,
                "ct_first_seen": None, "ct_last_seen": None,
                "days_since_first_ct_seen": None, "ct_distinct_issuers": 0,
                "ct_distinct_names": 0}

    def _parse(value):
        try:
            return _dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        except Exception:
            return None

    starts = [d for d in (_parse(r.get("not_before", "")) for r in rows) if d]
    first = min(starts) if starts else None
    last = max(starts) if starts else None
    now = _dt.datetime.now(_dt.timezone.utc)
    if first and first.tzinfo is None:
        first = first.replace(tzinfo=_dt.timezone.utc)
    return {
        "domain": domain,
        "observed_at": observed_at,
        "ct_log_count": len(rows),
        "ct_first_seen": first.isoformat() if first else None,
        "ct_last_seen": last.isoformat() if last else None,
        "days_since_first_ct_seen": (now - first).days if first else None,
        "ct_distinct_issuers": len({r.get("issuer_name") for r in rows}),
        "ct_distinct_names": len({r.get("common_name") for r in rows}),
    }
