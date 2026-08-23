"""Assembling the labelled domain universe from public sources.

Each loader normalises one source into the same schema:

    domain | label | family | source | first_seen

label:      0 = benign, 1 = malicious
family:     DGA family or threat class; None for benign. Drives the
            family-disjoint split, so it must be populated for malicious rows.
first_seen: earliest known date; drives the temporal split. None is tolerated
            (those rows are simply excluded from the temporal split).

Live feeds (PhishTank, OpenPhish, URLhaus) change daily. Snapshot them with a
date in the filename and never re-download over an existing snapshot - the raw
tier is write-once precisely so a reviewer can be told which day the feed was
captured.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

SCHEMA = ["domain", "label", "family", "source", "first_seen"]

# Public suffixes needing two labels to reach the registrable domain.
_TWO_LEVEL = {"co.uk", "org.uk", "ac.uk", "gov.uk", "co.in", "net.in", "org.in",
              "co.jp", "com.au", "net.au", "com.br", "co.za", "com.cn"}


def registrable(domain: str) -> str:
    """Reduce a hostname to its registrable domain.

    A crude public-suffix approximation - adequate here because it is applied
    identically to every source, so it cannot bias one class relative to
    another. Swap in the `publicsuffix2` package if exactness is needed.
    """
    d = str(domain).strip().lower().rstrip(".")
    d = re.sub(r"^https?://", "", d).split("/")[0].split(":")[0]
    parts = d.split(".")
    if len(parts) < 3:
        return d
    if ".".join(parts[-2:]) in _TWO_LEVEL:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def _frame(rows: pd.DataFrame) -> pd.DataFrame:
    for col in SCHEMA:
        if col not in rows.columns:
            rows[col] = None
    return rows[SCHEMA]


# --------------------------------------------------------------------------
# Benign sources
# --------------------------------------------------------------------------

def load_tranco(path, top_n: int = 300_000) -> pd.DataFrame:
    """Tranco CSV: rank,domain (no header). Keep the list ID in the filename -
    it is the permanent citation for the exact snapshot used."""
    df = pd.read_csv(path, header=None, names=["rank", "domain"], nrows=top_n)
    df["domain"] = df["domain"].map(registrable)
    df["label"] = 0
    df["source"] = f"tranco:{Path(path).stem}"
    return _frame(df.drop_duplicates("domain"))


# --------------------------------------------------------------------------
# Malicious sources
# --------------------------------------------------------------------------

def load_umudga(root, families=None, per_family: int | None = 20_000) -> pd.DataFrame:
    """UMUDGA ships one file per DGA family; the filename is the family label."""
    root = Path(root)
    frames = []
    for f in sorted(root.rglob("*.csv")) + sorted(root.rglob("*.txt")):
        family = f.stem.lower()
        if families and family not in families:
            continue
        col = pd.read_csv(f, header=None, usecols=[0], names=["domain"],
                          nrows=per_family, on_bad_lines="skip")
        col["family"] = family
        frames.append(col)
    if not frames:
        return _frame(pd.DataFrame(columns=["domain"]))
    df = pd.concat(frames, ignore_index=True)
    df["domain"] = df["domain"].map(registrable)
    df["label"] = 1
    df["source"] = "umudga"
    return _frame(df.drop_duplicates("domain"))


def load_dgarchive(path, per_family: int | None = 20_000) -> pd.DataFrame:
    """DGArchive CSV export: domain,family,first_seen (column names vary by
    export version - adjust the rename map if yours differs)."""
    df = pd.read_csv(path, on_bad_lines="skip")
    df = df.rename(columns={c: c.lower() for c in df.columns})
    df = df.rename(columns={"dga": "family", "date": "first_seen",
                            "domain_name": "domain"})
    df["domain"] = df["domain"].map(registrable)
    df["label"] = 1
    df["source"] = "dgarchive"
    if per_family:
        df = df.groupby("family", group_keys=False).head(per_family)
    return _frame(df.drop_duplicates("domain"))


def load_phish_feed(path, source: str, family: str = "phishing") -> pd.DataFrame:
    """PhishTank / OpenPhish / URLhaus. URLs are reduced to registrable domains,
    which loses per-URL granularity - correct here, since the unit of analysis
    is the domain and its certificate, not the individual URL."""
    text = Path(path).read_text(errors="replace")
    if path.endswith(".csv"):
        df = pd.read_csv(path, on_bad_lines="skip")
        col = next((c for c in df.columns if "url" in c.lower()), df.columns[0])
        urls = df[col]
        dates = next((df[c] for c in df.columns
                      if "date" in c.lower() or "added" in c.lower()), None)
    else:
        urls = pd.Series([l for l in text.splitlines() if l and not l.startswith("#")])
        dates = None
    out = pd.DataFrame({"domain": urls.map(registrable)})
    out["first_seen"] = list(dates) if dates is not None else None
    out["label"] = 1
    out["family"] = family
    out["source"] = source
    return _frame(out.drop_duplicates("domain"))


def load_cic_bell(path) -> pd.DataFrame:
    """CIC-Bell-DNS2021. Only the domain and class label are taken here.

    Its precomputed third-party reputation features are deliberately NOT
    imported: several encode the ground-truth label. They stay quarantined in
    features.yaml and are re-examined in the leakage audit, not silently used.
    """
    df = pd.read_csv(path, on_bad_lines="skip", low_memory=False)
    df = df.rename(columns={c: c.lower().strip() for c in df.columns})
    dom_col = next((c for c in df.columns if c in ("domain", "domain_name", "url")),
                   df.columns[0])
    lab_col = next((c for c in df.columns if c in ("label", "class", "type")), None)
    out = pd.DataFrame({"domain": df[dom_col].map(registrable)})
    if lab_col:
        cls = df[lab_col].astype(str).str.lower()
        out["label"] = (~cls.str.contains("benign|legit")).astype(int)
        out["family"] = cls.where(out["label"] == 1)
    else:
        out["label"] = 1
        out["family"] = "cic_unlabelled"
    out["source"] = "cic_bell_dns2021"
    return _frame(out.drop_duplicates("domain"))


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------

def combine(frames, drop_cross_label_conflicts: bool = True) -> pd.DataFrame:
    """Merge sources, resolve duplicates, report conflicts.

    A domain appearing as both benign and malicious is a label conflict. These
    are dropped rather than arbitrated: keeping them injects noise into the
    ground truth that no model can overcome and that inflates the apparent
    irreducible error.
    """
    df = pd.concat([f for f in frames if len(f)], ignore_index=True)
    df["domain"] = df["domain"].astype(str)
    df = df[df["domain"].str.contains(r"\.", regex=True)]
    df = df[df["domain"].str.len().between(4, 253)]

    conflicts = df.groupby("domain")["label"].nunique()
    conflicting = set(conflicts[conflicts > 1].index)
    if drop_cross_label_conflicts and conflicting:
        df = df[~df["domain"].isin(conflicting)]

    df = df.sort_values("label", ascending=False).drop_duplicates("domain", keep="first")
    df.attrs["n_conflicts_dropped"] = len(conflicting)
    return df.reset_index(drop=True)


def probe_universe(df: pd.DataFrame, n_total: int = 50_000,
                   malicious_fraction: float = 0.30, seed: int = 42,
                   min_per_family: int = 50) -> pd.DataFrame:
    """Stratified subset for the (expensive) certificate probe.

    Probing everything is not feasible; probing a biased subset is worse than
    probing fewer domains. Malicious rows are sampled per family so that no
    single large family dominates and the family-disjoint split stays viable.
    """
    rng_seed = seed
    n_mal = int(n_total * malicious_fraction)
    mal = df[df["label"] == 1]
    ben = df[df["label"] == 0]

    mal = mal.assign(_f=mal["family"].fillna("unknown"))
    per_fam = max(min_per_family, n_mal // max(mal["_f"].nunique(), 1))
    parts = [g.sample(min(len(g), per_fam), random_state=rng_seed)
             for _, g in mal.groupby("_f", sort=False)]
    mal_s = pd.concat(parts, ignore_index=True).drop(columns="_f")
    if len(mal_s) > n_mal:
        mal_s = mal_s.sample(n_mal, random_state=rng_seed)

    ben_s = ben.sample(min(len(ben), n_total - len(mal_s)), random_state=rng_seed)
    out = pd.concat([mal_s, ben_s], ignore_index=True).sample(frac=1, random_state=rng_seed)
    return out.reset_index(drop=True)
