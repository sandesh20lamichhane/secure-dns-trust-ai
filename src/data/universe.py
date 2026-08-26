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


_IPV4 = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")


def is_ip_literal(host: str) -> bool:
    """URLhaus carries many bare-IP URLs. An IP has no registrable domain, no
    DNS name to score lexically, and no TLSA record, so it falls outside the
    unit of analysis and is dropped rather than mangled into a pseudo-domain."""
    h = str(host).strip().lower()
    return bool(_IPV4.match(h)) or ":" in h


def registrable(domain: str) -> str:
    """Reduce a hostname to its registrable domain.

    A crude public-suffix approximation - adequate here because it is applied
    identically to every source, so it cannot bias one class relative to
    another. Swap in the `publicsuffix2` package if exactness is needed.
    """
    d = str(domain).strip().lower().rstrip(".")
    d = re.sub(r"^https?://", "", d).split("/")[0].split(":")[0]
    if is_ip_literal(d):
        return ""
    parts = d.split(".")
    if len(parts) < 3:
        return d
    if ".".join(parts[-2:]) in _TWO_LEVEL:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def _frame(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()                      # avoid mutating a caller's slice
    for col in SCHEMA:
        if col not in rows.columns:
            rows[col] = None
    return rows[SCHEMA]


# --------------------------------------------------------------------------
# Benign sources
# --------------------------------------------------------------------------

def load_tranco(path, top_n: int = 1_000_000) -> pd.DataFrame:
    """Tranco CSV: rank,domain (no header). Keep the list ID in the filename -
    it is the permanent citation for the exact snapshot used.

    Full 1M by default. The benign side is the binding constraint on both
    diversity and class ratio, so it is taken deep while the malicious side is
    capped per family.
    """
    df = pd.read_csv(path, header=None, names=["rank", "domain"], nrows=top_n)
    df["domain"] = df["domain"].map(registrable)
    df["label"] = 0
    df["source"] = f"tranco:{Path(path).stem}"
    return _frame(df.drop_duplicates("domain"))


# --------------------------------------------------------------------------
# Malicious sources
# --------------------------------------------------------------------------

def load_umudga(root, per_family: int = 20_000, size: str = "50000",
                families=None) -> pd.DataFrame:
    """UMUDGA. Layout is <family>/list/<count>.txt, one bare domain per line.

    Note the sibling `arff/` and `csv/` directories are ignored: those hold
    UMUDGA's own precomputed feature sets, and this study computes its features
    from the raw strings so that the same pipeline applies to every source.

    20k per family rather than the full 1M file. Within a family every domain
    comes from one algorithm under a different seed, so information saturates
    quickly; what generalises is the number of families, which the
    family-disjoint split holds out. Loading more rows per family inflates the
    corpus without adding discriminative signal and pushes the class ratio far
    from any realistic base rate. The sample-size ablation tests this claim
    rather than asserting it.
    """
    root = Path(root)
    frames = []
    for list_dir in sorted(root.rglob("list")):
        if not list_dir.is_dir():
            continue
        family = list_dir.parent.name.lower()
        if families and family not in families:
            continue
        f = list_dir / f"{size}.txt"
        if not f.exists():
            candidates = sorted(list_dir.glob("*.txt"),
                                key=lambda p: p.stat().st_size, reverse=True)
            if not candidates:
                continue
            f = candidates[0]
        col = pd.read_csv(f, header=None, names=["domain"],
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
    """PhishTank / OpenPhish / URLhaus.

    URLhaus prefixes its dump with a comment block and hides the column header
    inside it, so a naive read_csv picks the wrong column and silently returns
    a handful of rows. The header is recovered from the comment block when
    present.

    URLs are reduced to registrable domains, losing per-URL granularity. That
    is correct here: the unit of analysis is the domain and its certificate,
    not the individual URL.
    """
    from io import StringIO

    path = str(path)
    lines = Path(path).read_text(errors="replace").splitlines()
    comments = [l for l in lines if l.lstrip().startswith("#")]
    data = [l for l in lines if l.strip() and not l.lstrip().startswith("#")]
    if not data:
        return _frame(pd.DataFrame(columns=["domain"]))

    if path.endswith(".csv"):
        header = None
        for c in reversed(comments):                    # header is the last comment
            fields = [x.strip() for x in c.lstrip("#").strip().split(",")]
            if len(fields) > 2 and any("url" in f.lower() for f in fields):
                header = fields
                break
        buf = StringIO("\n".join(data))
        if header:
            df = pd.read_csv(buf, header=None, names=header, on_bad_lines="skip")
        else:
            df = pd.read_csv(StringIO("\n".join(lines)), on_bad_lines="skip")
        url_col = next((c for c in df.columns if "url" in str(c).lower()
                        and "link" not in str(c).lower()), df.columns[0])
        urls = df[url_col].astype(str)
        date_col = next((c for c in df.columns
                         if "dateadded" in str(c).lower() or "date" in str(c).lower()
                         or "added" in str(c).lower()), None)
        dates = df[date_col] if date_col is not None else None
    else:
        urls = pd.Series(data, dtype=str)
        dates = None

    out = pd.DataFrame({"domain": urls.map(registrable)})
    out["first_seen"] = list(dates) if dates is not None else None
    out["label"] = 1
    out["family"] = family
    out["source"] = source
    out = out[out["domain"].str.contains(r"\.", regex=True, na=False)]
    out = out[out["domain"].str.len() > 0]
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
