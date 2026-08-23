"""Certificate features derived from collected TLS + CT observations.

Nothing here re-probes the network: this is a pure transformation of
data/01_collected/, which is what makes 02_interim reproducible.
"""
from __future__ import annotations

import pandas as pd

SHORT_VALIDITY_DAYS = 100     # Let's Encrypt-style 90-day certs
VERY_FRESH_DAYS = 7


def build(tls_df: pd.DataFrame, ct_df: pd.DataFrame | None = None) -> pd.DataFrame:
    df = tls_df.copy()
    if df.empty:
        return df

    df["has_certificate"] = True
    df["short_validity"] = df["validity_days"] < SHORT_VALIDITY_DAYS
    df["very_fresh_cert"] = df["cert_age_days"] < VERY_FRESH_DAYS
    df["cn_san_mismatch"] = ~df["cn_in_san"].fillna(False)
    df["weak_key"] = df["key_bits"].fillna(0).lt(2048) & df["key_algorithm"].eq("RSA")
    df["san_bucket"] = pd.cut(df["san_count"].fillna(0),
                              bins=[-1, 1, 2, 5, 20, 100, 1e9],
                              labels=["1", "2", "3-5", "6-20", "21-100", "100+"])

    if ct_df is not None and not ct_df.empty:
        df = df.merge(ct_df, on="domain", how="left", suffixes=("", "_ct"))
        df["ct_log_count"] = df["ct_log_count"].fillna(0)
        df["no_ct_presence"] = df["ct_log_count"].eq(0)

    return df


def align_with_universe(features: pd.DataFrame, all_domains) -> pd.DataFrame:
    """Domains with NO certificate must remain rows, not disappear.

    Absence of a certificate is itself informative; silently inner-joining them
    away would bias the evaluation set toward reachable hosts.
    """
    universe = pd.DataFrame({"domain": list(all_domains)})
    out = universe.merge(features, on="domain", how="left")
    out["has_certificate"] = out["has_certificate"].fillna(False)
    return out
