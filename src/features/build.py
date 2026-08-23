"""Assemble the final modelling matrix from feature groups named in features.yaml."""
from __future__ import annotations

import pandas as pd

from . import certificate as cert_features
from . import lexical


def assemble(domains: pd.DataFrame, tls_df=None, ct_df=None, dns_df=None,
             groups=("lexical", "dns", "tlsa", "certificate"),
             quarantined=()) -> pd.DataFrame:
    """domains: DataFrame with at least ['domain', 'label']."""
    out = domains[["domain", "label"]].copy()

    if "lexical" in groups:
        out = out.merge(lexical.extract_frame(out["domain"]), on="domain", how="left")

    if ("dns" in groups or "tlsa" in groups) and dns_df is not None:
        out = out.merge(dns_df, on="domain", how="left")

    if "certificate" in groups and tls_df is not None:
        cf = cert_features.build(tls_df, ct_df)
        cf = cert_features.align_with_universe(cf, out["domain"])
        out = out.merge(cf.drop(columns=["label"], errors="ignore"),
                        on="domain", how="left")

    dropped = [c for c in out.columns if c in set(quarantined)]
    if dropped:
        out = out.drop(columns=dropped)
        out.attrs["quarantined_dropped"] = dropped
    return out


def to_matrix(df: pd.DataFrame, categorical=("tld", "issuer_org", "key_algorithm",
                                             "sig_algorithm", "san_bucket")):
    """Return (X, y, feature_names). Categoricals become pandas 'category' dtype
    so XGBoost handles them natively - no target encoding, which leaks."""
    y = df["label"].values
    X = df.drop(columns=["label", "domain"], errors="ignore")
    for col in categorical:
        if col in X.columns:
            X[col] = X[col].astype("category")
    drop_cols = [c for c in X.columns if X[c].dtype == "object"]
    X = X.drop(columns=drop_cols)
    return X, y, list(X.columns)
