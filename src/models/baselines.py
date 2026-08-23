"""Baselines required for the ablation. Not the contribution - the yardstick."""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def build(name: str, params: dict):
    if name == "logistic_regression":
        return Pipeline([("scale", StandardScaler()),
                         ("clf", LogisticRegression(**params))])
    if name == "random_forest":
        return RandomForestClassifier(**params)
    if name == "svm_rbf":
        params = {k: v for k, v in params.items() if k != "subsample_n"}
        return Pipeline([("scale", StandardScaler()),
                         ("clf", SVC(probability=True, **params))])
    raise ValueError(f"unknown baseline: {name}")
