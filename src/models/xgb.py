"""XGBoost trainer: the workhorse for the heterogeneous tabular features."""
from __future__ import annotations

import numpy as np
import xgboost as xgb


def build(params: dict, y_train=None):
    p = dict(params)
    p.pop("name", None)
    if p.get("scale_pos_weight") == "auto":
        if y_train is None:
            p.pop("scale_pos_weight")
        else:
            pos = float(np.sum(y_train == 1))
            p["scale_pos_weight"] = float(np.sum(y_train == 0)) / max(pos, 1.0)
    return xgb.XGBClassifier(enable_categorical=True, **p)


def fit(model, X_train, y_train, X_val, y_val, verbose: int = 100):
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=verbose)
    return model
