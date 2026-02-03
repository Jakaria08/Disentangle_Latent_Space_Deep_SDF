# Wrapper utilities for computing DCI per latent subset.

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score

from . import dci as dci_metric


def dci_by_subset(factors, codes, subsets, subset_cfg):
    """
    Compute DCI score dict per subset.

    factors: (N, F)
    codes:   (N, D)
    subsets: dict name -> list of latent indices
    subset_cfg: dict name -> {"label_indices": list[int], "continuous_factors": bool}
    """
    results = {}
    if factors is None or codes is None:
        return results
    for name, dims in subsets.items():
        cfg = subset_cfg.get(name)
        if cfg is None:
            continue
        label_indices = cfg.get("label_indices")
        if label_indices is None:
            continue
        if isinstance(label_indices, int):
            label_indices = [label_indices]
        factors_sub = factors[:, label_indices]
        codes_sub = codes[:, dims]
        results[name] = dci_metric.dci(
            factors_sub,
            codes_sub,
            continuous_factors=cfg.get("continuous_factors", True),
        )
    return results


def predictability_by_subset(factors, codes, subsets, subset_cfg):
    """
    Return simple predictability (accuracy for classification or R2 for regression)
    per subset using linear/logistic models.
    """
    results = {}
    if factors is None or codes is None:
        return results
    for name, dims in subsets.items():
        cfg = subset_cfg.get(name)
        if cfg is None:
            continue
        label_indices = cfg.get("label_indices")
        if label_indices is None:
            continue
        if isinstance(label_indices, int):
            label_indices = [label_indices]
        factors_sub = factors[:, label_indices]
        codes_sub = codes[:, dims]
        if factors_sub.shape[1] != 1:
            # only support single factor per subset
            continue
        y = factors_sub[:, 0]
        if cfg.get("continuous_factors", True):
            model = LinearRegression()
            model.fit(codes_sub, y)
            y_pred = model.predict(codes_sub)
            results[name] = float(r2_score(y, y_pred))
        else:
            y_int = y.astype(int)
            model = LogisticRegression(max_iter=1000, solver="liblinear")
            model.fit(codes_sub, y_int)
            y_pred = model.predict(codes_sub)
            results[name] = float(accuracy_score(y_int, y_pred))
    return results
