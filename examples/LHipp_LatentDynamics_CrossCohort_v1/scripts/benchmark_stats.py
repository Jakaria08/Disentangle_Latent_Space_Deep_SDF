#!/usr/bin/env python3
"""Statistics for the benchmark reports: subject bootstrap, paired tests, Holm correction.

The unit is always the subject (one row per subject, already averaged over seeds where needed).
No torch, so these are testable anywhere.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy import stats


def bootstrap_mean_ci(values: Iterable[float], draws: int = 2000, seed: int = 12345) -> tuple[float, float]:
    values = np.asarray(list(values), dtype=np.float64)
    if len(values) < 2 or draws <= 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))


def paired_difference(a: Iterable[float], b: Iterable[float], draws: int = 2000, seed: int = 12345) -> dict[str, float]:
    """a - b on the same subjects: mean, subject-bootstrap 95% CI, Wilcoxon signed-rank p."""
    a, b = np.asarray(list(a), dtype=np.float64), np.asarray(list(b), dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("paired arrays must align subject by subject")
    diff = a - b
    low, high = bootstrap_mean_ci(diff, draws, seed)
    if len(diff) < 2 or np.allclose(diff, 0.0):
        p = 1.0
    else:
        p = float(stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided").pvalue)
    return {"n": int(len(diff)), "mean_difference": float(diff.mean()), "ci95_low": low, "ci95_high": high, "wilcoxon_p": p}


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values (monotone, capped at 1)."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted, running = {}, 0.0
    for rank, (key, p) in enumerate(items):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[key] = running
    return adjusted


def correlations(x: Iterable[float], y: Iterable[float]) -> dict[str, float]:
    x, y = np.asarray(list(x), dtype=np.float64), np.asarray(list(y), dtype=np.float64)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return {"n": int(len(x)), "pearson_r": float("nan"), "spearman_rho": float("nan")}
    return {"n": int(len(x)), "pearson_r": float(stats.pearsonr(x, y)[0]), "spearman_rho": float(stats.spearmanr(x, y)[0])}
