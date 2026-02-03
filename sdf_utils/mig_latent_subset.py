# Wrapper utilities for computing MIG per latent subset.

from . import mig as mig_metric


def mig_by_subset(factors, codes, subsets, subset_cfg):
    """
    Compute MIG score dict per subset.

    factors: (N, F)
    codes:   (N, D)
    subsets: dict name -> list of latent indices
    subset_cfg: dict name -> {"label_indices": list[int], "continuous_factors": bool, "nb_bins": int}
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
        results[name] = mig_metric.mig(
            factors_sub,
            codes_sub,
            continuous_factors=cfg.get("continuous_factors", True),
            continuous_codes=True,
            nb_bins=cfg.get("nb_bins", 10),
        )
    return results
