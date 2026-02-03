# Wrapper utilities for computing SAP per latent subset.

from . import sap as sap_metric


def sap_by_subset(factors, codes, subsets, subset_cfg):
    """
    Compute SAP score per subset.

    factors: (N, F) full label matrix
    codes:   (N, D) full latent matrix
    subsets: dict name -> list of latent indices
    subset_cfg: dict name -> {
        "label_indices": list[int],
        "regression": bool,
        "continuous_factors": bool,
        "nb_bins": int
    }
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
        results[name] = sap_metric.sap(
            factors_sub,
            codes_sub,
            continuous_factors=cfg.get("continuous_factors", True),
            nb_bins=cfg.get("nb_bins", 10),
            regression=cfg.get("regression", True),
        )
    return results
