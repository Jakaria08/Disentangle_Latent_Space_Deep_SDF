# coding=utf-8
# Copyright 2018 Ubisoft La Forge Authors.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np

from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import minmax_scale


def _get_bin_index(x, nb_bins):
    bins = np.linspace(0, 1, nb_bins + 1)
    return np.digitize(x, bins[:-1], right=False).astype(int)


def _entropy_discrete(x):
    return mutual_info_score(x, x)


def mig(
    factors,
    codes,
    continuous_factors=True,
    continuous_codes=True,
    nb_bins=10,
):
    """Compute MIG (Mutual Information Gap).

    factors: (N, F) array of ground-truth factors.
    codes:   (N, C) array of latent codes.
    """
    if factors.ndim != 2 or codes.ndim != 2:
        raise ValueError("factors and codes must be 2D arrays.")

    if continuous_factors:
        factors = minmax_scale(factors)
        factors = _get_bin_index(factors, nb_bins)
    else:
        factors = factors.astype(int)

    if continuous_codes:
        codes = minmax_scale(codes)
        codes = _get_bin_index(codes, nb_bins)
    else:
        codes = codes.astype(int)

    num_factors = factors.shape[1]
    num_codes = codes.shape[1]
    mi_matrix = np.zeros((num_factors, num_codes))

    for f in range(num_factors):
        for c in range(num_codes):
            mi_matrix[f, c] = mutual_info_score(factors[:, f], codes[:, c])

    mig_scores = []
    for f in range(num_factors):
        mi_sorted = np.sort(mi_matrix[f, :])
        top1 = mi_sorted[-1] if num_codes >= 1 else 0.0
        top2 = mi_sorted[-2] if num_codes >= 2 else 0.0
        ent = _entropy_discrete(factors[:, f])
        if ent == 0:
            mig_scores.append(0.0)
        else:
            mig_scores.append((top1 - top2) / ent)

    return {
        "mig": float(np.mean(mig_scores)) if mig_scores else float("nan"),
        "mi_matrix": mi_matrix,
    }
