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

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score


def _normalize_rows(mat, axis=1):
    denom = mat.sum(axis=axis, keepdims=True)
    denom[denom == 0] = 1.0
    return mat / denom


def _entropy(p):
    p = np.clip(p, 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def dci(factors, codes, continuous_factors=True):
    """Compute DCI scores (disentanglement/completeness/informativeness).

    factors:   (N, F) array of ground-truth factors.
    codes:     (N, C) array of latent codes.
    """
    if factors.ndim != 2 or codes.ndim != 2:
        raise ValueError("factors and codes must be 2D arrays.")

    num_factors = factors.shape[1]
    num_codes = codes.shape[1]

    importance = np.zeros((num_codes, num_factors))
    informativeness = []

    for f in range(num_factors):
        y = factors[:, f]
        if continuous_factors:
            model = LinearRegression()
            model.fit(codes, y)
            y_pred = model.predict(codes)
            informativeness.append(r2_score(y, y_pred))
            w = np.abs(model.coef_)
        else:
            y_int = y.astype(int)
            model = LogisticRegression(
                max_iter=1000, solver="liblinear", multi_class="ovr"
            )
            model.fit(codes, y_int)
            y_pred = model.predict(codes)
            informativeness.append(accuracy_score(y_int, y_pred))
            w = np.abs(model.coef_)
            if w.ndim == 2:
                w = w.mean(axis=0)
        importance[:, f] = w

    # Disentanglement per code (how focused each code is on a single factor)
    p_c = _normalize_rows(importance, axis=1)
    if num_factors > 1:
        disent_per_code = 1.0 - (_entropy(p_c) / np.log(num_factors))
    else:
        disent_per_code = np.ones((num_codes,))

    code_importance = importance.sum(axis=1)
    if code_importance.sum() > 0:
        disentanglement = float(
            np.sum(disent_per_code * code_importance) / code_importance.sum()
        )
    else:
        disentanglement = float("nan")

    # Completeness per factor (how concentrated factor info is in a few codes)
    p_f = _normalize_rows(importance.T, axis=1)
    if num_codes > 1:
        comp_per_factor = 1.0 - (_entropy(p_f) / np.log(num_codes))
    else:
        comp_per_factor = np.ones((num_factors,))

    factor_importance = importance.sum(axis=0)
    if factor_importance.sum() > 0:
        completeness = float(
            np.sum(comp_per_factor * factor_importance) / factor_importance.sum()
        )
    else:
        completeness = float("nan")

    informativeness_mean = float(np.mean(informativeness)) if informativeness else float("nan")

    return {
        "disentanglement": disentanglement,
        "completeness": completeness,
        "informativeness": informativeness_mean,
        "importance_matrix": importance,
    }
