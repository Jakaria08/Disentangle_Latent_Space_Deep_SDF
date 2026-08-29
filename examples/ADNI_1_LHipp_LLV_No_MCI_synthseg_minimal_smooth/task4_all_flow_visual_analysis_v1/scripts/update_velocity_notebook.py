#!/usr/bin/env python3
"""Replace only the instantaneous-velocity cells in the results notebook."""

from __future__ import annotations

import copy
import json
from pathlib import Path


TASK = Path(__file__).resolve().parents[1]
NOTEBOOK = TASK / "notebooks" / "all_flow_results.ipynb"


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}


CURRENT_BLOCK = [
    markdown(r"""## 7. Instantaneous velocity: definitions and limits

For an encoded shape $z$ at age $t$ and condition $d$, the model-implied latent generator is

$$v_\theta(z,t,d)=\left.\partial_\tau\Phi_\theta(z,t,\tau,d)\right|_{\tau=t}.$$

For DirectC4, $\Phi(z,s,t,d)=z+(t-s)\phi(z,s,t,d)$, so the diagonal value is $v_\theta(z,t,d)=\phi(z,t,t,d)$. It describes what the fitted model predicts at that state; it is not a directly measured biological velocity.

The plots use **Observed (GT estimate)** for the dataset reference. At an interior visit it is computed from the recorded scans immediately before and after that visit; endpoints use the nearest recorded interval. Therefore, “GT” in the plot labels means *computed from observed longitudinal scans*. It is not a direct measurement of continuous instantaneous motion between scans. Current intervals range from about 0.47 to 3.36 years.

Latent results use standardized-coordinate units per year and should be interpreted within each representation. Surface results decode a 0.05-year model step and report signed normal motion in mm/year. For INR, the equivalent implicit level-set derivative is used. Negative surface-normal velocity means inward motion; positive means outward motion.
"""),
    markdown(r"""### Latent speed magnitude

**RMS standardized coordinate/year** is the square root of the mean squared velocity across latent coordinates. It measures total latent-change speed without displaying latent width. The horizontal axis is the Observed (GT estimate); the vertical axis is the model generator. The dashed identity line indicates equal magnitude. Points below it mean the model changes more slowly than the observed estimate.
"""),
    code("H.velocity_scatter(D).show()\n"),
    markdown(r"""### Latent direction agreement

**Cosine similarity** compares the directions of the full latent velocity vectors and ignores their magnitudes. A value of $+1$ means the model and Observed (GT estimate) point in the same latent direction, $0$ means no directional alignment, and $-1$ means opposite directions.
"""),
    code("H.velocity_alignment(D).show()\n"),
    markdown(r"""### Latent speed trend with age

Each point is the median latent RMS speed within an age band. Solid/dashed series distinguish the model generator from the Observed (GT estimate), while the two panels separate CN and AD. This shows whether speed rises or falls with age; it does not establish a continuously observed trajectory between visits.
"""),
    code("H.velocity_by_age(D).show()\n"),
    markdown(r"""### Same-shape latent condition effect

For every observed shape and age, the model is evaluated twice: once with the CN condition and once with the AD condition. The metric is the RMS magnitude of $v_{AD}-v_{CN}$ per coordinate/year. Zero means the condition input does not change the local field. This is a controlled model sensitivity, not the difference between the actual AD and CN subject groups.
"""),
    code("H.condition_velocity_gap(D).show()\n"),
    markdown(r"""### Surface speed trend with age

**Area-weighted RMS normal speed (mm/year)** measures the magnitude of inward/outward motion across the whole surface. Squaring removes direction, so both contraction and expansion increase this value. Model and Observed (GT estimate) use the same physical unit and can be compared across representations.
"""),
    code("H.surface_speed_by_age(D).show()\n"),
    markdown(r"""### Signed surface trend with age

**Area-weighted mean normal velocity (mm/year)** preserves direction. Negative values indicate average inward movement or contraction; positive values indicate average outward movement or expansion. The zero line marks no net signed surface movement. Local inward and outward changes can cancel in this mean, so it should be read together with RMS speed.
"""),
    code("H.surface_signed_trend_by_age(D).show()\n"),
    markdown(r"""### Surface agreement metrics

Four complementary metrics are shown:

- **MAE (mm/year):** mean absolute vertex-wise normal-velocity error; lower is better.
- **Spatial Pearson correlation:** similarity of the surface pattern after removing overall offset and scale; $+1$ is best, $0$ indicates no linear pattern match, and $-1$ is reversed.
- **Sign agreement:** surface-area fraction where model and Observed (GT estimate) agree on inward versus outward motion; higher is better.
- **Speed ratio:** model RMS speed divided by Observed RMS speed; 1 is magnitude matched, below 1 is slower, and above 1 is faster.
"""),
    code("H.surface_velocity_agreement(D).show()\n"),
    markdown(r"""### Same-shape surface condition effect with age

This is the area-weighted RMS of the surface field obtained by subtracting CN-condition velocity from AD-condition velocity while holding shape and age fixed. Larger values mean stronger use of the disease-condition input. The CN and AD panels indicate whether that controlled sensitivity depends on the type of shape supplied to the model.
"""),
    code("H.surface_condition_effect_by_age(D).show()\n"),
    markdown(r"""### Spatial AD-minus-CN velocity pattern

Each mesh shows one average field per method. For both Observed (GT estimate) and models, the displayed value is the mean instantaneous normal velocity in AD subjects minus the mean in CN subjects at corresponding surface locations. Negative values mean AD is more inward-moving than CN; positive values mean AD is more outward-moving. This is an observed-group comparison and is different from switching the condition on the same shape.
"""),
    code("H.instantaneous_surface_group_maps(D).show()\n"),
    markdown(r"""### Internal diagonal-generator consistency check

This compares the analytic diagonal generator with the velocity obtained from a 0.05-year model transport step. The metric is RMS difference per coordinate/year on a logarithmic axis; smaller is better. It verifies numerical implementation and local continuity only. It does **not** measure agreement with observed anatomy.
"""),
    code("""H.diagonal_check(D).show()
velocity_audit = D['velocity_summary'][[
    'method_label','diagnosis','velocity_rmse_per_coordinate_per_year_mean',
    'velocity_cosine_mean','diagonal_fd_rmse_per_coordinate_per_year_mean'
]].rename(columns={
    'method_label': 'Method', 'diagnosis': 'Diagnosis',
    'velocity_rmse_per_coordinate_per_year_mean': 'Model vs Observed RMS error/coordinate/year',
    'velocity_cosine_mean': 'Mean direction cosine',
    'diagonal_fd_rmse_per_coordinate_per_year_mean': 'Internal diagonal-step RMS difference/coordinate/year',
})
display(velocity_audit)
"""),
]


LEGACY_BLOCK = [
    markdown(r"""### Instantaneous velocity in the existing Cocycle/ODE/BrainODE cohort

These models use a different completed test cohort and protocol, so their values remain separate from the current matched analysis. Cocycle models use their diagonal generator; Latent ODE and BrainODE use their learned ODE vector field. **Observed (GT estimate)** has the same meaning as above: it is calculated from neighboring recorded scans, not directly measured continuous motion.
"""),
    markdown(r"""#### Latent speed magnitude

The axes show RMS standardized latent speed per coordinate/year. Dividing the standardized vector norm by the square root of coordinate count removes the direct width effect. The dashed line indicates equal model and Observed (GT estimate) magnitude. This normalization still does not make the different latent coordinate systems anatomically identical.
"""),
    code("H.legacy_velocity_plot(D).show()\n"),
    markdown(r"""#### Latent direction agreement

Cosine similarity measures whether the model vector and Observed (GT estimate) point in the same latent direction: $+1$ is the same direction, $0$ is no alignment, and $-1$ is opposite. It does not assess speed magnitude.
"""),
    code("H.legacy_velocity_alignment(D).show()\n"),
    markdown(r"""#### Latent speed trend with age

Each point is the median RMS standardized-coordinate speed in an age band. Model and Observed (GT estimate) lines are shown separately for CN and AD. Use this for within-architecture age trends, not for pooling this cohort with the current experiment.
"""),
    code("H.legacy_velocity_by_age(D).show()\n"),
    markdown(r"""#### Surface speed trend with age

Area-weighted RMS normal speed reports the magnitude of surface movement in mm/year and can be compared across PCA, INR, Latent ODE, and BrainODE decoders. It has no inward/outward sign.
"""),
    code("H.surface_speed_by_age(D, legacy=True).show()\n"),
    markdown(r"""#### Signed surface trend with age

Area-weighted mean normal velocity preserves direction: negative is inward contraction and positive is outward expansion. Because inward and outward regions can cancel, interpret this together with RMS surface speed.
"""),
    code("H.surface_signed_trend_by_age(D, legacy=True).show()\n"),
    markdown(r"""#### Surface agreement metrics

**MAE** is vertex-wise error in mm/year (lower is better); **Pearson correlation** measures spatial-pattern agreement; **sign agreement** is the area fraction with matching inward/outward direction; and **speed ratio** compares model RMS magnitude with Observed (GT estimate), with 1 indicating equal speed.
"""),
    code("H.surface_velocity_agreement(D, legacy=True).show()\n"),
    markdown(r"""#### Same-shape surface condition effect with age

The model is evaluated under AD and CN conditions at the same shape and age. The plotted RMS difference in mm/year measures controlled condition sensitivity. It is not the actual AD-subject minus CN-subject group difference.
"""),
    code("H.surface_condition_effect_by_age(D, legacy=True).show()\n"),
    markdown(r"""#### Spatial AD-minus-CN velocity pattern

Each mesh contains one average field. The value is mean AD-subject normal velocity minus mean CN-subject normal velocity at each corresponding location. Negative means more inward movement in AD; positive means more outward movement in AD. The Observed (GT estimate) mesh is calculated from the recorded scans.
"""),
    code("H.instantaneous_surface_group_maps(D, legacy=True).show()\n"),
]


def main() -> int:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    original = copy.deepcopy(notebook["cells"])
    cells = notebook["cells"]
    current_start = next(i for i, cell in enumerate(cells) if "## 7. Instantaneous velocity" in "".join(cell.get("source", [])))
    current_end = next(i for i, cell in enumerate(cells) if "## 8. One averaged surface-change mesh" in "".join(cell.get("source", [])))
    preserved_before = copy.deepcopy(cells[:current_start])
    preserved_after = copy.deepcopy(cells[current_end:])
    cells[current_start:current_end] = CURRENT_BLOCK
    if cells[:current_start] != preserved_before:
        raise RuntimeError("Unexpected change before current velocity block")

    legacy_index = next(i for i, cell in enumerate(cells) if "H.legacy_endpoint_plot" in "".join(cell.get("source", [])))
    legacy_source = "".join(cells[legacy_index].get("source", []))
    if "H.legacy_horizon_plot" not in legacy_source or "H.legacy_velocity_plot" not in legacy_source:
        raise RuntimeError("Unexpected legacy comparison cell")
    cells[legacy_index] = code("H.legacy_endpoint_plot(D, 'assd').show()\nH.legacy_horizon_plot(D, 'assd').show()\n")
    cells[legacy_index + 1:legacy_index + 1] = LEGACY_BLOCK

    interpretation = next(cell for cell in cells if "## 13. Interpretation contract" in "".join(cell.get("source", [])))
    source = "".join(interpretation["source"])
    source = source.replace(
        "- Velocity magnitude uses RMS per coordinate to avoid representation-width bias.\n",
        "- Latent velocity magnitude uses RMS per standardized coordinate; surface velocity uses mm/year.\n"
        "- Observed (GT estimate) is calculated from recorded longitudinal scans and is not directly measured continuous motion.\n",
    )
    interpretation["source"] = source.splitlines(keepends=True)

    # Confirm that every original cell outside the authorized velocity locations still appears unchanged.
    allowed_original = set(range(current_start, current_end))
    legacy_original = next(i for i, cell in enumerate(original) if "H.legacy_endpoint_plot" in "".join(cell.get("source", [])))
    interpretation_original = next(i for i, cell in enumerate(original) if "## 13. Interpretation contract" in "".join(cell.get("source", [])))
    allowed_original.update({legacy_original, interpretation_original})
    unchanged = [cell for i, cell in enumerate(original) if i not in allowed_original]
    remaining = copy.deepcopy(cells)
    for cell in unchanged:
        try:
            remaining.remove(cell)
        except ValueError as exc:
            raise RuntimeError("A non-velocity cell would change") from exc

    NOTEBOOK.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "notebook": str(NOTEBOOK),
        "original_cells": len(original),
        "updated_cells": len(cells),
        "preserved_non_velocity_cells": len(unchanged),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
