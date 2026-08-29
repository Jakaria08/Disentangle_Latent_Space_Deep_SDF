from __future__ import annotations

from compare_models import paired_bootstrap
from evaluate import balanced_surface_indices


def _records() -> list[dict[str, str]]:
    rows = []
    for diagnosis in ("CN", "AD"):
        for pair_type in ("adjacent", "nonadjacent"):
            for subject in ("01", "02", "03"):
                for visit in range(2):
                    rows.append(
                        {
                            "diagnosis": diagnosis,
                            "pair_type": pair_type,
                            "subject": f"{diagnosis}_{subject}",
                            "source_scan_id": f"{diagnosis}_{subject}_{visit}",
                            "target_scan_id": f"{diagnosis}_{subject}_{visit + 1}",
                        }
                    )
    return rows


def test_surface_subset_is_deterministic_and_stratified():
    records = _records()
    selected = balanced_surface_indices(records, limit=8, seed=1701)
    assert selected == balanced_surface_indices(records, limit=8, seed=1701)
    assert len(selected) == len(set(selected)) == 8
    assert {(records[index]["diagnosis"], records[index]["pair_type"]) for index in selected} == {
        ("CN", "adjacent"),
        ("CN", "nonadjacent"),
        ("AD", "adjacent"),
        ("AD", "nonadjacent"),
    }


def test_bootstrap_respects_metric_direction():
    spiral = {"a": 1.0, "b": 2.0}
    adaptive = {"a": 2.0, "b": 3.0}
    high = paired_bootstrap(spiral, adaptive, samples=100, seed=1, higher_is_better=True)
    low = paired_bootstrap(spiral, adaptive, samples=100, seed=1, higher_is_better=False)
    assert high["probability_adaptive_better"] == 1.0
    assert low["probability_adaptive_better"] == 0.0

