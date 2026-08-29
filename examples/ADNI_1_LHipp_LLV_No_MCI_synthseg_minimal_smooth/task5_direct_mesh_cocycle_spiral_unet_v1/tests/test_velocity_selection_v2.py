from __future__ import annotations

import copy

import numpy as np
import torch

import common as C
import objectives as O
from data import PreparedSplit
from train import validate_config


class ConstantVelocityModel:
    def __init__(self, velocity: tuple[float, float, float]):
        self.faces = torch.tensor([[0, 1, 2]], dtype=torch.long)
        self.velocity = torch.tensor(velocity, dtype=torch.float32)

    def eval(self):
        return self

    def instantaneous_velocity(
        self,
        vertices: torch.Tensor,
        age: torch.Tensor,
        disease: torch.Tensor,
    ) -> torch.Tensor:
        del age, disease
        return self.velocity.to(vertices).reshape(1, 1, 3).expand_as(vertices)


def small_split() -> PreparedSplit:
    vertices = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    reference = torch.zeros_like(vertices)
    reference[:, :, 2] = -1.0
    return PreparedSplit(
        split="val",
        vertices=vertices,
        ages=torch.tensor([70.0, 75.0]),
        years_from_baseline=torch.tensor([0.0, 0.0]),
        labels=torch.tensor([0.0, 1.0]),
        volumes=torch.tensor([1.0, 1.0]),
        velocity_reference=reference,
        velocity_reference_weight=torch.tensor([1.0, 0.25]),
        scan_ids=np.asarray(["cn", "ad"]),
        subject_ids=np.asarray(["1", "2"]),
        diagnoses=np.asarray(["CN", "AD"]),
        visit_orders=np.asarray([0, 0]),
        subject_visit_offsets=np.asarray([0, 1, 2]),
    )


def first_last_fixture() -> dict:
    return {
        "groups": {
            "CN": {
                "error_to_nochange_ratio": 0.90,
                "predicted_log_volume_rate_per_year": -0.01,
            },
            "AD": {
                "error_to_nochange_ratio": 0.92,
                "predicted_log_volume_rate_per_year": -0.04,
            },
            "overall": {"flipped_face_fraction": 0.0},
        }
    }


def test_velocity_selection_is_normalized_against_zero_velocity():
    split = small_split()
    perfect = O.evaluate_velocity_selection(ConstantVelocityModel((0.0, 0.0, -1.0)), split, batch_size=2)
    zero = O.evaluate_velocity_selection(ConstantVelocityModel((0.0, 0.0, 0.0)), split, batch_size=2)
    perfect_overall = perfect["groups"]["overall"]
    zero_overall = zero["groups"]["overall"]
    assert perfect_overall["normalized_error_ratio"] == 0.0
    assert abs(perfect_overall["speed_ratio"] - 1.0) < 1.0e-7
    assert abs(perfect_overall["vector_cosine"] - 1.0) < 1.0e-7
    assert abs(zero_overall["vector_error_to_zero_ratio"] - 1.0) < 1.0e-7
    assert abs(zero_overall["normal_error_to_zero_ratio"] - 1.0) < 1.0e-7
    assert zero_overall["speed_ratio"] == 0.0
    assert abs(float(perfect_overall["reference_weight_sum"]) - 1.25) < 1.0e-7


def test_velocity_v2_score_and_feasibility_gates():
    config = C.read_json(C.TASK_ROOT / "configs" / "spiral_direct_c4_velocity_v2_s42.json")
    validate_config(config)
    defects = {"relative_cocycle_defect_mean": 0.005, "relative_inverse_defect_mean": 0.006}
    velocity = {"groups": {"overall": {"normalized_error_ratio": 0.8, "speed_ratio": 0.7}}}
    report = O.validation_selection(first_last_fixture(), defects, config, velocity)
    expected = 0.91 + 0.01 * 0.005 + 0.05 * 0.8
    assert abs(report["score"] - expected) < 1.0e-10
    assert report["feasible"] is True
    assert all(report["gates"].values())

    collapsed = copy.deepcopy(velocity)
    collapsed["groups"]["overall"]["speed_ratio"] = 0.2
    assert O.validation_selection(first_last_fixture(), defects, config, collapsed)["feasible"] is False

    wrong_order = first_last_fixture()
    wrong_order["groups"]["AD"]["predicted_log_volume_rate_per_year"] = 0.01
    assert O.validation_selection(wrong_order, defects, config, velocity)["feasible"] is False


def test_v1_validation_score_is_unchanged_without_velocity_metrics():
    config = C.read_json(C.TASK_ROOT / "configs" / "spiral_direct_c4_s42.json")
    defects = {"relative_cocycle_defect_mean": 0.005, "relative_inverse_defect_mean": 0.006}
    score, feasible = O.validation_score(first_last_fixture(), defects, config)
    assert abs(score - (0.91 + 0.01 * 0.005)) < 1.0e-10
    assert feasible is True


def test_flipped_face_fraction_detects_orientation_reversal():
    source = small_split().vertices[:1]
    faces = torch.tensor([[0, 1, 2]], dtype=torch.long)
    assert O.flipped_face_fraction(source, source.clone(), faces).item() == 0.0
    reversed_prediction = source.clone()
    reversed_prediction[:, 2, 1] = -1.0
    assert O.flipped_face_fraction(source, reversed_prediction, faces).item() == 1.0
