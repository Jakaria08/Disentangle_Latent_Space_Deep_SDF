#!/usr/bin/env python3
"""Fast no-training tests for the exact-SDF sampling ablation."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

import numpy as np


TEST_DIR = Path(__file__).resolve().parent
TASK_DIR = TEST_DIR.parent
SCRIPT_DIR = TASK_DIR / "scripts"
BASE_SCRIPTS = TASK_DIR.parent / "task2_inr_multires_single_field_v1" / "scripts"
for path in (SCRIPT_DIR, BASE_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ablation_common import load_matrix, materialize_config, require_bulk_path  # noqa: E402
from sampling_dataset import (  # noqa: E402
    _count_plan,
    sample_shell_stratified_pair,
    validate_shell_sampling,
)


class SamplingAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.matrix = load_matrix()
        cls.configs = {
            name: materialize_config(cls.matrix, name)
            for name in cls.matrix["experiments"]
        }

    def test_only_training_sampling_varies_between_phase1_arms(self) -> None:
        signatures = []
        for name in ("control_u030_n100", "medium_u005_n030", "tight_u002_n010"):
            config = copy.deepcopy(self.configs[name])
            for key in ("name", "description", "output_dir", "sampling", "sampling_ablation"):
                config.pop(key, None)
            signatures.append(config)
        self.assertEqual(signatures[0], signatures[1])
        self.assertEqual(signatures[0], signatures[2])
        self.assertEqual(
            self.configs["control_u030_n100"]["latent_fit"]["sampling"],
            self.configs["tight_u002_n010"]["latent_fit"]["sampling"],
        )

    def test_population_transfer_policy_is_unambiguous(self) -> None:
        for config in self.configs.values():
            settings = config["population_initialization"]
            self.assertTrue(settings["load_model"])
            self.assertTrue(settings["load_train_latents"])
            self.assertFalse(settings["load_optimizer"])
            self.assertFalse(settings["load_rng"])
            self.assertTrue(settings["reset_epoch"])
            self.assertEqual(config["latent_size"], 256)

    def test_all_runtime_outputs_are_on_bulk(self) -> None:
        for config in self.configs.values():
            require_bulk_path(config["output_dir"])
        with self.assertRaises(ValueError):
            require_bulk_path(TASK_DIR / "forbidden_runtime_output")

    def test_sampling_totals_and_fixed_eikonal(self) -> None:
        for config in self.configs.values():
            sampling = config["sampling"]
            broad = sum(int(sampling[key]) for key in (
                "global_near_samples_per_scene", "global_positive_samples_per_scene", "global_negative_samples_per_scene"
            ))
            near = sum(int(sampling[key]) for key in (
                "local_ultra_near_samples_per_scene", "local_positive_samples_per_scene", "local_negative_samples_per_scene"
            ))
            self.assertEqual(broad, 8192)
            self.assertEqual(near, 8192)
            self.assertEqual(config["eikonal"]["target_band"], 0.03)
            self.assertEqual(config["eikonal"]["weight"], 0.01)
            self.assertFalse(config["second_order"]["enabled"])

    def test_largest_remainder_count_plan_is_exact(self) -> None:
        shells = [
            {"fraction": 0.4},
            {"fraction": 0.3},
            {"fraction": 0.3},
        ]
        counts = _count_plan(257, shells)
        self.assertEqual(sum(counts), 257)
        self.assertLessEqual(max(counts) - min(counts), 26)

    def test_shell_sampler_respects_shell_counts_and_sign_balance(self) -> None:
        config = copy.deepcopy(self.configs["shell_stratified"])
        sampling = config["sampling"]
        sampling.update(
            {
                "global_near_samples_per_scene": 128,
                "global_positive_samples_per_scene": 64,
                "global_negative_samples_per_scene": 64,
                "local_ultra_near_samples_per_scene": 128,
                "local_positive_samples_per_scene": 64,
                "local_negative_samples_per_scene": 64,
            }
        )
        validate_shell_sampling(sampling)
        rng = np.random.default_rng(123)
        xyz = rng.uniform(-0.9, 0.9, size=(20000, 3)).astype(np.float32)
        positive_sdf = np.exp(rng.uniform(np.log(1.0e-5), np.log(0.5), size=20000)).astype(np.float32)
        negative_sdf = -np.exp(rng.uniform(np.log(1.0e-5), np.log(0.5), size=20000)).astype(np.float32)
        pos = np.column_stack((xyz, positive_sdf)).astype(np.float32)
        neg = np.column_stack((xyz[::-1], negative_sdf)).astype(np.float32)
        broad, near = sample_shell_stratified_pair(
            pos,
            neg,
            sampling,
            np.random.default_rng(456),
            config["network_specs"]["grid_aabb"],
            config["network_specs"]["sampling_balance_resolution"],
        )
        self.assertEqual(broad.shape, (256, 4))
        self.assertEqual(near.shape, (256, 4))
        for samples, group in ((broad, "global"), (near, "local")):
            planned = _count_plan(256, sampling["shells"][group])
            absolute = np.abs(samples[:, 3])
            for count, shell in zip(planned, sampling["shells"][group]):
                lower = float(shell["min_abs_sdf"])
                upper = shell["max_abs_sdf"]
                mask = absolute >= lower
                if upper is not None:
                    mask &= absolute < float(upper)
                self.assertEqual(int(mask.sum()), count)
                values = samples[mask, 3]
                self.assertLessEqual(abs(int(np.sum(values > 0)) - int(np.sum(values < 0))), 1)

    def test_validation_and_test_are_not_used_for_periodic_selection(self) -> None:
        for config in self.configs.values():
            self.assertEqual(config["periodic_evaluation"]["splits"], ["val"])
            self.assertNotIn("test", config["periodic_evaluation"]["splits"])
            self.assertEqual(
                config["periodic_evaluation"]["fscore_thresholds_mm"],
                [0.1, 0.25, 0.5, 1.0],
            )


if __name__ == "__main__":
    unittest.main()
