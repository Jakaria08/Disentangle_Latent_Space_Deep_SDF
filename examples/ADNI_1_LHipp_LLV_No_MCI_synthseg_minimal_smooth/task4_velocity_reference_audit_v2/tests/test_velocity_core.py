#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from velocity_core import Candidate, fit_trajectory, generalized_rigid_alignment, kabsch_align, vector_cosine


class VelocityCoreTests(unittest.TestCase):
    def test_kabsch_removes_rigid_motion_without_scaling(self):
        rng = np.random.default_rng(2)
        points = rng.normal(size=(100, 3))
        angle = 0.3
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]])
        moved = points @ rotation + np.array([2.0, -1.0, 0.5])
        aligned, recovered, _ = kabsch_align(moved, points)
        self.assertAlmostEqual(np.linalg.det(recovered), 1.0, places=10)
        self.assertLess(np.sqrt(np.mean((aligned - points) ** 2)), 1.0e-10)
        aligned_scale = np.sqrt(np.mean(np.sum((aligned - aligned.mean(axis=0)) ** 2, axis=1)))
        moved_scale = np.sqrt(np.mean(np.sum((moved - moved.mean(axis=0)) ** 2, axis=1)))
        self.assertAlmostEqual(aligned_scale, moved_scale, places=10)

    def test_generalized_alignment_removes_pose_but_keeps_shape(self):
        rng = np.random.default_rng(3)
        base = rng.normal(size=(80, 3))
        visits = []
        for angle, shift in [(0.0, [0, 0, 0]), (0.2, [1, 2, 0]), (-0.15, [-2, 1, 1])]:
            rotation = np.array([[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]])
            visits.append(base @ rotation + np.asarray(shift))
        aligned, _ = generalized_rigid_alignment(np.stack(visits))
        self.assertLess(np.max(np.sqrt(np.mean((aligned - aligned[0]) ** 2, axis=(1, 2)))), 1.0e-9)

    def test_linear_derivative_is_exact(self):
        ages = np.array([70.0, 71.5, 74.0, 77.0])
        intercept = np.arange(18, dtype=float).reshape(6, 3)
        slope = np.linspace(-0.2, 0.3, 18).reshape(6, 3)
        values = intercept[None] + (ages - 70.0)[:, None, None] * slope
        model = fit_trajectory(ages, values, Candidate("linear", "linear", "raw", "surface", 1))
        self.assertLess(np.max(np.abs(model.derivative(73.0)[0] - slope)), 1.0e-10)
        self.assertLess(np.max(np.abs(model.predict(72.0)[0] - (intercept + 2.0 * slope))), 1.0e-10)

    def test_quadratic_derivative_is_exact_in_basis(self):
        rng = np.random.default_rng(4)
        components, _ = np.linalg.qr(rng.normal(size=(12, 4)))
        components = components.T
        mean = rng.normal(size=12)
        ages = np.array([68.0, 70.0, 72.0, 75.0, 78.0])
        center = ages.mean()
        a = rng.normal(size=4)
        b = rng.normal(size=4)
        c = rng.normal(size=4) * 0.1
        scores = a + (ages - center)[:, None] * b + (ages - center)[:, None] ** 2 * c
        values = (scores @ components + mean).reshape(len(ages), 4, 3)
        candidate = Candidate("quadratic", "quadratic", "raw", "pca", 2, 4)
        model = fit_trajectory(ages, values, candidate, mean, components)
        expected = (b + 2.0 * (73.0 - center) * c) @ components
        self.assertLess(np.max(np.abs(model.derivative(73.0).reshape(-1) - expected)), 1.0e-8)

    def test_vector_cosine(self):
        self.assertAlmostEqual(vector_cosine([1, 2, 3], [2, 4, 6]), 1.0)
        self.assertAlmostEqual(vector_cosine([1, 0], [-1, 0]), -1.0)


if __name__ == "__main__":
    unittest.main()
