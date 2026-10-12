"""Bounded arithmetic checks; no native physics or exposed targets are read."""
from __future__ import annotations

from itertools import combinations
import unittest

import numpy as np

from src.a32.frozen_representations import haar24_transform
from src.a32.priors import build_priors
from src.a32.qp import solve_qp


def _enumerated_reference(H, h, A, lower, upper):
    """Independent small-QP reference enumerates all possible active faces."""
    dimension = len(h)
    constraints = np.vstack((A, -A))
    limits = np.r_[upper, -lower]
    best = None
    for count in range(dimension + 1):
        for selected in combinations(range(len(limits)), count):
            active = constraints[list(selected)]
            if count:
                system = np.block([[H, active.T],
                                   [active, np.zeros((count, count))]])
                try:
                    solution = np.linalg.solve(system, np.r_[h, limits[list(selected)]])
                except np.linalg.LinAlgError:
                    continue
                point, dual = solution[:dimension], solution[dimension:]
                if np.any(dual < -1e-8):
                    continue
            else:
                point = np.linalg.solve(H, h)
            if np.any(constraints @ point - limits > 1e-8):
                continue
            objective = float(0.5 * point @ H @ point - h @ point)
            if best is None or objective < best[0]:
                best = objective, point.copy()
    if best is None:
        raise AssertionError("synthetic reference unexpectedly infeasible")
    return best[1]


class QPTests(unittest.TestCase):
    def assert_solution(self, result, H, h, A, lower, upper):
        self.assertTrue(result["success"], (result["status"], result["reason"], result["kkt_relative"]))
        self.assertEqual(result["status"], "SUCCESS")
        self.assertLessEqual(result["iterations"], 200)
        self.assertLessEqual(result["feasibility"], 1e-10)
        self.assertLessEqual(result["kkt_relative"], 1e-8)
        self.assertTrue(np.all(result["multipliers"]["lower"] >= 0))
        self.assertTrue(np.all(result["multipliers"]["upper"] >= 0))
        np.testing.assert_allclose(result["normal"],
                                   A.T @ (result["multipliers"]["upper"] - result["multipliers"]["lower"]),
                                   rtol=1e-14, atol=1e-14)
        # H is symmetrized only within its admitted roundoff asymmetry.  The
        # diagnostic identity therefore uses an operation-scaled roundoff
        # bound rather than an absolute tolerance on a nearly zero residual.
        arithmetic_scale = max(1., np.linalg.norm(H, ord=np.inf) * np.linalg.norm(result["c"], ord=np.inf),
                               np.linalg.norm(h, ord=np.inf), np.linalg.norm(result["normal"], ord=np.inf))
        np.testing.assert_allclose(result["stationarity"], H @ result["c"] - h + result["normal"],
                                   rtol=1e-13, atol=128 * np.finfo(float).eps * arithmetic_scale)
        self.assertAlmostEqual(result["objective"], float(0.5 * result["c"] @ H @ result["c"] - h @ result["c"]),
                               delta=1e-10 * max(1, abs(result["objective"])))
        self.assertGreaterEqual(result["time"]["wall_seconds"], 0)
        self.assertGreaterEqual(result["time"]["cpu_seconds"], 0)
        self.assertFalse(result["jitter_added"])
        self.assertFalse(result["pseudoinverse_used"])
        self.assertTrue(result["no_pixel_clipping"])

    def test_general_spd_unconstrained(self):
        H = np.array([[8., 2., -1.], [2., 3., .5], [-1., .5, 1.]])
        h = np.array([1., -2., .4])
        A = np.empty((0, 3))
        result = solve_qp(H, h, A, [], [])
        self.assert_solution(result, H, h, A, np.empty(0), np.empty(0))
        np.testing.assert_allclose(result["c"], np.linalg.solve(H, h), atol=1e-12, rtol=1e-12)

    def test_coupled_active_normals_and_duplicate_pixels(self):
        H = np.array([[4., 1.5], [1.5, 1.]])
        h = np.array([10., 6.])
        A = np.array([[1., 0.], [0., 1.], [1., 1.], [2., 2.], [0., 0.]])
        lower = np.array([0., 0., -np.inf, -np.inf, 0.])
        upper = np.array([1.5, 1.2, 1.8, 3.6, 0.])
        result = solve_qp(H, h, A, lower, upper)
        self.assert_solution(result, H, h, A, lower, upper)
        np.testing.assert_allclose(result["c"], [1.5, .3], atol=1e-10, rtol=1e-10)

    def test_duplicate_equalities(self):
        H, h = np.diag([2., 5.]), np.array([.2, 1.1])
        A = np.array([[1., 1.], [2., 2.], [1., 0.], [0., 1.]])
        lower, upper = np.array([1., 2., 0., 0.]), np.array([1., 2., 1., 1.])
        result = solve_qp(H, h, A, lower, upper, initial=[.5, .5])
        self.assert_solution(result, H, h, A, lower, upper)
        np.testing.assert_allclose(result["c"], [4.1 / 7, 2.9 / 7], atol=1e-11, rtol=1e-11)
        self.assertEqual(len(result["equality_pixel_rows"]), 1)

    def test_nonzero_feasible_set_phase_I(self):
        H, h, A = np.eye(1), np.array([2.5]), np.eye(1)
        result = solve_qp(H, h, A, 2., 3., initial=[-4.])
        self.assert_solution(result, H, h, A, np.array([2.]), np.array([3.]))
        np.testing.assert_allclose(result["c"], [2.5], atol=1e-12)
        self.assertIsNotNone(result["phase_I"])
        self.assertEqual(result["iterations"], result["phase_iterations"] + result["active_set_iterations"])

    def test_infeasible_and_failed_solutions_are_retained(self):
        contradictory = solve_qp(np.eye(1), [1.], [[1.], [1.]], [1., -np.inf], [np.inf, 0.])
        self.assertFalse(contradictory["success"])
        self.assertEqual(contradictory["status"], "INFEASIBLE")
        self.assertEqual(contradictory["c"].shape, (1,))
        zero_row = solve_qp(np.eye(1), [0.], [[0.]], [1.], [2.], initial=[.4])
        self.assertEqual(zero_row["status"], "INFEASIBLE")
        np.testing.assert_array_equal(zero_row["c"], [.4])
        reversed_bounds = solve_qp(np.eye(1), [0.], [[1.]], [2.], [1.])
        self.assertEqual(reversed_bounds["status"], "INFEASIBLE")
        capped = solve_qp(np.eye(2), [2., 3.], np.eye(2), 0., 1., config={"max_iterations": 1})
        self.assertEqual(capped["status"], "MAX_ITERATIONS")
        self.assertFalse(capped["success"])
        self.assertEqual(capped["iterations"], 1)
        self.assertLessEqual(capped["feasibility"], 1e-10)
        np.testing.assert_allclose(capped["c"], [2 / 3, 1.], atol=1e-12)

    def test_budget_hook_retains_current_point(self):
        stopped = solve_qp(np.eye(2), [2., 3.], np.eye(2), 0., 1.,
                           initial=[.2, .3], check=lambda: False)
        self.assertEqual(stopped["status"], "CHECK_STOPPED")
        self.assertEqual(stopped["iterations"], 0)
        np.testing.assert_array_equal(stopped["c"], [.2, .3])
        def failing_check():
            raise RuntimeError("synthetic budget exhausted")
        failed = solve_qp(np.eye(1), [0.], [[1.]], 0., 1., check=failing_check)
        self.assertEqual(failed["status"], "CHECK_FAILED")
        self.assertIn("synthetic budget", failed["reason"])

    def test_frozen_config_and_malformed_H_rejected(self):
        for settings in ({"max_iterations": 201}, {"kkt_rtol": 1e-7},
                         {"feasibility_tolerance": 1e-9}, {"no_pixel_clipping": False}):
            with self.assertRaises(ValueError):
                solve_qp(np.eye(1), [0.], [[1.]], 0., 1., config=settings)
        for H in (np.zeros((2, 2)), np.array([[1., .1], [0., 1.]])):
            with self.assertRaises(ValueError):
                solve_qp(H, [0., 0.], np.eye(2), 0., 1.)
        result = solve_qp(np.eye(1), [2.], [[1.]], 0., 1., config={"solver": {"max_iterations": 20}})
        self.assertTrue(result["success"])

    def test_small_synthetic_qps_against_exhaustive_faces(self):
        rng = np.random.default_rng(3202)
        for case in range(6):
            orthogonal, _ = np.linalg.qr(rng.normal(size=(4, 4)))
            H = orthogonal @ np.diag(np.geomspace(1, 500, 4)) @ orthogonal.T
            h = rng.normal(size=4) * 100
            A = rng.normal(size=(5, 4))
            lower = -rng.uniform(.1, 1.5, size=5)
            upper = rng.uniform(.1, 1.5, size=5)
            result = solve_qp(H, h, A, lower, upper)
            with self.subTest(case=case):
                self.assert_solution(result, H, h, A, lower, upper)
                expected = _enumerated_reference(H, h, A, lower, upper)
                np.testing.assert_allclose(result["c"], expected, atol=2e-8, rtol=2e-8)

    def test_material_sized_anisotropic_QP_with_known_KKT_point(self):
        rng = np.random.default_rng(3264)
        basis, _ = np.linalg.qr(rng.normal(size=(1152, 64)))
        A = 1.2 * 24 * basis
        rotation, _ = np.linalg.qr(rng.normal(size=(64, 64)))
        H = rotation @ np.diag(np.geomspace(1, 1e4, 64)) @ rotation.T
        expected = rng.normal(size=64) * .1
        material = A @ expected
        upper_indices = np.flatnonzero(material > .1)[:3]
        lower_indices = np.flatnonzero(material < -.1)[:2]
        lower, upper = np.full(1152, -20.), np.full(1152, 20.)
        upper[upper_indices], lower[lower_indices] = material[upper_indices], material[lower_indices]
        normal = A[upper_indices].T @ np.array([.5, 1., 1.5]) - A[lower_indices].T @ np.array([.4, .8])
        h = H @ expected + normal
        result = solve_qp(H, h, A, lower, upper)
        self.assert_solution(result, H, h, A, lower, upper)
        np.testing.assert_allclose(result["c"], expected, atol=2e-8, rtol=2e-8)


class PriorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.haar = haar24_transform()[0]

    def _inputs(self, dimension=5, seed=3200):
        rng = np.random.default_rng(seed)
        coefficients = rng.normal(size=(64, dimension)) @ np.diag(np.geomspace(.1, 3, dimension))
        coefficients += np.linspace(-.3, .7, dimension)
        T = rng.normal(size=(1152, dimension))
        mass_haar = rng.normal(size=(64, 1152))
        for channel in range(2):
            for level, (start, stop) in enumerate(((0, 9), (9, 36), (36, 144), (144, 576))):
                mass_haar[:, channel * 576 + start:channel * 576 + stop] *= (level + 1) * (channel + 1) / 10
        train_mass = np.hstack((mass_haar[:, :576] @ self.haar,
                                mass_haar[:, 576:] @ self.haar))
        return coefficients, T, train_mass

    def test_exact_original_OAS_and_covariance_trace_matched_ridge(self):
        coefficients, T, mass = self._inputs()
        priors = build_priors(coefficients, T, mass, self.haar)
        self.assertEqual(set(priors), {"gaussian", "ridge", "multiscale"})
        centered = coefficients - coefficients.mean(axis=0)
        sample = centered.T @ centered / 64
        dimension = len(sample)
        trace, trace_square = np.trace(sample), np.trace(sample @ sample)
        rho = min(1., ((1 - 2 / dimension) * trace_square + trace ** 2) /
                  ((64 + 1 - 2 / dimension) * (trace_square - trace ** 2 / dimension)))
        expected = (1 - rho) * sample + rho * trace / dimension * np.eye(dimension)
        np.testing.assert_allclose(priors["gaussian"]["covariance"], expected, atol=1e-13, rtol=1e-13)
        self.assertAlmostEqual(priors["gaussian"]["metadata"]["shrinkage"], rho, places=14)
        ridge_variance = np.trace(expected) / dimension
        np.testing.assert_allclose(priors["ridge"]["covariance"], ridge_variance * np.eye(dimension))
        np.testing.assert_allclose(priors["ridge"]["precision"], np.eye(dimension) / ridge_variance)
        self.assertAlmostEqual(priors["gaussian"]["metadata"]["covariance_trace"],
                               priors["ridge"]["metadata"]["covariance_trace"], places=13)
        self.assertNotAlmostEqual(priors["gaussian"]["metadata"]["precision_trace"],
                                  priors["ridge"]["metadata"]["precision_trace"], places=4)
        for entry in priors.values():
            np.testing.assert_allclose(entry["mean"], coefficients.mean(axis=0), rtol=0, atol=0)
            np.linalg.cholesky(entry["covariance"])
            np.linalg.cholesky(entry["precision"])
            np.testing.assert_allclose(entry["precision"] @ entry["covariance"], np.eye(dimension), atol=1e-11, rtol=1e-11)
            self.assertEqual(entry["metadata"]["independent_train_parents"], 64)
            self.assertFalse(entry["metadata"]["numerical_damping_added"])

    def test_multiscale_is_precision_pullback_through_nonorthogonal_basis(self):
        coefficients, T, mass = self._inputs(dimension=3, seed=3201)
        priors = build_priors(coefficients, T, mass, (self.haar, None))
        transformed = np.hstack((mass[:, :576] @ self.haar.T, mass[:, 576:] @ self.haar.T))
        centered = transformed - transformed.mean(axis=0)
        transformed_basis = np.vstack((self.haar @ T[:576] / 24, self.haar @ T[576:] / 24))
        expected = np.zeros((3, 3))
        for channel in range(2):
            for start, stop in ((0, 9), (9, 36), (36, 144), (144, 576)):
                selection = slice(channel * 576 + start, channel * 576 + stop)
                variance = np.sum(centered[:, selection] ** 2) / (64 * (stop - start))
                expected += transformed_basis[selection].T @ transformed_basis[selection] / variance
        np.testing.assert_allclose(priors["multiscale"]["precision"], expected, atol=1e-11, rtol=1e-12)
        np.testing.assert_allclose(priors["multiscale"]["covariance"], np.linalg.inv(expected), atol=1e-12, rtol=1e-12)
        self.assertEqual(len(priors["multiscale"]["metadata"]["levels"]), 8)

    def test_rank_deficient_64_coordinate_sample_has_proper_OAS(self):
        coefficients, T, mass = self._inputs(dimension=64, seed=3265)
        priors = build_priors(coefficients, T, mass, self.haar)
        self.assertGreater(priors["gaussian"]["metadata"]["shrinkage"], 0)
        self.assertGreater(np.linalg.eigvalsh(priors["gaussian"]["covariance"])[0], 0)
        np.testing.assert_allclose(priors["gaussian"]["precision"] @ priors["gaussian"]["covariance"],
                                   np.eye(64), atol=1e-11, rtol=1e-11)

    def test_inactive_zero_variance_channel_and_explicit_degenerate_failure(self):
        coefficients, T, mass = self._inputs(dimension=3)
        T[576:] = 0
        mass[:, 576:] = 0
        result = build_priors(coefficients, T, mass, self.haar)
        self.assertTrue(all(record["omitted_zero_variance_zero_response"]
                            for record in result["multiscale"]["metadata"]["levels"][4:]))
        T[576, 0] = 1
        with self.assertRaisesRegex(ValueError, "zero TRAIN Haar variance"):
            build_priors(coefficients, T, mass, self.haar)
        with self.assertRaisesRegex(ValueError, "variance is zero"):
            build_priors(np.zeros((64, 3)), T, mass, self.haar)

    def test_train_shape_and_fixed_real_coordinates_required(self):
        coefficients, T, mass = self._inputs()
        with self.assertRaises(ValueError):
            build_priors(coefficients[:63], T, mass, self.haar)
        with self.assertRaises(ValueError):
            build_priors(coefficients.astype(complex), T, mass, self.haar)
        with self.assertRaises(ValueError):
            build_priors(coefficients, T[:576], mass, self.haar)


if __name__ == "__main__":
    unittest.main()
