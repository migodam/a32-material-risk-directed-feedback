"""Synthetic array/provenance arithmetic only; no Maxwell execution."""
from __future__ import annotations

import csv
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from src.a32 import charts, data
from src.a32 import frozen_representations as frozen


def _save(path, **values):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **values)


def _fixture(root):
    rows = [dict(case_id=f"p{pid}_{bg}_a1_snr20", parent_id=pid,
                 family="shell", background_id=bg, best_id="FORBIDDEN")
            for pid in data.ONLINE_PARENTS for bg in data.BACKGROUNDS]
    csv_path = root / "results/a29/COVERAGE_CASES.csv"
    csv_path.parent.mkdir(parents=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    backgrounds = {}
    for bg, real in (("weak", .01), ("ring", .04)):
        chi_b = np.full(576, real + .02j, dtype=np.complex128)
        backgrounds[bg] = chi_b
        _save(root / f"inputs/a25/background/{bg}.npz", chi_b=chi_b,
              Db=np.zeros((32, 576), complex), eb=np.ones((576, 8), complex),
              yb=np.zeros((32, 8), complex), G=np.zeros((576, 576), complex),
              S=np.zeros((32, 576), complex), incident=np.ones((576, 8), complex),
              L_b=np.eye(576, dtype=complex), J=np.zeros((512, 1152)),
              volume=np.array(1 / 576), grid=np.array(24), points=np.zeros((576, 2)),
              source_points=np.zeros((8, 2)), receiver_points=np.zeros((32, 2)),
              descriptor=np.array("{}"), geometry_descriptor=np.array("{}"),
              Ug=np.array([{"forbidden": True}], dtype=object))
    truths, n3s, bgs = [], [], []
    for row in rows:
        bg = backgrounds[row["background_id"]]
        n3 = bg + np.full(576, .1 + .04j, dtype=np.complex128)
        _save(root / f"results/a29/pools/{row['case_id']}.npz",
              observation=np.ones((32, 8), complex), sigma=np.array(.1),
              n3=n3, background=bg, chi=np.array([{"forbidden": True}], dtype=object),
              truth=np.array([{"forbidden": True}], dtype=object))
        n3s.append(n3)
        bgs.append(bg)
        truths.append(np.full(576, row["parent_id"] / 100000 + .1j, complex))
    _save(root / "results/a29/OFFLINE_TARGETS.npz",
          case_id=np.array([row["case_id"] for row in rows][::-1]),
          truth=np.array(truths[::-1]), background=np.array(bgs[::-1]), N3=np.array(n3s[::-1]))
    metadata = {key: [] for key in ("parent_id", "split", "background_id", "amplitude",
                                  "case_id", "noise_db", "status")}
    materials = []
    expected = []
    for pid in data.TRAIN_PARENTS:
        delta = np.full(576, (pid - 250000) / 200 + .03j, complex)
        expected.append(delta)
        for bg, amplitude, noise, suffix in (("weak", 1.2, float("nan"), "a1_clean"),
                                             ("weak", 1.2, 20., "a1_snr20"),
                                             ("ring", 1.2, float("nan"), "a1_clean"),
                                             ("weak", .3, float("nan"), "a0_clean")):
            values = (pid, "train", bg, amplitude, f"p{pid}_{bg}_{suffix}", noise, "OK")
            for key, value in zip(metadata, values):
                metadata[key].append(value)
            materials.append(backgrounds[bg] + amplitude * delta)
    _save(root / "inputs/a25/train.npz", **{key: np.asarray(value) for key, value in metadata.items()},
          chi=np.asarray(materials))
    return rows, np.asarray(expected)


class _ArchiveSpy:
    def __init__(self, load, path, trace):
        self.archive = load(path, allow_pickle=False)
        self.path, self.trace = Path(path), trace
        self.files = self.archive.files

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.archive.close()

    def __getitem__(self, key):
        self.trace.append((self.path, key))
        if self.path.parent.name == "pools" and key not in {"observation", "sigma", "n3", "background"}:
            raise AssertionError("Forbidden pool key: " + key)
        if self.path.parent.name == "background" and key not in data.BACKGROUND_KEYS:
            raise AssertionError("Forbidden background key: " + key)
        if self.path.parent.name == "exact_states" and key not in {"data", "chi", "background", "cache_protocol"}:
            raise AssertionError("Forbidden paid-cache key: " + key)
        return self.archive[key]


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name) / "release"
        cls.rows, cls.expected_train = _fixture(cls.root)
        cls.cases = data.online_cases(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_online_whitelist_and_csv_order(self):
        trace, load = [], np.load
        with patch.object(data.np, "load", side_effect=lambda path, **kwargs: _ArchiveSpy(load, path, trace)):
            cases = data.online_cases(self.root)
        self.assertEqual([case.case_id for case in cases], [row["case_id"] for row in self.rows])
        self.assertEqual(set(cases[0].metadata), {"case_id", "parent_id", "family", "background_id"})
        self.assertEqual(set(cases[0].background), set(data.BACKGROUND_KEYS))
        self.assertFalse(any(path.name == "OFFLINE_TARGETS.npz" for path, _ in trace))
        self.assertEqual(len([key for path, key in trace if path.parent.name == "pools"]), 32 * 4)
        with self.assertRaises(ValueError):
            cases[0].n3.setflags(write=True)

    def test_train_clean64_and_normalization(self):
        trace, load = [], np.load
        with patch.object(data.np, "load", side_effect=lambda path, **kwargs: _ArchiveSpy(load, path, trace)):
            normalized, metadata = data.train_deltas(self.root)
        np.testing.assert_allclose(normalized, self.expected_train, rtol=0, atol=1e-16)
        self.assertEqual(normalized.shape, (64, 576))
        self.assertEqual(metadata["selected_case_ids"], [f"p{pid}_weak_a1_clean" for pid in data.TRAIN_PARENTS])
        self.assertEqual(metadata["independent_parents"], 64)
        self.assertFalse(metadata["background_replication"])
        train_keys = [key for path, key in trace if path.name == "train.npz"]
        self.assertEqual(train_keys[:2], ["parent_id", "split"])

    def test_train_split_leakage_refused_before_chi(self):
        path = self.root / "inputs/a25/train.npz"
        original = path.read_bytes()
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        arrays["split"][0] = "dev"
        _save(path, **arrays)
        trace, load = [], np.load
        try:
            with patch.object(data.np, "load", side_effect=lambda path, **kwargs: _ArchiveSpy(load, path, trace)):
                with self.assertRaisesRegex(ValueError, "BEFORE_MATERIAL_READ"):
                    data.train_deltas(self.root)
            self.assertFalse(any(key == "chi" for _, key in trace))
        finally:
            path.write_bytes(original)

    def _cache(self, chi):
        case = self.cases[0]
        _save(case.cache_path, data=np.ones((1, 32, 8), complex), chi=chi[None],
              background=case.chi_b, cache_protocol=np.array("A28-EXACT-STATE-v1"),
              mass=np.array([{"forbidden": True}], dtype=object),
              lu=np.array([{"forbidden": True}], dtype=object))
        return case

    def test_cache_four_keys_and_exact_pack_unpack(self):
        n3 = self.cases[0].n3
        scale = np.sqrt(1 / 576)
        packed = scale * np.r_[n3.real, n3.imag]
        canonical = (packed[:576] + 1j * packed[576:]) / scale
        case = self._cache(canonical)
        trace, load = [], np.load
        with patch.object(data.np, "load", side_effect=lambda path, **kwargs: _ArchiveSpy(load, path, trace)):
            cached = data.cached_n3(case)
        self.assertEqual({key for _, key in trace}, {"data", "chi", "background", "cache_protocol"})
        np.testing.assert_array_equal(cached.chi, canonical)
        self.assertIn(cached.same_material_identity, {"EXACT_N3", "EXACT_FROZEN_MASS_PACK_UNPACK_N3"})
        self.assertEqual(cached.data.shape, (32, 8))

    def test_cache_rejects_arbitrary_nearby_material(self):
        changed = self.cases[0].n3.copy()
        changed[0] += 1e-12
        case = self._cache(changed)
        with self.assertRaisesRegex(ValueError, "FOREIGN_OR_STALE"):
            data.cached_n3(case)

    def _completed_predictions(self, name):
        out = Path(self.tmp.name) / name
        out.mkdir()
        ids = [case.case_id for case in self.cases]
        for case in self.cases:
            _save(out / f"{case.case_id}.npz", case_id=np.array(case.case_id), chi_gaussian=case.n3)
        token = out / "ONLINE_COMPLETE.json"
        token.write_text(json.dumps({"status": "ONLINE_COMPLETE", "case_ids": ids,
                                     "prediction_files": [case_id + ".npz" for case_id in ids]}))
        return ids, token

    def test_offline_refuses_missing_token_before_target_open(self):
        load, opened = np.load, []

        def record(path, **kwargs):
            opened.append(Path(path))
            return load(path, **kwargs)

        with patch.object(data.np, "load", side_effect=record):
            with self.assertRaisesRegex(ValueError, "COMPLETION_TOKEN"):
                data.offline_targets(self.root, [case.case_id for case in self.cases],
                                     completion_token_path=Path(self.tmp.name) / "missing.json")
        self.assertEqual(opened, [])

    def test_offline_saved_token_and_target_reordering(self):
        ids, token = self._completed_predictions("valid_online")
        targets = data.offline_targets(self.root, ids, completion_token_path=token)
        self.assertEqual(targets.case_ids, tuple(ids))
        self.assertEqual(targets.truth.shape, (32, 576))
        for index, case in enumerate(self.cases):
            np.testing.assert_array_equal(targets.truth[index], np.full(576, case.parent_id / 100000 + .1j))

    def test_offline_bad_saved_identity_never_opens_target(self):
        ids, token = self._completed_predictions("invalid_online")
        _save(token.parent / f"{ids[0]}.npz", case_id=np.array("wrong"), chi_gaussian=self.cases[0].n3)
        load, opened = np.load, []

        def record(path, **kwargs):
            opened.append(Path(path))
            return load(path, **kwargs)

        with patch.object(data.np, "load", side_effect=record):
            with self.assertRaisesRegex(ValueError, "PREDICTION_CONTENT"):
                data.offline_targets(self.root, ids, completion_token_path=token)
        self.assertFalse(any(path.name == "OFFLINE_TARGETS.npz" for path in opened))


def _arithmetic_model():
    rng = np.random.default_rng(32012)
    weights = np.full(576, 1 / 576)
    real_weights = np.tile(weights, 2)
    train = .01 * (rng.normal(size=(64, 576)) + 1j * rng.normal(size=(64, 576)))
    haar, metadata = frozen.haar24_transform()
    basis = np.eye(1152, 64) / np.sqrt(real_weights)[:, None]
    coefficients = np.hstack(((train.real * np.sqrt(weights)) @ haar.T,
                             (train.imag * np.sqrt(weights)) @ haar.T))
    return frozen.TrainModel(train, weights, real_weights, haar, metadata, coefficients,
                             basis, np.ones(64), 64, (), {})


class ChartTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = _arithmetic_model()
        cls.background = np.full(576, .2 + .2j)
        cls.n3 = cls.background + 1.2 * cls.model.train_deltas[0]

    def test_preparation_reuses_frozen_function(self):
        with patch.object(frozen, "prepare_train", return_value=self.model) as prepare:
            self.assertIs(charts.prepare_charts(self.model.train_deltas), self.model)
            prepare.assert_called_once_with(self.model.train_deltas, cell_weights=None)

    def test_physical_scale_train_projection_and_metric(self):
        chart = charts.freeze_chart(self.model, "pca64", self.n3, self.background)
        np.testing.assert_array_equal(chart.T, 1.2 * self.model.single_basis[:, :64])
        np.testing.assert_allclose(chart.material_metric, 1.44 * np.eye(64), rtol=0, atol=1e-15)
        actual = 1.2 * np.hstack((self.model.train_deltas.real, self.model.train_deltas.imag))
        np.testing.assert_allclose(chart.train_mass, actual * np.sqrt(self.model.real_weights), rtol=0, atol=0)
        normal_equation = chart.train_coefficients @ chart.material_metric
        np.testing.assert_allclose(normal_equation, (actual * self.model.real_weights) @ chart.T,
                                   rtol=1e-13, atol=1e-18)
        projected_n3 = self.background + (chart.T @ chart.initial_coefficients)[:576]
        np.testing.assert_allclose(chart.decode(chart.initial_coefficients).real, projected_n3.real,
                                   rtol=0, atol=1e-16)
        with self.assertRaises(ValueError):
            chart.T.setflags(write=True)

    def test_sparse_support_is_first_frozen_n3_start(self):
        chart = charts.freeze_chart(self.model, "sparse32", self.n3, self.background)
        normalized = (self.n3 - self.background) / 1.2
        coefficients = np.r_[self.model.haar_analysis @ (normalized.real * np.sqrt(self.model.cell_weights)),
                             self.model.haar_analysis @ (normalized.imag * np.sqrt(self.model.cell_weights))]
        expected = np.argsort(-np.abs(coefficients), kind="stable")[:32]
        np.testing.assert_array_equal(chart.support_ids, expected)
        self.assertEqual(chart.metadata["start_id"], 0)
        self.assertEqual(chart.metadata["source"], "N3")
        self.assertEqual(chart.dimension, 32)

    def test_first_eligible_selection_ignores_any_later_start(self):
        real_freeze = frozen.freeze_case

        def freeze(*args, **kwargs):
            choices = real_freeze(*args, **kwargs)
            starts = [dict(start) for start in choices["starts"]]
            starts[0].update(eligible=False, reasons=["SYNTHETIC_INELIGIBLE"])
            return {**choices, "starts": tuple(starts)}

        with patch.object(frozen, "freeze_case", side_effect=freeze):
            chart = charts.freeze_chart(self.model, "sparse64", self.n3, self.background)
        self.assertEqual(chart.metadata["start_id"], 1)
        self.assertEqual(chart.metadata["selection_rule"], "FIRST_ELIGIBLE_FROZEN_START")

    def test_no_rank_fallback_and_no_pixel_clipping(self):
        model = replace(self.model, single_rank=63)
        with self.assertRaisesRegex(ValueError, "NO_ELIGIBLE_FROZEN_START"):
            charts.freeze_chart(model, "pca64", self.n3, self.background)
        chart = charts.freeze_chart(self.model, "pca64", self.n3, self.background)
        coefficients = np.ones(64)
        self.assertGreater(chart.decode(coefficients).real.max(), frozen.REAL_BOUNDS[1])

    def test_projection_qp_uses_truth_free_material_constraints(self):
        chart = charts.freeze_chart(self.model, "pca64", self.n3, self.background)
        problem = chart.projection_problem()
        self.assertEqual(set(problem), {"H", "h", "A", "lower", "upper"})
        np.testing.assert_array_equal(problem["A"], chart.T)
        np.testing.assert_allclose(problem["H"] @ chart.initial_coefficients, problem["h"],
                                   rtol=1e-13, atol=1e-18)
        receipt = {"success": True, "c": chart.frozen_feasible_initial}
        check = lambda: None
        solver = Mock(return_value=receipt)
        self.assertIs(chart.feasible_projection_initial(solver=solver, check=check), receipt)
        self.assertIs(solver.call_args.kwargs["check"], check)
        np.testing.assert_array_equal(solver.call_args.kwargs["initial"], chart.frozen_feasible_initial)


if __name__ == "__main__":
    unittest.main()
