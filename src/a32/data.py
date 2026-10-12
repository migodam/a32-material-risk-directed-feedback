"""Whitelisted A32 inputs, with a separate, completion-gated target loader.

Online identities come only from the four metadata columns of COVERAGE_CASES.
Online pools are read through observation/sigma/n3/background, never pooled
candidate arrays.  The paid N3 reader does not load factors or run physics.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np


PIXELS = 24 * 24
TRAIN_PARENTS = tuple(range(250001, 250065))
ONLINE_PARENTS = tuple(range(250065, 250081))
BACKGROUNDS = ("weak", "ring")
AMPLITUDE = 1.2
BACKGROUND_KEYS = frozenset({
    "J", "chi_b", "Db", "eb", "yb", "G", "S", "incident", "L_b",
    "volume", "grid", "points", "source_points", "receiver_points",
    "descriptor", "geometry_descriptor",
})
_COMPLEX_BACKGROUND_SHAPES = {
    "chi_b": (PIXELS,), "Db": (32, PIXELS), "eb": (PIXELS, 8),
    "yb": (32, 8), "G": (PIXELS, PIXELS), "S": (32, PIXELS),
    "incident": (PIXELS, 8), "L_b": (PIXELS, PIXELS),
}


def _readonly(value: Any) -> np.ndarray:
    array = np.asarray(value)
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)


def _input_root(config_or_input_root: Mapping[str, Any] | str | Path) -> Path:
    if isinstance(config_or_input_root, Mapping):
        config = config_or_input_root
        for key, expected in (("parent_ids", ONLINE_PARENTS),
                              ("train_parent_ids", TRAIN_PARENTS),
                              ("backgrounds", BACKGROUNDS)):
            if key in config and tuple(config[key]) != expected:
                raise ValueError("A32_FROZEN_INPUT_IDENTITY_MISMATCH:" + key)
        if (float(config.get("amplitude", AMPLITUDE)) != AMPLITUDE
                or int(config.get("grid", 24)) != 24):
            raise ValueError("A32_FROZEN_AMPLITUDE_OR_GRID_MISMATCH")
        path = Path(config["input_root"])
    else:
        path = Path(config_or_input_root)
    if path.is_symlink() or not path.is_dir():
        raise ValueError("A32_DECLARED_INPUT_ROOT_REQUIRED")
    root = path.resolve()
    if any("locked" in part.lower() or part.lower() in {"test", "id_test", "ood_test"}
           for part in root.parts):
        raise ValueError("A32_LOCKED_PATH_FORBIDDEN")
    return root


def _safe(root: Path, member: str | Path) -> Path:
    relative = Path(member)
    if relative.is_absolute() or any(part in {"..", "."} for part in relative.parts):
        raise ValueError("A32_RELATIVE_INPUT_MEMBER_REQUIRED")
    path = root / relative
    if any("locked" in part.lower() or part.lower() in {"test", "id_test", "ood_test"}
           for part in path.parts):
        raise ValueError("A32_LOCKED_PATH_FORBIDDEN")
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("A32_SYMLINK_INPUT_FORBIDDEN")
    if not path.is_file() or not path.resolve().is_relative_to(root):
        raise ValueError("A32_INPUT_MISSING_OR_UNSAFE:" + str(relative))
    return path


def _complex_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or array.dtype != np.complex128 or not np.isfinite(array).all():
        raise ValueError("A32_COMPLEX128_FINITE_LAYOUT_REQUIRED:" + name)
    return _readonly(array)


def case_metadata(config_or_input_root: Mapping[str, Any] | str | Path) -> tuple[dict[str, Any], ...]:
    """Extract only identities/family; never consult a target archive for ordering."""
    root = _input_root(config_or_input_root)
    keys = ("case_id", "parent_id", "family", "background_id")
    with _safe(root, "results/a29/COVERAGE_CASES.csv").open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if any(header.count(key) != 1 for key in keys):
            raise ValueError("A32_CASE_METADATA_COLUMNS_REQUIRED")
        positions = [header.index(key) for key in keys]
        rows = []
        for record in reader:
            if len(record) != len(header):
                raise ValueError("A32_CASE_METADATA_ROW_LAYOUT")
            row = dict(zip(keys, (record[index] for index in positions)))
            row["parent_id"] = int(row["parent_id"])
            pid, bg = row["parent_id"], row["background_id"]
            if (pid not in ONLINE_PARENTS or bg not in BACKGROUNDS or not row["family"]
                    or row["case_id"] != f"p{pid}_{bg}_a1_snr20"):
                raise ValueError("A32_ONLINE_CASE_IDENTITY_MISMATCH")
            rows.append(row)
    if len(rows) != 32 or len({row["case_id"] for row in rows}) != 32:
        raise ValueError("A32_FULL32_ONLINE_CASES_REQUIRED")
    for pid in ONLINE_PARENTS:
        pair = [row for row in rows if row["parent_id"] == pid]
        if (len(pair) != 2 or {row["background_id"] for row in pair} != set(BACKGROUNDS)
                or len({row["family"] for row in pair}) != 1):
            raise ValueError("A32_PAIRED_BACKGROUND_METADATA_REQUIRED")
    return tuple(rows)


def _known_background(root: Path, background_id: str) -> Mapping[str, np.ndarray]:
    if background_id not in BACKGROUNDS:
        raise ValueError("A32_KNOWN_BACKGROUND_REQUIRED")
    with np.load(_safe(root, f"inputs/a25/background/{background_id}.npz"), allow_pickle=False) as archive:
        if BACKGROUND_KEYS - set(archive.files):
            raise ValueError("A32_INCOMPLETE_KNOWN_BACKGROUND")
        arrays = {key: _readonly(archive[key]) for key in sorted(BACKGROUND_KEYS)}
    for key, shape in _COMPLEX_BACKGROUND_SHAPES.items():
        _complex_array(arrays[key], shape, key)
    real_shapes = {"J": (512, 2 * PIXELS), "points": (PIXELS, 2),
                   "source_points": (8, 2), "receiver_points": (32, 2)}
    for key, shape in real_shapes.items():
        if (arrays[key].shape != shape or arrays[key].dtype != np.float64
                or not np.isfinite(arrays[key]).all()):
            raise ValueError("A32_KNOWN_GEOMETRY_OR_J_LAYOUT:" + key)
    if (arrays["volume"].shape != () or float(arrays["volume"]) != 1 / PIXELS
            or arrays["grid"].shape != () or int(arrays["grid"]) != 24
            or any(arrays[key].shape != () or arrays[key].dtype.kind not in "US"
                   for key in ("descriptor", "geometry_descriptor"))):
        raise ValueError("A32_KNOWN_BACKGROUND_DESCRIPTOR_LAYOUT")
    return MappingProxyType(arrays)


@dataclass(frozen=True)
class OnlineCase:
    case_id: str
    parent_id: int
    family: str
    background_id: str
    observation: np.ndarray
    sigma: float
    n3: np.ndarray
    background: Mapping[str, np.ndarray]
    cache_path: Path

    @property
    def chi_b(self) -> np.ndarray:
        return self.background["chi_b"]

    @property
    def metadata(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in ("case_id", "parent_id", "family", "background_id")}


def online_cases(config_or_input_root: Mapping[str, Any] | str | Path) -> tuple[OnlineCase, ...]:
    """Return 32 online cases; pool candidate and offline target keys are untouched."""
    root = _input_root(config_or_input_root)
    rows = case_metadata(root)
    backgrounds = {name: _known_background(root, name) for name in BACKGROUNDS}
    cases = []
    for row in rows:
        path = _safe(root, f"results/a29/pools/{row['case_id']}.npz")
        with np.load(path, allow_pickle=False) as archive:
            observation = _complex_array(archive["observation"], (32, 8), "observation")
            n3 = _complex_array(archive["n3"], (PIXELS,), "n3")
            pool_background = _complex_array(archive["background"], (PIXELS,), "pool_background")
            sigma_array = np.asarray(archive["sigma"])
            if sigma_array.shape != () or not np.isfinite(sigma_array) or float(sigma_array) <= 0:
                raise ValueError("A32_POSITIVE_FINITE_SCALAR_SIGMA_REQUIRED")
            sigma = float(sigma_array)
        background = backgrounds[row["background_id"]]
        if not np.array_equal(pool_background, background["chi_b"]):
            raise ValueError("A32_POOL_KNOWN_BACKGROUND_MISMATCH")
        cache_path = root / "inputs/a28/results/a28/exact_states" / (row["case_id"] + ".npz")
        cases.append(OnlineCase(**row, observation=observation, sigma=sigma, n3=n3,
                                background=background, cache_path=cache_path))
    return tuple(cases)


def train_deltas(config_or_input_root: Mapping[str, Any] | str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Exactly one weak, clean amplitude-1.2 sample per independent TRAIN parent.

    Every split and parent identity is checked before accessing TRAIN materials.
    The returned contrast is normalized by the saved physical amplitude.
    """
    root = _input_root(config_or_input_root)
    with np.load(_safe(root, "inputs/a25/train.npz"), allow_pickle=False) as archive:
        parents, splits = np.asarray(archive["parent_id"]), np.asarray(archive["split"]).astype(str)
        if (parents.ndim != 1 or parents.dtype.kind not in "iu" or splits.shape != parents.shape
                or set(map(int, parents)) != set(TRAIN_PARENTS) or set(splits) != {"train"}):
            raise ValueError("A32_TRAIN64_IDENTITY_REQUIRED_BEFORE_MATERIAL_READ")
        metadata = {key: np.asarray(archive[key]) for key in
                    ("background_id", "amplitude", "case_id", "noise_db", "status")}
        if any(value.shape != parents.shape for value in metadata.values()):
            raise ValueError("A32_TRAIN_METADATA_LAYOUT")
        bg_ids = metadata["background_id"].astype(str)
        case_ids = metadata["case_id"].astype(str)
        selected = []
        for pid in TRAIN_PARENTS:
            mask = ((parents == pid) & (bg_ids == "weak") & (metadata["amplitude"] == AMPLITUDE)
                    & (case_ids == f"p{pid}_weak_a1_clean") & np.isnan(metadata["noise_db"])
                    & (metadata["status"].astype(str) == "OK"))
            indices = np.flatnonzero(mask)
            if len(indices) != 1:
                raise ValueError("A32_ONE_CLEAN_WEAK_TRAIN_SAMPLE_PER_PARENT_REQUIRED")
            selected.append(int(indices[0]))
        saved = np.asarray(archive["chi"])
        if saved.shape != (len(parents), PIXELS) or saved.dtype != np.complex128:
            raise ValueError("A32_TRAIN_COMPLEX128_MATERIAL_LAYOUT")
        chi = saved[selected].copy()
        selected_case_ids = case_ids[selected].tolist()
    with np.load(_safe(root, "inputs/a25/background/weak.npz"), allow_pickle=False) as archive:
        background = _complex_array(archive["chi_b"], (PIXELS,), "train_known_background")
    normalized = (chi - background[None, :]) / AMPLITUDE
    if not np.isfinite(normalized).all():
        raise ValueError("A32_NONFINITE_TRAIN_CONTRAST")
    return _readonly(normalized), {
        "parents": list(TRAIN_PARENTS), "selected_case_ids": selected_case_ids,
        "independent_parents": 64, "background": "weak", "amplitude": AMPLITUDE,
        "noise": "clean", "split": "train", "background_replication": False,
        "normalization": "(chi-chi_b)/saved amplitude", "split_leakage": False,
    }


@dataclass(frozen=True)
class CachedN3:
    data: np.ndarray
    chi: np.ndarray
    background: np.ndarray
    cache_protocol: str
    path: Path
    same_material_identity: str
    relative_chi_difference: float


def cached_n3(case: OnlineCase) -> CachedN3:
    """Read only four paid-cache keys, with no solve or nearest-material fallback.

    Exact direct identity or the exact frozen pack/unpack identity is required.
    A native material may differ from N3 by the pack/unpack rounding operation;
    this precise alternative never accepts an arbitrary nearby material.  The
    full frozen adapter separately owns exact saved-mass and equation audits.
    """
    path = Path(case.cache_path)
    root = path.parent
    path = _safe(root, path.name)
    if path.name != case.case_id + ".npz":
        raise ValueError("A32_PAID_CACHE_CASE_IDENTITY_MISMATCH")
    with np.load(path, allow_pickle=False) as archive:
        data = _complex_array(archive["data"], (1, 32, 8), "cached_data")[0]
        chi = _complex_array(archive["chi"], (1, PIXELS), "cached_chi")[0]
        background = _complex_array(archive["background"], (PIXELS,), "cached_background")
        protocol = np.asarray(archive["cache_protocol"])
        if protocol.shape != () or str(protocol) != "A28-EXACT-STATE-v1":
            raise ValueError("A32_PAID_CACHE_PROTOCOL_MISMATCH")
    if not np.array_equal(background, case.chi_b):
        raise ValueError("A32_PAID_CACHE_BACKGROUND_MISMATCH")
    root_mass = np.sqrt(1 / PIXELS)
    mass = root_mass * np.r_[case.n3.real, case.n3.imag]
    canonical = (mass[:PIXELS] + 1j * mass[PIXELS:]) / root_mass
    if np.array_equal(chi, case.n3):
        identity = "EXACT_N3"
    elif np.array_equal(chi, canonical):
        identity = "EXACT_FROZEN_MASS_PACK_UNPACK_N3"
    else:
        raise ValueError("A32_FOREIGN_OR_STALE_PAID_N3_CACHE")
    norm = float(np.linalg.norm(case.n3))
    difference = float(np.linalg.norm(chi - case.n3)) / (norm if norm else 1.0)
    return CachedN3(data, chi, background, str(protocol), path, identity, difference)


def _validate_completion(case_ids: Sequence[str], completion_token_path: str | Path) -> Path:
    """Token and all32 saved predictions must be valid before any target load."""
    path = Path(completion_token_path)
    if path.is_symlink() or path.suffix != ".json" or not path.is_file():
        raise ValueError("A32_EXPLICIT_ONLINE_COMPLETION_TOKEN_REQUIRED")
    token_root = path.parent.resolve()
    path = _safe(token_root, path.name)
    token = json.loads(path.read_text())
    expected = list(case_ids)
    if token.get("status") != "ONLINE_COMPLETE" or token.get("case_ids") != expected:
        raise ValueError("A32_ALL32_ONLINE_COMPLETION_IDENTITY_REQUIRED")
    members = token.get("prediction_files")
    if not isinstance(members, list) or len(members) != 32 or len(set(members)) != 32:
        raise ValueError("A32_ALL32_SAVED_PREDICTION_FILES_REQUIRED")
    for case_id, member in zip(expected, members):
        if not isinstance(member, str) or Path(member).name != case_id + ".npz":
            raise ValueError("A32_SAVED_PREDICTION_FILE_CASE_MISMATCH")
        prediction_path = _safe(token_root, member)
        with np.load(prediction_path, allow_pickle=False) as archive:
            identity = np.asarray(archive["case_id"])
            prediction_keys = [key for key in archive.files if key.startswith("chi_")]
            if identity.shape != () or str(identity) != case_id or not prediction_keys:
                raise ValueError("A32_SAVED_PREDICTION_CONTENT_REQUIRED")
            for key in prediction_keys:
                _complex_array(archive[key], (PIXELS,), "saved_prediction:" + key)
    return path


@dataclass(frozen=True)
class OfflineTargets:
    case_ids: tuple[str, ...]
    truth: np.ndarray
    background: np.ndarray
    n3: np.ndarray
    completion_token_path: Path

    @property
    def baseline(self) -> np.ndarray:
        return self.n3


def offline_targets(config_or_input_root: Mapping[str, Any] | str | Path,
                    case_ids: Sequence[str], *, completion_token_path: str | Path) -> OfflineTargets:
    """Independent offline scorer access, after an explicit all32 saved-output gate.

    Token schema: {"status":"ONLINE_COMPLETE", "case_ids":[online order],
    "prediction_files":["<case_id>.npz", ...]} under the online output directory.
    Each referenced NPZ requires scalar case_id and finite complex128 chi_*.
    """
    root = _input_root(config_or_input_root)
    expected = tuple(row["case_id"] for row in case_metadata(root))
    if tuple(case_ids) != expected:
        raise ValueError("A32_OFFLINE_CASE_ORDER_MISMATCH")
    token = _validate_completion(expected, completion_token_path)
    with np.load(_safe(root, "results/a29/OFFLINE_TARGETS.npz"), allow_pickle=False) as archive:
        saved_ids = tuple(np.asarray(archive["case_id"]).astype(str).tolist())
        if len(saved_ids) != 32 or len(set(saved_ids)) != 32 or set(saved_ids) != set(expected):
            raise ValueError("A32_OFFLINE_TARGET_IDENTITIES_MISMATCH")
        order = np.asarray([saved_ids.index(case_id) for case_id in expected])
        truth = _complex_array(archive["truth"], (32, PIXELS), "offline_truth")[order]
        background = _complex_array(archive["background"], (32, PIXELS), "offline_background")[order]
        n3 = _complex_array(archive["N3"], (32, PIXELS), "offline_n3")[order]
    return OfflineTargets(expected, _readonly(truth), _readonly(background), _readonly(n3), token)
