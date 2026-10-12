"""A30 paid-state adapter over the unmodified, vendored A29 backend.

Only saved known-background arrays are admitted.  Construction never calls a
geometry/Green builder, background solve, or current-material factorization.
Missing or stale paid A28/N3 caches are errors.  A book supplied by the parent
must implement ``counts``, ``span``, ``reserve_state``, and ``reserve_rhs``.
The latter counts complex columns passed to an actual LU solve, including
audits; cached matrix applications retain separate action counters.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import json
from pathlib import Path
import sys
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np


SOURCE_COMMIT = "5d5e31fcfc56ac8ef9d64c46b4b7bbdedef96bb4"
TOLERANCE = 1e-10
_VENDOR_SRC = Path(__file__).resolve().parents[2] / "vendor/frozen_a29/src"
_BACKGROUND_ARRAYS = frozenset({"chi_b", "eb", "yb", "Db", "G", "S", "incident"})
_BACKGROUND_KEYS = _BACKGROUND_ARRAYS | frozenset({
    "L_b", "J", "volume", "grid", "points", "source_points", "receiver_points",
    "descriptor", "geometry_descriptor", "initializer_id", "closure_provenance",
    "Ug", "Vg", "step", "fixed_step", "data_norm2", "beta", "D_indices", "D_blocks",
})
_CACHE_KEYS = frozenset({"mass", "chi", "current", "field", "w", "data", "lu",
                         "pivots", "residual", "background", "cache_protocol"})


class ReservationBook(Protocol):
    counts: Any

    def span(self, label: str, **counts: int): ...
    def reserve_state(self, material, *, resolution: int, purpose: str): ...
    def reserve_rhs(self, count: int, role: str): ...


def _frozen_module(name):
    """Use canonical imports required by unchanged source; reject collisions."""
    for ancestor, existing in tuple(sys.modules.items()):
        if existing is not None and ancestor.split(".")[0] in {"a20", "a24", "a25", "a27"}:
            filename = getattr(existing, "__file__", None)
            if filename is None or not Path(filename).resolve().is_relative_to(_VENDOR_SRC):
                raise ImportError("A30_NONFROZEN_MODULE_COLLISION:" + ancestor)
    if str(_VENDOR_SRC) not in sys.path:
        sys.path.insert(0, str(_VENDOR_SRC))
    module = importlib.import_module(name)
    if not Path(module.__file__).resolve().is_relative_to(_VENDOR_SRC):
        raise ImportError("A30_NONFROZEN_IMPORT:" + name)
    return module


_contracts = _frozen_module("a24.contracts")
_cpu = _frozen_module("a24.physics")
NumericalFailure = _contracts.NumericalFailure
as_columns = _cpu.as_columns


def _readonly(value, dtype=None):
    array = np.asarray(value, dtype=dtype)
    # Immutable backing bytes also prevent re-enabling the writable flag.
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)


def _relative(residual, scale):
    denominator = float(np.linalg.norm(scale))
    return float(np.linalg.norm(residual)) / (denominator if denominator else 1.)


def _safe_path(path):
    value = Path(path).resolve()
    if any("locked" in part.lower() for part in value.parts):
        raise ValueError("A30_LOCKED_PATH_FORBIDDEN")
    if any(part.lower() in {"labels", "oracle", "offline", "offline_eval", "teacher"}
           for part in value.parts):
        raise ValueError("A30_LABEL_OR_ORACLE_PATH_FORBIDDEN")
    if value.suffix != ".npz":
        raise ValueError("A30_EXPLICIT_NPZ_PATH_REQUIRED")
    return value


def torch_pivots_to_scipy(pivots, n):
    """LAPACK/Torch swaps are 1-based; SciPy applies the same swaps 0-based.

    This changes only indices, never the packed LU or permutation order.
    The arithmetic suite verifies pivoting and conjugate-adjoint solves with
    a synthetic matrix.  No actual cached material is used for this test.
    """
    values = np.asarray(pivots)
    if values.shape != (n,) or values.dtype.kind not in "iu":
        raise ValueError("A30_TORCH_PIVOTS_SHAPE_OR_DTYPE")
    if np.any(values < np.arange(1, n + 1)) or np.any(values > n):
        raise ValueError("A30_TORCH_LAPACK_PIVOTS_INVALID")
    return np.asarray(values - 1, dtype=np.int32)


@dataclass(frozen=True, slots=True, eq=False)
class AnchorState:
    chi: np.ndarray
    current: np.ndarray
    field: np.ndarray
    data: np.ndarray
    matrix: np.ndarray
    factor: tuple[np.ndarray, np.ndarray]
    residual: float
    cached: bool
    cache_path: str | None
    _owner: object = field(repr=False)
    _native: Any = field(repr=False)

    @property
    def L(self):
        return self.matrix

    @property
    def exciting(self):
        return self.field


class _SavedPhysics2D(_cpu.Physics2D):
    """Initialize frozen methods from saved arrays; bypass geometry setup."""
    def __init__(self, adapter):
        self.book = adapter.book
        self.config = {"backward_residual_tolerance": TOLERANCE}
        self.n = self.N = adapter.n
        self.grid = adapter.grid
        self.components = 1
        self.volume = adapter.volume
        self.sqrt_volume = adapter.sqrt_volume
        self.q, self.s = adapter.q, adapter.s
        self.G, self.S, self.incident = adapter.G, adapter.S, adapter.incident

    def solve_factor(self, factor, rhs, *, adjoint=False, role="full"):
        columns, _ = as_columns(rhs, self.n, role)
        self.book.reserve_rhs(int(columns.shape[1]), role + ("_adjoint" if adjoint else "_forward"))
        return super().solve_factor(factor, rhs, adjoint=adjoint, role=role)


class FrozenMaxwellAdapter:
    """Frozen 24x24 scalar-TM F and known-background L, without fallback.

    ``device='cpu'`` uses SciPy.  ``device='cuda:0'`` uses the already-installed
    Torch runtime and frozen ``a27.ExactMaxwell``.  ``backend='torch'`` also
    permits CPU-only arithmetic verification; CUDA execution is never implied.
    All public arrays use complex128 NumPy layouts (n,s) and (q,s).
    """
    def __init__(self, arrays, book: ReservationBook, *, volume=None, grid=None,
                 descriptor=None, allowed_cache_paths=(), device="cpu", backend=None,
                 purpose_prefix="A30_E1", _synthetic=False):
        if not all(hasattr(book, name) for name in ("counts", "span", "reserve_state", "reserve_rhs")):
            raise TypeError("A30_RESERVATION_BOOK_REQUIRED")
        if set(arrays) - _BACKGROUND_KEYS:
            raise ValueError("A30_UNREGISTERED_BACKGROUND_KEYS")
        if _BACKGROUND_ARRAYS - set(arrays):
            raise ValueError("A30_INCOMPLETE_SAVED_BACKGROUND")
        self.book, self._owner = book, object()
        self._synthetic = bool(_synthetic)
        self.purpose_prefix = str(purpose_prefix)
        self.volume = float(arrays.get("volume") if volume is None else volume)
        self.grid = int(arrays.get("grid", 0) if grid is None else grid)
        self.sqrt_volume = self.volume ** .5
        if not np.isfinite(self.volume) or self.volume <= 0:
            raise ValueError("A30_POSITIVE_CELL_VOLUME_REQUIRED")
        if self._synthetic:
            self.n = int(np.shape(arrays["chi_b"])[0])
            if not 1 <= self.n <= 16:
                raise ValueError("A30_SMALL_GENERIC_MATRIX_ONLY")
        else:
            if self.grid != 24 or self.volume != 1 / 576:
                raise ValueError("A30_FROZEN_24X24_MASS_CHART_REQUIRED")
            self.n = 576
        if descriptor is None:
            descriptor = arrays.get("descriptor", {})
        if not isinstance(descriptor, dict):
            descriptor = json.loads(str(descriptor))
        self.descriptor = json.loads(json.dumps(descriptor))
        self.allowed_cache_paths = frozenset(_safe_path(p) for p in allowed_cache_paths)
        self._transfers = {}
        with book.span("a30_saved_background_import", saved_background_imports=1):
            saved = {}
            for name in _BACKGROUND_ARRAYS:
                if np.asarray(arrays[name]).dtype != np.complex128:
                    raise ValueError("A30_COMPLEX128_BACKGROUND_REQUIRED:" + name)
                saved[name] = _readonly(arrays[name])
                if not np.isfinite(saved[name]).all():
                    raise NumericalFailure("A30_NONFINITE_BACKGROUND:" + name)
                setattr(self, name, saved[name])
            self.s = self.eb.shape[1] if self.eb.ndim == 2 else 0
            self.q = self.yb.shape[0] if self.yb.ndim == 2 else 0
            expected = {"chi_b": (self.n,), "eb": (self.n, self.s), "yb": (self.q, self.s),
                        "Db": (self.q, self.n), "G": (self.n, self.n),
                        "S": (self.q, self.n), "incident": (self.n, self.s)}
            if not self.s or not self.q or any(saved[k].shape != v for k, v in expected.items()):
                raise ValueError("A30_SAVED_BACKGROUND_SHAPE_MISMATCH")
            if not self._synthetic:
                self._validate_descriptor(arrays)
            self.arrays = MappingProxyType(saved)
            book.counts["a30_saved_background_array_bytes"] += sum(v.nbytes for v in saved.values())
        self._cpu = _SavedPhysics2D(self)
        self._cpu.validate_material(self.chi_b)
        self._audit_background(arrays)
        self.device = str(device)
        self.backend = backend or ("torch" if self.device.startswith("cuda") else "scipy")
        if self.backend not in {"scipy", "torch"} or (self.backend == "scipy" and self.device != "cpu"):
            raise ValueError("A30_SCIPY_CPU_OR_TORCH_CPU_CUDA_REQUIRED")
        self._torch_model = None
        if self.backend == "torch":
            self._initialize_torch()

    @classmethod
    def from_background_npz(cls, path, book, **kwargs):
        """Read exactly one named saved background, without searching data."""
        path = _safe_path(path)
        if path.name not in {"weak.npz", "ring.npz"} or path.parent.name != "background":
            raise ValueError("A30_EXPLICIT_KNOWN_BACKGROUND_FILE_REQUIRED")
        if not path.is_file():
            raise FileNotFoundError("A30_SAVED_BACKGROUND_MISSING:" + str(path))
        with book.span("a30_background_file_read", saved_background_file_reads=1):
            with np.load(path, allow_pickle=False) as archive:
                if set(archive.files) - _BACKGROUND_KEYS:
                    raise ValueError("A30_UNREGISTERED_BACKGROUND_FILE_KEYS")
                arrays = {k: archive[k].copy() for k in archive.files}
        return cls(arrays, book, **kwargs)

    @classmethod
    def for_synthetic_matrices(cls, arrays, book, *, volume=1., allowed_cache_paths=(),
                               backend="scipy"):
        """Arithmetic-only fixtures, never a Maxwell geometry constructor."""
        return cls(arrays, book, volume=volume, grid=0, descriptor={},
                   allowed_cache_paths=allowed_cache_paths, backend=backend,
                   purpose_prefix="SYNTHETIC_MATRIX", _synthetic=True)

    def _validate_descriptor(self, arrays):
        descriptor = self.descriptor
        geometry = descriptor.get("geometry", {})
        if (self.s != 8 or self.q != 32 or descriptor.get("adapter") != "a24_2d_tm_a2"
                or descriptor.get("precision") != "complex128"
                or descriptor.get("material_law") != "scalar_chi"
                or descriptor.get("current_normalization") != "sqrt_cell_volume"
                or descriptor.get("cell_volume") != self.volume
                or geometry != {"grid": 24, "side": 1., "n_tx": 8, "n_rx": 32,
                                "radius_tx": .75, "radius_rx": .8, "aperture": "full"}):
            raise ValueError("A30_FROZEN_BACKGROUND_DESCRIPTOR_MISMATCH")
        if not np.array_equal(np.asarray(descriptor.get("background")),
                              np.c_[self.chi_b.real, self.chi_b.imag]):
            raise ValueError("A30_BACKGROUND_MATERIAL_DESCRIPTOR_MISMATCH")
        for name, key, shape in (("source_points", "source_coordinates_in_order", (8, 2)),
                                 ("receiver_points", "receiver_coordinates_in_order", (32, 2))):
            if name not in arrays or np.shape(arrays[name]) != shape or not np.array_equal(
                    arrays[name], np.asarray(descriptor.get(key))):
                raise ValueError("A30_APERTURE_ORDER_MISMATCH:" + name)

    def _audit_background(self, arrays):
        with self.book.span("a30_cached_background_equation_audit", background_equation_audits=1,
                            cached_background_field_columns=self.s, cached_background_transfer_columns=self.n):
            current = self.chi_b[:, None] * self.eb
            matrix = np.eye(self.n, dtype=np.complex128) - self.chi_b[:, None] * self.G
            checks = {"field": _relative(self.incident + self.G @ current - self.eb, self.eb),
                      "data": _relative(self.S @ current - self.yb, self.yb),
                      "receiver_transfer": _relative(self.Db @ matrix - self.S, self.S)}
            if "L_b" in arrays:
                checks["saved_background_matrix"] = _relative(np.asarray(arrays["L_b"]) - matrix, matrix)
            if "J" in arrays:
                b = np.einsum("qn,ns->sqn", self.Db, self.eb) / self.sqrt_volume
                re = np.concatenate((b.real, b.imag), axis=1).reshape(2 * self.q * self.s, self.n)
                im = np.concatenate((-b.imag, b.real), axis=1).reshape(2 * self.q * self.s, self.n)
                derived = np.concatenate((re, im), axis=1)
                if np.shape(arrays["J"]) != derived.shape:
                    raise ValueError("A30_BACKGROUND_MASS_J_SHAPE_MISMATCH")
                checks["saved_real_mass_J"] = _relative(np.asarray(arrays["J"]) - derived, derived)
            if any(not np.isfinite(v) or v > TOLERANCE for v in checks.values()):
                raise NumericalFailure("A30_SAVED_BACKGROUND_IDENTITY_FAILED:" + repr(checks))
            self.background_audit = MappingProxyType(checks)

    def _initialize_torch(self):
        with self.book.span("a30_optional_torch_import_and_transfer", optional_torch_adapters=1):
            torch = importlib.import_module("torch")
            exact = _frozen_module("a27.exact")
            tp = _frozen_module("a25.torch_physics")
            device = tp.require_device(self.device)

            class SavedReference(tp.TorchReference):
                def __init__(reference):
                    torch.nn.Module.__init__(reference)
                    reference.book = self.book
                    reference.config = {"backward_residual_tolerance": TOLERANCE,
                                        "feasibility_tolerance": TOLERANCE}
                    reference.n, reference.grid = self.n, self.grid
                    reference.q, reference.s = self.q, self.s
                    reference.volume, reference.sqrt_volume = self.volume, self.sqrt_volume
                    for name in ("G", "S", "incident", "chi_b"):
                        reference.register_buffer(name, torch.tensor(np.array(getattr(self, name), copy=True),
                                                                     dtype=torch.complex128, device=device))

            class ReservedExact(exact.ExactMaxwell):
                def _solve(model, state, rhs, *, adjoint=False, role="full"):
                    self.book.reserve_rhs(int(len(rhs) * rhs.shape[-1]),
                                          role + ("_adjoint" if adjoint else "_forward"))
                    return super()._solve(state, rhs, adjoint=adjoint, role=role)

            self._torch, self._torch_exact = torch, exact
            self._torch_reference = SavedReference()
            self._torch_model = ReservedExact(self._torch_reference)

    def _material(self, chi):
        return self._cpu.validate_material(chi).copy()

    def _state(self, state):
        if not isinstance(state, AnchorState) or state._owner is not self._owner:
            raise ValueError("A30_FOREIGN_ANCHOR_STATE")
        return state

    def _export_state(self, native, *, cached=False, path=None):
        if self.backend == "scipy":
            values = [native.chi, native.current, native.field, native.data, native.L]
            factor = native.factor
            residual = float(native.residual)
        else:
            values = [getattr(native, name)[0].detach().cpu().numpy()
                      for name in ("chi", "current", "field", "data", "matrix")]
            factor = (native.lu[0].detach().cpu().numpy(),
                      torch_pivots_to_scipy(native.pivots[0].detach().cpu().numpy(), self.n))
            residual = float(native.residual[0])
        return AnchorState(*(_readonly(a) for a in values),
                           (_readonly(factor[0]), _readonly(factor[1])), residual,
                           cached, None if path is None else str(path), self._owner, native)

    def full_state(self, chi, *, purpose="endpoint"):
        """One attempted fresh state; reservation precedes matrix/LU/solve."""
        chi = self._material(chi)
        self.book.reserve_state(chi, resolution=self.grid,
                                purpose=self.purpose_prefix + ":" + str(purpose))
        with self.book.span("a30_fresh_full_state", a30_fresh_full_state_calls=1):
            if self.backend == "scipy":
                native = self._cpu.full_state(chi)
            else:
                mass = _contracts.mass_pack(chi, self.volume)[None]
                tensor = self._torch.tensor(mass, dtype=self._torch.float64, device=self.device)
                native = self._torch_model(tensor)
            return self._export_state(native)

    def cached_anchor(self, chi, path):
        """Read a paid same-material cache; LU audit costs s new RHS, 0 states."""
        chi = self._material(chi)
        path = _safe_path(path)
        if path not in self.allowed_cache_paths:
            raise ValueError("A30_CACHE_PATH_NOT_WHITELISTED")
        if not path.is_file():
            raise FileNotFoundError("A30_PAID_N3_CACHE_MISSING_NO_FALLBACK:" + str(path))
        with self.book.span("a30_paid_anchor_cache_load", paid_anchor_cache_loads=1):
            with np.load(path, allow_pickle=False) as archive:
                if set(archive.files) != _CACHE_KEYS:
                    raise ValueError("A30_PAID_CACHE_KEYS_MISMATCH")
                a = {name: archive[name].copy() for name in archive.files}
            mass = _contracts.mass_pack(chi, self.volume)[None]
            shapes = {"mass": (1, 2 * self.n), "chi": (1, self.n), "current": (1, self.n, self.s),
                      "field": (1, self.n, self.s), "w": (1, self.n, self.s),
                      "data": (1, self.q, self.s), "lu": (1, self.n, self.n),
                      "pivots": (1, self.n), "residual": (1,), "background": (self.n,)}
            if any(a[key].shape != shape for key, shape in shapes.items()):
                raise ValueError("A30_PAID_CACHE_LAYOUT_MISMATCH")
            if (str(a["cache_protocol"]) != "A28-EXACT-STATE-v1"
                    or not np.array_equal(a["background"], self.chi_b)
                    or not np.array_equal(a["mass"], mass)):
                raise ValueError("A30_FOREIGN_OR_STALE_PAID_CACHE")
            for name in ("chi", "current", "field", "w", "data", "lu", "background"):
                if a[name].dtype != np.complex128 or not np.isfinite(a[name]).all():
                    raise ValueError("A30_PAID_CACHE_PRECISION_OR_FINITE_MISMATCH:" + name)
            if a["mass"].dtype != np.float64 or not np.isfinite(a["residual"]).all():
                raise ValueError("A30_PAID_CACHE_MASS_OR_RESIDUAL_INVALID")
            unpacked_chi = _contracts.mass_unpack(mass[0], self.volume)
            if _relative(a["chi"][0] - unpacked_chi, unpacked_chi) > TOLERANCE:
                raise ValueError("A30_PAID_CACHE_CHI_MASS_MISMATCH")
            actual_chi = a["chi"][0]
            matrix = np.eye(self.n, dtype=np.complex128) - actual_chi[:, None] * self.G
            rhs = actual_chi[:, None] * self.incident
            checks = {"current": _relative(matrix @ a["current"][0] - rhs, rhs),
                      "field": _relative(a["field"][0] - self.incident - self.G @ a["current"][0], a["field"][0]),
                      "data": _relative(a["data"][0] - self.S @ a["current"][0], a["data"][0]),
                      "relative_current": _relative(a["w"][0] - (actual_chi - self.chi_b)[:, None] * a["field"][0], a["w"][0])}
            if (any(not np.isfinite(v) or v > TOLERANCE for v in checks.values())
                    or np.any(a["residual"] < 0) or np.any(a["residual"] > TOLERANCE)):
                raise NumericalFailure("A30_PAID_CACHE_EQUATION_MISMATCH:" + repr(checks))
            pivots = torch_pivots_to_scipy(a["pivots"][0], self.n)
            if self.backend == "scipy":
                native = _cpu.FullState(actual_chi, a["current"][0], a["field"][0], a["data"][0],
                                        matrix, (a["lu"][0], pivots), checks["current"])
            else:
                t = {name: self._torch.tensor(a[name], device=self.device) for name in shapes
                     if name != "background"}
                native = self._torch_exact.ExactState(t["mass"], t["chi"], t["current"], t["field"],
                    t["w"], t["data"], self._torch.tensor(matrix[None], device=self.device),
                    t["lu"], t["pivots"], t["residual"], id(self._torch_model))
            state = self._export_state(native, cached=True, path=path)
            check = self.solve_anchor(rhs, state, role="cached_LU_audit")
            if _relative(check - state.current, state.current) > TOLERANCE:
                raise NumericalFailure("A30_PAID_CACHE_LU_MISMATCH")
            return state

    def L_response(self, chi):
        """Frozen A24/A29 known-background affine response; no new solves."""
        chi = self._material(chi)
        with self.book.span("a30_cached_L_apply", cached_L_material_directions=1,
                            cached_L_source_columns=self.s):
            return self.yb + self.Db @ ((chi - self.chi_b)[:, None] * self.eb)

    def solve_anchor(self, rhs, state, *, adjoint=False, role="certificate"):
        state = self._state(state)
        columns, vector = as_columns(rhs, self.n, role)
        if self.backend == "scipy":
            value = self._cpu.solve_factor(state._native.factor, columns, adjoint=adjoint, role=role)
        else:
            with self.book.span("a30_anchor_RHS_device_transfer", anchor_RHS_transfer_columns=columns.shape[1]):
                tensor = self._torch.tensor(np.array(columns, copy=True), dtype=self._torch.complex128,
                                            device=self.device)[None]
                value, _ = self._torch_model._solve(state._native, tensor, adjoint=adjoint, role=role)
                value = value[0].detach().cpu().numpy()
        with self.book.span("a30_anchor_solve_residual_audit", anchor_residual_columns=columns.shape[1]):
            residual = (state.matrix.conj().T if adjoint else state.matrix) @ value - columns
            error = _relative(residual, columns)
            if not np.isfinite(error) or error > TOLERANCE:
                raise NumericalFailure("A30_ANCHOR_BACKWARD_RESIDUAL:" + str(error))
        return value[:, 0] if vector else value

    def anchor_jvp(self, dchi, state):
        """Eq. (1), raw complex-material directions; each s-column solve paid."""
        state = self._state(state)
        directions, vector = as_columns(dchi, self.n, "material_direction")
        count = directions.shape[1]
        with self.book.span("a30_anchor_material_jvp", anchor_material_directions=count):
            rhs = (state.field[:, :, None] * directions[:, None, :]).reshape(self.n, self.s * count)
            current = self.solve_anchor(rhs, state, role="anchor_jvp")
            value = self._cpu.observe(current).reshape(self.q, self.s, count)
        return value[:, :, 0] if vector else value

    def receiver_transfer(self, state):
        """Return cached T=S L^-1; first construction costs q adjoint RHS."""
        state = self._state(state)
        if state not in self._transfers:
            with self.book.span("a30_anchor_receiver_transfer", receiver_transfer_builds=1):
                z = self.solve_anchor(self.S.conj().T, state, adjoint=True, role="receiver_transfer")
                self._transfers[state] = _readonly(z.conj().T)
        else:
            self.book.counts["a30_anchor_receiver_transfer_reuses"] += 1
        return self._transfers[state]

    def cached_material_dot_identity(self, dchi, load, state):
        """Real material-adjoint identity using saved field and cached T only.

        This never calls a new state; a missing receiver transfer is constructed
        once by q paid adjoint RHS.  ``load`` is a complex (q,s) data load.
        Both raw-pixel and fixed mass-coordinate dot products are returned.
        """
        state = self._state(state)
        direction, vector = as_columns(dchi, self.n, "dot_material_direction")
        if not vector:
            raise ValueError("A30_DOT_SINGLE_DIRECTION_REQUIRED")
        load = np.asarray(load, dtype=np.complex128)
        if load.shape != (self.q, self.s) or not np.isfinite(load).all():
            raise ValueError("A30_COMPLEX_DATA_LOAD_REQUIRED")
        with self.book.span("a30_cached_material_adjoint_dot", cached_material_dot_checks=1):
            transfer = self.receiver_transfer(state)
            jvp = transfer @ (direction * state.field)
            gradient = np.sum(state.field.conj() * (transfer.conj().T @ load), axis=1)
            lhs = float(np.vdot(jvp, load).real)
            rhs = float(np.vdot(direction[:, 0], gradient).real)
            mass_direction = _contracts.mass_pack(direction[:, 0], self.volume)
            mass_gradient = np.r_[gradient.real, gradient.imag] / self.sqrt_volume
            mass_rhs = float(np.dot(mass_direction, mass_gradient))
            error = max(abs(lhs - rhs), abs(lhs - mass_rhs)) / max(1., abs(lhs), abs(rhs))
            if not np.isfinite(error) or error > TOLERANCE:
                raise NumericalFailure("A30_CACHED_MATERIAL_DOT_MISMATCH")
        return {"data_dot": lhs, "material_dot": rhs, "mass_material_dot": mass_rhs,
                "relative_error": error}


# Concise alias for callers; the implementation and frozen source are identical.
MaxwellAdapter = FrozenMaxwellAdapter
