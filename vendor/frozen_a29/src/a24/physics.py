"""Separately named 2-D TM adapter around the unchanged A2 discretization.

Current and exciting field are mass normalized: ``c = sqrt(v) j`` and
``e = sqrt(v) E + G c``.  The 2-D material law is simply ``a(chi)=chi``.
All native solves use one SciPy LU for a collection of right-hand sides.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
import warnings

import numpy as np
import scipy.linalg as la

from .contracts import NumericalFailure, get_ledger, mass_pack, mass_unpack, pack, unpack


_VENDOR_NAME = "_a24_unchanged_a2_physics"


def _vendor():
    """Load by path; importing this adapter does not initialize a geometry."""
    if _VENDOR_NAME not in sys.modules:
        path = Path(__file__).resolve().parents[2] / "vendor" / "a2" / "physics.py"
        spec = importlib.util.spec_from_file_location(_VENDOR_NAME, path)
        if spec is None or spec.loader is None:
            raise ImportError("A2_VENDOR_PHYSICS_UNAVAILABLE")
        module = importlib.util.module_from_spec(spec)
        # dataclasses needs the module registered while executing its classes.
        sys.modules[_VENDOR_NAME] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(_VENDOR_NAME, None)
            raise
    return sys.modules[_VENDOR_NAME]


def as_columns(value, rows, name="rhs"):
    """Validate a complex vector/matrix while preserving vector return shape."""
    a = np.asarray(value, dtype=np.complex128)
    vector = a.ndim == 1
    if vector:
        a = a[:, None]
    if a.ndim != 2 or a.shape[0] != int(rows):
        raise ValueError("COMPLEX_COLUMNS_REQUIRED:" + name)
    if not np.isfinite(a).all():
        raise NumericalFailure("NONFINITE_ARRAY:" + name)
    return a, vector


@dataclass(slots=True)
class FullState:
    chi: np.ndarray
    current: np.ndarray
    field: np.ndarray
    data: np.ndarray
    L: np.ndarray
    factor: tuple
    residual: float

    @property
    def exciting(self):
        return self.field


class Physics2D:
    """Finite-grid Helmholtz VIE; this is not an A20 3-D scalar switch."""

    def __init__(self, grid=24, book=None, config=None):
        self.config = dict(config or {})
        self.book = get_ledger() if book is None else book
        self.grid = int(grid)
        self.N = self.n = self.grid * self.grid
        self.components = 1
        self.s, self.q = 8, 32
        kernel = _vendor()
        self.frequency_hz = float(kernel.C0)
        with self.book.span("tm2d_geometry_setup", physics_geometry_builds=1,
                            G_dense_assemblies=1, receiver_matrix_assemblies=1):
            self.geometry = kernel.Geometry(n=self.grid, side=1.0, n_tx=self.s,
                                            n_rx=self.q, radius_tx=.75,
                                            radius_rx=.8, aperture="full")
            self.vie = kernel.VIE(self.geometry, frequency_hz=self.frequency_hz)
            self.points = np.asarray(self.vie.points, dtype=np.float64)
            self.volume = float(self.vie.h ** 2)
            self.sqrt_volume = float(np.sqrt(self.volume))
            self.k = float(self.vie.k)
            self.G = np.asarray(self.vie.D.dense_direct(), dtype=np.complex128)
            self.S = np.asarray(self.vie.S / self.sqrt_volume, dtype=np.complex128)
            self.incident = np.asarray(self.sqrt_volume * self.vie.E,
                                       dtype=np.complex128)
            self.book.counts["physics_array_bytes"] += (
                self.points.nbytes + self.G.nbytes + self.S.nbytes + self.incident.nbytes
                + self.vie.D.kernel.nbytes + self.vie.D._fk.nbytes
                + self.vie.D._fk_adj.nbytes + self.vie.E.nbytes + self.vie.S.nbytes
                + self.vie.direct.nbytes + self.vie.tx.nbytes + self.vie.rx.nbytes
            )

    def bind_book(self, book):
        self.book = book
        return self

    def validate_material(self, chi):
        a = np.asarray(chi, dtype=np.complex128)
        if a.shape != (self.N,):
            raise ValueError("COMPLEX_MATERIAL_N_REQUIRED")
        if not np.isfinite(a).all():
            raise NumericalFailure("NONFINITE_MATERIAL")
        tol = 1e-10
        if (np.any(a.real < -.5 - tol) or np.any(a.real > 4. + tol)
                or np.any(a.imag < -tol) or np.any(a.imag > .5 + tol)):
            raise NumericalFailure("MATERIAL_DOMAIN_VIOLATION")
        return a

    def material_to_a(self, chi):
        return self.validate_material(chi).copy()

    def material_derivative(self, chi):
        self.validate_material(chi)
        return np.ones(self.N, dtype=np.complex128)

    def green(self, value, *, adjoint=False):
        v, vector = as_columns(value, self.n, "G")
        label = "G_adjoint" if adjoint else "G"
        with self.book.span(label, **{label + "_actions": v.shape[1],
                                     "Green_kernel_rhs": v.shape[1]}):
            out = (self.G.conj().T if adjoint else self.G) @ v
            self.book.counts["green_result_bytes"] += out.nbytes
        return out[:, 0] if vector else out

    def observe(self, value, *, adjoint=False):
        v, vector = as_columns(value, self.q if adjoint else self.n, "S")
        label = "S_adjoint" if adjoint else "S"
        with self.book.span(label, **{label + "_actions": v.shape[1],
                                     "receiver_kernel_rhs": v.shape[1]}):
            out = (self.S.conj().T if adjoint else self.S) @ v
            self.book.counts["receiver_result_bytes"] += out.nbytes
        return out[:, 0] if vector else out

    def factor_matrix(self, matrix, *, role="full", **counters):
        """General complex LU; singular factors fail without a replacement."""
        a = np.asarray(matrix, dtype=np.complex128)
        if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] == 0:
            raise ValueError("NONEMPTY_SQUARE_FACTOR_REQUIRED")
        count = {"LU_factorizations": 1, role + "_LU_factorizations": 1, **counters}
        with self.book.span(role + "_LU_factor", **count):
            if not np.isfinite(a).all():
                raise NumericalFailure("NONFINITE_FACTOR:" + role)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", la.LinAlgWarning)
                    factor = la.lu_factor(a.copy(), overwrite_a=True, check_finite=False)
            except (la.LinAlgError, la.LinAlgWarning, ValueError) as exc:
                raise NumericalFailure("LU_FAILED:" + role + ":" + str(exc)) from exc
            self.book.counts["LU_array_bytes"] += factor[0].nbytes + factor[1].nbytes
        return factor

    def solve_factor(self, factor, rhs, *, adjoint=False, role="full"):
        v, vector = as_columns(rhs, factor[0].shape[0], role + "_solve")
        name = role + ("_adjoint" if adjoint else "_forward")
        with self.book.span(name + "_solve", linear_solve_rhs=v.shape[1],
                            **{name + "_RHS": v.shape[1]}):
            try:
                out = la.lu_solve(factor, v, trans=2 if adjoint else 0,
                                  check_finite=False)
            except (la.LinAlgError, ValueError) as exc:
                raise NumericalFailure("SOLVE_FAILED:" + name + ":" + str(exc)) from exc
            if not np.isfinite(out).all():
                raise NumericalFailure("NONFINITE_SOLVE:" + name)
            self.book.counts["solve_result_bytes"] += out.nbytes
        return out[:, 0] if vector else out

    def apply_matrix(self, matrix, rhs, *, adjoint=False, role="L"):
        a = np.asarray(matrix, dtype=np.complex128)
        v, vector = as_columns(rhs, a.shape[0] if adjoint else a.shape[1], role)
        label = role + ("_adjoint" if adjoint else "")
        with self.book.span(label + "_apply", **{label + "_actions": v.shape[1]}):
            out = (a.conj().T if adjoint else a) @ v
            self.book.counts["matrix_result_bytes"] += out.nbytes
        return out[:, 0] if vector else out

    def full_state(self, chi):
        chi = self.validate_material(chi).copy()
        with self.book.span("full_forward", full_forward_calls=1,
                            full_forward_source_rhs=self.s):
            with self.book.span("full_material_assembly", full_L_assemblies=1):
                L = np.eye(self.n, dtype=np.complex128) - chi[:, None] * self.G
                rhs = chi[:, None] * self.incident
                self.book.counts["full_assembly_bytes"] += L.nbytes + rhs.nbytes
            factor = self.factor_matrix(L, role="full")
            current = self.solve_factor(factor, rhs, role="full")
            field = self.incident + self.green(current)
            data = self.observe(current)
            with self.book.span("full_state_residual_audit", residual_audits=1,
                                full_state_residual_rhs=self.s):
                residual_vector = self.apply_matrix(L, current, role="residual_L") - rhs
                rhs_norm = float(la.norm(rhs))
                residual = float(la.norm(residual_vector) / (rhs_norm if rhs_norm else 1.))
                self.book.counts["residual_array_bytes"] += residual_vector.nbytes
                tolerance = float(self.config.get("backward_residual_tolerance", 1e-10))
                if not np.isfinite(residual) or residual > tolerance:
                    raise NumericalFailure("FULL_STATE_BACKWARD_RESIDUAL:" + str(residual))
            self.book.counts["full_field_array_bytes"] += field.nbytes + chi.nbytes
        return FullState(chi, current, field, data, L, factor, residual)

    def material_pullback(self, field, z):
        """Real mass-coordinate adjoint of ``delta_chi * field``."""
        e, _ = as_columns(field, self.n, "material_field")
        v, _ = as_columns(z, self.n, "material_load")
        if e.shape != v.shape:
            raise ValueError("MATCHED_FIELD_SOURCE_SHAPE_REQUIRED")
        with self.book.span("material_pullback", material_adjoint_directions=1,
                            material_adjoint_source_rhs=e.shape[1]):
            t = np.sum(e.conj() * v, axis=1) / self.sqrt_volume
            result = np.r_[t.real, t.imag]
            self.book.counts["material_pullback_bytes"] += t.nbytes + result.nbytes
        return result

    def pack_real(self, data):
        return pack(data)

    def unpack_real(self, data):
        return unpack(data, self.q, self.s)

    def pack_material(self, chi):
        return mass_pack(self.validate_material(chi), self.volume)

    def unpack_material(self, value):
        chi = mass_unpack(value, self.volume)
        if chi.shape != (self.N,):
            raise ValueError("MATERIAL_MASS_DIMENSION_MISMATCH")
        return chi
