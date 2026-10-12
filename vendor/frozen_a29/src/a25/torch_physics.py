"""Differentiable A24 relative physics in real material mass coordinates.

Only fixed, known-background arrays enter this module.  State-dependent
gradients stay in the Torch graph, including differentiation through an
explicit adjoint.  Guards detach their *decision* only; they never replace a
failed solve.  CPU/CUDA double precision is required; MPS is unsupported.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from a24.contracts import NumericalFailure, get_ledger


def require_device(device):
    device = torch.device(device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("A25_DOUBLE_PHYSICS_REQUIRES_CPU_OR_CUDA")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA_UNAVAILABLE")
    return device


def mass_pack(chi, volume):
    if not chi.is_complex():
        raise ValueError("COMPLEX_MATERIAL_REQUIRED")
    return (float(volume) ** .5) * torch.cat((chi.real, chi.imag), dim=-1)


def mass_unpack(value, volume):
    if value.is_complex() or value.shape[-1] % 2:
        raise ValueError("REAL_MATERIAL_MASS_COORDINATES_REQUIRED")
    n = value.shape[-1] // 2
    return torch.complex(value[..., :n], value[..., n:]) / (float(volume) ** .5)


def pack_data(data):
    """(...,q,s) -> (...,s*[Re q,Im q]), matching A24 exactly."""
    if not data.is_complex() or data.ndim < 2:
        raise ValueError("COMPLEX_RECEIVER_SOURCE_ARRAY_REQUIRED")
    transposed = data.transpose(-1, -2)
    joined = torch.cat((transposed.real, transposed.imag), dim=-1)
    return joined.flatten(-2)


def unpack_data(value, q, s):
    if value.is_complex() or value.shape[-1] != 2 * q * s:
        raise ValueError("REAL_PACKED_DATA_REQUIRED")
    shaped = value.reshape(*value.shape[:-1], s, 2 * q)
    return torch.complex(shaped[..., :q], shaped[..., q:]).transpose(-1, -2)


def _batched(value, dimensions):
    single = value.ndim == dimensions - 1
    if single:
        value = value.unsqueeze(0)
    if value.ndim != dimensions:
        raise ValueError("INVALID_TORCH_ARRAY_DIMENSION")
    return value, single


def validate_closure_provenance(provenance, descriptor, *, rank, config=None,
                                required=True):
    """Bind a portable closure to its frozen, known-background construction.

    This checks metadata only.  It neither rebuilds the basis nor introduces
    a numerical oracle or a content hash into cache/checkpoint loading.
    """
    config = dict(config or {})
    if not provenance:
        if required:
            raise ValueError("A25_CLOSURE_PROVENANCE_REQUIRED")
        return None
    if not isinstance(provenance, dict):
        raise ValueError("CLOSURE_PROVENANCE_OBJECT_REQUIRED")
    fields = {"builder", "rank", "probe_seed", "source_commit", "protocol_id",
              "background_id", "background_descriptor", "basis_identity",
              "truth_or_oracle_used"}
    if fields - provenance.keys():
        raise ValueError("INCOMPLETE_CLOSURE_PROVENANCE")
    builder = provenance["builder"]
    if builder not in {"TASK_R32", "R16", "R24", "R32"}:
        raise ValueError("UNKNOWN_FROZEN_CLOSURE_BUILDER")
    expected_rank = int(builder.rsplit("R", 1)[1])
    if (type(provenance["rank"]) is not int or provenance["rank"] != expected_rank
            or int(rank) != expected_rank):
        raise ValueError("CLOSURE_PROVENANCE_RANK_MISMATCH")
    if type(provenance["probe_seed"]) is not int or provenance["probe_seed"] < 0:
        raise ValueError("EXPLICIT_CLOSURE_PROBE_SEED_REQUIRED")
    background_id = provenance["background_id"]
    if background_id not in {"weak", "ring"}:
        raise ValueError("KNOWN_BACKGROUND_ID_REQUIRED")
    if provenance["background_descriptor"] != descriptor:
        raise ValueError("CLOSURE_KNOWN_BACKGROUND_DESCRIPTOR_MISMATCH")
    if provenance["basis_identity"] != "A25-frozen-background-" + background_id + "-" + builder:
        raise ValueError("CLOSURE_BASIS_IDENTITY_MISMATCH")
    if provenance["truth_or_oracle_used"] is not False:
        raise ValueError("CLOSURE_TRUTH_OR_ORACLE_FORBIDDEN")
    if (not isinstance(provenance["source_commit"], str) or not provenance["source_commit"]
            or not isinstance(provenance["protocol_id"], str)
            or not provenance["protocol_id"].startswith("A25-")):
        raise ValueError("EXPLICIT_A25_SOURCE_PROTOCOL_REQUIRED")
    for key in ("source_commit", "protocol_id"):
        if key in config and provenance[key] != config[key]:
            raise ValueError("CLOSURE_ADMITTED_" + key.upper() + "_MISMATCH")
    if "selected_nn_closure" in config and builder != config["selected_nn_closure"]:
        raise ValueError("CLOSURE_ADMITTED_BUILDER_MISMATCH")
    if builder == "TASK_R32":
        if (provenance.get("independent_material_probes") != 4
                or provenance.get("independent_receiver_probes") != 8):
            raise ValueError("TASK_R32_PROBE_CONSTRUCTION_MISMATCH")
        expected_seed = int(config["master_seed"]) + 537 if "master_seed" in config else None
    else:
        expected_seed = (int(config.get("control_seed", 241008)) + int(background_id == "ring"))
    if expected_seed is not None and provenance["probe_seed"] != expected_seed:
        raise ValueError("CLOSURE_ADMITTED_PROBE_SEED_MISMATCH")
    return deepcopy(provenance)


@dataclass
class TorchState:
    mass: torch.Tensor
    chi: torch.Tensor
    h: torch.Tensor
    data: torch.Tensor
    w: torch.Tensor
    field: torch.Tensor
    margins: dict = dataclass_field(default_factory=dict)
    extra: dict = dataclass_field(default_factory=dict, repr=False)
    current: torch.Tensor | None = None
    canonical_field: torch.Tensor | None = None


class TorchReference(nn.Module):
    """Frozen cache; construction/factor preparation must be metered by caller.

    Every array is copied.  No full inverse or current-material Jacobian is
    retained.  The background LU is fixed and its RHS actions are charged.
    """
    def __init__(self, arrays, *, volume, grid, descriptor=None, book=None,
                 config=None, device="cpu", fixed_step=None, require_provenance=None):
        super().__init__()
        self.book = get_ledger() if book is None else book
        self.config = dict(config or {})
        self.descriptor = deepcopy(dict(descriptor or {}))
        self.volume, self.grid = float(volume), int(grid)
        self.sqrt_volume, self.n = self.volume ** .5, self.grid ** 2
        device = require_device(device)
        if self.volume <= 0 or self.grid < 1:
            raise ValueError("POSITIVE_GEOMETRY_REQUIRED")
        for key, described in (("source_points", "source_coordinates_in_order"),
                               ("receiver_points", "receiver_coordinates_in_order")):
            if key in arrays and described in self.descriptor and not np.array_equal(
                    np.asarray(arrays[key]), np.asarray(self.descriptor[described])):
                raise ValueError("CACHE_APERTURE_ORDER_DESCRIPTOR_MISMATCH")
        if self.descriptor and (self.descriptor.get("precision") != "complex128"
                                or self.descriptor.get("cell_volume") != self.volume
                                or self.descriptor.get("geometry", {}).get("grid") != self.grid):
            raise ValueError("CACHE_PRECISION_GEOMETRY_DESCRIPTOR_MISMATCH")
        required = ("chi_b", "eb", "yb", "Db", "G", "S", "incident", "Ug", "Vg")
        if set(required) - set(arrays):
            raise ValueError("INCOMPLETE_BACKGROUND_CACHE")
        provenance = arrays.get("closure_provenance")
        if provenance is not None and not isinstance(provenance, dict):
            try:
                provenance = json.loads(str(provenance))
            except (TypeError, ValueError) as exc:
                raise ValueError("INVALID_CLOSURE_PROVENANCE_JSON") from exc
        if require_provenance is None:
            require_provenance = (str(self.config.get("protocol_id", "")).startswith("A25-")
                                  or "selected_nn_closure" in self.config)
        if np.ndim(arrays["Ug"]) != 2 or np.shape(arrays["Ug"]) != np.shape(arrays["Vg"]):
            raise ValueError("LOWRANK_CACHE_SHAPE_MISMATCH")
        self.closure_provenance = validate_closure_provenance(provenance, self.descriptor,
            rank=np.shape(arrays["Ug"])[1], config=self.config, required=require_provenance)
        if self.closure_provenance is not None:
            # Saved reference_descriptors now bind checkpoints to construction
            # and portable basis identity as well as aperture/background order.
            self.descriptor["closure_provenance"] = deepcopy(self.closure_provenance)
        with self.book.span("torch_reference_transfer", torch_reference_builds=1):
            for name in required:
                value = torch.tensor(np.array(arrays[name], copy=True),
                                     dtype=torch.complex128, device=device)
                if not torch.isfinite(value).all():
                    raise NumericalFailure("NONFINITE_TORCH_CACHE:" + name)
                self.register_buffer(name, value)
                self.book.counts["torch_reference_buffer_bytes"] += value.numel() * value.element_size()
            self.s, self.q = self.eb.shape[1], self.yb.shape[0]
            expected = {"chi_b": (self.n,), "eb": (self.n, self.s),
                        "yb": (self.q, self.s), "Db": (self.q, self.n),
                        "G": (self.n, self.n), "S": (self.q, self.n),
                        "incident": (self.n, self.s)}
            if any(tuple(getattr(self, name).shape) != shape for name, shape in expected.items()):
                raise ValueError("BACKGROUND_CACHE_SHAPE_MISMATCH")
            if self.Ug.shape != self.Vg.shape or self.Ug.shape[0] != self.n:
                raise ValueError("LOWRANK_CACHE_SHAPE_MISMATCH")
            self.rank = self.Ug.shape[1]
            self.register_buffer("cb", self.chi_b[:, None] * self.eb)
            lb = torch.eye(self.n, dtype=torch.complex128, device=device) - self.chi_b[:, None] * self.G
            if "L_b" in arrays:
                supplied = torch.tensor(np.array(arrays["L_b"], copy=True), dtype=torch.complex128, device=device)
                if supplied.shape != lb.shape or not torch.allclose(supplied, lb, rtol=1e-13, atol=1e-13):
                    raise ValueError("BACKGROUND_MATRIX_DESCRIPTOR_MISMATCH")
                lb = supplied
            self.register_buffer("Lb", lb)
            norm = self.yb.abs().square().sum()
            if float(norm) <= 0:
                raise NumericalFailure("ZERO_BACKGROUND_DATA_NORM")
            self.register_buffer("yb_norm2", norm)
            indices = np.asarray(arrays.get("D_indices", np.empty((0, 4))), dtype=np.int64)
            blocks = np.asarray(arrays.get("D_blocks", np.empty((0, 4, 4))), dtype=np.complex128)
            if indices.ndim != 2 or blocks.shape != (*indices.shape, indices.shape[-1]):
                raise ValueError("INVALID_LOCAL_BLOCK_CACHE")
            if indices.size and (len(np.unique(indices)) != indices.size
                                 or indices.min() < 0 or indices.max() >= self.n):
                raise ValueError("LOCAL_BLOCKS_MUST_BE_DISJOINT")
            self.register_buffer("D_indices", torch.tensor(indices, dtype=torch.int64, device=device))
            self.register_buffer("D_blocks", torch.tensor(blocks, dtype=torch.complex128, device=device))
        with self.book.span("torch_background_LU", LU_factorizations=1,
                            background_LU_factorizations=1):
            try:
                lu, pivots = torch.linalg.lu_factor(lb)
            except RuntimeError as exc:
                raise NumericalFailure("BACKGROUND_LU_FAILED") from exc
            self.register_buffer("Lb_lu", lu)
            self.register_buffer("Lb_pivots", pivots)
        if fixed_step is None:
            fixed_step = self.background_step()
        if not np.isfinite(fixed_step) or fixed_step <= 0:
            raise ValueError("POSITIVE_FROZEN_BACKGROUND_STEP_REQUIRED")
        self.register_buffer("step", torch.tensor(float(fixed_step), dtype=torch.float64, device=device))
        self._observable_ready = False

    @classmethod
    def from_npz(cls, path, *, book=None, config=None, device="cpu"):
        with np.load(Path(path), allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        descriptor = json.loads(str(arrays.pop("descriptor", "{}")))
        volume = float(arrays.pop("volume"))
        grid = int(arrays.pop("grid"))
        step = float(arrays.pop("fixed_step")) if "fixed_step" in arrays else (float(arrays.pop("step")) if "step" in arrays else None)
        return cls(arrays, volume=volume, grid=grid, descriptor=descriptor,
                   book=book, config=config, device=device, fixed_step=step,
                   require_provenance=True)

    @classmethod
    def from_cache(cls, cache, *, rank=16, seed=241008, local=False,
                   oversampling=8, book=None, config=None, device="cpu"):
        ug, vg, d = cache.prepare_lowrank(rank, local, seed, oversampling)
        p = cache.physics
        arrays = {"chi_b": cache.chi_b, "eb": cache.eb, "yb": cache.yb,
                  "Db": cache.Db, "G": p.G, "S": p.S, "incident": p.incident,
                  "Ug": ug, "Vg": vg}
        if d.blocks:
            if len({len(ind) for ind in d.indices}) != 1:
                raise ValueError("TORCH_LOCAL_BLOCKS_REQUIRE_EQUAL_BLOCK_SIZE")
            arrays.update(D_indices=np.stack(d.indices), D_blocks=np.stack(d.blocks))
        return cls(arrays, volume=p.volume, grid=p.grid, descriptor=cache.descriptor,
                   book=cache.book if book is None else book,
                   config={**cache.config, **dict(config or {})}, device=device)

    def bind_book(self, book):
        self.book = book
        return self

    def _reverse_counter(self, value, role, rhs):
        # Hooks count reverse-operation invocations, not a fabricated LU count.
        # Their actual CPU/GPU time is included in the enclosing backward span.
        if value.requires_grad:
            def hook(gradient):
                self.book.check()
                self.book.counts["torch_autograd_reverse_" + role + "_rhs"] += rhs
                return gradient
            value.register_hook(hook)
        return value

    def apply(self, matrix, value, role, *, adjoint=False):
        count = int(value.numel() // value.shape[-2])
        label = role + ("_adjoint" if adjoint else "")
        with self.book.span("torch_" + label, **{label + "_actions": count}):
            out = (matrix.mH if adjoint else matrix) @ value
        return self._reverse_counter(out, label, count)

    def green(self, value, *, adjoint=False):
        return self.apply(self.G, value, "G", adjoint=adjoint)

    def observe(self, value, *, adjoint=False):
        return self.apply(self.S, value, "S", adjoint=adjoint)

    def apply_db(self, value, *, adjoint=False):
        return self.apply(self.Db, value, "Db", adjoint=adjoint)

    def solve_background(self, value, *, adjoint=False):
        rhs = int(value.numel() // self.n)
        role = "background_adjoint" if adjoint else "background_forward"
        with self.book.span("torch_" + role + "_solve", linear_solve_rhs=rhs,
                            **{role + "_RHS": rhs}):
            out = torch.linalg.lu_solve(self.Lb_lu, self.Lb_pivots, value, adjoint=adjoint)
            if not torch.isfinite(out).all():
                raise NumericalFailure("NONFINITE_BACKGROUND_SOLVE")
        return self._reverse_counter(out, role, rhs)

    def background_step(self):
        with self.book.span("torch_background_spectral_step", background_material_directions=2 * self.n,
                            background_step_SVDs=1):
            b = torch.einsum("qn,ns->sqn", self.Db, self.eb) / self.sqrt_volume
            re = torch.cat((b.real, b.imag), dim=1).reshape(2 * self.q * self.s, self.n)
            im = torch.cat((-b.imag, b.real), dim=1).reshape(2 * self.q * self.s, self.n)
            j = torch.cat((re, im), dim=1)
            top = torch.linalg.svdvals(j)[0].square()
            if float(top) <= 0:
                raise NumericalFailure("ZERO_BACKGROUND_JACOBIAN")
            return float(.8 * self.yb_norm2 / top)

    def project(self, mass):
        chi = mass_unpack(mass, self.volume)
        real = chi.real.clamp(float(self.config.get("real_lower", -.5)),
                              float(self.config.get("real_upper", 4.)))
        imag = chi.imag.clamp(float(self.config.get("imag_lower", 0.)),
                              float(self.config.get("imag_upper", .5)))
        return mass_pack(torch.complex(real, imag), self.volume)

    def common_initial(self, observation):
        observation, single = _batched(observation, 3)
        with self.book.span("torch_common_initial", common_initial_images=len(observation)):
            load = (self.yb - observation) / self.yb_norm2
            z = self.apply_db(load, adjoint=True)
            gradient = mass_pack((self.eb.conj() * z).sum(-1), 1.) / self.sqrt_volume
            mass = mass_pack(self.chi_b, self.volume)[None] - self.step * gradient
            mass = self.project(mass)
        return mass[0] if single else mass

    def prepare_observable(self, retained=15):
        retained = min(int(retained), self.q, self.n)
        if retained < 1:
            raise ValueError("POSITIVE_OBSERVABLE_RANK_REQUIRED")
        if self._observable_ready:
            if self.observable_rank != retained:
                raise ValueError("OBSERVABLE_RANK_ALREADY_FROZEN")
            return
        with self.book.span("torch_SOM_receiver_SVD", receiver_SVDs=1):
            u, singular, vh = torch.linalg.svd(self.S, full_matrices=False)
            if float(singular[retained - 1] / singular[0]) < float(self.config.get("orthogonal_rank_rtol", 1e-10)):
                raise NumericalFailure("UNSAFE_DETERMINISTIC_CURRENT_RANK")
            self.register_buffer("observable_left", u[:, :retained])
            self.register_buffer("observable_right", vh[:retained].mH)
            self.register_buffer("observable_singular", singular[:retained])
        self.observable_rank, self._observable_ready = retained, True

    def som_initial(self, observation, retained=15):
        """Truncated receiver-SVD current and independent analytic BP material."""
        self.prepare_observable(retained)
        observation, single = _batched(observation, 3)
        with self.book.span("torch_SOM_initial", deterministic_current_source_rhs=len(observation) * self.s):
            current = self.observable_right @ ((self.observable_left.mH @ observation)
                                               / self.observable_singular[:, None])
            back = self.observe(observation, adjoint=True)
            projected = self.observe(back)
            denominator = projected.abs().square().sum(-2, keepdim=True)
            if bool((denominator <= 0).any()):
                raise NumericalFailure("ZERO_BP_SOURCE_DENOMINATOR")
            gamma = (projected.conj() * observation).sum(-2, keepdim=True) / denominator
            bp_current = back * gamma
            field = self.incident + self.green(bp_current)
            mass = self.project(self.material_ratio(bp_current, field))
        return (current[0], mass[0]) if single else (current, mass)

    def material_ratio(self, current, field):
        numerator = (field.conj() * current).sum(-1)
        denominator = field.abs().square().sum(-1)
        floor = float(self.config.get("pole_relative_floor", 1e-10))
        threshold = floor * denominator.detach().amax(-1, keepdim=True)
        if bool((denominator <= threshold).any()) or not torch.isfinite(denominator).all():
            raise NumericalFailure("UNSAFE_SHARED_MATERIAL_RATIO")
        with self.book.span("torch_source_shared_material_ratio", material_ratio_images=int(current.numel() // (self.n * self.s))):
            return mass_pack(numerator / denominator, self.volume)

    def canonical_current(self, state):
        if state.current is None:
            state.current = self.cb + self.solve_background(state.w)
        return state.current

    def consistency(self, state):
        current = self.canonical_current(state)
        field = self.incident + self.green(current)
        state.canonical_field = field
        residual = current - state.chi[..., :, None] * field
        with self.book.span("torch_material_consistency", material_consistency_images=int(current.numel() // (self.n * self.s))):
            norm = torch.linalg.vector_norm(current, dim=(-2, -1))
            ratio = torch.linalg.vector_norm(residual, dim=(-2, -1)) / norm.clamp_min(torch.finfo(torch.float64).tiny)
            discrepancy = torch.linalg.vector_norm(field - state.field, dim=(-2, -1)) / torch.linalg.vector_norm(field, dim=(-2, -1)).clamp_min(torch.finfo(torch.float64).tiny)
        return {"Rmat": ratio, "field_closure_discrepancy": discrepancy,
                "material_residual": residual}


class TorchRelativeModel(nn.Module):
    """L or package-Woodbury R, including strict real-mass JVP/VJP."""
    def __init__(self, reference: TorchReference, kind="R"):
        super().__init__()
        self.reference = reference
        self.kind = str(kind).upper()
        if self.kind not in {"L", "R"}:
            raise ValueError("TORCH_RELATIVE_KIND_MUST_BE_L_OR_R")
        self.rank = reference.rank if self.kind == "R" else 0

    @property
    def book(self):
        return self.reference.book

    def _guard(self, matrix, name):
        with self.book.span("torch_core_guard_" + name, core_guards=int(np.prod(matrix.shape[:-2]))):
            singular = torch.linalg.svdvals(matrix.detach())
            ratio = singular[..., -1] / singular[..., 0]
            condition = ratio.reciprocal()
            if (not torch.isfinite(condition).all()
                    or bool((ratio < float(self.reference.config.get("core_relative_sigma_min", 1e-10))).any())
                    or bool((condition > float(self.reference.config.get("core_condition_cap", 1e10))).any())):
                raise NumericalFailure("UNSAFE_TORCH_CORE:" + name)
            return {"relative_sigma_min": ratio, "condition": condition}

    def _solve(self, matrix, rhs, role, *, adjoint=False):
        systems = int(np.prod(matrix.shape[:-2]))
        directions = int(rhs.shape[-1])
        with self.book.span("torch_" + role + ("_adjoint_solve" if adjoint else "_solve"),
                            LU_factorizations=systems, linear_solve_rhs=systems * directions,
                            **{role + "_LU_factorizations": systems}):
            try:
                result = torch.linalg.solve(matrix.mH if adjoint else matrix, rhs)
            except RuntimeError as exc:
                raise NumericalFailure("TORCH_SOLVE_FAILED:" + role) from exc
            if not torch.isfinite(result).all():
                raise NumericalFailure("NONFINITE_TORCH_SOLVE:" + role)
        return self.reference._reverse_counter(result, role, systems * directions)

    def _solve_P(self, rhs, state, *, adjoint=False):
        ref = self.reference
        if not ref.D_indices.numel():
            return rhs
        local = rhs[:, ref.D_indices, :]
        solved = self._solve(state.extra["P"], local, "local_P", adjoint=adjoint)
        return rhs.index_copy(1, ref.D_indices.flatten(), solved.flatten(1, 2))

    def feedback_apply(self, value, *, adjoint=False):
        ref = self.reference
        if self.kind == "L":
            return torch.zeros_like(value)
        batch, single = _batched(value, 3)
        out = torch.zeros_like(batch)
        if ref.D_indices.numel():
            local = batch[:, ref.D_indices, :]
            blocks = ref.D_blocks.mH if adjoint else ref.D_blocks
            local = blocks @ local
            out = out.index_copy(1, ref.D_indices.flatten(), local.flatten(1, 2))
        if self.rank:
            with self.book.span("torch_feedback_apply", lowrank_feedback_rhs=len(batch) * batch.shape[-1]):
                out = out + (ref.Vg @ (ref.Ug.mH @ batch) if adjoint
                             else ref.Ug @ (ref.Vg.mH @ batch))
        return out[0] if single else out

    def solve_A(self, rhs, state, *, adjoint=False):
        rhs, single = _batched(rhs, 3)
        if state.extra.get("model_id") != id(self):
            raise ValueError("TORCH_STATE_MODEL_MISMATCH")
        if self.kind == "L":
            return rhs[0] if single else rhs
        ref = self.reference
        t = self._solve_P(rhs, state, adjoint=adjoint)
        if self.rank:
            if adjoint:
                gamma = self._solve(state.extra["K"],
                                    ref.Ug.mH @ (state.extra["h_batch"].conj()[..., None] * t),
                                    "feedback_K", adjoint=True)
                out = t + self._solve_P(ref.Vg @ gamma, state, adjoint=True)
            else:
                gamma = self._solve(state.extra["K"], ref.Vg.mH @ t, "feedback_K")
                out = t + state.extra["CU"] @ gamma
        else:
            out = t
        return out[0] if single else out

    def forward(self, mass, *, recover_current=True):
        ref = self.reference
        mass, single = _batched(mass, 2)
        if mass.shape[-1] != 2 * ref.n or mass.dtype != torch.float64:
            raise ValueError("FLOAT64_MATERIAL_MASS_REQUIRED")
        chi = mass_unpack(mass, ref.volume)
        if not torch.isfinite(chi).all():
            raise NumericalFailure("NONFINITE_TORCH_MATERIAL")
        tolerance = float(ref.config.get("feasibility_tolerance", 1e-10))
        if (bool((chi.real < float(ref.config.get("real_lower", -.5)) - tolerance).any())
                or bool((chi.real > float(ref.config.get("real_upper", 4.)) + tolerance).any())
                or bool((chi.imag < float(ref.config.get("imag_lower", 0.)) - tolerance).any())
                or bool((chi.imag > float(ref.config.get("imag_upper", .5)) + tolerance).any())):
            raise NumericalFailure("TORCH_MATERIAL_DOMAIN_VIOLATION")
        h = chi - ref.chi_b
        state = TorchState(mass, chi, h, None, None, None,
                           extra={"model_id": id(self), "h_batch": h, "single": single})
        with self.book.span("torch_relative_" + self.kind + "_forward",
                            relative_forward_calls=len(mass), relative_forward_source_rhs=len(mass) * ref.s):
            if self.kind == "R":
                if ref.D_indices.numel():
                    p = torch.eye(ref.D_indices.shape[-1], dtype=torch.complex128, device=mass.device) - h[:, ref.D_indices, None] * ref.D_blocks
                    state.extra["P"] = p
                    state.margins["P"] = self._guard(p, "P")
                if self.rank:
                    cu = self._solve_P(h[..., None] * ref.Ug, state)
                    k = torch.eye(self.rank, dtype=torch.complex128, device=mass.device) - ref.Vg.mH @ cu
                    state.extra.update(CU=cu, K=k)
                    state.margins["K"] = self._guard(k, "K")
            w = self.solve_A(h[..., None] * ref.eb, state)
            field = ref.eb + self.feedback_apply(w)
            data = ref.yb + ref.apply_db(w)
            state.w, state.field, state.data = w, field, data
            if recover_current:
                ref.canonical_current(state)
            if single:
                for name in ("mass", "chi", "h", "w", "field", "data", "current"):
                    value = getattr(state, name)
                    if value is not None:
                        setattr(state, name, value[0])
        return state

    def _match(self, mass, state):
        if state is None:
            return self.forward(mass, recover_current=False)
        if state.extra.get("model_id") != id(self) or state.mass.shape != mass.shape:
            raise ValueError("TORCH_STATE_MISMATCH")
        if not torch.equal(state.mass.detach(), mass.detach()):
            raise ValueError("TORCH_STATE_MATERIAL_MISMATCH")
        return state

    def jvp(self, mass, direction, state=None):
        state = self._match(mass, state)
        field, single = _batched(state.field, 3)
        direction, _ = _batched(direction, 2)
        if direction.shape != (len(field), 2 * self.reference.n):
            raise ValueError("MATCHED_REAL_MASS_DIRECTION_REQUIRED")
        with self.book.span("torch_explicit_jvp", material_tangent_directions=len(field),
                            material_tangent_source_rhs=len(field) * self.reference.s):
            dchi = mass_unpack(direction, self.reference.volume)
            dw = self.solve_A(dchi[..., None] * field, state)
            data = self.reference.apply_db(dw)
            result = pack_data(data)
        return result[0] if single else result

    def adjoint(self, mass, real_load, state=None):
        state = self._match(mass, state)
        load = unpack_data(real_load, self.reference.q, self.reference.s)
        return self.vjp_complex(mass, load, state)

    def vjp_complex(self, mass, load, state=None):
        state = self._match(mass, state)
        field, single = _batched(state.field, 3)
        load, _ = _batched(load, 3)
        if load.shape != (len(field), self.reference.q, self.reference.s):
            raise ValueError("MATCHED_COMPLEX_DATA_LOAD_REQUIRED")
        with self.book.span("torch_explicit_adjoint", data_adjoint_directions=len(field),
                            data_adjoint_source_rhs=len(field) * self.reference.s):
            z = self.solve_A(self.reference.apply_db(load, adjoint=True), state, adjoint=True)
            pullback = (field.conj() * z).sum(-1) / self.reference.sqrt_volume
            result = mass_pack(pullback, 1.)
        return result[0] if single else result

    def data_gradient(self, mass, observation, state=None):
        state = self._match(mass, state)
        return self.vjp_complex(mass, (state.data - observation) / self.reference.yb_norm2, state)
