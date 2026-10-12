"""A32 linear charts over the unchanged TRAIN-only A30 representation freeze."""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping

import numpy as np

from . import frozen_representations as frozen


AMPLITUDE = 1.2
REPRESENTATIONS = {"pca64": ("single_chart", 64),
                   "sparse32": ("sparse", 32), "sparse64": ("sparse", 64)}


def _readonly(value: Any) -> np.ndarray:
    array = np.asarray(value)
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)


def _real(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.complex128)
    if array.shape != (frozen.PIXELS,) or not np.isfinite(array).all():
        raise ValueError("A32_FINITE_COMPLEX576_MATERIAL_REQUIRED")
    return np.r_[array.real, array.imag]


def prepare_charts(train_deltas: Any, *, cell_weights: Any = None) -> frozen.TrainModel:
    """Reuse the whole frozen preparation, including its unused geometry work."""
    return frozen.prepare_train(train_deltas, cell_weights=cell_weights)


@dataclass(frozen=True)
class FrozenChart:
    representation: str
    T: np.ndarray
    material_metric: np.ndarray
    support_ids: np.ndarray
    background: np.ndarray
    real_weights: np.ndarray
    train_mass: np.ndarray
    train_coefficients: np.ndarray
    initial_coefficients: np.ndarray
    frozen_feasible_initial: np.ndarray
    n3: np.ndarray
    amplitude: float
    metadata: Mapping[str, Any]

    @property
    def dimension(self) -> int:
        return self.T.shape[1]

    @property
    def mass_basis(self) -> np.ndarray:
        return _readonly(np.sqrt(self.real_weights)[:, None] * self.T)

    def decode(self, coefficients: Any) -> np.ndarray:
        """Decode actual material without pixel clipping or support changes."""
        value = np.asarray(coefficients)
        if (np.iscomplexobj(value) or value.shape != (self.dimension,)
                or not np.isfinite(value).all()):
            raise ValueError("A32_FINITE_REAL_CHART_COEFFICIENTS_REQUIRED")
        delta = self.T @ np.asarray(value, dtype=float)
        return self.background + delta[:frozen.PIXELS] + 1j * delta[frozen.PIXELS:]

    def projection_problem(self) -> dict[str, np.ndarray]:
        """Coefficient QP for the weighted projection of the saved N3 material."""
        background_real = _real(self.background)
        lower = np.r_[np.full(frozen.PIXELS, frozen.REAL_BOUNDS[0]),
                      np.full(frozen.PIXELS, frozen.IMAG_BOUNDS[0])] - background_real
        upper = np.r_[np.full(frozen.PIXELS, frozen.REAL_BOUNDS[1]),
                      np.full(frozen.PIXELS, frozen.IMAG_BOUNDS[1])] - background_real
        return {"H": self.material_metric,
                "h": _readonly(self.T.T @ (self.real_weights * _real(self.n3 - self.background))),
                "A": self.T, "lower": _readonly(lower), "upper": _readonly(upper)}

    def feasible_projection_initial(self, *, solver: Callable[..., dict[str, Any]] | None = None,
                                    config: Mapping[str, Any] | None = None,
                                    check: Callable[[], Any] | None = None) -> dict[str, Any]:
        """Use the parent coefficient QP, imported lazily; propagate its receipt."""
        if solver is None:
            from .qp import solve_qp
            solver = solve_qp
        return solver(**self.projection_problem(), initial=self.frozen_feasible_initial,
                      config=config, check=check)


def freeze_chart(train_model: frozen.TrainModel, representation: str, n3: Any,
                 background: Any, *, amplitude: float = AMPLITUDE) -> FrozenChart:
    """Select the first eligible frozen start; no target or best-start input exists.

    B is frozen using normalized (N3-chi_b)/1.2.  The physical basis is T=1.2B;
    TRAIN coefficients project the *actual* amplitude-1.2 TRAIN contrasts into T.
    """
    if representation not in REPRESENTATIONS or amplitude != AMPLITUDE:
        raise ValueError("A32_FROZEN_REPRESENTATION_OR_AMPLITUDE_REQUIRED")
    if isinstance(background, Mapping):
        background = background["chi_b"]
    background = np.asarray(background, dtype=np.complex128)
    n3 = np.asarray(n3, dtype=np.complex128)
    _real(background)
    _real(n3)
    method, dimension = REPRESENTATIONS[representation]
    normalized_n3 = (n3 - background) / amplitude
    choices = frozen.freeze_case(train_model, method, dimension, normalized_n3, background,
                                 {"amplitude_scale": amplitude})
    selected = next((start for start in choices["starts"] if start["eligible"]), None)
    if selected is None:
        reasons = [reason for start in choices["starts"] for reason in start["reasons"]]
        raise ValueError("A32_NO_ELIGIBLE_FROZEN_START:" + ",".join(sorted(set(reasons))))
    T = amplitude * np.asarray(selected["basis"], dtype=float)
    weights = np.asarray(train_model.real_weights, dtype=float)
    if T.shape != (2 * frozen.PIXELS, dimension) or weights.shape != (2 * frozen.PIXELS,):
        raise ValueError("A32_FROZEN_CHART_LAYOUT_MISMATCH")
    material_metric = T.T @ (weights[:, None] * T)
    actual_train = amplitude * np.hstack((train_model.train_deltas.real, train_model.train_deltas.imag))
    train_coefficients = np.linalg.solve(material_metric, ((actual_train * weights[None, :]) @ T).T).T
    initial = np.linalg.solve(material_metric, T.T @ (weights * _real(n3 - background)))
    support = selected.get("support", np.arange(dimension, dtype=int))
    metadata = MappingProxyType({
        "scope": "TRUTH_FREE_TRAIN_ONLY_FIXED_CHART", "selection_rule": "FIRST_ELIGIBLE_FROZEN_START",
        "start_id": int(selected["id"]), "source": selected["source"],
        "source_train_index": selected["source_train_index"], "amplitude": amplitude,
        "support_kind": "REAL_HAAR_COORDINATES" if method == "sparse" else "TRAIN_SVD_COORDINATES",
        "train_coefficient_scope": "WEIGHTED_PROJECTION_OF_ACTUAL_AMPLITUDE_1.2_TRAIN",
        "n3_scope": "INITIALIZATION_AND_CONDITIONAL_SUPPORT_ONLY", "target_access": False,
    })
    return FrozenChart(representation, _readonly(T), _readonly(material_metric), _readonly(support),
                       _readonly(background), _readonly(weights),
                       _readonly(actual_train * np.sqrt(weights)[None, :]),
                       _readonly(train_coefficients), _readonly(initial), _readonly(selected["params"]),
                       _readonly(n3), amplitude, metadata)
