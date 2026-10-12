"""Frozen TRAIN-only A30 charts and explicitly OFFLINE capacity fits.

No function in this module reads files, runs physics, trains a network, or computes
the research gate.  ``freeze_case`` has no target argument.  ``fit_capacity`` is
the only target-taking entry point and is an OFFLINE material oracle diagnostic.
All budgets are real coordinates, including independent imaginary contrasts.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter, process_time
from typing import Any, Mapping

import numpy as np
from scipy import ndimage
from scipy.optimize import minimize
from scipy.special import softmax


GRID = 24
PIXELS = GRID * GRID
BUDGETS = (16, 32, 64)
METHODS = ("atlas", "level_set", "sparse", "single_chart")
REAL_BOUNDS = (-0.5, 4.0)
IMAG_BOUNDS = (0.0, 0.5)
SMOOTHING_WIDTH = 0.5
_FOUR = ndimage.generate_binary_structure(2, 1)
_EIGHT = ndimage.generate_binary_structure(2, 2)


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


def _real(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.complex128).reshape(PIXELS)
    return np.concatenate((value.real, value.imag))


def _complex(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=float).reshape(2 * PIXELS)
    return value[:PIXELS] + 1j * value[PIXELS:]


def _field(value: Any, name: str, scalar: bool = False) -> np.ndarray:
    result = np.asarray(value, dtype=np.complex128)
    if scalar and result.size == 1:
        result = np.full(PIXELS, result.item(), dtype=np.complex128)
    if result.size != PIXELS or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite complex 576-vector")
    return result.reshape(PIXELS)


def box_feasibility(image: Any) -> dict[str, Any]:
    """Exact comparisons: the physical box never has a numerical tolerance."""
    image = _field(image, "image")
    lo_r, hi_r = REAL_BOUNDS
    lo_i, hi_i = IMAG_BOUNDS
    violation = max(0.0, float(lo_r - image.real.min()),
                    float(image.real.max() - hi_r),
                    float(lo_i - image.imag.min()),
                    float(image.imag.max() - hi_i))
    return {"feasible": bool(np.all(image.real >= lo_r)
                             and np.all(image.real <= hi_r)
                             and np.all(image.imag >= lo_i)
                             and np.all(image.imag <= hi_i)),
            "strict_interior": bool(np.all(image.real > lo_r)
                                    and np.all(image.real < hi_r)
                                    and np.all(image.imag > lo_i)
                                    and np.all(image.imag < hi_i)),
            "real_min": float(image.real.min()), "real_max": float(image.real.max()),
            "imag_min": float(image.imag.min()), "imag_max": float(image.imag.max()),
            "max_violation": violation, "feasibility_tolerance": 0.0}


def digital_topology(mask: Any) -> dict[str, int]:
    """4-connected foreground / 8-connected background, with exterior padding."""
    mask = np.asarray(mask, dtype=bool).reshape(GRID, GRID)
    components = int(ndimage.label(mask, _FOUR)[1])
    outside = np.pad(~mask, 1, constant_values=True)
    holes = int(ndimage.label(outside, _EIGHT)[1]) - 1
    return {"components": components, "holes": holes,
            "euler": components - holes}


def _partition_signature(labels: np.ndarray, phases: int) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=int).reshape(GRID, GRID)
    adjacency = set()
    for left, right in ((labels[:-1], labels[1:]),
                        (labels[:, :-1], labels[:, 1:])):
        for a, b in zip(left.ravel(), right.ravel()):
            if a > 0 and b > 0 and a != b:
                adjacency.add(tuple(sorted((int(a), int(b)))))
    return {"support": digital_topology(labels > 0),
            "phases": [digital_topology(labels == r + 1) for r in range(phases)],
            "phase_adjacency": [list(pair) for pair in sorted(adjacency)]}


def _soft_partition(fields: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probabilities = softmax(np.vstack((np.zeros(PIXELS), fields / SMOOTHING_WIDTH)),
                            axis=0)
    phase_weights = probabilities[1:].T
    occupied = phase_weights.sum(axis=1) >= 0.5
    labels = np.where(occupied, np.argmax(phase_weights, axis=1) + 1, 0)
    return phase_weights, labels.reshape(GRID, GRID)


def _dct(size: int) -> np.ndarray:
    x = np.arange(size) + 0.5
    matrix = np.cos(np.pi * np.arange(size)[:, None] * x[None, :] / size)
    matrix[0] *= np.sqrt(1 / size)
    matrix[1:] *= np.sqrt(2 / size)
    return matrix


def haar24_transform() -> tuple[np.ndarray, tuple[dict[str, Any], ...]]:
    """576 x 576 orthogonal analysis matrix; no 32-grid padding.

    2-D Haar approximations descend 24 -> 12 -> 6 -> 3.  A fixed separable
    orthonormal 3x3 DCT completes the nine remaining coarse coordinates.
    """
    current = np.eye(PIXELS).reshape(GRID, GRID, PIXELS)
    stages = []
    size = GRID
    while size > 3:
        a, b = current[::2, ::2], current[::2, 1::2]
        c, d = current[1::2, ::2], current[1::2, 1::2]
        details = ((a - b + c - d) / 2,
                   (a + b - c - d) / 2,
                   (a - b - c + d) / 2)
        stages.append((size, details))
        current = (a + b + c + d) / 2
        size //= 2
    coarse = np.einsum("ui,ijn,vj->uvn", _dct(3), current, _dct(3))
    rows = [coarse.reshape(9, PIXELS)]
    metadata = [{"kind": "coarse_dct", "size": 3, "row": r, "col": c}
                for r in range(3) for c in range(3)]
    for fine, details in reversed(stages):
        for name, block in zip(("horizontal", "vertical", "diagonal"), details):
            rows.append(block.reshape(-1, PIXELS))
            metadata.extend({"kind": name, "size": fine,
                             "row": r, "col": c}
                            for r in range(fine // 2) for c in range(fine // 2))
    return _readonly(np.vstack(rows)), tuple(metadata)


def _rank(matrix: np.ndarray) -> int:
    values = np.linalg.svd(matrix, compute_uv=False)
    if not values.size or values[0] == 0:
        return 0
    tolerance = (32 * np.finfo(float).eps * max(matrix.shape) * values[0])
    return int(np.count_nonzero(values > tolerance))


def _allocate(total: int, capacities: list[int], weights: list[int]) -> list[int] | None:
    if total < len(capacities) or sum(capacities) < total or min(capacities) < 1:
        return None
    counts = [1] * len(capacities)
    desired = np.asarray(weights, float) / sum(weights) * total
    for _ in range(total - len(capacities)):
        available = [r for r, capacity in enumerate(capacities) if counts[r] < capacity]
        selected = max(available, key=lambda r: (desired[r] - counts[r], -r))
        counts[selected] += 1
    return counts


def _normal_basis(mask: np.ndarray, count: int) -> np.ndarray:
    boundary = mask & ~ndimage.binary_erosion(mask, _FOUR, border_value=0)
    points = np.argwhere(boundary)
    if count > len(points):
        raise ValueError("not enough distinct TRAIN boundary controls")
    selected = [0]
    distance = np.sum((points - points[0]) ** 2, axis=1)
    for _ in range(count - 1):
        index = int(np.argmax(distance))
        selected.append(index)
        distance = np.minimum(distance, np.sum((points - points[index]) ** 2, axis=1))
    centers = points[selected].astype(float)
    indices = ndimage.distance_transform_edt(~boundary, return_distances=False,
                                             return_indices=True)
    projected = np.stack(indices, axis=-1).reshape(PIXELS, 2)
    distance2 = np.sum((projected[:, None] - centers[None, :]) ** 2, axis=2)
    if count > 1:
        center_dist = np.sqrt(np.sum((centers[:, None] - centers[None, :]) ** 2, axis=2))
        np.fill_diagonal(center_dist, np.inf)
        width = max(1.25, 1.25 * float(np.median(center_dist.min(axis=1))))
    else:
        width = 1.25
    # The log-softmax avoids underflow and preserves a positive PoU everywhere.
    basis = softmax(-distance2 / (2 * width * width), axis=1)
    return _readonly(basis)


def _level_basis(count: int) -> np.ndarray:
    frequencies = sorted(((r, c) for r in range(GRID) for c in range(GRID)),
                         key=lambda pair: (pair[0] ** 2 + pair[1] ** 2, *pair))
    x = np.arange(GRID) + 0.5
    columns = []
    for r, c in frequencies[:count]:
        field = np.cos(np.pi * r * x[:, None] / GRID) * np.cos(np.pi * c * x[None, :] / GRID)
        columns.append(field.ravel() / np.max(np.abs(field)))
    return _readonly(np.column_stack(columns))


@dataclass(frozen=True)
class GeometryChart:
    method: str
    budget: int
    basis: tuple[np.ndarray, ...]
    geometry_counts: tuple[int, ...]
    intrinsic_rank: int
    eligible: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class TrainTemplate:
    index: int
    masks: np.ndarray
    fields: np.ndarray
    reference_contrasts: np.ndarray
    topology: dict[str, Any]
    soft_topology: dict[str, Any]
    phase_quantization_mse: float
    smoothing_mse: float
    charts: Mapping[tuple[str, int], GeometryChart]


@dataclass(frozen=True)
class TrainModel:
    train_deltas: np.ndarray
    cell_weights: np.ndarray
    real_weights: np.ndarray
    haar_analysis: np.ndarray
    haar_metadata: tuple[dict[str, Any], ...]
    train_haar_coefficients: np.ndarray
    single_basis: np.ndarray
    singular_values: np.ndarray
    single_rank: int
    templates: tuple[TrainTemplate, ...]
    metadata: dict[str, Any]


def _geometry_decode(template: TrainTemplate, chart: GeometryChart,
                     params: np.ndarray, jacobian: bool = False) -> tuple[np.ndarray, Any, Any]:
    phases = len(template.masks)
    geometry = sum(chart.geometry_counts)
    fields = template.fields.copy()
    position = 0
    for r, basis in enumerate(chart.basis):
        count = chart.geometry_counts[r]
        fields[r] += basis @ params[position:position + count]
        position += count
    weights, labels = _soft_partition(fields)
    contrasts = params[geometry:geometry + phases] + 1j * params[geometry + phases:]
    delta = weights @ contrasts
    if not jacobian:
        return delta, labels, weights
    derivatives = np.empty((PIXELS, chart.budget), dtype=np.complex128)
    position = 0
    for r, basis in enumerate(chart.basis):
        count = chart.geometry_counts[r]
        derivatives[:, position:position + count] = (
            weights[:, r] * (contrasts[r] - delta) / SMOOTHING_WIDTH)[:, None] * basis
        position += count
    derivatives[:, geometry:geometry + phases] = weights
    derivatives[:, geometry + phases:] = 1j * weights
    return delta, labels, np.vstack((derivatives.real, derivatives.imag))


def prepare_train(train_deltas: Any, *, cell_weights: Any = None) -> TrainModel:
    """Freeze charts from exactly 64 independent TRAIN parents, never backgrounds.

    The caller owns provenance; this function accepts arrays rather than data
    paths, split identifiers, targets, case labels or family labels.
    """
    train = np.asarray(train_deltas, dtype=np.complex128)
    if train.shape != (64, PIXELS) or not np.isfinite(train).all():
        raise ValueError("TRAIN must contain exactly 64 finite independent-parent 576-vectors")
    weights = np.full(PIXELS, 1 / PIXELS) if cell_weights is None else np.asarray(cell_weights, float)
    if weights.shape != (PIXELS,) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("cell_weights must be a fixed positive 576-vector")
    real_weights = np.tile(weights, 2)
    train_real = np.hstack((train.real, train.imag))
    mass_train = train_real * np.sqrt(real_weights)[None, :]
    # Explicitly zero inactive TRAIN coordinates before and after SVD.  Their
    # contribution is exactly zero; no numerical ghost pixels are introduced.
    active = np.any(mass_train != 0, axis=0)
    _, singular, vt = np.linalg.svd(mass_train[:, active], full_matrices=False)
    weighted_basis = np.zeros((2 * PIXELS, vt.shape[0]))
    weighted_basis[active] = vt.T
    for col in range(weighted_basis.shape[1]):
        pivot = int(np.argmax(np.abs(weighted_basis[:, col])))
        if weighted_basis[pivot, col] < 0:
            weighted_basis[:, col] *= -1
    single_basis = weighted_basis / np.sqrt(real_weights)[:, None]
    single_rank = _rank(mass_train)
    haar, haar_metadata = haar24_transform()
    train_coeff = np.hstack(((train.real * np.sqrt(weights)) @ haar.T,
                            (train.imag * np.sqrt(weights)) @ haar.T))
    templates = []
    for index, delta in enumerate(train):
        magnitude = np.abs(delta).reshape(GRID, GRID)
        if magnitude.max() == 0:
            raise ValueError(f"TRAIN parent {index} has empty zero-amplitude occupancy")
        occupied = magnitude >= 0.25 * magnitude.max()
        cuts = np.quantile(magnitude[occupied], [1 / 3, 2 / 3])
        original_bands = np.searchsorted(cuts, magnitude, side="right")
        masks = np.stack([occupied & (original_bands == r) for r in range(3)
                          if np.any(occupied & (original_bands == r))])
        phases = len(masks)
        labels = np.zeros((GRID, GRID), dtype=int)
        fields = []
        boundary_sizes = []
        contrasts = []
        for r, mask in enumerate(masks):
            labels[mask] = r + 1
            signed = ndimage.distance_transform_edt(mask) - ndimage.distance_transform_edt(~mask)
            signed -= np.where(mask, 0.5, -0.5)
            fields.append(signed.ravel())
            boundary_sizes.append(int(np.count_nonzero(
                mask & ~ndimage.binary_erosion(mask, _FOUR, border_value=0))))
            contrasts.append(np.mean(delta[mask.ravel()]))
        fields = _readonly(np.stack(fields))
        contrasts = _readonly(np.asarray(contrasts))
        topology = _partition_signature(labels, phases)
        soft_weights, soft_labels = _soft_partition(fields)
        soft_topology = _partition_signature(soft_labels, phases)
        hard_delta = np.sum(masks.reshape(phases, PIXELS).T * contrasts[None, :], axis=1)
        smooth_delta = soft_weights @ contrasts
        template = TrainTemplate(index, _readonly(masks), fields, contrasts,
                                 topology, soft_topology,
                                 float(np.sum(weights * np.abs(delta - hard_delta) ** 2)),
                                 float(np.sum(weights * np.abs(hard_delta - smooth_delta) ** 2)), {})
        charts = {}
        for method in ("atlas", "level_set"):
            for budget in BUDGETS:
                geometry = budget - 2 * phases
                capacities = boundary_sizes if method == "atlas" else [PIXELS] * phases
                counts = _allocate(geometry, capacities, boundary_sizes)
                reasons = []
                basis = ()
                rank = 0
                if counts is None:
                    reasons.append("INSUFFICIENT_DISTINCT_TRAIN_GEOMETRY_CONTROLS")
                    counts = [0] * phases
                else:
                    basis = tuple(_normal_basis(mask, count) if method == "atlas"
                                  else _level_basis(count)
                                  for mask, count in zip(masks, counts))
                    provisional = GeometryChart(method, budget, basis, tuple(counts), 0, False, ())
                    # Intrinsic dimension is checked at a generic independent
                    # complex-contrast point, not at a zero-contrast singularity.
                    generic = np.r_[np.zeros(geometry),
                                    0.7 + 0.3 * np.arange(phases),
                                    0.08 + 0.07 * np.arange(phases) ** 2]
                    _, _, tangent = _geometry_decode(template, provisional, generic, True)
                    rank = _rank(tangent * np.sqrt(real_weights)[:, None])
                    if rank != budget:
                        reasons.append("DEFICIENT_GENERIC_MATERIAL_TANGENT_RANK")
                if method == "atlas" and soft_topology != topology:
                    reasons.append("ZERO_GEOMETRY_SMOOTHING_CHANGES_DIGITAL_TOPOLOGY")
                charts[method, budget] = GeometryChart(method, budget, basis, tuple(counts),
                                                       rank, not reasons, tuple(reasons))
        object.__setattr__(template, "charts", charts)
        templates.append(template)
    metadata = {"scope": "TRAIN_ONLY", "parents": 64, "background_replication": False,
                "centering": "UNCENTERED", "single_numerical_rank": single_rank,
                "real_budgets": list(BUDGETS), "smoothing_width_pixels": SMOOTHING_WIDTH,
                "haar_levels": [24, 12, 6, 3], "coarse_completion": "orthonormal_3x3_DCT",
                "physical_bounds": {"real": list(REAL_BOUNDS), "imag": list(IMAG_BOUNDS)},
                "templates": [{"index": t.index, "phases": len(t.masks),
                               "topology": t.topology, "soft_topology": t.soft_topology,
                               "phase_quantization_mse": t.phase_quantization_mse,
                               "smoothing_mse": t.smoothing_mse,
                               "charts": [{"method": c.method, "budget": c.budget,
                                           "geometry_parameters": sum(c.geometry_counts),
                                           "real_contrast_parameters": 2 * len(t.masks),
                                           "intrinsic_rank": c.intrinsic_rank,
                                           "eligible": c.eligible, "reasons": list(c.reasons)}
                                          for c in t.charts.values()]} for t in templates]}
    return TrainModel(_readonly(train), _readonly(weights), _readonly(real_weights), haar,
                      haar_metadata, _readonly(train_coeff), _readonly(single_basis),
                      _readonly(singular), single_rank, tuple(templates), metadata)


def _configuration(config: Mapping[str, Any] | None) -> dict[str, Any]:
    result = dict(config or {})
    result.setdefault("amplitude_scale", 1.0)
    result.setdefault("max_starts", 8)
    result.setdefault("max_iterations", 100)
    result.setdefault("certificate_tolerance", 1e-8)
    if not (np.isfinite(result["amplitude_scale"]) and result["amplitude_scale"] > 0):
        raise ValueError("amplitude_scale must be finite and positive")
    if not (1 <= int(result["max_starts"]) <= 8 and
            1 <= int(result["max_iterations"]) <= 100):
        raise ValueError("hard limits: at most 8 starts and 100 iterations per start")
    if not (0 < result["certificate_tolerance"] <= 1e-8):
        raise ValueError("certificate_tolerance must be explicit and <=1e-8")
    return result


def _material_limits(background: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bg = _real(background)
    lower = np.r_[np.full(PIXELS, REAL_BOUNDS[0]), np.full(PIXELS, IMAG_BOUNDS[0])] - bg
    upper = np.r_[np.full(PIXELS, REAL_BOUNDS[1]), np.full(PIXELS, IMAG_BOUNDS[1])] - bg
    return lower, upper


def _ray_repair(params: np.ndarray, basis: np.ndarray,
                background: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Contract coefficients toward zero; preserve chart/support, never clip pixels."""
    params = np.asarray(params, float).copy()
    if box_feasibility(background + _complex(basis @ params))["feasible"]:
        return params, {"contracted": False, "factor": 1.0, "inward_ulps": 0}
    lower, upper = _material_limits(background)
    direction = basis @ params
    ratios = [1.0]
    positive, negative = direction > 0, direction < 0
    ratios.extend((upper[positive] / direction[positive]).tolist())
    ratios.extend((lower[negative] / direction[negative]).tolist())
    factor = max(0.0, min(ratios))
    ulps = 0
    while factor > 0 and ulps < 32:
        factor = np.nextafter(factor, 0.0)
        ulps += 1
        repaired = params * factor
        if box_feasibility(background + _complex(basis @ repaired))["feasible"]:
            return repaired, {"contracted": True, "factor": factor, "inward_ulps": ulps}
    return np.zeros_like(params), {"contracted": True, "factor": 0.0,
                                  "inward_ulps": ulps, "zero_ray_fallback": True}


def _sparse_basis(model: TrainModel, support: np.ndarray) -> np.ndarray:
    basis = np.zeros((2 * PIXELS, len(support)))
    synthesis = model.haar_analysis.T / np.sqrt(model.cell_weights)[:, None]
    for col, coefficient in enumerate(support):
        channel, index = divmod(int(coefficient), PIXELS)
        basis[channel * PIXELS:(channel + 1) * PIXELS, col] = synthesis[:, index]
    return _readonly(basis)


def _contrast_bounds(background: np.ndarray, amplitude: float, phases: int) -> list[tuple[float, float]]:
    intervals = [(REAL_BOUNDS[0] - background.real.min(),
                  REAL_BOUNDS[1] - background.real.max()),
                 (IMAG_BOUNDS[0] - background.imag.min(),
                  IMAG_BOUNDS[1] - background.imag.max())]
    bounds = []
    for lo, hi in intervals:
        lo, hi = float(lo / amplitude), float(hi / amplitude)
        # A declared eight-ULP inward guard on regional endpoints.  This is a
        # parameter-domain safeguard; actual-material checks still use zero tol.
        for _ in range(8):
            lo, hi = float(np.nextafter(lo, hi)), float(np.nextafter(hi, lo))
        bounds.extend([(lo, hi)] * phases)
    return bounds


def freeze_case(model: TrainModel, method: str, budget: int, baseline_delta: Any,
                background: Any, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Freeze all discrete choices and starts without accepting or inspecting truth."""
    aliases = {"pca": "single_chart", "analytic_atlas": "atlas", "levelset": "level_set"}
    method = aliases.get(method, method)
    if method not in METHODS or budget not in BUDGETS:
        raise ValueError("method/budget outside frozen A30 design")
    settings = _configuration(config)
    baseline = _field(baseline_delta, "baseline_delta")
    background = _field(background, "background", scalar=True)
    if not box_feasibility(background)["feasible"]:
        raise ValueError("known background is outside actual-material box")
    distances = np.sum(model.cell_weights[None, :] * np.abs(model.train_deltas - baseline) ** 2, axis=1)
    nearest = np.argsort(distances, kind="stable")
    sources = [("N3", None, baseline)] + [("TRAIN", int(i), model.train_deltas[i])
                                           for i in nearest[:7]]
    starts = []
    amplitude = float(settings["amplitude_scale"])
    for start_id, (source, parent_index, anchor) in enumerate(sources[:settings["max_starts"]]):
        start = {"id": start_id, "source": source, "source_train_index": parent_index,
                 "eligible": True, "reasons": [], "budget": budget}
        if method in ("sparse", "single_chart"):
            if method == "sparse":
                coefficients = np.r_[model.haar_analysis @ (anchor.real * np.sqrt(model.cell_weights)),
                                     model.haar_analysis @ (anchor.imag * np.sqrt(model.cell_weights))]
                # Literal top-s rule is frozen.  No mandatory DC or truth-based
                # support repair is allowed, including zero-loss corner cases.
                support = np.argsort(-np.abs(coefficients), kind="stable")[:budget]
                basis = _sparse_basis(model, support)
                start["support"] = _readonly(support)
            else:
                if model.single_rank < budget or model.single_basis.shape[1] < budget:
                    start.update(eligible=False, reasons=["TRAIN_SVD_RANK_BELOW_REAL_BUDGET"])
                    starts.append(start)
                    continue
                basis = _readonly(model.single_basis[:, :budget])
            params = basis.T @ (model.real_weights * _real(anchor))
            physical_basis = amplitude * basis
            params, repair = _ray_repair(params, physical_basis, background)
            start.update(basis=basis, params=_readonly(params),
                         intrinsic_rank=_rank(basis * np.sqrt(model.real_weights)[:, None]),
                         initial_ray_repair=repair)
        else:
            eligible = [t for t in model.templates if t.charts[method, budget].eligible]
            if not eligible:
                start.update(eligible=False, reasons=["NO_ELIGIBLE_TRAIN_GEOMETRY_CHART"])
                starts.append(start)
                continue
            template = min(eligible, key=lambda t: (np.sum(model.cell_weights * np.abs(
                model.train_deltas[t.index] - anchor) ** 2), t.index))
            chart = template.charts[method, budget]
            phases = len(template.masks)
            geometry = sum(chart.geometry_counts)
            weights, _ = _soft_partition(template.fields)
            contrasts = np.linalg.lstsq(weights * np.sqrt(model.cell_weights)[:, None],
                                        anchor * np.sqrt(model.cell_weights), rcond=None)[0]
            bound = 2.0 if method == "atlas" else 3.0 / geometry
            bounds = [(-bound, bound)] * geometry + _contrast_bounds(background, amplitude, phases)
            params = np.r_[np.zeros(geometry), contrasts.real, contrasts.imag]
            params = np.array([np.clip(x, lo, hi) for x, (lo, hi) in zip(params, bounds)])
            start.update(template_index=template.index, topology_key=template.topology,
                         phase_count=phases, geometry_count=geometry,
                         params=_readonly(params), bounds=tuple(bounds),
                         intrinsic_rank=chart.intrinsic_rank,
                         contrast_bound_restriction="COMMON_BACKGROUND_EXTREMA_REGIONAL_BOX",
                         contrast_bounds_inward_ulps=8)
        if start.get("intrinsic_rank", budget) != budget:
            start.update(eligible=False, reasons=["MATERIAL_RANK_BELOW_REAL_BUDGET"])
        starts.append(start)
    return {"scope": "TRUTH_FREE_FROZEN_CASE", "method": method, "budget": budget,
            "settings": settings, "background": _readonly(background),
            "baseline_delta": _readonly(baseline), "nearest_train_order": _readonly(nearest),
            "starts": tuple(starts)}


def _losses(model: TrainModel, image: np.ndarray, background: np.ndarray,
            target: np.ndarray) -> dict[str, Any]:
    difference = image - target
    target_mse = float(np.sum(model.cell_weights * np.abs(target - background) ** 2))
    mse = float(np.sum(model.cell_weights * np.abs(difference) ** 2))
    return {"scope": "OFFLINE_CAPACITY_LOSS", "fixed_mass_squared_error": mse,
            "real_squared_error": float(np.sum(model.cell_weights * difference.real ** 2)),
            "imag_squared_error": float(np.sum(model.cell_weights * difference.imag ** 2)),
            "target_delta_squared_norm": target_mse,
            "delta_nrmse": float(np.sqrt(mse / target_mse)) if target_mse else None,
            "zero_target_normalization": target_mse == 0,
            "morphology": "NOT_COMPUTED_HERE_PARENT_INDEPENDENT_SCORER"}


def _box_dimension(basis: np.ndarray, background: np.ndarray,
                   weights: np.ndarray) -> dict[str, Any]:
    budget = basis.shape[1]
    rank = _rank(basis * np.sqrt(weights)[:, None])
    strictly_inside = box_feasibility(background)["strict_interior"]
    result = {"unconstrained_material_rank": rank,
              "zero_delta_actual_box_strict_interior": strictly_inside,
              "feasible_affine_dimension": rank if strictly_inside else None,
              "dimension_certificate": "STRICT_INTERIOR_BACKGROUND" if strictly_inside
                                       else "UNRESOLVED_BOUNDARY_BACKGROUND"}
    forced = []
    # Exact zero-mean boundary corner: a nonnegative weighted-zero-mean channel
    # is identically zero, hence these chart directions are not free in the box.
    for channel, limits in ((0, REAL_BOUNDS), (1, IMAG_BOUNDS)):
        bg_channel = background.real if channel == 0 else background.imag
        block = basis[channel * PIXELS:(channel + 1) * PIXELS]
        at_boundary = np.all(bg_channel == limits[0]) or np.all(bg_channel == limits[1])
        weighted_means = weights[channel * PIXELS:(channel + 1) * PIXELS] @ block
        if at_boundary and np.all(np.abs(weighted_means) <= 32 * np.finfo(float).eps):
            forced.append(block)
    if forced:
        result.update(feasible_affine_dimension=budget - _rank(np.vstack(forced)),
                      dimension_certificate="WEIGHTED_ZERO_MEAN_CHANNEL_FORCED_AT_BOX_BOUNDARY")
    return result


def _qp_fit(model: TrainModel, frozen: dict[str, Any], start: dict[str, Any],
            target: np.ndarray) -> dict[str, Any]:
    """Primal active-set convex QP; at most the frozen number of iterations.

    The only rounding repair contracts parameters within the selected chart.
    There is no second optimizer, target-generated support, or pixel projection.
    """
    settings, background = frozen["settings"], frozen["background"]
    amplitude = float(settings["amplitude_scale"])
    basis = amplitude * start["basis"]
    weight = model.real_weights
    target_delta = _real(target - background)
    lower, upper = _material_limits(background)
    hessian = basis.T @ (weight[:, None] * basis)
    linear = basis.T @ (weight * target_delta)
    h_inv = np.linalg.inv(hessian)
    unconstrained = h_inv @ linear
    raw_c = np.vstack((basis, -basis))
    raw_d = np.r_[upper, -lower]
    norms = np.linalg.norm(raw_c, axis=1)
    keep = norms > 0
    constraint = raw_c[keep] / norms[keep, None]
    limit = raw_d[keep] / norms[keep]
    params = start["params"].copy()
    active: list[int] = []
    multipliers = np.empty(0)
    trajectory = []
    best = None
    target_mse = float(np.sum(weight * target_delta ** 2))
    repairs = []

    def record(iteration: int, point: np.ndarray, status: str) -> None:
        nonlocal best
        image = background + _complex(basis @ point)
        feasible = box_feasibility(image)["feasible"]
        loss = float(np.sum(weight * (basis @ point - target_delta) ** 2))
        entry = {"iteration": iteration, "fixed_mass_squared_error": loss,
                 "params": point.copy(), "box_feasible": feasible, "status": status}
        trajectory.append(entry)
        if feasible and (best is None or loss < best[0]):
            best = (loss, point.copy(), iteration)

    record(0, params, "INITIAL_FROZEN_SOURCE")
    reason = "MAXIMUM_ITERATIONS"
    iterations = 0
    if box_feasibility(background + _complex(basis @ unconstrained))["feasible"]:
        params = unconstrained
        record(0, params, "UNCONSTRAINED_CLOSED_FORM_IS_BOX_FEASIBLE")
        reason = "UNCONSTRAINED_GLOBAL_SOLUTION"
    else:
        for iteration in range(1, int(settings["max_iterations"]) + 1):
            iterations = iteration
            if active:
                ca = constraint[active]
                gram = ca @ h_inv @ ca.T
                multipliers = np.linalg.lstsq(gram, ca @ unconstrained - limit[active],
                                              rcond=32 * np.finfo(float).eps)[0]
                equality_optimum = unconstrained - h_inv @ ca.T @ multipliers
            else:
                multipliers = np.empty(0)
                equality_optimum = unconstrained
            direction = equality_optimum - params
            step_tolerance = 128 * np.finfo(float).eps * (1 + np.linalg.norm(params))
            if np.linalg.norm(direction) <= step_tolerance:
                if not active or np.min(multipliers) >= -step_tolerance:
                    reason = "ACTIVE_SET_KKT_STATIONARY"
                    record(iteration, params, reason)
                    break
                removed = int(np.argmin(multipliers))
                active.pop(removed)
                record(iteration, params, "REMOVE_NEGATIVE_DUAL_CONSTRAINT")
                continue
            movement = constraint @ direction
            slack = limit - constraint @ params
            candidates = np.flatnonzero(movement > step_tolerance)
            if active:
                candidates = candidates[~np.isin(candidates, active)]
            fraction = 1.0
            blocking = None
            if len(candidates):
                ratios = np.maximum(slack[candidates], 0) / movement[candidates]
                minimum = int(np.argmin(ratios))
                if ratios[minimum] < 1.0:
                    fraction = float(ratios[minimum])
                    blocking = int(candidates[minimum])
            params = params + fraction * direction
            params, repair = _ray_repair(params, basis, background)
            if repair["contracted"]:
                repairs.append({"iteration": iteration, **repair})
            record(iteration, params, "ACTIVE_SET_FEASIBLE_STEP")
            if blocking is not None and blocking not in active:
                active.append(blocking)
            elif fraction == 1.0:
                # The equality optimum has been reached.  Recheck dual signs at
                # the next iteration instead of claiming convergence from a move.
                continue
        if active:
            ca = constraint[active]
            multipliers = np.linalg.lstsq(ca @ h_inv @ ca.T,
                                          ca @ unconstrained - limit[active],
                                          rcond=32 * np.finfo(float).eps)[0]
    if best is None:
        return {"id": start["id"], "status": "NO_EXACT_BOX_FEASIBLE_ITERATE",
                "trajectory": trajectory, "iterations": iterations, "eligible": False}
    _, best_params, best_iteration = best
    image = background + _complex(basis @ best_params)
    gradient = hessian @ best_params - linear
    dual = np.maximum(multipliers, 0.0)
    stationarity = gradient + (constraint[active].T @ dual if active else 0)
    slack = limit[active] - constraint[active] @ best_params if active else np.empty(0)
    complementarity = float(np.dot(dual, np.abs(slack)))
    radius = float(np.sqrt(np.sum(weight * np.maximum(np.abs(lower), np.abs(upper)) ** 2)) / amplitude)
    # W-orthogonality bounds every feasible coefficient norm by radius.  For
    # nonnegative lambda, convexity bounds suboptimality by this KKT gap estimate.
    gap = float((radius + np.linalg.norm(best_params)) * np.linalg.norm(stationarity)
                + complementarity)
    normalized_gap = gap / target_mse if target_mse else gap
    certificate_ok = bool(box_feasibility(image)["feasible"]
                          and normalized_gap <= settings["certificate_tolerance"])
    loss = _losses(model, image, background, target)
    upper_error = loss["delta_nrmse"]
    lower_error = (float(np.sqrt(max(0, upper_error ** 2 - 2 * normalized_gap)))
                   if upper_error is not None else None)
    certificate = {"scope": "CONVEX_FIXED_FROZEN_CHART_ONLY", "numerical_only": True,
                   "certified": certificate_ok,
                   "stationarity_norm": float(np.linalg.norm(stationarity)),
                   "dual_feasibility_violation": 0.0,
                   "complementarity": complementarity,
                   "primal_violation": box_feasibility(image)["max_violation"],
                   "kkt_suboptimality_gap_upper": gap,
                   "normalized_kkt_gap": normalized_gap,
                   "tolerance": settings["certificate_tolerance"],
                   "nrmse_interval": [lower_error, upper_error],
                   "active_constraint_indices": np.asarray(active, dtype=int),
                   "nonnegative_multipliers": dual.copy(),
                   "no_interval_arithmetic_proof": True}
    return {"id": start["id"], "eligible": True, "status": reason,
            "optimizer": "PRIMAL_ACTIVE_SET_CONVEX_QP_NO_FALLBACK",
            "iterations": iterations, "function_evaluations": len(trajectory),
            "trajectory": trajectory, "best_iteration": best_iteration,
            "image": image, "params": best_params, "losses": loss,
            "box": box_feasibility(image), "certificate": certificate,
            "global_vs_local": "CONVEX_FIXED_CHART_NUMERICAL_CERTIFIED" if certificate_ok
                               else "CONVEX_FIXED_CHART_UNCERTIFIED_ACHIEVED_UPPER_BOUND",
            "dimensions": _box_dimension(start["basis"], background, weight),
            "ray_repairs": repairs}


def _geometry_fit(model: TrainModel, frozen: dict[str, Any], start: dict[str, Any],
                  target: np.ndarray) -> dict[str, Any]:
    settings, background = frozen["settings"], frozen["background"]
    method, budget = frozen["method"], frozen["budget"]
    template = model.templates[start["template_index"]]
    chart = template.charts[method, budget]
    amplitude = float(settings["amplitude_scale"])
    target_real = _real(target)
    norm = float(np.sum(model.cell_weights * np.abs(target - background) ** 2))
    objective_scale = norm if norm > 0 else 1.0
    trajectory = []
    best = None
    diagnostic_best = None

    def evaluate(params: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        delta, labels, derivative = _geometry_decode(template, chart, params, True)
        image = background + amplitude * delta
        residual = _real(image) - target_real
        value = 0.5 * float(np.sum(model.real_weights * residual ** 2)) / objective_scale
        gradient = amplitude * derivative.T @ (model.real_weights * residual) / objective_scale
        return value, gradient, image, labels

    def objective(params: np.ndarray) -> tuple[float, np.ndarray]:
        value, gradient, _, _ = evaluate(params)
        return value, gradient

    def record(params: np.ndarray, iteration: int) -> None:
        nonlocal best, diagnostic_best
        value, _, image, labels = evaluate(params)
        topology = _partition_signature(labels, len(template.masks))
        topology_valid = topology == template.topology
        feasible = box_feasibility(image)["feasible"]
        accepted = feasible and (method != "atlas" or topology_valid)
        trajectory.append({"iteration": iteration, "params": params.copy(),
                           "fixed_mass_squared_error": 2 * value * objective_scale,
                           "box_feasible": feasible, "digital_topology": topology,
                           "topology_preserved": topology_valid,
                           "accepted_for_primary": accepted})
        if feasible and (diagnostic_best is None or value < diagnostic_best[0]):
            diagnostic_best = (value, params.copy(), iteration, topology_valid)
        if accepted and (best is None or value < best[0]):
            best = (value, params.copy(), iteration)

    record(start["params"], 0)
    if method == "atlas" and not trajectory[0]["topology_preserved"]:
        return {"id": start["id"], "eligible": False,
                "status": "ZERO_GEOMETRY_SMOOTHING_CHANGES_DIGITAL_TOPOLOGY",
                "iterations": 0, "trajectory": trajectory}
    result = minimize(objective, start["params"].copy(), jac=True, method="L-BFGS-B",
                      bounds=start["bounds"],
                      options={"maxiter": int(settings["max_iterations"]),
                               "ftol": 1e-12, "gtol": 1e-8, "maxls": 20},
                      callback=lambda point: record(point, len(trajectory)))
    # SciPy's final point is already an accepted-iteration callback unless there
    # was no move; record a distinct final point only within the same iteration.
    if not np.array_equal(result.x, trajectory[-1]["params"]):
        record(result.x, int(result.nit))
    if best is None:
        return {"id": start["id"], "eligible": False,
                "status": "NO_EXACT_BOX_AND_REQUIRED_TOPOLOGY_FEASIBLE_ITERATE",
                "iterations": int(result.nit), "trajectory": trajectory}
    _, params, best_iteration = best
    delta, labels, weights = _geometry_decode(template, chart, params)
    image = background + amplitude * delta
    phases, geometry = len(template.masks), sum(chart.geometry_counts)
    contrasts = params[geometry:geometry + phases] + 1j * params[geometry + phases:]
    hard_delta = contrasts[np.maximum(labels.ravel() - 1, 0)] * (labels.ravel() > 0)
    violations = sum(not entry["topology_preserved"] for entry in trajectory)
    return {"id": start["id"], "eligible": True,
            "status": str(result.message), "optimizer_success": bool(result.success),
            "optimizer_status": int(result.status), "optimizer": "L-BFGS-B_ANALYTIC_JACOBIAN",
            "iterations": int(result.nit), "function_evaluations": int(result.nfev),
            "trajectory": trajectory, "best_iteration": best_iteration,
            "image": image, "params": params, "losses": _losses(model, image, background, target),
            "box": box_feasibility(image), "digital_topology": _partition_signature(labels, phases),
            "topology_preserved": _partition_signature(labels, phases) == template.topology,
            "topology_violation_iterates": violations,
            "best_unrestricted_diagnostic": {
                "scope": "OFFLINE_DIAGNOSTIC_NOT_ACCEPTED_AS_ATLAS_PRIMARY",
                "params": diagnostic_best[1], "iteration": diagnostic_best[2],
                "topology_preserved": diagnostic_best[3],
                "fixed_mass_squared_error": 2 * diagnostic_best[0] * objective_scale},
            "global_vs_local": "ACHIEVED_LOCAL_NONCONVEX_UPPER_BOUND",
            "certificate": {"scope": "NO_GLOBAL_NONCONVEX_CAPACITY_CERTIFICATE",
                            "certified": False, "continuous_topology_guarantee": False,
                            "digital_predicate_required": method == "atlas"},
            "dimensions": {"real_parameter_count": budget, "intrinsic_rank": chart.intrinsic_rank,
                           "geometry_count": geometry, "contrast_real_count": 2 * phases,
                           "phase_count": phases},
            "bias": {"scope": "TRAIN_TEMPLATE_AND_RASTERIZATION_DIAGNOSTICS",
                     "phase_quantization_mse_normalized_train": template.phase_quantization_mse,
                     "zero_geometry_smoothing_mse_normalized_train": template.smoothing_mse,
                     "achieved_soft_vs_hard_mse_actual": float(np.sum(model.cell_weights *
                         np.abs(amplitude * (delta - hard_delta)) ** 2)),
                     "smoothing_width_pixels": SMOOTHING_WIDTH,
                     "regional_common_background_extrema_box": True}}


def offline_best_s_tail(model: TrainModel, truth: Any, background: Any,
                        budget: int) -> dict[str, Any]:
    """Truth-selected support only diagnoses unconstrained best-s-term capacity.

    This function returns no material candidate and is never used by freeze_case.
    The real/imaginary coefficient tail counts exactly ``budget`` real entries.
    """
    if budget not in BUDGETS:
        raise ValueError("budget outside frozen protocol")
    delta = _field(truth, "OFFLINE truth") - _field(background, "background", scalar=True)
    coefficients = np.r_[model.haar_analysis @ (delta.real * np.sqrt(model.cell_weights)),
                         model.haar_analysis @ (delta.imag * np.sqrt(model.cell_weights))]
    order = np.argsort(-np.abs(coefficients), kind="stable")
    tail = float(np.sum(coefficients[order[budget:]] ** 2))
    total = float(np.sum(coefficients ** 2))
    return {"scope": "OFFLINE_UNCONSTRAINED_ORTHOGONAL_BEST_S_TAIL",
            "online_candidate": False, "real_budget": budget,
            "truth_selected_support": order[:budget].copy(),
            "fixed_mass_squared_tail": tail,
            "delta_nrmse_tail": float(np.sqrt(tail / total)) if total else None,
            "coefficient_squared_norm": total,
            "actual_material_box_enforced": False,
            "same_as_frozen_support_chart": False}


def fit_capacity(model: TrainModel, method: str, budget: int, baseline_delta: Any,
                 background: Any, truth: Any,
                 config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """OFFLINE best achieved fit over the frozen, truth-free discrete choices.

    Targets affect objective optimization and reporting only.  Constructing
    ``frozen`` precedes target conversion; online code uses freeze_case instead.
    """
    wall_start, cpu_start = perf_counter(), process_time()
    frozen = freeze_case(model, method, budget, baseline_delta, background, config)
    target = _field(truth, "OFFLINE truth")
    fits = []
    for start in frozen["starts"]:
        if not start["eligible"]:
            fits.append({"id": start["id"], "eligible": False, "iterations": 0,
                         "status": "FROZEN_START_INELIGIBLE", "reasons": start["reasons"],
                         "trajectory": []})
        elif frozen["method"] in ("sparse", "single_chart"):
            fits.append(_qp_fit(model, frozen, start, target))
        else:
            fits.append(_geometry_fit(model, frozen, start, target))
    feasible = [fit for fit in fits if fit.get("eligible") and "image" in fit
                and fit["box"]["feasible"]]
    result = {"scope": "OFFLINE_CAPACITY_ONLY", "method": frozen["method"],
              "budget": budget, "real_intrinsic_budget": budget,
              "frozen_starts": frozen["starts"], "fits": fits,
              "settings": frozen["settings"],
              "selection_scope": "BEST_ACHIEVED_OFFLINE_OVER_FINITE_TRUTH_FREE_CHOICES",
              "morphology": "NOT_COMPUTED_HERE_PARENT_INDEPENDENT_SCORER",
              "online_measurement_selection_validated": False,
              "physics_calls": 0, "training_calls": 0}
    if feasible:
        best = min(feasible, key=lambda fit: (fit["losses"]["fixed_mass_squared_error"], fit["id"]))
        result.update(status="ACHIEVED_OFFLINE_FIT", best_start_id=best["id"],
                      image=best["image"], delta=(best["image"] - frozen["background"]) /
                      frozen["settings"]["amplitude_scale"],
                      params=best["params"], losses=best["losses"], box=best["box"],
                      certificate=best["certificate"], dimensions=best["dimensions"],
                      global_vs_local=best["global_vs_local"],
                      trajectory=best["trajectory"],
                      primary_topology_preserved=best.get("topology_preserved"))
    else:
        result.update(status="NO_ELIGIBLE_EXACT_FEASIBLE_FIT", image=None, params=None,
                      losses=None, box={"feasible": False, "feasibility_tolerance": 0.0},
                      certificate={"certified": False}, global_vs_local="NOT_RUN_OR_INFEASIBLE")
    if frozen["method"] == "sparse":
        tail = offline_best_s_tail(model, target, frozen["background"], budget)
        result["offline_best_s_tail"] = tail
        result["fixed_support_box_vs_best_s_mse_gap"] = (
            result["losses"]["fixed_mass_squared_error"] - tail["fixed_mass_squared_tail"]
            if result["losses"] is not None else None)
    result["cost"] = {"wall_seconds": perf_counter() - wall_start,
                      "cpu_seconds": process_time() - cpu_start,
                      "starts": len(fits), "material_optimizer_iterations": sum(
                          fit["iterations"] for fit in fits),
                      "physics_calls": 0, "data_scope": "EXPLICIT_OFFLINE_TARGET_ARGUMENT"}
    return result
