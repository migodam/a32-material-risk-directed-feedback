"""Small SPD material QP with linear physical pixel bounds.

The objective is exactly .5*c.T@H@c-h.T@c and lower <= A@c <= upper.
H is supplied by the caller, which keeps proper prior precision and numerical
damping separate.  This solver never changes H, adds jitter, uses a pinv, clips
pixels, evaluates physics, or substitutes a different optimization problem.
"""
from __future__ import annotations

from time import perf_counter, process_time
from typing import Any, Callable, Mapping

import numpy as np
from scipy.linalg import cho_solve, qr, solve_triangular
from scipy.optimize import linprog


_DEFAULTS = {"max_iterations": 200, "kkt_rtol": 1e-8,
             "feasibility_tolerance": 1e-10, "no_pixel_clipping": True}


def _real_array(value: Any, name: str) -> np.ndarray:
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    result = np.asarray(value, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _configuration(config: Mapping[str, Any] | None) -> dict[str, Any]:
    settings = dict(_DEFAULTS)
    if config is not None:
        if not isinstance(config, Mapping):
            raise ValueError("config must be a mapping")
        source = config.get("solver", config)
        if not isinstance(source, Mapping):
            raise ValueError("config['solver'] must be a mapping")
        settings.update({key: source[key] for key in _DEFAULTS if key in source})
    maximum = settings["max_iterations"]
    if isinstance(maximum, bool) or int(maximum) != maximum or not 1 <= int(maximum) <= 200:
        raise ValueError("frozen QP cap: max_iterations must be an integer in [1,200]")
    settings["max_iterations"] = int(maximum)
    for name, maximum in (("kkt_rtol", 1e-8), ("feasibility_tolerance", 1e-10)):
        value = float(settings[name])
        if not np.isfinite(value) or value < 0 or value > maximum or (name == "kkt_rtol" and value == 0):
            raise ValueError(f"frozen QP tolerance: {name} must be <= {maximum}")
        settings[name] = value
    if settings["no_pixel_clipping"] is not True:
        raise ValueError("frozen QP forbids pixel clipping")
    return settings


def _limits(value: Any, rows: int, name: str) -> np.ndarray:
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    result = np.asarray(value, dtype=float)
    if result.ndim == 0:
        result = np.full(rows, float(result))
    if result.shape != (rows,) or np.isnan(result).any():
        raise ValueError(f"{name} must be a scalar or {rows}-vector without NaNs")
    if (name == "lower" and np.isposinf(result).any()) or (name == "upper" and np.isneginf(result).any()):
        raise ValueError(f"{name} has an impossible infinite bound")
    return result


def _linf(value: np.ndarray) -> float:
    return float(np.max(np.abs(value), initial=0.0))


def _primal_violation(point: np.ndarray, matrix: np.ndarray,
                      lower: np.ndarray, upper: np.ndarray) -> float:
    value = matrix @ point
    return float(max(0.0, np.max(lower - value, initial=-np.inf),
                     np.max(value - upper, initial=-np.inf)))


def _independent_rows(matrix: np.ndarray) -> list[int]:
    if len(matrix) == 0:
        return []
    _, triangular, pivots = qr(matrix.T, mode="economic", pivoting=True,
                               check_finite=False)
    diagonal = np.abs(np.diag(triangular))
    tolerance = 64 * np.finfo(float).eps * max(matrix.shape) * float(diagonal.max(initial=0.0))
    rank = int(np.count_nonzero(diagonal > tolerance))
    return [int(index) for index in pivots[:rank]]


def _adds_rank(matrix: np.ndarray, row: np.ndarray) -> bool:
    if len(matrix) == 0:
        return True
    if len(matrix) >= matrix.shape[1]:
        return False
    orthogonal, _ = qr(matrix.T, mode="economic", check_finite=False)
    residual = row - orthogonal @ (orthogonal.T @ row)
    return bool(np.linalg.norm(residual) >
                64 * np.finfo(float).eps * max(matrix.shape) * np.linalg.norm(row))


def _equality_optimum(H: np.ndarray, h: np.ndarray, working: np.ndarray,
                       limits: np.ndarray, h_factor: np.ndarray,
                       current: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """QR null-space solve with only independent working normals."""
    dimension = len(h)
    count = len(working)
    if count == 0:
        return cho_solve((h_factor, True), h, check_finite=False), np.empty(0), np.empty(0)
    orthogonal, triangular = qr(working.T, mode="full", check_finite=False)
    triangular = triangular[:count, :count]
    normal_basis, null_basis = orthogonal[:, :count], orthogonal[:, count:]
    normal_coordinates = solve_triangular(triangular.T, limits, lower=True,
                                          check_finite=False)
    optimum = normal_basis @ normal_coordinates
    if count < dimension:
        reduced = null_basis.T @ H @ null_basis
        reduced = (reduced + reduced.T) / 2
        reduced_factor = np.linalg.cholesky(reduced)
        rhs = null_basis.T @ (h - H @ optimum)
        optimum += null_basis @ cho_solve((reduced_factor, True), rhs,
                                          check_finite=False)
    dual_at_optimum = solve_triangular(triangular,
                                       -normal_basis.T @ (H @ optimum - h),
                                       check_finite=False)
    dual_at_current = solve_triangular(triangular,
                                       -normal_basis.T @ (H @ current - h),
                                       check_finite=False)
    return optimum, dual_at_optimum, dual_at_current


def solve_qp(H: Any, h: Any, A: Any, lower: Any, upper: Any,
             initial: Any = None, config: Mapping[str, Any] | None = None,
             check: Callable[[], Any] | None = None) -> dict[str, Any]:
    """Solve one frozen convex material QP and retain failed solutions.

    ``check()`` is an optional no-argument budget/cancellation hook.  None or
    True permits work; False or an exception returns the current point with a
    failed status.  Malformed inputs (including a non-SPD H) raise ValueError.
    Numerical failure, infeasibility and the iteration cap return diagnostics.

    ``multipliers`` has nonnegative ``lower`` and ``upper`` arrays in the raw
    pixel-constraint units, and ``normal=A.T@(upper-lower)``.  Equal bounds use
    the positive/negative parts of their unrestricted equality multiplier.
    The maximum raw bound violation is ``feasibility``.  ``kkt_relative`` is
    the maximum of relative stationarity, dual violation and complementarity.
    Stationarity is scaled by max(||Hc||inf,||h||inf,||normal||inf); products are
    scaled by that quantity times max(||c||inf,||Ac||inf).  A nonzero tiny floor
    only protects division by zero; it never changes the QP or its matrices.

    A feasibility LP is used only when neither the provided point nor zero is
    feasible.  Its reported iterations and every active-set iteration share
    the same cap of 200.  Constraint-row normalization changes neither bounds
    nor the feasible set.  Feasible coefficient steps remain in the chart;
    there is no post-solve repair or pixel clipping.
    """
    wall_start, cpu_start = perf_counter(), process_time()
    settings = _configuration(config)
    hessian = _real_array(H, "H")
    linear = _real_array(h, "h")
    matrix = _real_array(A, "A")
    if linear.ndim != 1 or not linear.size or hessian.shape != (linear.size, linear.size):
        raise ValueError("H must be (d,d) and h must be a nonempty d-vector")
    dimension = len(linear)
    if matrix.ndim != 2 or matrix.shape[1] != dimension:
        raise ValueError("A must have shape (m,d)")
    symmetry_scale = max(_linf(hessian), np.finfo(float).tiny)
    if _linf(hessian - hessian.T) > 64 * np.finfo(float).eps * symmetry_scale:
        raise ValueError("H must be symmetric; an asymmetric objective is not silently changed")
    hessian = (hessian + hessian.T) / 2
    try:
        h_factor = np.linalg.cholesky(hessian)
    except np.linalg.LinAlgError as error:
        raise ValueError("H must be SPD; no jitter/pinv is allowed") from error
    rows = len(matrix)
    low, high = _limits(lower, rows, "lower"), _limits(upper, rows, "upper")
    point = np.zeros(dimension) if initial is None else _real_array(initial, "initial").copy()
    if point.shape != (dimension,):
        raise ValueError("initial must be a d-vector")
    if check is not None and not callable(check):
        raise ValueError("check must be callable")
    iterations, phase_iterations, active_iterations = 0, 0, 0
    phase_record: dict[str, Any] | None = None
    equality_rows: list[int] = []
    active: list[int] = []
    working_dual = np.empty(0)
    source_rows = np.empty(0, dtype=int)
    source_side = np.empty(0, dtype=int)
    row_norm = np.linalg.norm(matrix, axis=1)

    def budget_status() -> tuple[str, str] | None:
        if check is None:
            return None
        try:
            permitted = check()
        except Exception as error:
            return "CHECK_FAILED", str(error)
        if permitted is not None and not bool(permitted):
            return "CHECK_STOPPED", "budget/cancellation hook returned false"
        return None

    def finish(status: str, reason: str = "") -> dict[str, Any]:
        raw_lower, raw_upper = np.zeros(rows), np.zeros(rows)
        count_equal = len(equality_rows)
        if len(working_dual) == count_equal + len(active):
            for position, row in enumerate(equality_rows):
                value = float(working_dual[position] / row_norm[row])
                raw_upper[row], raw_lower[row] = max(value, 0.0), max(-value, 0.0)
            for position, index in enumerate(active):
                value = max(float(working_dual[count_equal + position]), 0.0) / row_norm[source_rows[index]]
                if source_side[index] > 0:
                    raw_upper[source_rows[index]] = value
                else:
                    raw_lower[source_rows[index]] = value
        normal = matrix.T @ (raw_upper - raw_lower)
        material = matrix @ point
        gradient = hessian @ point - linear
        stationarity = gradient + normal
        scale = max(_linf(hessian @ point), _linf(linear), _linf(normal), np.finfo(float).tiny)
        variable_scale = max(_linf(point), _linf(material), np.finfo(float).tiny)
        product_scale = max(scale * variable_scale, np.finfo(float).tiny)
        lower_finite, upper_finite = np.isfinite(low), np.isfinite(high)
        complementarity = max(_linf(raw_lower[lower_finite] * (material[lower_finite] - low[lower_finite])),
                              _linf(raw_upper[upper_finite] * (high[upper_finite] - material[upper_finite])))
        stationarity_relative = _linf(stationarity) / scale
        complementarity_relative = complementarity / product_scale
        dual_violation = max(0.0, -float(raw_lower.min(initial=0.0)), -float(raw_upper.min(initial=0.0)))
        dual_relative = dual_violation / max(_linf(raw_lower), _linf(raw_upper), np.finfo(float).tiny)
        kkt_relative = max(stationarity_relative, complementarity_relative, dual_relative)
        feasibility = _primal_violation(point, matrix, low, high)
        feasible = bool(feasibility <= settings["feasibility_tolerance"])
        certified = bool(feasible and np.isfinite(kkt_relative) and kkt_relative <= settings["kkt_rtol"])
        success = certified and status == "SUCCESS"
        elapsed, cpu = perf_counter() - wall_start, process_time() - cpu_start
        return {"c": point.copy(), "status": status, "reason": reason,
                "success": success, "iterations": iterations,
                "phase_iterations": phase_iterations, "active_set_iterations": active_iterations,
                "objective": float(0.5 * point @ hessian @ point - linear @ point),
                "kkt_relative": float(kkt_relative),
                "stationarity_relative": float(stationarity_relative),
                "stationarity_norm": _linf(stationarity), "stationarity": stationarity,
                "multipliers": {"lower": raw_lower, "upper": raw_upper},
                "normal": normal, "feasibility": feasibility, "feasible": feasible,
                "complementarity": float(complementarity),
                "complementarity_relative": float(complementarity_relative),
                "dual_feasibility": float(dual_violation),
                "kkt_scale": scale, "complementarity_scale": product_scale,
                "active_pixel_rows": source_rows[np.asarray(active, dtype=int)].copy(),
                "active_sides": source_side[np.asarray(active, dtype=int)].copy(),
                "equality_pixel_rows": np.asarray(equality_rows, dtype=int),
                "optimizer": "PRIMAL_FEASIBLE_QR_NULLSPACE_ACTIVE_SET",
                "settings": dict(settings), "phase_I": phase_record,
                "no_pixel_clipping": True, "jitter_added": False, "pseudoinverse_used": False,
                "wall_seconds": elapsed, "cpu_seconds": cpu,
                "time": {"wall_seconds": elapsed, "cpu_seconds": cpu}}

    stopped = budget_status()
    if stopped is not None:
        return finish(*stopped)
    if np.any(low > high):
        return finish("INFEASIBLE", "a lower bound exceeds its upper bound")
    zero_rows = row_norm == 0
    if np.any((low[zero_rows] > 0) | (high[zero_rows] < 0)):
        return finish("INFEASIBLE", "a zero pixel row excludes zero")
    equal = np.isfinite(low) & np.isfinite(high) & (low == high) & ~zero_rows
    all_equal_rows = np.flatnonzero(equal)
    equality_matrix = matrix[all_equal_rows] / row_norm[all_equal_rows, None]
    equality_limit = high[all_equal_rows] / row_norm[all_equal_rows]
    selected = _independent_rows(equality_matrix)
    equality_rows = [int(all_equal_rows[index]) for index in selected]
    fixed_matrix = equality_matrix[selected]
    fixed_limit = equality_limit[selected]
    upper_rows = np.flatnonzero(np.isfinite(high) & ~equal & ~zero_rows)
    lower_rows = np.flatnonzero(np.isfinite(low) & ~equal & ~zero_rows)
    constraint = np.vstack((matrix[upper_rows] / row_norm[upper_rows, None],
                             -matrix[lower_rows] / row_norm[lower_rows, None]))
    limit = np.r_[high[upper_rows] / row_norm[upper_rows], -low[lower_rows] / row_norm[lower_rows]]
    source_rows = np.r_[upper_rows, lower_rows]
    source_side = np.r_[np.ones(len(upper_rows), dtype=int), -np.ones(len(lower_rows), dtype=int)]
    if not np.isfinite(constraint).all() or not np.isfinite(limit).all() or not np.isfinite(equality_limit).all():
        return finish("NUMERICAL_FAILURE", "constraint normalization overflowed")
    if _primal_violation(point, matrix, low, high) > settings["feasibility_tolerance"]:
        zero = np.zeros(dimension)
        if _primal_violation(zero, matrix, low, high) <= settings["feasibility_tolerance"]:
            point = zero
        else:
            stopped = budget_status()
            if stopped is not None:
                return finish(*stopped)
            phase = linprog(np.zeros(dimension),
                            A_ub=constraint if len(constraint) else None,
                            b_ub=limit if len(constraint) else None,
                            A_eq=equality_matrix if len(equality_matrix) else None,
                            b_eq=equality_limit if len(equality_matrix) else None,
                            bounds=[(None, None)] * dimension, method="highs-ds",
                            options={"maxiter": settings["max_iterations"],
                                     "primal_feasibility_tolerance": 1e-10,
                                     "dual_feasibility_tolerance": 1e-10})
            phase_iterations = int(phase.nit)
            iterations = phase_iterations
            phase_record = {"optimizer": "HIGHS_DS_FEASIBILITY_LP", "iterations": phase_iterations,
                            "success": bool(phase.success), "status": int(phase.status),
                            "message": str(phase.message)}
            if phase.x is not None and np.isfinite(phase.x).all():
                point = np.asarray(phase.x, float).copy()
            if not phase.success or _primal_violation(point, matrix, low, high) > settings["feasibility_tolerance"]:
                return finish("INFEASIBLE" if phase.status == 2 else "PHASE_I_FAILED", str(phase.message))
    reason = "the shared phase-I/active-set iteration cap was reached"
    for _ in range(settings["max_iterations"] - iterations):
        stopped = budget_status()
        if stopped is not None:
            return finish(*stopped)
        iterations += 1
        active_iterations += 1
        working = np.vstack((fixed_matrix, constraint[active]))
        working_limit = np.r_[fixed_limit, limit[active]]
        try:
            optimum, dual_at_optimum, working_dual = _equality_optimum(
                hessian, linear, working, working_limit, h_factor, point)
        except (np.linalg.LinAlgError, ValueError) as error:
            return finish("NUMERICAL_FAILURE", str(error))
        candidate_diagnostics = finish("SUCCESS")
        if candidate_diagnostics["success"]:
            return candidate_diagnostics
        direction = optimum - point
        step_tolerance = 64 * np.finfo(float).eps * dimension * (
            1 + np.linalg.norm(point) + np.linalg.norm(optimum))
        if np.linalg.norm(direction) <= step_tolerance:
            inequalities_dual = dual_at_optimum[len(equality_rows):]
            if len(inequalities_dual) and np.min(inequalities_dual) < 0:
                active.pop(int(np.argmin(inequalities_dual)))
                working_dual = np.empty(0)
                continue
            point = optimum
            working_dual = dual_at_optimum
            polished = finish("SUCCESS")
            if polished["success"]:
                return polished
            return finish("NUMERICAL_KKT_STALL", "equality optimum failed the unchanged-QP KKT/feasibility tolerances")
        movement = constraint @ direction
        slack = limit - constraint @ point
        movement_tolerance = 64 * np.finfo(float).eps * dimension * max(1.0, np.linalg.norm(direction))
        eligible = movement > movement_tolerance
        if active:
            eligible[np.asarray(active, dtype=int)] = False
        indices = np.flatnonzero(eligible)
        ratios = np.maximum(slack[indices], 0.0) / movement[indices]
        fraction = min(1.0, float(ratios.min(initial=np.inf)))
        new_point = optimum.copy() if fraction == 1 else point + fraction * direction
        # Only shorten this feasible coefficient segment to absorb rounding.
        # This is not a pixel projection or an after-solve feasibility repair.
        for _retry in range(16):
            if _primal_violation(new_point, matrix, low, high) <= settings["feasibility_tolerance"]:
                break
            fraction = float(np.nextafter(fraction, 0.0))
            new_point = point + fraction * direction
        if _primal_violation(new_point, matrix, low, high) > settings["feasibility_tolerance"]:
            return finish("NUMERICAL_FEASIBILITY_STALL", "a coefficient step could not retain physical bounds")
        point = new_point
        if fraction < 1:
            tie_tolerance = 64 * np.finfo(float).eps * max(1.0, abs(fraction))
            blockers = indices[np.abs(ratios - fraction) <= tie_tolerance]
            added = False
            for index in blockers:
                index = int(index)
                if _adds_rank(working, constraint[index]):
                    active.append(index)
                    working = np.vstack((working, constraint[index]))
                    added = True
            if not added:
                return finish("NUMERICAL_DEPENDENT_BLOCKER", "blocking normals were numerically dependent on the working set")
        working_dual = np.empty(0)
    # Recompute duals at the retained final point without taking another step.
    working = np.vstack((fixed_matrix, constraint[active]))
    working_limit = np.r_[fixed_limit, limit[active]]
    try:
        _, _, working_dual = _equality_optimum(hessian, linear, working, working_limit, h_factor, point)
    except (np.linalg.LinAlgError, ValueError):
        working_dual = np.empty(0)
    final = finish("SUCCESS")
    return final if final["success"] else finish("MAX_ITERATIONS", reason)
