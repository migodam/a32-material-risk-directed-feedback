"""Pure A32 exposed-development summaries; no fitting or physics.

The independent unit is a parent. Weak/ring metrics are averaged first and
parents then receive equal weight. The material gate is a ratio of those MSE
means, *not* a mean of parent-relative gains. A missing/failed/duplicated case
never disappears from a gate: its parent remains present and incomplete.
File access occurs only in explicitly called read/write helpers.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from .frozen_metrics import FAMILIES, VALIDATION_PARENTS

METRICS = ("fixed_mass_squared_error", "delta_nrmse", "support_iou", "edge_f1")
PRIMARY_FAMILIES = ("shell", "asymmetric_piecewise", "mixed_scale")
BACKGROUNDS = ("weak", "ring")
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 321012
OK_STATUSES = frozenset(("OK", "COMPLETE"))


def parent_family(parent_id: int) -> str | None:
    """Metadata-fixed family assignment, independent of reconstruction quality."""
    parent = int(parent_id)
    return FAMILIES[(parent - 250065) // 4] if parent in VALIDATION_PARENTS else None


def finite_number(value) -> float | None:
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def row_ok(row: Mapping) -> bool:
    """Use A32's authoritative success flag; retain QP statuses as provenance.

    CSV booleans are accepted. Frozen OK/COMPLETE validity is only a fallback
    for caller-supplied aggregate tables that have no raw success column.
    """
    truthy = (True, 1, "True", "true", "1")
    if "success" in row:
        return row["success"] in truthy
    return (row.get("valid", True) in truthy and
            row.get("complete", True) in truthy and
            row.get("status", "OK") in OK_STATUSES)


def expected_methods(config: Mapping) -> list[str]:
    """The frozen A32 method inventory, including explicitly offline capacity."""
    names = ["N3"]
    for representation in config.get("representations", ()):
        names.extend((f"{representation}_projected_N3", f"{representation}_L_gaussian"))
        names.extend(f"{representation}_calibrated_L_{prior}" for prior in config.get("prior_types", ()))
        names.append(f"{representation}_fixed_chart_OFFLINE_capacity")
    names.append(str(config["primary_method"]))
    return list(dict.fromkeys(names))


def _valid_metric(value, metric: str) -> float | None:
    value = finite_number(value)
    if value is None or value < 0:
        return None
    if metric in {"support_iou", "edge_f1"} and value > 1:
        return None
    return value


def _mean(values: Sequence[float]) -> float | None:
    # Divide before summing to avoid overflow from an otherwise finite mean.
    return finite_number(math.fsum(value / len(values) for value in values)) if values else None


def _gain(candidate: float | None, reference: float | None) -> float | None:
    return finite_number(1 - candidate / reference) if candidate is not None and reference is not None and reference > 0 else None


def _configuration(config: Mapping) -> tuple[list[int], list[str], str]:
    parents = [int(parent) for parent in config.get("parent_ids", VALIDATION_PARENTS)]
    if len(parents) != len(VALIDATION_PARENTS) or set(parents) != set(VALIDATION_PARENTS):
        raise ValueError("A32_COMPLETE_FROZEN_16_PARENT_WHITELIST_REQUIRED")
    if len(set(parents)) != len(parents):
        raise ValueError("A32_DISTINCT_PARENT_IDS_REQUIRED")
    families = list(config.get("primary_families", PRIMARY_FAMILIES))
    if len(families) != 3 or set(families) != set(PRIMARY_FAMILIES):
        raise ValueError("A32_FROZEN_PRIMARY_FAMILIES_REQUIRED")
    backgrounds = list(config.get("backgrounds", BACKGROUNDS))
    if len(backgrounds) != 2 or set(backgrounds) != set(BACKGROUNDS):
        raise ValueError("A32_WEAK_RING_BACKGROUND_PAIR_REQUIRED")
    bootstrap = config.get("bootstrap", {})
    if (int(bootstrap.get("samples", BOOTSTRAP_SAMPLES)) != BOOTSTRAP_SAMPLES or
            int(bootstrap.get("seed", BOOTSTRAP_SEED)) != BOOTSTRAP_SEED or
            bootstrap.get("unit", "parent") != "parent" or
            bootstrap.get("average_backgrounds_first", True) is not True):
        raise ValueError("A32_FROZEN_PAIRED_PARENT_BOOTSTRAP_REQUIRED")
    primary = str(config["primary_method"])
    if not primary:
        raise ValueError("A32_PRIMARY_METHOD_REQUIRED")
    return parents, families, primary


def aggregate_parent_rows(
    rows: Iterable[Mapping], config: Mapping, *, methods: Sequence[str] | None = None,
    metrics: Sequence[str] = METRICS,
) -> list[dict]:
    """Return all expected method/parent rows, including missing or failed ones.

    Each expected parent/method requires exactly one weak and one ring row,
    distinct case IDs, matching frozen family metadata, and successful status.
    ``available_condition_descriptive_mean`` is explicitly non-gating. Missing
    diagnostic metrics are tracked per metric; structural/status failures make
    every metric incomplete. Unrecognized input parents are retained separately
    by :func:`e0_statistics`, rather than being substituted for expected parents.
    """
    parents, _, primary = _configuration(config)
    records = [dict(row) for row in rows]
    metric_names = tuple(metrics)
    if not metric_names or len(set(metric_names)) != len(metric_names):
        raise ValueError("DISTINCT_NONEMPTY_METRICS_REQUIRED")
    declared = expected_methods(config) if methods is None else list(methods)
    names = list(dict.fromkeys(["N3", primary, *declared]))
    names.extend(sorted({str(row["method"]) for row in records} - set(names)))
    if any(not name for name in names):
        raise ValueError("NONEMPTY_METHOD_NAMES_REQUIRED")
    grouped = defaultdict(list)
    for row in records:
        try:
            grouped[(str(row["method"]), int(row["parent_id"]))].append(row)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("METHOD_AND_INTEGER_PARENT_ID_REQUIRED") from error
    output = []
    for method in names:
        for parent in parents:
            local = grouped.get((method, parent), [])
            expected_family = parent_family(parent)
            by_background = defaultdict(list)
            for row in local:
                by_background[str(row.get("background_id", ""))].append(row)
            issues = []
            if not local:
                issues.append("MISSING_PARENT")
            missing = [name for name in BACKGROUNDS if not by_background[name]]
            duplicates = [name for name in BACKGROUNDS if len(by_background[name]) > 1]
            unexpected = sorted(set(by_background) - set(BACKGROUNDS))
            if missing:
                issues.append("MISSING_BACKGROUNDS")
            if duplicates:
                issues.append("DUPLICATED_BACKGROUND_ROWS")
            if unexpected:
                issues.append("UNEXPECTED_BACKGROUNDS")
            identifiers = [str(row["case_id"]) if row.get("case_id") is not None else "" for row in local]
            if local and (any(not value for value in identifiers) or len(set(identifiers)) != len(local)):
                issues.append("MISSING_OR_DUPLICATED_CASE_ID")
            if any(row.get("family") != expected_family for row in local):
                issues.append("FAMILY_MISMATCH")
            if any(not row_ok(row) for row in local):
                issues.append("FAILED_CASE_STATUS")
            structural_complete = len(local) == 2 and not issues
            metric_complete, means, descriptive, undefined = {}, {}, {}, []
            for metric in metric_names:
                values = [_valid_metric(row.get(metric), metric) for row in local]
                complete = structural_complete and all(value is not None for value in values)
                mean = _mean(values) if complete else None
                complete = complete and mean is not None
                metric_complete[metric] = complete
                means[metric] = mean
                available = [value for row, value in zip(local, values) if row_ok(row) and value is not None]
                descriptive[metric] = _mean(available)
                if structural_complete and not complete:
                    undefined.append(metric)
            if undefined:
                issues.append("UNDEFINED_METRICS")
            complete = structural_complete and all(metric_complete.values())
            output.append({
                "parent_id": parent, "family": expected_family, "method": method,
                "aggregation": "parent_equal", "background_count_expected": 2,
                "observations": len(local), "case_ids": identifiers,
                "source_statuses": [str(row.get("status", "OK")) for row in local],
                "structural_complete": structural_complete, "complete": complete,
                "valid": complete, "status": "OK" if complete else "INCOMPLETE",
                "issues": issues, "missing_backgrounds": missing,
                "duplicated_backgrounds": duplicates, "unexpected_backgrounds": unexpected,
                "undefined_metrics": undefined, "metric_complete": metric_complete,
                "available_condition_descriptive_mean": descriptive, **means,
            })
    return output


def _parent_metric(row: Mapping, metric: str) -> float | None:
    if row.get("aggregation") == "parent_equal" and "metric_complete" in row:
        if not row.get("structural_complete", False) or not row["metric_complete"].get(metric, False):
            return None
    elif not row_ok(row):
        return None
    return _valid_metric(row.get(metric), metric)


def paired_parent_statistic(
    parent_rows: Iterable[Mapping], candidate: str, reference: str, parent_ids: Sequence[int],
    *, metric: str = "fixed_mass_squared_error", samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Paired ratio-of-means gain with descriptive 2000-parent-resample CI.

    The point estimate and CI require every explicitly expected parent. A
    separate available-pair ratio is returned for diagnosis and cannot become a
    gate. Parent-relative gains are diagnostic only. Zero reference error in an
    individual parent does not invalidate a positive group denominator.
    """
    if int(samples) != BOOTSTRAP_SAMPLES or int(seed) != BOOTSTRAP_SEED:
        raise ValueError("A32_FROZEN_2000_SEED321012_BOOTSTRAP_REQUIRED")
    parents = [int(value) for value in parent_ids]
    if not parents or len(set(parents)) != len(parents):
        raise ValueError("DISTINCT_EXPLICIT_EXPECTED_PARENTS_REQUIRED")
    grouped = defaultdict(list)
    for row in parent_rows:
        if str(row.get("method")) in {candidate, reference}:
            grouped[(str(row["method"]), int(row["parent_id"]))].append(row)
    pairs, failures = [], []
    for parent in parents:
        left, right = grouped.get((candidate, parent), []), grouped.get((reference, parent), [])
        reasons = []
        if len(left) != 1 or len(right) != 1:
            reasons.append("MISSING_OR_DUPLICATED_PARENT_ROW")
        values = [None, None]
        if not reasons:
            for index, row in enumerate((left[0], right[0])):
                values[index] = _parent_metric(row, metric)
                if values[index] is None:
                    reasons.append(("CANDIDATE" if index == 0 else "REFERENCE") + "_FAILED_OR_UNDEFINED")
                if parent_family(parent) is not None and row.get("family") != parent_family(parent):
                    reasons.append("FAMILY_MISMATCH")
        av, bv = values
        valid = not reasons
        pair = {"parent_id": parent, "family": parent_family(parent),
                "candidate": candidate, "reference": reference,
                "candidate_value": av, "reference_value": bv,
                "complete": valid, "status": "OK" if valid else "INCOMPLETE", "issues": reasons,
                "parent_relative_gain": _gain(av, bv) if valid else None}
        pairs.append(pair)
        if reasons:
            failures.append({"parent_id": parent, "issues": reasons})
    available = [pair for pair in pairs if pair["complete"]]
    av_mean = _mean([pair["candidate_value"] for pair in available])
    bv_mean = _mean([pair["reference_value"] for pair in available])
    available_gain = _gain(av_mean, bv_mean)
    complete = not failures and available_gain is not None
    result = {
        "candidate": candidate, "reference": reference, "metric": metric,
        "formula": "1 - equal_parent_mean(candidate)/equal_parent_mean(reference)",
        "unit": "independent_parent", "samples": int(samples), "seed": int(seed),
        "bootstrap_is_descriptive_only": True, "expected_parent_ids": parents,
        "expected_parents": len(parents), "paired_parents": len(available),
        "coverage_complete": complete, "status": "OK" if complete else "INCOMPLETE",
        "undefined_or_failed_parents": failures, "parent_pairs": pairs,
        "candidate_equal_parent_mean": av_mean if complete else None,
        "reference_equal_parent_mean": bv_mean if complete else None,
        "point_estimate": available_gain if complete else None, "ci95": None,
        "available_pair_descriptive_candidate_mean": av_mean,
        "available_pair_descriptive_reference_mean": bv_mean,
        "available_pair_descriptive_gain": available_gain,
        "bootstrap_valid_resamples": 0, "bootstrap_undefined_resamples": 0,
    }
    if complete:
        values = np.asarray([[pair["candidate_value"], pair["reference_value"]] for pair in pairs], dtype=float)
        indices = np.random.default_rng(int(seed)).integers(0, len(parents), size=(int(samples), len(parents)))
        sampled = (values[indices] / len(parents)).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            gains = 1 - sampled[:, 0] / sampled[:, 1]
        valid = (sampled[:, 1] > 0) & np.isfinite(gains)
        result["bootstrap_valid_resamples"] = int(valid.sum())
        result["bootstrap_undefined_resamples"] = int((~valid).sum())
        if valid.all():
            result["ci95"] = [float(value) for value in np.quantile(gains, [.025, .975])]
    elif not failures:
        result["undefined_group_ratio"] = True
    return result


def e0_statistics(
    rows: Iterable[Mapping], config: Mapping, *, methods: Sequence[str] | None = None,
    comparison_pairs: Sequence[tuple[str, str]] | None = None,
) -> dict:
    """Summarize E0 and evaluate only its frozen, completeness-sensitive gate."""
    records = [dict(row) for row in rows]
    parents, primary_families, primary = _configuration(config)
    parent_rows = aggregate_parent_rows(records, config, methods=methods)
    names = list(dict.fromkeys(row["method"] for row in parent_rows))
    pairs = list(dict.fromkeys(comparison_pairs if comparison_pairs is not None else
                               [(method, "N3") for method in names if method != "N3"]))
    if (primary, "N3") not in pairs:
        pairs.append((primary, "N3"))
    groups = {"complex_families": [parent for parent in parents if parent_family(parent) in primary_families],
              "smooth": [parent for parent in parents if parent_family(parent) == "smooth"],
              "all_families": parents}
    groups.update({"family_" + family: [parent for parent in parents if parent_family(parent) == family]
                   for family in FAMILIES})
    comparisons = {}
    for candidate, reference in pairs:
        comparisons[candidate + "_vs_" + reference] = {
            group: paired_parent_statistic(parent_rows, candidate, reference, ids)
            for group, ids in groups.items()}
    primary_stat = comparisons[primary + "_vs_N3"]["complex_families"]
    unexpected = [row for row in records if int(row["parent_id"]) not in parents]
    primary_ids = set(groups["complex_families"])
    incomplete_primary = [row["parent_id"] for row in parent_rows
                          if row["method"] in {primary, "N3"} and row["parent_id"] in primary_ids
                          and not (row["structural_complete"] and
                                   row["metric_complete"]["fixed_mass_squared_error"])]
    complete = primary_stat["coverage_complete"] and not incomplete_primary and not unexpected
    threshold = finite_number(config.get("E0_gate", {}).get("material_MSE_gain_min", .05))
    if threshold != .05:
        raise ValueError("A32_FROZEN_5_PERCENT_MATERIAL_MSE_GATE_REQUIRED")
    if config.get("E0_gate", {}).get("complete_primary_required", True) is not True:
        raise ValueError("A32_COMPLETE_PRIMARY_GATE_REQUIRED")
    gain = primary_stat["point_estimate"] if complete else None
    gate = {"primary_method": primary, "reference": "N3", "metric": "fixed_mass_squared_error",
            "primary_families": primary_families, "expected_parent_count": len(primary_ids),
            "formula": primary_stat["formula"], "gain_min": threshold,
            "complete_primary_required": True, "coverage_complete": complete,
            "gain": gain, "status": "INCOMPLETE" if not complete else "PASS" if gain >= threshold else "FAIL",
            "incomplete_primary_parent_ids": sorted(set(incomplete_primary)),
            "unexpected_parent_rows": len(unexpected), "bootstrap_is_descriptive_only": True}
    return {"protocol_id": config.get("protocol_id"), "phase": "E0_OFFLINE_EXPOSED_DEVELOPMENT",
            "holdout": False, "unit": "independent_parent", "backgrounds_averaged_first": list(BACKGROUNDS),
            "expected_parent_ids": parents, "primary_families": primary_families,
            "smooth_reported_separately": True, "methods": names, "parent_rows": parent_rows,
            "unexpected_parent_rows": unexpected, "comparisons": comparisons, "E0_gate": gate}


def read_metrics_csv(path: str | Path) -> list[dict]:
    """Read only the path explicitly supplied by the caller."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows_csv(path: str | Path, rows: Iterable[Mapping], *, columns: Sequence[str] | None = None) -> Path:
    """Write nested fields as JSON; preserve missing cells as empty, never zero."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [dict(row) for row in rows]
    names = list(columns) if columns is not None else list(dict.fromkeys(key for row in records for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="raise")
        writer.writeheader()
        for row in records:
            writer.writerow({key: json.dumps(value, allow_nan=False, sort_keys=True)
                             if isinstance(value, (dict, list, tuple)) else value for key, value in row.items()})
    return path


def write_e0_statistics(summary: Mapping, output_dir: str | Path) -> dict[str, str]:
    """Persist the requested summary, complete parent table, and pair table."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "E0_STATISTICS.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False, sort_keys=True) + "\n", encoding="utf-8")
    parent_path = write_rows_csv(output / "E0_PARENT_METRICS.csv", summary["parent_rows"])
    pairs = []
    for comparison, groups in summary["comparisons"].items():
        for group, statistic in groups.items():
            pairs.extend({"comparison": comparison, "group": group, "metric": statistic["metric"], **row}
                         for row in statistic["parent_pairs"])
    pair_path = write_rows_csv(output / "E0_PAIRED_PARENT_METRICS.csv", pairs)
    return {"summary": str(summary_path), "parent_metrics": str(parent_path), "paired_parent_metrics": str(pair_path)}
