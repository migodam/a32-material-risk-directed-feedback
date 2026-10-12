"""A32 plots from explicit, already-produced arrays/tables; never fit or solve.

All plots save PNG, SVG and their underlying CSV. Online charts require one
row per method/parent after any condition averaging. Missing/failed parents
remain in the plot table and are visibly marked, never replaced by zero gain.
The reconstruction examples are fixed by metadata, not selected by quality.
"""
from __future__ import annotations

import math
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

from .statistics import finite_number, paired_parent_statistic, parent_family, row_ok, write_rows_csv

FIXED_EXAMPLES = (("shell", 250065), ("asymmetric_piecewise", 250069),
                  ("smooth", 250073), ("mixed_scale", 250077))
FAMILY_COLORS = {"shell": "#4477AA", "asymmetric_piecewise": "#EE6677",
                 "smooth": "#228833", "mixed_scale": "#AA3377"}


def _label(name: str, labels: Mapping[str, str]) -> str:
    return labels.get(name, name.replace("_", " "))


def _save(fig, rows: Iterable[Mapping], output_dir: str | Path, basename: str) -> dict[str, str]:
    if not basename or Path(basename).name != basename:
        raise ValueError("SIMPLE_NONEMPTY_FIGURE_BASENAME_REQUIRED")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {kind: output / (basename + "." + kind) for kind in ("png", "svg", "csv")}
    try:
        with plt.rc_context({"svg.fonttype": "none"}):
            fig.savefig(paths["png"], dpi=220, bbox_inches="tight")
            fig.savefig(paths["svg"], bbox_inches="tight")
        write_rows_csv(paths["csv"], rows)
    finally:
        plt.close(fig)
    return {key: str(value) for key, value in paths.items()}


def _identifier(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def read_reconstructions(path: str | Path) -> dict[str, np.ndarray]:
    """Read only an explicitly supplied NPZ, with object/pickle loading disabled."""
    with np.load(Path(path), allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def _case_image(arrays: Mapping, name: str, index: int, count: int) -> np.ndarray | None:
    if name not in arrays:
        return None
    values = np.asarray(arrays[name])
    if values.ndim < 2 or values.shape[0] != count:
        raise ValueError("CASE_FIRST_RECONSTRUCTION_ARRAY_REQUIRED:" + name)
    image = np.asarray(values[index], dtype=np.complex128)
    side = math.isqrt(image.size)
    if side * side != image.size or image.ndim > 2:
        raise ValueError("SQUARE_FULL_GRID_RECONSTRUCTION_REQUIRED:" + name)
    return image.reshape(side, side) if np.isfinite(image).all() else None


def plot_reconstruction_grid(
    reconstructions: Mapping, metric_rows: Iterable[Mapping], *, methods: Sequence[str],
    output_dir: str | Path, basename: str = "E0_FIXED_FAMILY_EXAMPLES",
    component: str = "abs_delta", color_limits: tuple[float, float] | None = None,
    method_labels: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Four fixed first-family parents under weak background, common delta scale.

    ``truth``, ``background``, ``case_id``, ``N3`` and method-named arrays use
    the explicit NPZ schema. ``abs_delta`` displays |chi-background|. Signed
    real/imaginary components can also be requested and share a symmetric
    absolute bound. No example can be replaced because it is missing or failed.
    """
    if component not in {"abs_delta", "real_delta", "imag_delta"}:
        raise ValueError("EXPLICIT_DELTA_DISPLAY_COMPONENT_REQUIRED")
    labels = dict(method_labels or {})
    names = ["truth", *dict.fromkeys(["N3", *methods])]
    if "background" in names or len(set(names)) != len(names):
        raise ValueError("DISTINCT_NONRESERVED_METHOD_NAMES_REQUIRED")
    case_ids = [_identifier(value) for value in np.asarray(reconstructions.get("case_id", [])).reshape(-1)]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("DISTINCT_RECONSTRUCTION_CASE_IDS_REQUIRED")
    index_by_case = {case: index for index, case in enumerate(case_ids)}
    records = [dict(row) for row in metric_rows]
    panels, plot_rows, finite_images = {}, [], []
    for row_index, (family, parent) in enumerate(FIXED_EXAMPLES):
        local = [row for row in records if int(row["parent_id"]) == parent and
                 row.get("family") == family and row.get("background_id") == "weak"]
        matching_cases = sorted({_identifier(row["case_id"]) for row in local})
        case = matching_cases[0] if len(matching_cases) == 1 else None
        array_index = index_by_case.get(case)
        for column, name in enumerate(names):
            status, image = "OK", None
            if case is None:
                status = "MISSING_OR_DUPLICATED_FIXED_CASE"
            elif array_index is None:
                status = "MISSING_RECONSTRUCTION_CASE"
            else:
                if name != "truth":
                    method_rows = [row for row in local if row.get("method") == name]
                    if len(method_rows) != 1:
                        status = "MISSING_OR_DUPLICATED_METRIC_ROW"
                    elif not row_ok(method_rows[0]):
                        status = "FAILED_CASE: " + str(method_rows[0].get("status"))
                if status == "OK":
                    estimate = _case_image(reconstructions, name, array_index, len(case_ids))
                    background = _case_image(reconstructions, "background", array_index, len(case_ids))
                    if estimate is None or background is None:
                        status = "MISSING_OR_NONFINITE_ARRAY"
                    elif estimate.shape != background.shape:
                        raise ValueError("MATCHED_RECONSTRUCTION_BACKGROUND_GRID_REQUIRED")
                    else:
                        delta = estimate - background
                        image = np.abs(delta) if component == "abs_delta" else delta.real if component == "real_delta" else delta.imag
                        if not np.isfinite(image).all():
                            image, status = None, "NONFINITE_DELTA"
            panels[(row_index, column)] = (image, status)
            common = {"evidence_role": "offline_exposed_image_diagnostic", "family": family,
                      "parent_id": parent, "background_id": "weak", "case_id": case,
                      "method": name, "component": component, "status": status}
            if image is None:
                plot_rows.append({**common, "pixel_y": None, "pixel_x": None, "display_value": None})
            else:
                finite_images.append(image)
                plot_rows.extend({**common, "pixel_y": y, "pixel_x": x, "display_value": float(image[y, x])}
                                 for y in range(image.shape[0]) for x in range(image.shape[1]))
    if color_limits is not None:
        low, high = map(float, color_limits)
        if not math.isfinite(low) or not math.isfinite(high) or high <= low:
            raise ValueError("FINITE_INCREASING_COMMON_COLOR_LIMITS_REQUIRED")
        if component == "abs_delta" and low < 0:
            raise ValueError("NONNEGATIVE_ABSOLUTE_DELTA_SCALE_REQUIRED")
    elif finite_images:
        bound = max(float(np.max(np.abs(image))) for image in finite_images)
        # A displayed all-zero panel requires a finite rendering scale; the
        # CSV retains its exact zeros and records the chosen limits.
        bound = bound if bound > 0 else 1.
        low, high = (0., bound) if component == "abs_delta" else (-bound, bound)
    else:
        low, high = None, None
    for row in plot_rows:
        row.update(color_min=low, color_max=high, color_scale="common_all_displayed_panels")
    fig, axes = plt.subplots(4, len(names), figsize=(2.7 * len(names), 11), squeeze=False, constrained_layout=True)
    shown = None
    for row_index, (family, parent) in enumerate(FIXED_EXAMPLES):
        for column, name in enumerate(names):
            ax = axes[row_index, column]
            image, status = panels[(row_index, column)]
            if image is not None:
                shown = ax.imshow(image, origin="lower", vmin=low, vmax=high,
                                  cmap="viridis" if component == "abs_delta" else "RdBu_r", interpolation="nearest")
            else:
                ax.set_facecolor("#F2F2F2")
                ax.text(.5, .5, textwrap.fill(status, 23), ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_index == 0:
                ax.set_title(textwrap.fill("Truth (offline)" if name == "truth" else _label(name, labels), 25), fontsize=10)
            if column == 0:
                ax.set_ylabel(f"{family}\nparent {parent} / weak", fontsize=10)
    if shown is not None:
        fig.colorbar(shown, ax=axes.ravel().tolist(), shrink=.75, label={
            "abs_delta": "|chi − background|", "real_delta": "Re(chi − background)",
            "imag_delta": "Im(chi − background)"}[component])
    fig.suptitle("E0 offline image diagnostics · fixed metadata examples · common absolute delta limits", fontsize=12)
    return _save(fig, plot_rows, output_dir, basename)


def _index_parent_rows(rows: Iterable[Mapping]) -> dict:
    indexed = defaultdict(list)
    for row in rows:
        indexed[(str(row["method"]), int(row["parent_id"]))].append(dict(row))
    return indexed


def _parent_value(local: list[Mapping], parent: int, field: str) -> tuple[float | None, str]:
    if len(local) != 1:
        return None, "MISSING_OR_DUPLICATED_PARENT_ROW"
    row = local[0]
    if row.get("aggregation") == "parent_equal" and "metric_complete" in row:
        valid = row.get("structural_complete", False) and row["metric_complete"].get(field, False)
    else:
        valid = row_ok(row)
    if not valid:
        return None, "FAILED_OR_INCOMPLETE_PARENT"
    if parent_family(parent) is not None and row.get("family") != parent_family(parent):
        return None, "FAMILY_MISMATCH"
    value = finite_number(row.get(field))
    if value is None or value < 0:
        return None, "UNDEFINED_METRIC"
    return value, "OK"


def _explicit_parents(parent_ids: Sequence[int]) -> list[int]:
    parents = [int(value) for value in parent_ids]
    if not parents or len(set(parents)) != len(parents):
        raise ValueError("DISTINCT_EXPLICIT_EXPECTED_PARENTS_REQUIRED")
    return parents


def plot_online_parent_gains(
    parent_rows: Iterable[Mapping], *, candidate_methods: Sequence[str], reference_method: str,
    parent_ids: Sequence[int], output_dir: str | Path,
    error_metric: str = "fixed_mass_squared_error", basename: str = "E1_ACTUAL_ONLINE_PARENT_GAINS",
    method_labels: Mapping[str, str] | None = None,
    evidence_label: str = "Actual measured online material fits",
) -> dict[str, str]:
    """Actual online gains for every explicitly expected parent; no oracle input.

    Supply measured online final/initial parent metrics with their real status.
    This helper has no capacity estimate, uncertainty proxy, or predicted-risk
    gain input. Positive gain means lower measured candidate material error.
    """
    parents, indexed = _explicit_parents(parent_ids), _index_parent_rows(parent_rows)
    methods, labels = list(dict.fromkeys(candidate_methods)), dict(method_labels or {})
    if not methods:
        raise ValueError("EXPLICIT_ONLINE_CANDIDATE_METHODS_REQUIRED")
    fig, ax = plt.subplots(figsize=(max(12, .72 * len(parents)), 5.7), constrained_layout=True)
    plot_rows, missing_count, missing_by_parent = [], 0, defaultdict(int)
    colors = plt.get_cmap("tab10")
    for method_index, method in enumerate(methods):
        offset = (method_index - (len(methods) - 1) / 2) * min(.16, .65 / len(methods))
        xs, ys = [], []
        for index, parent in enumerate(parents):
            av, candidate_status = _parent_value(indexed.get((method, parent), []), parent, error_metric)
            bv, reference_status = _parent_value(indexed.get((reference_method, parent), []), parent, error_metric)
            gain = finite_number(1 - av / bv) if av is not None and bv is not None and bv > 0 else None
            status = "OK" if gain is not None else "INCOMPLETE"
            plot_rows.append({"evidence_role": "actual_online_measured_gain", "parent_id": parent,
                              "family": parent_family(parent), "candidate": method, "reference": reference_method,
                              "metric": error_metric, "candidate_value": av, "reference_value": bv, "gain": gain,
                              "status": status, "candidate_status": candidate_status, "reference_status": reference_status})
            if gain is not None:
                xs.append(index + offset)
                ys.append(gain)
            else:
                missing_count += 1
                missing_by_parent[parent] += 1
        ax.scatter(xs, ys, s=38, color=colors(method_index % 10),
                   label=f"{_label(method, labels)} ({len(ys)}/{len(parents)} complete)")
    ax.axhline(0, color="#555555", linewidth=.8)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.set_xticks(range(len(parents)), [f"{parent}\n{parent_family(parent) or 'unassigned'}" +
                  (f"\n× {missing_by_parent[parent]} incomplete" if missing_by_parent[parent] else "") for parent in parents],
                  rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Actual material-error gain: 1 − candidate / reference")
    ax.set_title(f"{evidence_label} · all expected parents")
    ax.legend(loc="best", fontsize=8)
    ax.grid(axis="y", alpha=.2)
    if missing_count:
        fig.supxlabel(f"× in parent labels: {missing_count} missing, failed or undefined comparisons; no numerical gain plotted", fontsize=9)
    return _save(fig, plot_rows, output_dir, basename)


def plot_quality_runtime(
    parent_rows: Iterable[Mapping], *, methods: Sequence[str], parent_ids: Sequence[int],
    output_dir: str | Path, runtime_field: str, quality_field: str = "fixed_mass_squared_error",
    runtime_label: str = "Charged CPU seconds", quality_label: str = "Full-grid material MSE (lower is better)",
    evidence_label: str = "E1 actual online", basename: str = "QUALITY_RUNTIME",
    method_labels: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Measured per-parent quality/time points; equal-parent means only if complete.

    The caller supplies the actual runtime column and its unit. No proxy time,
    extrapolated time, or same-quality speedup is calculated here.
    """
    parents, indexed = _explicit_parents(parent_ids), _index_parent_rows(parent_rows)
    names, labels = list(dict.fromkeys(methods)), dict(method_labels or {})
    if not names:
        raise ValueError("EXPLICIT_METHODS_REQUIRED")
    fig, ax = plt.subplots(figsize=(9.5, 5.8), constrained_layout=True)
    plot_rows, notes = [], []
    colors = plt.get_cmap("tab10")
    for method_index, method in enumerate(names):
        xs, ys = [], []
        for parent in parents:
            local = indexed.get((method, parent), [])
            runtime, time_status = _parent_value(local, parent, runtime_field)
            quality, quality_status = _parent_value(local, parent, quality_field)
            complete = runtime is not None and quality is not None
            plot_rows.append({"evidence_role": evidence_label, "method": method, "parent_id": parent,
                              "family": parent_family(parent), "runtime_field": runtime_field,
                              "runtime": runtime, "quality_field": quality_field, "quality": quality,
                              "status": "OK" if complete else "INCOMPLETE", "time_status": time_status,
                              "quality_status": quality_status})
            if complete:
                xs.append(runtime)
                ys.append(quality)
        color = colors(method_index % 10)
        label = f"{_label(method, labels)} ({len(xs)}/{len(parents)} complete)"
        ax.scatter(xs, ys, s=30, color=color, alpha=.65, label=label)
        if len(xs) == len(parents):
            ax.scatter([np.mean(xs)], [np.mean(ys)], s=150, marker="*", color=color, edgecolors="#222222")
        else:
            notes.append(f"{_label(method, labels)}: {len(parents) - len(xs)} missing/failed quality-time pairs")
    ax.set_xlabel(runtime_label)
    ax.set_ylabel(quality_label)
    ax.set_title(f"{evidence_label} · observed quality and runtime\n★ equal-parent mean only with complete coverage")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=.2)
    if notes:
        fig.supxlabel("\n".join(notes), fontsize=8)
    return _save(fig, plot_rows, output_dir, basename)


def plot_method_comparisons(
    parent_rows: Iterable[Mapping], *, comparison_groups: Mapping[str, Sequence[tuple[str, str]]],
    parent_ids: Sequence[int], output_dir: str | Path, basename: str = "E0_PRIOR_COMPARISONS",
    metric: str = "fixed_mass_squared_error", method_labels: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Explicit Gaussian L/calibrated-L and Gaussian/ridge/multiscale pairs.

    ``comparison_groups`` supplies real method names and display group titles;
    method names are never inferred or invented. Bars use ratio-of-equal-parent
    MSE means. CIs are descriptive paired-parent resamples, never a gate.
    """
    records, parents = [dict(row) for row in parent_rows], _explicit_parents(parent_ids)
    groups, labels = list(comparison_groups.items()), dict(method_labels or {})
    if not groups or any(not pairs for _, pairs in groups):
        raise ValueError("NONEMPTY_EXPLICIT_COMPARISON_GROUPS_REQUIRED")
    fig, axes = plt.subplots(1, len(groups), figsize=(max(8, 5 * len(groups)), 6.1),
                             squeeze=False, constrained_layout=True)
    summary_rows, pair_rows = [], []
    for group_index, (group, pairs) in enumerate(groups):
        ax = axes[0, group_index]
        tick_labels = []
        complete_comparisons = 0
        for index, (candidate, reference) in enumerate(pairs):
            statistic = paired_parent_statistic(records, candidate, reference, parents, metric=metric)
            summary_rows.append({"evidence_role": "offline_exposed_method_comparison", "group": group,
                                 "candidate": candidate, "reference": reference, "metric": metric,
                                 "gain": statistic["point_estimate"], "ci95": statistic["ci95"],
                                 "expected_parents": len(parents), "paired_parents": statistic["paired_parents"],
                                 "status": statistic["status"], "formula": statistic["formula"],
                                 "bootstrap_samples": statistic["samples"], "bootstrap_seed": statistic["seed"],
                                 "bootstrap_is_descriptive_only": True})
            pair_rows.extend({"group": group, "metric": metric, **pair} for pair in statistic["parent_pairs"])
            gain = statistic["point_estimate"]
            if gain is None:
                ax.text(index, .5, f"INCOMPLETE\n{statistic['paired_parents']}/{len(parents)} paired",
                        ha="center", va="bottom", transform=ax.get_xaxis_transform(), fontsize=8)
            else:
                complete_comparisons += 1
                ax.bar(index, gain, color="#4477AA" if gain >= 0 else "#EE6677", width=.58)
                if statistic["ci95"] is not None:
                    low, high = statistic["ci95"]
                    # The percentile interval need not enclose the point; draw
                    # its actual endpoints rather than clipping error lengths.
                    ax.vlines(index, low, high, color="#222222", linewidth=1.2)
                    ax.hlines([low, high], index - .1, index + .1, color="#222222", linewidth=1.2)
            tick_labels.append(textwrap.fill(_label(candidate, labels), 25) + "\nvs\n" +
                               textwrap.fill(_label(reference, labels), 25))
        ax.axhline(0, color="#555555", linewidth=.8)
        ax.set_xticks(range(len(pairs)), tick_labels, fontsize=8)
        ax.set_title(group, fontsize=11)
        ax.set_ylabel("Gain: ratio of equal-parent MSE means")
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax.grid(axis="y", alpha=.2)
        if not complete_comparisons:
            ax.set_ylim(0, 1)
            ax.set_yticks([])
            ax.set_ylabel("No complete paired comparison")
    fig.suptitle("E0 offline method comparisons · descriptive paired-parent 95% intervals", fontsize=12)
    artifacts = _save(fig, summary_rows, output_dir, basename)
    artifacts["parent_pairs_csv"] = str(write_rows_csv(Path(output_dir) / (basename + "_PARENT_PAIRS.csv"), pair_rows))
    return artifacts


def plot_offline_capacity(
    parent_rows: Iterable[Mapping], *, methods: Sequence[str], parent_ids: Sequence[int],
    output_dir: str | Path, metric: str = "fixed_mass_squared_error", basename: str = "E0_OFFLINE_CAPACITY",
    method_labels: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """A separately labelled offline capacity diagnostic, without online claims."""
    parents, indexed = _explicit_parents(parent_ids), _index_parent_rows(parent_rows)
    names, labels = list(dict.fromkeys(methods)), dict(method_labels or {})
    if not names:
        raise ValueError("EXPLICIT_OFFLINE_METHODS_REQUIRED")
    fig, ax = plt.subplots(figsize=(max(9, 1.6 * len(names)), 6.2), constrained_layout=True)
    rows, labelled_families, completed_counts = [], set(), []
    for method_index, method in enumerate(names):
        values = []
        for parent_index, parent in enumerate(parents):
            value, status = _parent_value(indexed.get((method, parent), []), parent, metric)
            rows.append({"evidence_role": "offline_capacity_diagnostic", "online_gain_claim": False,
                         "method": method, "parent_id": parent, "family": parent_family(parent),
                         "metric": metric, "value": value, "status": status})
            x = method_index + (parent_index - (len(parents) - 1) / 2) * (.56 / len(parents))
            if value is not None:
                family = parent_family(parent)
                ax.scatter(x, value, color=FAMILY_COLORS.get(family, "#777777"), s=28, alpha=.8,
                           label=family if family not in labelled_families else None)
                labelled_families.add(family)
                values.append(value)
        completed_counts.append(len(values))
        if len(values) == len(parents):
            ax.plot([method_index - .3, method_index + .3], [np.mean(values)] * 2, color="#222222", linewidth=2)
    ax.set_xticks(range(len(names)), [textwrap.fill(_label(name, labels), 18) +
                  f"\n{count}/{len(parents)} complete" for name, count in zip(names, completed_counts)], fontsize=9)
    ax.set_ylabel(metric + " (lower is better)")
    ax.set_title("E0 offline capacity diagnostics · exposed-development targets\nEach point is one parent; black mean requires full coverage")
    ax.grid(axis="y", alpha=.2)
    if labelled_families:
        ax.legend(fontsize=9, loc="best")
    fig.supxlabel("Offline image evidence; completeness labels and CSV preserve missing/failed parents. Online gains are reported separately.", fontsize=8)
    return _save(fig, rows, output_dir, basename)
