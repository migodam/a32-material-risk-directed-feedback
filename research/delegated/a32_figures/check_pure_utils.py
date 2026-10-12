"""Synthetic arithmetic/render checks only; never read an A32 target/result.

All outputs in this directory are marked as synthetic utility fixtures. They
are not experimental measurements, physics checks or scientific validation.
The entire check, including imports and rendering, receives one CPU receipt.
"""
from __future__ import annotations

import csv
import json
import resource
import sys
import time
import traceback
from pathlib import Path

started_wall = time.perf_counter()
started_usage = resource.getrusage(resource.RUSAGE_SELF)
task_dir = Path(__file__).resolve().parent
repo = task_dir.parents[2]
checks, artifacts = [], {}
outcome = "FAILED"
error_text = None

try:
    sys.path.insert(0, str(repo / "src"))
    import numpy as np
    from a32 import plots, statistics

    config = json.loads((repo / "a32_config.json").read_text())
    primary = config["primary_method"]
    left = "pca64_L_gaussian"
    cases, records = [], []
    for index, parent in enumerate(config["parent_ids"], 1):
        family = statistics.parent_family(parent)
        candidate_factor = .05 if family == "smooth" else .8 if index % 2 else 1.2
        for background, multiplier in (("weak", 1.), ("ring", 2.)):
            case_id = f"SYNTHETIC_{parent}_{background}"
            cases.append(case_id)
            for method, factor in (("N3", 1.), (primary, candidate_factor), (left, .9)):
                records.append({"case_id": case_id, "parent_id": parent, "family": family,
                                "background_id": background, "method": method, "status": "OPTIMAL",
                                "success": True, "fixed_mass_squared_error": index * multiplier * factor,
                                "delta_nrmse": .5, "support_iou": .6, "edge_f1": .7})
    summary = statistics.e0_statistics(records, config)
    complex_ids = [parent for parent in config["parent_ids"] if statistics.parent_family(parent) in config["primary_families"]]
    primary_pair = summary["comparisons"][primary + "_vs_N3"]["complex_families"]
    expected_gain = 1 - 95.2 / 94.
    assert abs(primary_pair["point_estimate"] - expected_gain) < 1e-12
    mean_parent_gain = np.mean([pair["parent_relative_gain"] for pair in primary_pair["parent_pairs"]])
    assert abs(primary_pair["point_estimate"] - mean_parent_gain) > .01
    assert primary_pair["paired_parents"] == 12 and primary_pair["bootstrap_valid_resamples"] == 2000
    assert summary["E0_gate"]["status"] == "FAIL"
    assert len(summary["methods"]) == 19 and len(summary["parent_rows"]) == 19 * 16
    assert summary["comparisons"][primary + "_vs_N3"]["smooth"]["point_estimate"] > .9
    checks.append("Exact ratio-of-parent-means gate; distinct from mean parent gains; smooth separate; 2000 resamples.")

    repeat = statistics.paired_parent_statistic(summary["parent_rows"], primary, "N3", complex_ids)
    assert repeat["ci95"] == primary_pair["ci95"]
    checks.append("Bootstrap is deterministic with frozen seed 321012.")
    bad = [dict(row) for row in records]
    failed = next(row for row in bad if row["method"] == primary and row["parent_id"] == 250069)
    failed["success"] = False
    failed["status"] = "OPTIMAL"
    failed_summary = statistics.e0_statistics(bad, config)
    assert failed_summary["E0_gate"]["status"] == "INCOMPLETE"
    assert failed_summary["E0_gate"]["gain"] is None
    assert len(failed_summary["parent_rows"]) == len(summary["parent_rows"])
    checks.append("Authoritative success=False blocks gate despite finite metrics and OPTIMAL status; failed parent retained.")
    for label, changed in (
        ("missing", [row for row in records if not (row["method"] == primary and row["parent_id"] == 250069 and row["background_id"] == "ring")]),
        ("duplicated", records + [dict(next(row for row in records if row["method"] == primary and row["parent_id"] == 250069))]),
    ):
        local = statistics.e0_statistics(changed, config)
        assert local["E0_gate"]["status"] == "INCOMPLETE" and local["E0_gate"]["gain"] is None
        assert len(local["parent_rows"]) == 19 * 16
        checks.append(f"{label.capitalize()} required case blocks the gate without dropping an expected parent/method.")
    zero = [dict(row) for row in records]
    for row in zero:
        if row["method"] == "N3":
            row["fixed_mass_squared_error"] = 0.
    assert statistics.e0_statistics(zero, config)["E0_gate"]["status"] == "INCOMPLETE"
    checks.append("Undefined zero reference denominator remains incomplete.")

    output = task_dir / "SYNTHETIC_ONLY"
    original_save = plots._save
    def synthetic_save(fig, rows, output_dir, basename):
        title = fig._suptitle.get_text() if fig._suptitle is not None else ""
        fig.suptitle("SYNTHETIC UTILITY CHECK — no experiment result\n" + title, fontsize=11)
        return original_save(fig, [{"synthetic_check_only": True, **row} for row in rows], output_dir, basename)
    plots._save = synthetic_save

    background = np.full((len(cases), 576), .1 + .02j, dtype=np.complex128)
    truth = background.copy()
    for index in range(len(cases)):
        truth[index].reshape(24, 24)[6:18, 7:17] += .7 + .12j
    arrays = {"case_id": np.asarray(cases), "background": background, "truth": truth,
              "N3": truth + .04 + .01j, primary: truth + .02 + .005j, left: truth + .03 + .008j}
    artifacts["reconstructions"] = plots.plot_reconstruction_grid(
        arrays, bad, methods=[primary, left], output_dir=output, basename="SYNTHETIC_RECONSTRUCTIONS")
    with Path(artifacts["reconstructions"]["csv"]).open(newline="") as handle:
        plot_table = list(csv.DictReader(handle))
    assert {row["parent_id"] for row in plot_table} == {"250065", "250069", "250073", "250077"}
    assert {row["background_id"] for row in plot_table} == {"weak"}
    assert len({(row["color_min"], row["color_max"]) for row in plot_table}) == 1
    assert any(row["status"].startswith("FAILED_CASE") and row["display_value"] == "" for row in plot_table)
    checks.append("Fixed four-family weak examples use common absolute-delta limits; failed reconstruction stays blank and labelled.")
    artifacts["comparisons"] = plots.plot_method_comparisons(
        summary["parent_rows"], comparison_groups={
            "Gaussian L vs calibrated L": [(primary, left)],
            "Gaussian vs other priors": [(primary, "pca64_calibrated_L_ridge"), (primary, "pca64_calibrated_L_multiscale")]},
        parent_ids=complex_ids, output_dir=output, basename="SYNTHETIC_COMPARISONS")
    artifacts["capacity"] = plots.plot_offline_capacity(
        summary["parent_rows"], methods=["N3", "pca64_fixed_chart_OFFLINE_capacity"],
        parent_ids=config["parent_ids"], output_dir=output, basename="SYNTHETIC_OFFLINE_CAPACITY")
    online = [dict(row) for row in failed_summary["parent_rows"] if row["method"] in {primary, "N3"}]
    artifacts["online"] = plots.plot_online_parent_gains(
        online, candidate_methods=[primary], reference_method="N3", parent_ids=config["parent_ids"],
        output_dir=output, basename="SYNTHETIC_ONLINE_GAINS")
    with Path(artifacts["online"]["csv"]).open(newline="") as handle:
        online_table = list(csv.DictReader(handle))
    assert len(online_table) == 16
    assert next(row for row in online_table if row["parent_id"] == "250069")["gain"] == ""
    for index, row in enumerate(online):
        row["metric_complete"] = dict(row["metric_complete"])
        row["metric_complete"]["cpu_charged_seconds"] = row["structural_complete"]
        row["cpu_charged_seconds"] = 1. + index / 10
    artifacts["quality_runtime"] = plots.plot_quality_runtime(
        online, methods=[primary, "N3"], parent_ids=config["parent_ids"], runtime_field="cpu_charged_seconds",
        runtime_label="Synthetic fixture CPU units", evidence_label="SYNTHETIC CHECK — E1 actual online",
        output_dir=output, basename="SYNTHETIC_QUALITY_RUNTIME")
    checks.append("All five figure types render PNG/SVG plus CSV; missing online comparison is preserved as empty gain, never zero.")
    for paths in artifacts.values():
        for path in paths.values():
            assert Path(path).exists() and Path(path).stat().st_size > 0
    outcome = "PASS"
except Exception:
    error_text = traceback.format_exc()
finally:
    ended_usage = resource.getrusage(resource.RUSAGE_SELF)
    elapsed = time.perf_counter() - started_wall
    # This process exists solely for this check; charge interpreter startup too.
    user_cpu = ended_usage.ru_utime
    system_cpu = ended_usage.ru_stime
    cpu_total = user_cpu + system_cpu
    receipt = {"task": "A32 pure statistics and plotting synthetic utility check", "outcome": outcome,
               "is_research_experiment": False, "native_backend_invoked": False, "target_files_read": False,
               "wall_seconds": elapsed, "user_cpu_seconds": user_cpu, "system_cpu_seconds": system_cpu,
               "process_cpu_seconds": cpu_total, "cpu_charged_seconds": max(elapsed, cpu_total),
               "charge_rule": "max(process CPU, wall); parent can book its protocol ledger without double counting",
               "checks": checks, "artifacts": artifacts, "error": error_text}
    receipt_path = task_dir / f"CPU_RECEIPT_{len(list(task_dir.glob('CPU_RECEIPT_*.json'))) + 1:03d}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    (task_dir / "CPU_RECEIPT.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"outcome": outcome, "checks": len(checks), "cpu_charged_seconds": receipt["cpu_charged_seconds"],
                      "receipt": str(receipt_path), "error": error_text}))
if outcome != "PASS":
    raise SystemExit(1)
