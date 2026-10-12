# A32 pure statistics and figures

Status: **IMPLEMENTED; SYNTHETIC UTILITY CHECK PASS; ACTUAL-RESULT QA NOT RUN BY THIS WORKER.**

This worker owns only `src/a32/statistics.py`, `src/a32/plots.py` and this delegated artifact directory. No physics, fitting, parameter selection, target-file access, hashes or DeepSeek calls occurred. Numerical evidence and final scientific judgment remain with the parent Codex thread.

## Frozen statistical behavior

- Each expected method/parent requires one weak and one ring case, distinct case IDs, matching frozen family metadata, successful raw `success` flags and defined metrics. The raw solver `status` is retained as provenance; successful `OPTIMAL`/`ACTIVE_SET_CONVERGED` outcomes are accepted.
- Weak/ring metrics are averaged before equal-parent aggregation. The primary set is exactly 12 shell/asymmetric_piecewise/mixed_scale parents; smooth has its own descriptive summary. All four family summaries and the all-parent summary are also supplied.
- The E0 gain is `1 - mean_parent(method_MSE) / mean_parent(N3_MSE)`. It is not the average of parent-relative gains.
- The paired bootstrap resamples parent MSE pairs, then evaluates the same ratio of means: 2000 draws, seed 321012. Its interval is descriptive and never enters the 5% gate.
- Missing/failed/duplicate/family-mismatched/undefined required evidence yields an incomplete primary gate. Available-condition and available-pair means are explicitly descriptive and cannot replace a gate estimate.
- The 19 configured methods remain in the default expected method/parent table even when a whole method is absent. The primary gate requires the primary method and N3 on all 12 primary parents; an absent alternative method does not substitute for those requirements.

## Exact APIs

`statistics.py`:

```python
expected_methods(config)
aggregate_parent_rows(rows, config, *, methods=None, metrics=METRICS)
paired_parent_statistic(parent_rows, candidate, reference, parent_ids,
                        *, metric="fixed_mass_squared_error", samples=2000, seed=321012)
e0_statistics(rows, config, *, methods=None, comparison_pairs=None)
read_metrics_csv(path)
write_e0_statistics(summary, output_dir)
write_rows_csv(path, rows, *, columns=None)
```

`plots.py`:

```python
read_reconstructions(path)
plot_reconstruction_grid(reconstructions, metric_rows, *, methods, output_dir,
                         basename="E0_FIXED_FAMILY_EXAMPLES", component="abs_delta",
                         color_limits=None, method_labels=None)
plot_online_parent_gains(parent_rows, *, candidate_methods, reference_method,
                         parent_ids, output_dir, error_metric="fixed_mass_squared_error",
                         basename="E1_ACTUAL_ONLINE_PARENT_GAINS", method_labels=None)
plot_quality_runtime(parent_rows, *, methods, parent_ids, output_dir, runtime_field,
                      quality_field="fixed_mass_squared_error",
                      runtime_label="Charged CPU seconds",
                      quality_label="Full-grid material MSE (lower is better)",
                      evidence_label="E1 actual online", basename="QUALITY_RUNTIME",
                      method_labels=None)
plot_method_comparisons(parent_rows, *, comparison_groups, parent_ids, output_dir,
                        basename="E0_PRIOR_COMPARISONS",
                        metric="fixed_mass_squared_error", method_labels=None)
plot_offline_capacity(parent_rows, *, methods, parent_ids, output_dir,
                      metric="fixed_mass_squared_error", basename="E0_OFFLINE_CAPACITY",
                      method_labels=None)
```

Every plot returns PNG/SVG/CSV paths. `plot_method_comparisons` additionally returns a parent-pairs CSV path. Comparison pairs are explicit `(candidate, reference)` tuples grouped by caller-supplied panel titles; exact methods are never inferred from numeric results.

The reconstruction mapping requires `case_id` with shape `(32,)`, `truth`, `background`, `N3` and each requested exact method name with complex arrays of shape `(32,576)`. Case-first square grids are also accepted. The input must not use `chi_` prefixes. Examples are always parent 250065 shell, 250069 asymmetric_piecewise, 250073 smooth, 250077 mixed_scale, all weak background. The common color scale covers all displayed delta panels; absolute delta is the default, with optional real/imaginary delta views.

Online and quality/runtime inputs require one row per method/parent after condition averaging. Runtime columns and units are explicit. To use an aggregated runtime field, include it in `aggregate_parent_rows(..., metrics=(*METRICS, runtime_field))`. Per-parent observations remain visible; equal-parent quality/runtime means require complete coverage. Unavailable online gains are marked in parent labels and kept as empty CSV cells, without plotting a numerical substitute.

## Verification and receipts

`check_pure_utils.py` uses only synthetic fixtures, imports and configuration. It does not read A32 target/result files. Eight checks passed on the parent-provided A25 Python: ratio of means versus mean parent gains; deterministic bootstrap; frozen primary/smooth separation; authoritative success flag; failed/missing/duplicate retention; undefined zero-reference gate; fixed weak reconstruction examples/common scale; and all five PNG/SVG/CSV exports. All generated test artifacts are under `SYNTHETIC_ONLY/` and visibly labelled as utility checks, not experiment results.

Three representative synthetic figures were visually inspected. They retained a failed panel, the fixed example parents, common limits and explicit incomplete comparisons. Cosmetic changes after that check moved missing online states from the data region into parent labels, removed arbitrary percentage ticks from entirely incomplete comparison panels, and made method captions wrap at word boundaries. Per parent instruction these cosmetic changes were not rendered again; actual-result QA belongs to the parent.

- `CPU_RECEIPT_001.json`: failed system-Python NumPy import before arithmetic; recorded charge 0.0003035 s (startup not included in this first receipt).
- `CPU_RECEIPT_002.json`: failed bundled-Python Matplotlib import before arithmetic; recorded charge 0.085477916 s.
- `CPU_RECEIPT_003_LAUNCH_FAILURE.json`: shell quoting error before Python startup; CPU unavailable, launch wall time recorded; interpreter path corrected.
- `CPU_RECEIPT_004.json`: successful synthetic arithmetic/render check; recorded charge **1.639482959 s**, including interpreter CPU startup.
- `CPU_RECEIPT.json` is an alias of the latest successful receipt, **not another run**.

No actual material/online gains, same-quality speedup, scientific gate verdict, physical validation or aesthetic acceptance has been inferred from these synthetic checks. Parent intervention: run these utilities on actual completed result tables and review the final figures.
