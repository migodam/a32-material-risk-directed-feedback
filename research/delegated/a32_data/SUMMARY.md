# A32 data and chart interface receipt

Implementation complete. This worker changed only `src/a32/data.py`,
`src/a32/charts.py`, and `tests/test_data_charts.py`, plus this directory's
summary and CPU receipts. No config or pipeline edits were made.

## Input interfaces

- `online_cases(config_or_input_root) -> tuple[OnlineCase, ...]`: all32 cases
  in `COVERAGE_CASES.csv` order. Metadata has only case/parent/family/background
  identity. Pools expose only `observation`, `sigma`, `n3`, `background`.
  `case.background` is the whitelisted known-background mapping; `case.chi_b`
  is its material vector. Background and case arrays are immutable.
- `train_deltas(config_or_input_root) -> (normalized64x576, metadata)`: validates
  every split/parent identity before reading TRAIN material. Selects exactly one
  `p<parent>_weak_a1_clean`, amplitude1.2, NaN clean-noise marker, `OK` status
  per TRAIN parent250001–250064. Returns `(chi-chi_b)/1.2`.
- `cached_n3(case) -> CachedN3`: reads only `data`, `chi`, `background`,
  `cache_protocol`. Returns data32x8 and chi/background576. Requires exact N3
  identity or exact frozen `mass_unpack(mass_pack(N3))` identity, known
  background identity, and `A28-EXACT-STATE-v1`. Never accepts an arbitrary
  nearby material or recomputes physics. Full saved-mass, factor, and numerical
  equation validation remains the parent's frozen-adapter responsibility.
- `offline_targets(config_or_input_root, case_ids, *, completion_token_path)`
  is the independent target loader. It validates the explicit token and all32
  saved prediction files before opening `OFFLINE_TARGETS.npz`. Target arrays
  are returned in online order as `OfflineTargets.truth/background/n3`.

Token schema, in the directory containing all32 per-case prediction NPZs:

```json
{"status":"ONLINE_COMPLETE","case_ids":["online order"],
 "prediction_files":["<case_id>.npz"]}
```

Each of the32 referenced files must exist below the token directory, have the
matching scalar `case_id`, and contain at least one finite complex128 shape576
`chi_*` prediction. Missing/mismatched receipts refuse target access.

## Chart interfaces

- `prepare_charts(train_deltas, *, cell_weights=None)` reuses the unchanged
  frozen `prepare_train`, including its existing unused geometry preparation.
- `freeze_chart(model, representation, n3, background, *, amplitude=1.2)` accepts
  exactly `pca64`, `sparse32`, `sparse64`; it freezes normalized N3 contrast and
  selects the first eligible A30 start. No truth/best-start argument exists.
- `FrozenChart.T` is real1152xd, physically scaled as `1.2*B`;
  `material_metric = T.T @ W @ T`. `train_coefficients` are the exact weighted
  projections of actual amplitude1.2 TRAIN deltas into this same T.
  `train_mass` contains those actual TRAIN deltas multiplied by `sqrt(W)`.
- `initial_coefficients` is the raw weighted N3 projection;
  `frozen_feasible_initial` is the selected A30 feasible ray initialization.
  `decode(c)` never clips pixels. `projection_problem()` returns
  `H,h,A,lower,upper`. `feasible_projection_initial(solver=None, config=None,
  check=None)` lazily uses `.qp.solve_qp`, forwarding the optional zero-argument
  budget hook and returning its full QP receipt.

## Verification and accounting

14 synthetic arithmetic/access-boundary tests passed. They guard forbidden
pool/background/cache reads, TRAIN split leakage, target access before saved
online completion, physical basis scaling, coefficient normal equations,
conditional sparse support, first-eligible selection, and QP dispatch.

Real input whitelist validation found32 cases/16 parents and64 independent
clean TRAIN samples. All32 paid caches matched the exact frozen pack/unpack
identity; maximum relative chi difference from saved N3 was
`7.637592561734429e-17`. No real target archive was opened.

CPU receipts use the existing A25 Python interpreter with all requested BLAS,
OpenMP, and vector-library thread counts1:

| Receipt | CPU seconds |
|---|---:|
| `INPUT_SCHEMA_RECEIPT.json` | 0.094168 |
| `TEST_RECEIPT.json` | 0.465634 |
| `WHITELIST_INPUT_RECEIPT.json` | 0.109156 |
| Total for parent accounting | **0.668958** |

Real Maxwell calls0; real dev/locked target reads0; candidate reads0; hash
checks0. These checks establish interface arithmetic and access behavior, not
scientific reconstruction performance or E0/E1 acceptance. No real TRAIN
representation preparation or target fitting was run by this worker.
