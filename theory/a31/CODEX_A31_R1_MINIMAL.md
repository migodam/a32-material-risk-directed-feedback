# A31-R1 minimal execution specification

## Objective and decision

Test ONE hypothesis: replacing A17's full-GN-gap acceptance with a material-risk
metric, using a source-complete paired feedback proposal and one-HVP material
dual verification, yields better final materials at matched total cost.
Do not assume the hypothesis is true. Do not implement OPM, a current candidate
search, new networks, multi-topology generative inference, or uniform Neumann
certification. The earlier A31 DC-MDI execution plan is superseded for this task.

Read CORE_THEOREM.md and a31_r1_checks.py in this package first.

## Required fixed sources

- A30: migodam/a30-maxwell-observable-material-manifold,
  commit 717d3d26a290248bd2b9aeb55f3d03694b407b84.
  Read A30_E0_E1_DECISION.md, src/a30/capacity.py,
  src/a30/representations.py, src/a30/physics.py, src/a30/budget.py,
  and the corresponding saved capacity/anchor manifests.
- A17: migodam/a17-maxwell-certified-exchange,
  commit df61a1b533682a883cdaf65259b4528c354bd832.
  Read closure_final/manuscript/CORE_THEORY.tex,
  efficiency_followup/GPT_REVIEW_CONTEXT.md and the existing adapter contracts.
- A31 original: source-field derivative defect, not its old research verdict.

Missing files/backend modules: mark missing and stop that stage. Do not silently
rebuild missing cached full states, create new labels, search locked folders,
run inherited physics fixtures, or change the frozen backend.

## Non-negotiable interpretation of A30

Sparse-64's 35.69% and PCA-64's 21.91% are offline achieved capacity gains.
They are not online solver gains. Best-of-eight oracle starts/supports are not
available to the online algorithm. The -3.32% L-weighted bank result is NOT
failure of continuous coefficient fitting with L.
The implemented compensation and uniform-certificate routes did not pass.
Do not silently reactivate them.

## Representation and prior

1. Primary: the frozen TRAIN-only PCA-64 material subspace.
2. Secondary diagnostic: Sparse-64 with ONE deterministic online support from
   the saved A30 rule, namely the first eligible support in frozen order.
   Never load best_start_id or oracle coefficients to choose it. Report that
   this may lose some of the reported best-of-starts oracle headroom.
3. Define the global material class as x=background+T*c, with real material
   mass coordinates. Confirm T^T W T and freeze T within a trajectory.
   Start from the box-feasible material-metric projection of N3 into this
   class. Anchors a=background+T*c_t must stay in this SAME class. Do not
   retain an unmodeled data-dependent N3 residual outside the class. At each
   anchor z=c-c_t, recenter the SAME absolute prior, m=prior_mean-c_t.
   A cached F(N3) is NOT F(projected_N3); any new anchor solve is counted.
4. Fit a proper Gaussian/shrinkage coefficient prior from TRAIN only; freeze
   shrinkage and scaling before scoring any development material. Keep the
   actual material box. Keep optimizer damping separate from prior precision.
5. A data-selected sparse support gives conditional working-model uncertainty,
   not a globally calibrated posterior. PCA is the primary risk-theory test.
6. Do not recycle the preceding posterior as a prior and multiply the same
   data again. Reanchoring is numerical solution of one fixed inverse problem.
   No new weak-direction prior is multiplied in after seeing y. Any observable/
   weak decomposition is a coordinate re-expression of the SAME frozen prior.

## Algorithm PMRD: paired material-risk damping

At each physical anchor a:

A. Obtain the full source fields and full data residual. Compute the ordinary
   full adjoint gradient once. Form h=J_F^T r+Lambda*m without assembling full J.
B. Baseline J0 is the frozen calibrated-L slope. Build a cheap source-complete
   richer tangent Jhat from A17's FIXED receiver backbone and fixed richer
   Schur anchor. Reuse the original documented ranks and bank-construction
   rule; no one-out/one-in search, new current shortlist, rank sweep, or OPM.
   Refresh material-dependent Schur blocks and total-field injection at the
   actual state. Correct any source-field injection mismatch explicitly.
C. Use J0 and Jhat with the SAME exact h, prior, trust damping, physical box,
   and material metric to form baseline and paired trial displacements.
   Solve the small constrained coefficient problems, not full-grid currents.
   Let b=baseline displacement, d=paired displacement-b.
D. Compute qhat=(Jhat^T Jhat+Lambda)^{-1} M*d. The covariance here does not
   include trust-region damping. Apply one native JVP and tied native VJP to
   compute H_F*qhat. Form rho=M*d-H_F*qhat and Bdual.
E. For the actual experimental gate use the local-Gaussian lower-bound rule
   alpha=clip((Ahat-Bdual)/(2*d^T M*d),0,1), with alpha=0 for zero d.
   Record status LOCAL_GAUSSIAN_ONLY unless a separately valid nonlinear,
   boundary, and prior remainder bound is supplied. Do NOT label this a
   certified nonlinear posterior/material decision.
F. Trial image is a+T*(b+alpha*d). Evaluate the common full nonlinear objective
   as a safeguard. At most one half-step retry. A failed guard falls back to
   the declared baseline action/line search, not an uncounted solver.
G. Save source-field defect diagnostics and finite endpoint Taylor remainder.
   The latter is NOT a posterior-moment or uniform remainder certificate.
   Do not use truth for alpha, stopping, prior, support, or candidate generation.

Alpha=0 must retain the matched baseline update; it must NOT stop the whole
optimizer or produce a misleading advantage by always returning the old N3.

## Strong controls (same representation, prior, constraints)

- CL: one-anchor calibrated-L coefficient fit. Do not replace it with old bank
  weights; solve the actual continuous constrained problem.
- Full reduced-coordinate GN: exact full Jacobian in the same 64D material
  chart, assembled with reusable receiver adjoints when cheapest.
- Generic one-HVP material PCG/quasi-Newton: same full gradient and same extra
  HVP budget as PMRD, but no A17 Schur physics. This is essential to distinguish
  the proposed mechanism from ordinary material preconditioning.
- PMRD: paired Schur proposal plus material-risk gate.

In the shadow stage only, also score A17-style full quadratic gain and
always-accept paired proposals. No extra trajectory arms are automatically
licensed by adding diagnostics.

## Stage 0: algebra and zero-new-physics feasibility

Run a31_r1_checks.py. Reproduce paired mean/covariance support, exact local risk,
one-HVP dual bound, the nonlinear score identity, and BOTH counterexamples.
Use arithmetic/fake ledgers for budget tests. No native Maxwell fixtures here.
Audit whether all required A30 anchors and A17 operators are available.
Fit the continuous CL baseline and freeze parameter/box conventions without
reading offline targets in runtime. Profile cached algebra separately from
fresh physical actions. Do not estimate speedup from FLOPs or state counts.

## Stage 1: shadow audit, <=48 new 2D state evaluations

Use four metadata-preselected exposed parents:
250065 shell, 250069 asymmetric, 250077 mixed-scale, 250073 smooth control.
Use weak and ring backgrounds for each, original amplitude/noise convention.
PCA-64 and the declared Sparse-64 give 16 material-coordinate cases.
All are exposed development, not blind validation.

- At each representation-specific projected N3 anchor, form baseline/paired
  proposals before offline truth scoring. Reuse an existing state only if
  the material and physical configuration actually match. Up to 16 new
  projected-anchor states plus 32 baseline/paired endpoints fill the 48-call
  cap; failure/retry costs consume that same cap and may leave explicit
  missing endpoints. Cache fields only through the declared online adapter.
- Predictor sees h, cheap factors, and its one native HVP. It cannot read an
  exact assembled J/H or the exact local posterior mean from the audit process.
- Separate audit may assemble J, solve the exact 64D local Gaussian, and
  compare predicted signed gains with exact local Gaussian material gains.
  Charge this as AUDIT cost, never disguise it as online-free information.
- Audit source-field term, rank after real/source lifting, true HVP residual,
  effective paired material rank, physical box activity, numerical precision,
  and all boundary/nonlinear status labels.
- Evaluate baseline and full paired endpoints in real F to inspect finite
  source/field curvature. Truth is used only by a separate endpoint evaluator.
- Max 48 new full-state calls includes cache regeneration explicitly admitted
  by the ledger, endpoint checks, failed starts, retries and real fixtures.
  Missing inputs without a reserved action remain NOT_RUN.

Report exact local-Gaussian false acceptance separately from realized
single-object material error. An expected-risk bound does not guarantee that
every individual truth improves. Test whether predicted group-average gains
track realized group-average gains; do not fit a predictor on these same
scores and then call the test independent.

Stage-1 decisive stops:
- local algebra violates the validated bound beyond the declared FP tolerance;
- the source-field term is omitted or cannot be priced accurately;
- fewer than 50% of nonzero primary PCA proposals have a resolved sign
  (abs(predicted local gain)>Bdual); all-zero proposals also fail the
  feedback-value gate. This 50% is an engineering screen, not a theorem;
- paired directions offer no improvement over the generic one-HVP control;
- native profiling already erases the possible budget advantage.
Save negative evidence; do not fix failures by increasing rank or prior
precision. A promising but uncertified score is a different, explicitly
exploratory finding, not a certificate pass.

## Stage 2: short nonlinear trajectories, <=80 additional 2D calls

Only after the Stage-1 rule is frozen and its mechanism is not rejected.
Use the same four preselected parents, weak background, PCA-64 primary.
Run at most three outer attempts per nonlinear arm, at most one half-step
retry each. Controls are CL, full reduced-coordinate GN, generic one-HVP
material control, and PMRD. CL is an actual continuous solver and receives
its charged final full-F evaluation. Use common representation-projected starts and constraints, and report
original N3 quality separately so a worsened projection is not hidden.
This is a mechanism pilot, not a new independent statistical validation.

Primary metric: fixed mass-weighted squared complex material error.
Secondary: A30 delta-NRMSE, real/imag errors, support IoU, edge F1, topology
statistics. Residual and GN-gap are diagnostic only. Use the same physical
loss for predicted and realized signed gain; relative NRMSE is not identical
to the posterior squared-risk objective.

Predeclared engineering evidence target (not a mathematical constant):
- at matched inclusive work, positive mean complex-material improvement on
  the three complex parents vs BOTH CL and the generic material control;
- to expand, target >=5% parent-mean relative material MSE improvement, or
  <=2% material-MSE degradation vs exact chart-GN with >=20% lower inclusive
  time. Keep per-parent failures; no claim of significance from three parents.
- if CL matches full F in material quality, drop the feedback-specific claim.
- if full GN or one-HVP PCG is better, report that, preserve A17 as a standalone
  representation/design paper, and stop this inference mechanism.

Total Stage 1+2 cap: 128 new 2D full-state evaluations; <=8192 complex physics
RHS; CPU charged time <=1800 s. All are resource ceilings, NOT predictions.
No training and no locked data. Do not reset the ledger between stages.

## Conditional 3D check (not an automatic new campaign)

Only if 2D passes the declared gate. Use two existing A17 3D cached objects,
one smooth and one shell/near-contact, and one predeclared anchor each.
Verify vector realification, mass metric, source-specific fields and the
CM/RR polarizability derivative. Repeat a single paired decision, exact local
material-risk audit, and counted endpoint check. No 3D training/trajectory.
At most 8 new full states and <=600 s GPU-occupied time; whichever cap is
reached first stops the stage. Missing cache/backend is NOT_RUN, not permission
to create labels or to enlarge the budget. A pass is a vector-mechanism check,
not a 3D imaging superiority result.

## Cost contract and outputs

For 8 sources, the incremental one-HVP audit normally costs 8 tangent plus
8 adjoint complex RHS, not one RHS. A complete 32-receiver Jacobian may cost
only 32 receiver-adjoint RHS shared across sources. The full gradient can be
8 adjoint RHS; after a common anchor, PMRD might use 24 RHS versus 32 for
full-J construction, before Schur/setup costs. This is at most a count-based
opportunity, not predicted latency. Compare native batched solves, LU reuse,
source/receiver changes, memory, transfers, certificate assembly and fallback.

Output:
- A31_R1_DECISION.md
- SIGNED_RISK_AUDIT.csv
- NONLINEAR_MATERIAL_TRAJECTORIES.csv
- SOURCE_FIELD_DEFECT_AUDIT.json
- COST_LEDGER.jsonl and FAILURE_LEDGER.jsonl
- saved online proposals BEFORE offline target scoring

For each row label: EXACT_LOCAL_GAUSSIAN / CONDITIONAL_THEOREM /
EMPIRICAL_NONLINEAR / UNRESOLVED_BOUNDARY / NOT_RUN.
Do not average away missing conditions. Do not call count savings speedup.
