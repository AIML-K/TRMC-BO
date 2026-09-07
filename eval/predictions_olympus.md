# Olympus — predictions registered before the sweep

Written and committed **before** any Olympus campaign was run, and after the
frozen specs and the O1–O5 measurements below. The reason is stated plainly in
`results_issue_matformbench.md`: on MatFormBench the primary metric was promoted
from hit rate to distinct designs *after* seeing the data. That was defensible
(§8.6 already specified diversity, and δ-uniqueness comes from the Range-Aware BO
paper) but it cannot be repeated. Here the metric, the mechanism and the direction
of every expected effect are fixed in advance.

Only 9 smoke cells (branin/medium, 5 iterations, 1 seed) and 18 timing cells
(oer_plate_3496/narrow, 2 seeds) had been run when this was written. Both are
noted below where they bear on a prediction; neither is evidence for one.

## What is already measured, not predicted

| Fact | Value |
| --- | --- |
| Emulator fidelity vs. TFP, noiseless structural check | ≤ 4.5e-6, at the float32 accumulation floor |
| Emulator noise model | gap to TFP falls as 1/√K (3.9×, 4.7×, 2.8× per 16× K) |
| End-to-end R², all rows, Olympus's own protocol | pce10 0.890, wf3 0.936, oer_plate_3496 0.825 (published train 0.925 / 0.971 / 0.824) |
| Branin: three documented minima | all 0.39788736, global over 2M draws |
| Windows: α per width | 0.398 / 0.121 / 0.039 = target / P(y≤τ), the closed form |
| Achieved valid fraction | 9.97–10.04%, 3.00–3.04%, 0.97–0.98% |
| Branin valid components (wide/medium/narrow) | 4 / 3 / 3 at grid ≥ 1400 |
| δ=0.1 capacity of the valid region | branin ≈ 21, pce10 ≈ 75, oer_plate_3496 ≈ 465 |
| Nonlinear constraints attached at m=1 | 2, not 6 |
| Acquisition class at m=1 | qEI throughout, never qEHVI |
| Cost, hardest (task, width) under 18-way contention | 3.4 CPU-min/cell mean; 8.2 min for the slowest method |

## The mechanism under test

With one output, C1 runs through the shipped `CDFRangeObjective` + qEI path — a
path **no previous run has executed**, since every MatFormBench task is m = 3.
Because that objective is a deterministic function of `x`, the expectation over
posterior samples has nothing to average and qEI collapses to its own integrand:

    A_C1(x) = max(0, P(L ≤ f(x) ≤ U) − best_f)

verified to 6e-9 against a direct computation, with candidates below the incumbent
tied at *exactly* zero. Range-Aware TB under the new `box` mapping is the same
quantity with the hinge removed — its acquisition equals `P(L ≤ f(x) ≤ U)` to
6.7e-16, two independent implementations agreeing (a differentiable
Poisson-mixture noncentral χ² against a difference of normal CDFs).

So `M3` vs `M6c` differ in the hinge and nothing else: same target, same
surrogate, same restart count, same restart sampler. MatFormBench could only infer
the hinge's role indirectly (product-in-qEI beat the shipped path 10.00 vs 8.80 on
distinct designs at equal cost, while gradient-everywhere attractors collapsed 27
hits into 1 distinct design). Olympus measures it directly.

## Predictions

**P1 — the hinge trades hit rate for diversity.** `M6c` scores a higher hit rate
than `M3` and *fewer* distinct designs. Falsified if `M6c` matches or beats `M3`
on distinct designs, which would mean the plateau is not what preserved diversity
on MatFormBench and the §4.2 reading needs revisiting.

**P2 — the ablation ladder stays monotone in distinct designs**:
`M3 < M4 < M5`. The hinge leaves the acquisition flat below the incumbent, so C2
and C3 are what convert a flat landscape into spread-out proposals. This is the
component claim, and it must hold on a suite the components were not tuned on.

**P3 — the point-target baseline again wins hit rate and loses distinct designs.**
`M2` is a unimodal attractor with gradients everywhere, so it should reproduce its
MatFormBench signature (11.01 hits → 2.86 distinct, 26% concentration). Its
concentration ratio should again be far below TRMC-BO's 93–97%.

**P4 — extremum-seeking BO is the wrong tool.** `M1` lands at or below `M0` on
distinct designs. On MatFormBench it was 8× worse than random.

**P5 — cost ordering: hinge methods are cheap, no-hinge methods are not.** Ties
below the incumbent stop L-BFGS immediately, so `M3/M4/M5` should cost several
times less per iteration than `M2/M6/M6b/M6c`. *Already visible in the timing
cells* (0.46–1.04 vs 2.75–10.36 s/iter) — recorded here as an explanation held in
advance, not as a finding.

**P6 — Branin's disconnected components separate the methods.** With 3 components
at medium width, a method that converges on one basin caps near a third of the
achievable distinct count. Methods should rank by how many components they touch.

## Registered caveats

**M6, M6b and M6c are near-duplicates on this suite, and that is measured.** Their
three intervals overlap by more than 99% of the window width at every task and
width, because these windows are thin quantile slices of a smooth response so the
valid centroid sits essentially at the box midpoint. On MatFormBench it did not —
the median valid design sat at 0.13–0.18 of the y2 window, and the two mappings
gave visibly different results (1.73 vs 1.37 distinct). Consequences: the
box-to-ball *mapping sensitivity* question is empty here and any M6/M6b gap is
noise, not mapping; the `M3` vs `M6c` contrast is unaffected, since `M6c`'s
interval is the window by construction. All three still run, for table symmetry
with MatFormBench and because showing the mapping does not matter here is itself
worth reporting.

**Olympus contributes almost no input-feasibility difficulty.** pce10's
measurement hull covers 99.89% of its simplex and oer_plate_3496's hull *is* its
simplex, so 0–0.11% of designs are infeasible against MatFormBench L5-4's 65%.
Nothing about sparse feasibility can be claimed from this suite.

**The specification is constructed.** Olympus supplies only `minimize`; τ = q25 of
the response under the uniform measure is our synthesis, frozen in
`specs/olympus/native_targets.json`. τ's placement is not free — at q10 the widest
window would collapse to the one-sided native target.

**Two tasks, not three, are real mixtures**, and `photo_wf3` was dropped for
correlating with `pce10` at r = 0.742 (0.86 where they differ most). It stays
extracted and verified, so that choice is reversible.

---

# Addendum — second mixture task changed after branin, before any mixture sweep

Appended rather than edited into the text above, so the pre-registration stays
readable as what was actually registered. Nothing in P1–P6 changes: they are
statements about methods, not about which datasets they run on. Two of the
**caveats** above are now wrong, and one measurement that was reported as a
property of the suite turns out to be a property of one task.

**What happened.** `oer_plate_3496` was selected as the second mixture task on the
strength of its declared simplex constraint (`"parameters": "simplex"`, and all
2,121 rows sum to exactly 1.0), 6 components and 2,121 measurements. It was
rejected before any of its cells ran, on a property none of the earlier checks
looked at: **all four `oer_plate` datasets mix at most 4 of their 6 components.**
Not one measurement has 5 or 6 non-zero loadings. Uniform Dirichlet draws — our
initial design and our calibration reference measure — almost surely use every
component, so every design evaluated would have sat in a stratum the emulator
never trained on, and its published R² was measured only over 4-or-fewer-component
compositions.

**Why the existing checks missed it.** The convex-hull feasibility test reports
100% feasible there, because the hull of those faces *is* the whole simplex once
all six vertices are present — the same measurement that was written up as
"oer_plate's hull is its simplex, so it exercises no input feasibility" was this
defect, read as an absence of difficulty. Nearest-neighbour distance does not
reveal it either (median 0.098 against a 0.141 data grid), because the nearest
measurement is always on a face. `data.interior_support` now measures it directly
and `test_olympus.py` enforces a floor, with the rejected datasets pinned as the
counter-example.

**Replacement: `colors_bob`** — 5 components, 241 measured dye mixtures,
`difference_to_green`, minimize. Every one of its measurements has all five
components non-zero, so the interior is genuinely sampled. Its emulator passes the
same three-part gate (structure 1.7e-7, noise scaling 7.5x per 16x K, R² 0.847
against published train 0.806 / validation 0.947). It has a different activation
and transform combination from the polymer emulators — `linear`/`sigmoid` with
`identity` transforms, against `leaky_relu`/`relu` with `standardize`→`mean` — which
the loader's guards caught rather than silently mispredicting.

`branin` and `pce10` are untouched: their frozen τ and all six window specs are
byte-identical after regeneration, so the 270 branin and 242 pce10 cells already
completed remain valid.

**Corrections to the registered caveats.**

*"Olympus contributes almost no input-feasibility difficulty"* — true of `branin`
(none) and `pce10` (0.11% infeasible), **false of `colors_bob`**: only 22.6% of
uniform draws lie inside its measurement hull, and a 30-point initial design
yields 4–10 labeled points (median 7). That is a genuine sparse-feasibility
condition, comparable to MatFormBench L4. So the suite does exercise the
feasibility machinery after all, on one of its three tasks.

*The M6/M6b/M6c near-duplication* was measured on the branin and pce10 windows and
must be re-measured on `colors_bob` rather than assumed to carry over.

**One further correction, from the branin run.** The claim that Olympus exercises
no input feasibility was already too strong on `pce10`: `M1_qehvi_extremum` there
records `feasible_rate` 0.792 with a 21% random-fallback rate, against 1.000 and
0% for the other eight methods. Extremum-seeking walks to the simplex vertices,
which is exactly where the measurement hull stops. A rate quoted over a uniform
measure understates what a method that deliberately seeks corners will hit.
