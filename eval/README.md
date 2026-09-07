# eval/ — benchmark harness

Experiments for *Two-Sided Range-Constrained Bayesian Optimization for
Target-Driven Materials Formulation*. The optimizer core ships as compiled `.so`
binaries under `core/` (see the top-level README's "Why part of this is a
compiled binary"); this tree holds benchmarks, datasets, competing methods and
the run infrastructure. Dependency direction is **`eval/` → `core/`, never the
reverse.**

## Environment

MatFormBench ships its oracle and metrics as `.so` modules built for **CPython
3.10 on Linux x86_64**, while the production project pins 3.11. Experiments
therefore run in a dedicated 3.10 environment; `.venv/` (3.11) is untouched.

There is a **second, throwaway** environment, `.venv-olympus`, holding TensorFlow.
It is used once to lift the Olympus emulator weights out of their checkpoints and
is never used at sweep time -- see "Olympus" below for why TF must not be in
`.venv-eval`.

HOIP needs neither: it is a lookup over two CSVs and adds no dependency at all.

```bash
uv venv --python 3.10 .venv-eval
VIRTUAL_ENV=.venv-eval uv pip install -r eval/external/MatFormBench/requirements.txt
VIRTUAL_ENV=.venv-eval uv pip install \
    torch==2.6.0 botorch==0.15.0 gpytorch==1.14 linear-operator==0.6 \
    drs==2.0.0 scikit-learn==1.6.1 scipy==1.15.2 numpy==1.26.4 pandas==2.2.3 \
    threadpoolctl tqdm pyarrow

git clone --depth 1 https://github.com/DeepVerse/MatFormBench.git eval/external/MatFormBench
```

Install order matters: MatFormBench's unpinned requirements pull numpy 2.x, and
the second command pins it back to 1.26.4. The compiled oracle works under both,
but the production GP stack is pinned against 1.26.

Everything runs from the repo root with `core/` and the repo root on the path:

```bash
PYTHONPATH=core:. .venv-eval/bin/python -m eval.scripts.calibrate_ranges
```

## Layout

| Path | Role |
| --- | --- |
| `benchmarks/base.py` | `BenchmarkTask`, `RangeSpec`, `Observation`, `Oracle` protocol |
| `benchmarks/matformbench/` | task loading, oracle wrapper, range calibration, native plugin |
| `benchmarks/calibration.py` | the suite-neutral window rule; `*/ranges.py` are bindings |
| `benchmarks/registry.py` | `resolve(suite)`; the only code that knows which suites exist |
| `benchmarks/olympus/` | tasks, emulator/surface/pool oracles, extracted-weight access |
| `benchmarks/hoip/` | catalogue, lookup oracle with snapping, frozen catalogues and window |
| `methods/` | one file per compared method, all satisfying `methods.base.Method` |
| `methods/range_product.py` | the §4.2 product aggregation, for comparison against the shipped qEHVI-over-vector path |
| `harness/` | adapter, campaign loop, recorder, metrics, runner, aggregation |
| `specs/` | **committed** frozen target windows |
| `external/`, `data/`, `results/` | gitignored: third-party clone, caches, run output |

Adding a benchmark means implementing `Oracle` plus a task loader, and adding one
branch to `benchmarks/registry.resolve`. Adding a method means one file
implementing `propose`. Nothing else in `harness/` changes for either --
`aggregate` already takes `suite=` throughout.

## Protocol decisions

**Observation vs. truth.** Methods only ever see `oracle.observe` — noisy, and
carrying MatFormBench's random per-evaluation batch effect. Hit rate and range
violations are scored on `oracle.truth` (noiseless). Noise is fully reproducible
from the seed, but the batch cannot be pinned: `context={"batch_id": ...}` is
ignored by the compiled oracle, so the batch effect stays an inherent nuisance.

**Initial design.** 30 draws per run, matching `generate_and_export(n=30)`.
Infeasible draws keep their NaN response rather than being replaced — on L5-4
that leaves as few as 2 labeled points, and coping with it is precisely the
sparse-feasibility task. The design is drawn before any method acts and is
identical across methods at a given seed, so its cost is reported, not deducted
from the 50-evaluation optimization budget.

**Target windows.** Frozen ahead of every run by `scripts/calibrate_ranges.py`
and committed under `specs/`. The native one-sided threshold is kept as one
bound; the other is a quantile driven by a single scalar `alpha`, bisected until
the joint valid fraction among *input-feasible* designs hits 10 % / 3 % / 1 %.
Input feasibility and outcome-range validity are reported separately throughout.

## Olympus

Three tasks: `branin` (2-D box, analytical), `pce10` (4-D simplex, 1,040 measured
polymer blends) and `oer_plate_3496` (6-D simplex, 2,121 measured catalysts).
`photo_wf3` is deliberately excluded -- it is the same 1,040 blends as `pce10`
measured with a different polymer, correlating at r = 0.742 (0.86 on the rows
where they differ most), so the two would not be independent evidence.

**Emulators are extracted, not loaded.** Olympus ships a trained BayesNeuralNet
per dataset, but `pip install olymp` is not an option: its core pins
`SQLAlchemy==1.4.45` and `olympus/__init__.py` eagerly imports
`.databases`/`.emulators`, and mlflow (via MatFormBench) uses sqlalchemy too.
Putting TensorFlow in `.venv-eval` is also unacceptable -- it would sit in every
sweep worker competing for threads with 22 single-thread processes, while
`wall_clock_per_iter` is a reported metric. So the weights come out once into
`eval/data/olympus/` and the sweep runs on numpy. Suite isolation is then
structural, and a test asserts an Olympus cell imports neither TensorFlow nor
MatFormBench's extension.

**The specification window is constructed.** Olympus datasets carry only
`default_goal: minimize`. `tau = q25(response)` under the uniform measure becomes
the cap (frozen in `specs/olympus/native_targets.json`) and the shared calibrator
places the lower bound. q25 is forced, not chosen: with one output the calibration
is analytic (`valid fraction = P(y<=tau) * alpha`), so a cap at the target
quantile would drive `alpha` to 1, put the lower bound on the sample minimum and
turn the widest window back into the one-sided native target. Write-ups must say
the specification is constructed.

**`observe` vs `truth`.** The BNN is Bayesian, so it supplies its own noise:
`truth` is the noiseless posterior-mean pass, `observe` one local
reparameterization draw. Branin is **noiseless** -- `observe` is `truth` -- which
is how the Range-Aware BO paper uses it and which separates the acquisition
comparison from noise handling, since every MatFormBench task is noisy.

**What this suite cannot claim.** Input feasibility barely exists here: `pce10`'s
measurement hull covers 99.89% of its simplex and `oer_plate_3496`'s hull *is* its
simplex, against 65% unmanufacturable proposals on MatFormBench L5-4. And
`M6`/`M6b`/`M6c` overlap by >99% of the window width, because thin quantile slices
of a smooth response put the valid centroid at the box midpoint -- so the
box-to-ball mapping-sensitivity question is empty on Olympus, though it was not on
MatFormBench.

**What it can.** At m = 1 the shipped C1 path is `qEI(CDFRangeObjective)`, whose
objective is deterministic in `x`, so qEI collapses to `max(0, P(L<=f<=U) -
best_f)` -- the range probability behind a hinge, tied at exactly zero below the
incumbent. The `box` TB mapping is that same probability without the hinge
(agreeing to 6.7e-16, which also cross-checks two independent implementations of
it). `M3` vs `M6c` therefore isolates the hinge at matched target, surrogate and
restart count. `predictions_olympus.md` registers what that is expected to show,
before the sweep.

## HOIP

Three tasks, and they are **nested restrictions of one design space** rather than
three problems: `dense` (240 materials, 60.0% infeasible), `restricted` (408,
73.5%) and `full` (1,276, 91.3%). The suite is the perovskite application of
*Anubis: Bayesian optimization with unknown feasibility constraints for
scientific experimentation* -- 11 molecular cations x 29 metals x 4 halogens,
with the oracle a lookup in the near-hull DFT table upstream ships. A composition
absent from that table returns `(nan, nan)`: that absence is the unknown input
feasibility the suite exists to measure.

**Why an axis and not a task.** Across MatFormBench and Olympus, TRMC-BO's lead
in distinct qualifying designs holds at 24% and 41% infeasible and reverses at
69% -- on one task, with nothing else testing it. Olympus cannot help (its
mixture tasks are 99.8-100% feasible). The three catalogues bracket that crossing
inside a single chemistry.

**What is held fixed is the point.** Every rectangle the selection rule considers
must contain all seven valid materials of the full catalogue, so the valid set is
*identical* across the three tasks and the delta-separated capacity -- the ceiling
on the primary metric -- does not move along the axis. The valid share among
feasible designs lands at 7.3% / 6.5% / 6.3%. `scripts/freeze_hoip_catalogues.py`
carries the rule; it was frozen before any method ran, which is the same standing
the calibrated windows have on the other two suites.

**The specification is benchmark-supplied, and that is a first.** Upstream's
success test is `abs(bandgap - 1.25) < 0.5 and m_star < 4.`, i.e. `0.75 <= Eg <=
1.75` and `m* <= 4`. MatFormBench gave one-sided thresholds and Olympus gave only
`minimize`, so on both the window had to be constructed. HOIP therefore runs
**one condition, `native`**, and has no wide/medium/narrow axis: window width is
what the other two suites vary, and varying it here would mean discarding the one
window the paper did not have to invent. The raw gap is modelled, not upstream's
folded `abs(gap - 1.25)` -- folding destroys the two-sidedness under test and
gives the surrogate a crease at the target. Nothing is lost: `M2_target_distance`
aiming at the window midpoint *is* the folded objective.

`m*`'s lower bound is 0, which is physical rather than fitted, so C2 attaches
four constraints here (two sides x two properties) and `verify_variants` expects
that number.

**Categorical inputs, continuous path.** The design space is three categorical
choices. It is presented through the benchmark's own descriptors -- 6 per molcat,
4 per metal, 4 per halogen, so 14 dimensions -- which is the representation
Anubis itself found effective (their `desc_` arms), and which lets the production
acquisition path run exactly as it does on the other two suites. The oracle snaps
a proposal to a material **per block** (nearest molcat by its 6 descriptors, and
so on): the inverse of a concatenation is a concatenation of inverses, and a
joint nearest neighbour would let the widest-spread block decide the other two.

`Observation.X` carries the **snapped** design. That one decision is what keeps
the rest of the harness correct without changing it -- the GP trains on real
materials, `loop._is_duplicate` charges a re-proposal of an exhausted material as
the duplicate it is, and `metrics` scores what was evaluated. The relax-then-round
step is measured rather than assumed away: `oracle.diagnostics()` reports the snap
distance and the runner records it in `run.json`.

**Noiseless.** `observe` is `truth`. There is no emulator and no measurement
model, which makes HOIP the one suite whose conclusions cannot be an artifact of a
surrogate -- what the deferred Olympus pool protocol was for.

**Two guarded harness hooks**, both no-ops when an oracle does not publish them,
both pinned by a test so the two finished suites cannot move:
`adapter.BenchmarkVariables(restart_pool=...)` confines acquisition restarts to
the catalogue (the same idea as the Dirichlet mirror for simplex tasks, and it
applies to every method's restarts including random search), and
`oracle.canonicalize` lets the loop's duplicate check compare the design that will
actually be evaluated.

**What this suite cannot claim.** Its infeasibility label conflates two things:
not a stable perovskite, and stable but metallic so the effective mass is
undefined. Eight compositions are in the upstream SI table with a zero gap and no
effective mass, and upstream drops them, so the oracle calls them infeasible. It
changes no hit -- a zero gap is outside the window either way -- but it is what
the label means. Two feasible materials carry a censored effective mass (`>1000`
upstream), held as NaN because a sentinel is not a measurement.

## Interaction with the compiled core

`harness/adapter.BenchmarkVariables` presents a benchmark task with the
attribute surface `run_mobo` expects, so the three components under test run as
shipped rather than being reimplemented. Two coordinate systems are in play:
`item_bound` is raw and ordered `[lower; upper]`; `search_space` is normalized
`[0,1]` and ordered `[upper; lower]` — hence `.flip([0])` at every production
use site. Both conventions are inherited.

`bayesian_optimization.py` (compiled into `core/`) gained small `getattr`-guarded
hooks whose defaults preserve production behaviour exactly:

| Hook | Why |
| --- | --- |
| `semicontinuous_inputs` | Production collapses a coordinate's bounds to `[0,0]` when a restart starts at 0, because in the formulation data a ratio of 0 means "ingredient not added". MatFormBench coordinates are DoE-coded process variables on `[-1,1]` where 0 is the center point, so every coordinate must stay free. |
| `sample_raw_candidates` | The Dirichlet-Rescale restart sampler needs the Excel group/ratio schema, which benchmark tasks have no counterpart for. |
| `_`-prefixed config keys | Harness-only toggles and telemetry, stripped in `run_mobo` before the config is splatted into BoTorch. |

## Running things

```bash
# once: freeze the target windows (writes eval/specs/, committed)
PYTHONPATH=core:. .venv-eval/bin/python -m eval.scripts.calibrate_ranges --suite matformbench
#   --check   recalibrate and compare against the committed specs, writing nothing

# check that the C1/C2/C3 toggles reach BoTorch before trusting any ablation
PYTHONPATH=core:. .venv-eval/bin/python -m eval.scripts.verify_variants

# range-adapted sweep; re-running skips completed cells
PYTHONPATH=core:. .venv-eval/bin/python -m eval.scripts.run_sweep \
    --config eval/configs/matformbench_range_adapted.yaml --workers 22
#   --smoke     one task / one width / 1 seed / 5 evals
#   --dry-run   print the matrix and exit

# native protocol (batch recommend/topk/dss, MatFormBench's own metrics)
PYTHONPATH=core:. .venv-eval/bin/python -m eval.scripts.run_native \
    --tasks L1-1 L2-1 L3-1 L4-1 L5-4 --method M5_full --seeds 0 1 2

PYTHONPATH=core:. .venv-eval/bin/python -m pytest eval/tests -q
```

Olympus, in order. The first two steps need the throwaway TF venv; nothing after
them does.

```bash
git clone --depth 1 https://github.com/aspuru-guzik-group/olympus.git eval/external/olympus
uv venv --python 3.10 .venv-olympus
VIRTUAL_ENV=.venv-olympus uv pip install "tensorflow-cpu==2.15.*" \
    "tensorflow-probability==0.23.0" numpy==1.26.4

# lift the weights out of the checkpoints, then gate them against TFP
.venv-olympus/bin/python eval/scripts/extract_olympus_emulator.py \
    --datasets photo_pce10 photo_wf3 oer_plate_3496
.venv-olympus/bin/python eval/scripts/verify_olympus_emulator.py

# freeze the synthesized caps, then the windows (both committed)
PYTHONPATH=core:. .venv-eval/bin/python -m eval.scripts.synthesize_olympus_targets
PYTHONPATH=core:. .venv-eval/bin/python -m eval.scripts.calibrate_ranges --suite olympus

# what the valid region is, before any method runs: components and delta-capacity
PYTHONPATH=core:. .venv-eval/bin/python -m eval.scripts.valid_region_geometry --suite olympus

# toggles must reach the optimizer: expect 2 nonlinear constraints and qEI, not qEHVI
PYTHONPATH=core:. .venv-eval/bin/python -m eval.scripts.verify_variants \
    --suite olympus --task pce10

PYTHONPATH=core:. .venv-eval/bin/python -m eval.scripts.run_sweep \
    --config eval/configs/olympus_branin.yaml --workers 22
PYTHONPATH=core:. .venv-eval/bin/python -m eval.scripts.run_sweep \
    --config eval/configs/olympus_emulator.yaml --workers 22
```

HOIP, in order. No third-party environment is involved at any step.

```bash
# fetch the two CSVs and the descriptors; the shipped pickles are opened once
# here and re-emitted as JSON, never unpickled at sweep time
PYTHONPATH=core:. .venv-eval/bin/python -m eval.scripts.fetch_hoip_data

# check the catalogue against its own data before building on it
PYTHONPATH=core:. .venv-eval/bin/python -m eval.scripts.audit_hoip_catalogue

# freeze the feasibility ladder, then the windows (both committed)
PYTHONPATH=core:. .venv-eval/bin/python -m eval.scripts.freeze_hoip_catalogues
PYTHONPATH=core:. .venv-eval/bin/python -m eval.scripts.freeze_hoip_specs
#   --check on either: re-derive and compare, writing nothing

# exact, not sampled: the design space is enumerated
PYTHONPATH=core:. .venv-eval/bin/python -m eval.scripts.valid_region_geometry --suite hoip

# expect qEHVI / CDFRangeMultiOutputObjective and nonlinear=4 on M4/M5
PYTHONPATH=core:. .venv-eval/bin/python -m eval.scripts.verify_variants \
    --suite hoip --task dense

PYTHONPATH=core:. .venv-eval/bin/python -m eval.scripts.run_sweep \
    --config eval/configs/hoip_catalogue.yaml --workers 22
```

### Diagnostics

These exist because three separate claims about the results were argued in a chat
window before anyone could re-run them. Each is cheap and each answers one
question.

```bash
# Can any method see the answer key? Also: same initial design, same budget,
# and an inventory of which target_info fields each method actually reads.
# Run it on one task per geometry -- the constraint plumbing differs.
PYTHONPATH=core:. .venv-eval/bin/python -m eval.scripts.audit_leakage --task L4-1

# Does an acquisition rank passing candidates above failing ones? One GP fit per
# setting instead of one per iteration, so 15 settings x 5 seeds takes minutes.
# A one-step ranking; it says nothing about 50-iteration exploration.
PYTHONPATH=core:. .venv-eval/bin/python -m eval.scripts.acquisition_auc --seeds 0 1 2 3 4

# Do the windows themselves favour a midpoint-aiming baseline? Measures where the
# joint valid set sits inside its window and whether that predicts the baseline's
# margin. Withholds a verdict below 10 cells: on 5 cells it reverses.
PYTHONPATH=core:. .venv-eval/bin/python -m eval.scripts.window_geometry

# Where each method's qualifying designs landed, projected on a shared plane.
PYTHONPATH=core:. .venv-eval/bin/python -m eval.scripts.plot_distinct_designs \
    --task L3-1 --width medium

# How much of the design space the range acquisition cannot see, at any m.
# `plateau_fraction.py` answers this for one output only -- it hardcodes
# `mean[..., 0]` and a scalar window -- and the multi-output case turned out not
# to be safe, so this measures both conditions a plateau needs: whether the
# objective is deterministic in x, and whether the observed front dominates the
# rest of the space. The second is where the noise model enters.
PYTHONPATH=core:. .venv-eval/bin/python -m eval.scripts.acquisition_plateau \
    --suite hoip --task dense
```

`--smoke` writes to `results/_smoke/`, not into the real result tree. It produces
complete-looking cells at budget 5, and the sweep's resume marker is the presence
of `run.json` -- so a shakedown in the real tree would make the next full sweep
skip exactly the cells it had touched. (Checked: all 2,424 completed MatFormBench
and Olympus cells are budget 50, so this never happened.)

`eval/tests/test_fairness.py` runs the audit's four invariants on one small task,
so a protocol regression fails in CI rather than at the next audit.

Do not run a sweep and anything else CPU-heavy at the same time:
`wall_clock_per_iter` is a reported metric, and contention inflates it.

## Upstream defects

### MatFormBench

Two problems in the published repository, both worked around without modifying
it. Worth stating in any write-up, because they affect comparability.

**The published runner cannot execute.** `team_test_v2.main` calls
`_metrics1(data=validate_df, ...)` and `_metrics3(...)`, but the shipped compiled
metrics declare `data: str` and reject a DataFrame. Their own error text
mentions a DataFrame branch, but the Cython signature makes it unreachable.
This is not specific to our algorithm -- their own `algo_card_GA_GPR.yaml` fails
on the identical line. `_metrics6` additionally passes `targets` as a list where
`str` is declared. `eval/scripts/run_native.py` wraps both functions to spill a
frame to a temporary CSV and to JSON-encode list targets; the metric computation
itself is untouched. Their published leaderboard numbers were presumably
produced by a different internal version, so we do not claim direct comparison
against them.

**The native protocol writes into the clone.** `generate_and_export` regenerates
`synthetic_data/<level>/<dataset>.csv` in place, so running `run_native` leaves
the third-party checkout dirty. It is deterministic given the seed, and nothing
in the range-adapted path reads those CSVs (the oracle is called directly), so
results are unaffected -- but `eval/external/MatFormBench` is not pristine after
a native run, and `git pull` there will conflict. Delete and re-clone if that
matters. Our *code* still never modifies their repository; their own exporter
does.

**`tabpfn` is an undeclared dependency.** Without it their Ridge+TabPFN ensemble
scorer silently degrades to Ridge-only behind a warning, which changes every
score. It is installed explicitly above.

### Olympus

Four, all documented rather than worked around silently. The first two change what
can be claimed about the emulators; the last two cost a selected task each.

**`Dataset` splits with `random_seed=None`.** It calls `np.random.seed(None)`, so
the train/validation split behind every published emulator R2 is not reproducible
even in principle. Emulator fidelity is therefore established against TFP directly
-- structure to float32 round-off, sampled distribution converging as 1/sqrt(K) --
rather than by reproducing their number. `eval/scripts/verify_olympus_emulator.py`.

**The aleatoric noise scale is indexed by batch position.**
`wrapper_bayes_neural_net` builds it as
`softplus(Variable(tf.ones(y_pred.get_shape())))`, and `y_pred` is shaped
`(batch_size, targets_dim)`, so the observation noise is learned per *position
within the prediction batch*: the checkpoints hold 50 distinct values
(photo_pce10 sigma 0.196-0.232). The same design gets a different aleatoric scale
depending on where it lands in a batch. It never enters the returned prediction,
so it costs us nothing; it is extracted and recorded, and unused.

**`data.csv` has no header row.** Reading one silently drops a measurement
(1040 -> 1039) and shifts nothing else.

**The declared input constraint is unreliable in both directions.**
`scripts/audit_olympus_datasets.py` therefore ignores it and searches the
measurements for an affine dependency -- a column subset whose row-sum is constant
is an indicator vector in the null space of the centred design matrix. Of 43
datasets, 8 have such a subset and only 3 are usable:

* `colors_bob` **declares `"parameters": "simplex"` and is not one.** Not a single
  one of its 241 rows sums to 1 -- they range 0.64 to 4.40, median 2.68 -- because
  its five features are independent dye volumes, not composition fractions. (Its
  description is also the placeholder "Color mixing dataset. Blablabla".)
  Trusting the declaration confines the search to a 4-D slice of a box the
  emulator was trained across, and makes any hull test that drops a coordinate --
  valid only on a simplex -- project a box-shaped cloud. A 270-cell sweep was run
  and discarded before this was found.
* `p3ht` **declares `"none"` and is a mixture**: five composition percentages
  summing to 100 +- 0.1. Olympus simply missed it.
* the four `oer_plate` datasets declare simplex correctly but **have no interior**
  -- at most 4 of their 6 components are ever non-zero, so a uniform Dirichlet
  design queries a stratum with no training data, and their published R2 was
  measured only over 4-or-fewer-component compositions.
* `p3ht`'s emulator additionally **predicts exactly 0 for 41.6% of its own 178
  training designs**, though not one measured conductivity is 0: its `relu` output
  flattens negative pre-activations. Over a uniform measure 76.6% of hull-interior
  draws return 0, so its synthesized window calibrated to the empty interval
  `[0, 0]`.

A convex-hull feasibility test passes all of these -- for `colors_bob` it is
computed in the wrong space, and for `oer_plate` the hull of those faces is the
whole simplex. `benchmarks/olympus/tasks._measured_simplex_total` therefore checks
the declaration against the raw measurements (constant sum, `data.interior_support`,
`data.activation_floor_rate`), with the rejected datasets pinned as
counter-examples in `test_olympus.py`.

## Status

All three suites are complete: MatFormBench (1,344 cells), Olympus (1,200 cells,
+120 for M7_scbo), HOIP (420 cells, +60 for M8_comboo/M9_anubis) -- 2,964 cells
total, 0 errors. Environment, task loading, oracle wrappers, adapter, campaign
loop, range calibration, C1/C2/C3 variant toggles with a verification script,
instrumentation counters, metrics, sixteen registered methods (nine to fourteen
run per suite, per what each suite's outputs support and which of COMBOO/SCBO's
output-arity guards apply), sweep runner with resume, aggregation, native-protocol
plugin (30 results), the fairness audit and the acquisition/window/plateau
diagnostics.

Campaigns are reproducible as of the torch seeding in `harness/loop.py`. The
1200-cell MatFormBench sweep predates it: its statistics stand, but rerunning a
cell will not reproduce that cell's exact coordinates. Olympus and HOIP postdate
it and are exactly reproducible.

Stopped by design: the §4.2 product-aggregation sweep on MatFormBench, at 44 of
300 cells. It had already answered its question -- the specified product
acquisition converges onto a single basin, so the shipped hypervolume aggregation
is the one that matches the paper's diversity claim. `configs/matformbench_product.yaml`
carries the detail and resumes from cell 45 if wanted. HOIP's product variants ran
to completion (§ HOIP) and did not reproduce the collapse on that suite.

**HOIP's headline result is negative, and that is the finding.** The ladder was
built to locate where MatFormBench L5-4's distinct-design reversal (measured at
69% input-infeasible) falls on a second chemistry. It could not: TRMC-BO has no
distinct-design advantage over random search on HOIP's easiest rung (60%
infeasible), so there is no margin to track as infeasibility rises. The measured
cause is `m_star` -- unlearnable from composition at any kernel or representation
tried (held-out R² -0.85 to +0.10) -- which collapses the acquisition to a
near-flat surface and starves C3's in-band restart filter, so a large share of
what looked like a method's proposals were the loop's random-fallback safety net,
not the acquisition. `results_issue_hoip.md` has the full account, including a
non-fallback-only recount that erases most of the apparent wins.

**Update (2026-08-26): COMBOO, SCBO and Anubis added.** COMBOO's box-to-COMBOO
mapping bug is fixed (the hypervolume reference is now a low quantile of the
observed distribution, separated from the constraint threshold) and a second,
previously-masked bug it exposed -- the feasibility probe only ever checked one
side of the two-sided window, so its optimum was often not a valid seed for
stage 2's constrained solve -- is fixed too (`TwoSidedAuxiliaryUCB`). Re-run on
MatFormBench's 5D tasks (90 cells) and added to HOIP (30 cells): COMBOO still
loses badly to random search on MatFormBench (its probe now correctly declares
infeasibility on 34-96% of iterations, whichever tasks it can afford -- L4-1/L5-4
remain out of reach on cost alone) but *beats* both random search and
TRMC-BO's own M5_full on HOIP's two easier infeasibility rungs, losing only on
the hardest one -- see the "COMBOO 재개" / "부록" sections of
`results_issue_matformbench.md` / `results_issue_hoip.md`.

SCBO (Eriksson & Poloczek 2021, trust-region constrained BO) is implemented
(`methods/scbo.py`) and registered for Olympus's three single-output tasks --
exactly where COMBOO cannot run (it needs m>=2), so the two baselines now cover
complementary output arities. SCBO beats TRMC-BO's M5_full on every Olympus
task (30 cells x 4 tasks) and even beats the point-target baseline on `wf3` --
see `results_issue_olympus.md`'s appendix.

Anubis (Hickman et al. 2025, feasibility-classifier-weighted acquisition) is
implemented (`methods/anubis.py`, naive multiplicative combination only) and
registered for HOIP (the suite whose own source paper it is), run across all
three catalogues (30 cells). It shows no clear advantage over TRMC-BO --
plausibly because only the naive combination rule was implemented, not the
paper's more elaborate FIA/FCA variants.

alpha-GaBO remains excluded (not deferred) for the reason recorded above: no
public code exists. The Olympus pool protocol (model-assumption-free lookup
oracle) remains deferred.

**Post-hoc bug review (2026-08-26).** A second pass over all three new/touched
methods found three more issues, all fixed and re-verified:

- `anubis.py`: with zero feasible observations, every catalogue row scored
  identically (the bootstrap classifier), so `argsort` returned the exact same
  top-q indices every call -- on a deterministic lookup oracle this got the
  campaign stuck proposing one already-infeasible material for the entire
  budget. Fixed by excluding already-observed catalogue rows from selection
  (`_unexplored_mask`). HOIP's 30 M9_anubis cells were re-run; numbers shifted
  (mostly down) but the qualitative picture is the same as before the fix.
- `scbo.py`: the initial design was folded into the trust-region's
  success/failure bookkeeping as a single verdict, and a restart reset the
  trust-region length/counters but not its center, so it re-converged on the
  same incumbent instead of exploring elsewhere (restarts fire ~once per
  50-iteration campaign, not a rare case). Both fixed; Olympus's 120 M7_scbo
  cells were re-run -- numbers moved only slightly, SCBO still beats
  TRMC-BO's M5_full on every task.
- `comboo.py`: two latent crash risks that never fired in the completed 90+30
  cells (an all-NaN output column would crash `fit_gpytorch_mll`; `q != 1`
  would crash on a reshape) are now guarded rather than silently landmined --
  no re-run needed since neither affected the reported numbers.

`eval/tests/test_{anubis,scbo,comboo}.py` gained regression tests for all five
issues; 319 tests pass.

**Update (2026-08-27): C4, the feasibility component, added to the proposed
method.** The ladder had no channel through which "this design cannot be made at
all" could reach the search, and the production pipeline's binary/discrete ML
path cannot supply one: `ML_block.ml_inference` runs on `candidate_df`, i.e.
*after* `run_mobo` has chosen; `MLclassification.AutoClassifier` trains on binary
*property* labels, which an unevaluable design does not have; and
`initialize_model` drops all-NaN rows from every GP, so a failed evaluation
leaves no trace anywhere in C1/C2/C3.

C4 borrows Anubis's mechanism -- an online `P(feasible|x)` classifier over the
attempted designs -- and applies it in two places, both opt-in:

    C4a  feasibility-weighted ranking    score = P(feasible|x) * acq(x)
    C4b  feasibility-filtered restarts   keep the most-feasible restarts drawn

C1-C3 are statements about the outcome space (is `f(x)` in its window); C4 is a
statement about the input space (can `x` be evaluated). The axes are orthogonal,
so C4 multiplies a factor onto the ladder rather than replacing a rung of it.
Shared code lives in `utils/feasibility.py` (compiled into `core/utils/`) and `methods/anubis.py` now
imports it, so `M10`-`M12` vs `M9` isolates *where* the signal is applied rather
than *which* classifier produced it.

Two guards, both load-bearing and both tested: the naive product presumes a
non-negative acquisition, and would rank the *least* feasible point first if a
caller-supplied `_acqf_factory` went negative; and at `m == 1` the acquisition is
exactly 0 everywhere once any in-window point is observed, which makes the
product carry no ordering at all (it then ranks by feasibility alone).

**C4b selects by rank, not by a probability threshold, and that is a measured
correction rather than a design preference.** It was first built as
`P(feasible|x) >= tau` with `tau = 0.1`, chosen defensively so as not to starve
C3 further. Single 50-iteration verification cells showed it passed **1,280,000 of
1,280,000** drawn restarts across HOIP/dense and MatFormBench/L5-4 -- inert. A
2,000-draw probe at 30 observations explains why: the classifier's probabilities
are narrow, their *location* is problem-dependent, and they frequently collapse
to a constant. HOIP/dense spans 0.279-0.589; HOIP/full is identically 1.0 (one
feasible observation, so the bootstrap fallback is in force);
MatFormBench/L5-4 is identically 0.139. A threshold keeping 17% on HOIP/dense
keeps 0% on MatFormBench. Ranking within each drawn batch is invariant to all
three, and degrades correctly when the classifier knows nothing: a constant `P`
puts every row at the quantile, so every row survives. The default keeps half.

The same cells put the two filters in proportion: C3
admitted 2 of 256,000 draws on MatFormBench/L5-4 (0.0008%) and 13,983 of
1,024,000 on HOIP/dense (1.4%). C3 is the binding constraint by orders of
magnitude; C4b biases the pool C3 then cuts. `restart_passed_mean_range` and
`restart_passed_feasibility` are recorded per campaign so the two are never
confounded again.

**C4 is off by default and verifiably inert when off.** `M5_full` on
`hoip/catalogue/dense/native/seed000` was re-run against its committed cell: all
80 trace rows and every metric identical except wall-clock. The 2,964 existing
cells stand unchanged.

After the switch, C4b admits exactly 50.00% of draws on both suites -- it is
live. Notably, halving the pool by feasibility rank cost only 0.5% of the
*C3-admissible* restarts on HOIP/dense (13,983 -> 13,917 of 1,024,000 draws):
the in-band region and the high-feasibility region largely coincide there, so
the two filters are not fighting each other.

**What the verification cells could not measure.** On both hard cells the
downstream fallbacks dominate so completely that neither C4a nor C4b can reach
the outcome. HOIP/dense runs at `random_fallback_rate = 1.0` -- every proposal is
the loop's safety net, so M5_full, M10 and M11 produce byte-identical metrics and
the method's own candidate is never used. MatFormBench/L5-4 hits
`restart_filter_exhausted` on 49 of 50 iterations (C3 admitted 2 restarts in the
entire campaign), so the soft-fallback path -- which drops the filter -- runs
almost every iteration. Both are the contamination `results_issue_hoip.md` §3.2
warns about, at its limit. Whatever C4 is worth, the hardest rungs are not where
it can be read off.

**Where it can be read off, and a first single-seed signal.** MatFormBench/L3-1
is the cell that has both halves: genuine infeasibility (7-8 of 30 initial rows
across seeds 0-4) and a healthy loop (`random_fallback_rate = 0.0`, no filter
exhaustion). At `medium`, seed 0:

| | hit_rate | first_hit | diversity_norm |
|---|---|---|---|
| `M5_full` (C1+C2+C3) | 0.30 | 7 | 0.0714 |
| `M10_full_feasw` (+C4a) | 0.44 | 8 | 0.0627 |
| `M12_full_feas` (+C4a+C4b) | 0.44 | 8 | 0.0586 |

**One seed, one task, one width -- suggestive, not a result.** Reported because
it is the first cell in which C4 could act at all, and because it locates where
the sweep is worth spending: the middle rungs, not the hardest ones. Two
readings to carry into it. C4a and C4b land on the same hit rate, so on this cell
the ranking carries the effect and the restart filter adds nothing on top --
which is what `restart_passed_*` says too, C3 admitting 0.64% of draws against
C4b's 50%. And `feas_weight_acqf_degenerate` fired on 29 of 50 iterations, so
more than half of C4a's influence arrives through the flat guard -- ranking by
`P(feasible|x)` alone where the acquisition had no ordering to give -- rather
than through the product. Diversity moves the other way, slightly, in both.

New variants `M10_full_feasw` (C4a), `M11_full_feasf` (C4b) and `M12_full_feas`
(both) are registered and added to the HOIP and MatFormBench configs. The full
sweep has not been run.

