# HOIP — predictions registered before the sweep

Written and frozen **before** any HOIP campaign was run — this file reached its
final state before the 360-cell matrix in `configs/hoip_catalogue.yaml` was
started, and is committed together with the suite that produced it. The reason is the
one stated in `results_issue_matformbench.md` and repeated for Olympus: on
MatFormBench the primary metric was promoted from hit rate to distinct designs
*after* seeing the data. That was defensible — §8.6 already specified diversity,
and δ-uniqueness comes from the Range-Aware BO paper — but it cannot be repeated.
Here the metric, the mechanism and the direction of every expected effect are
fixed in advance.

Only the Stage-0 audit and a **single-seed protocol probe** had been run when this
was written. The probe is reported in full below, and where it bears on a
prediction that is said explicitly. One seed on one task is not evidence for a
prediction; it is a check that the protocol runs at all, and hiding it would be
worse than declaring it.

## What is already measured, not predicted

Exact counts, not estimates — the design space is finite and was enumerated.

| Fact | Value |
| --- | --- |
| Declared design space | 11 molcats × 29 metals × 4 halogens = 1,276 materials |
| Feasible (present in upstream's near-hull DFT table) | 111, i.e. **91.3% infeasible** |
| Valid under the native window | **7 materials**, 6.3% of feasible, 0.55% of the space |
| Ladder infeasibility (dense / restricted / full) | 60.0% / 73.5% / 91.3% |
| Ladder valid share of feasible | 7.3% / 6.5% / 6.3% |
| Valid set across the three tasks | **identical** — the same 7 materials, by construction |
| Campaign coverage (80 evaluations) | 33.3% / 19.6% / 6.3% |
| Catalogues nested | yes: dense ⊂ restricted ⊂ full |
| Composition parse ambiguities | 0 — `molcat+metal+halogen` is injective over all 1,276 |
| si_table compositions our parse cannot resolve | 0 |
| Compositions upstream drops as infeasible although computed | 8, all metallic (PBE gap 0.00, no effective mass) |
| Censored effective mass (`>1000`) | 2 of 111, held as NaN |
| Acquisition class | qEHVI throughout (m = 2), so Olympus's m = 1 hinge cannot apply — but see the collapse rows below, which are a *different* route to the same flatness |
| Nonlinear constraints attached on M4/M5 | **4** (two sides × two properties), C3 filtered on M5 only |
| Snap distance, proposal → material | median 0.0 across every probed method |
| Held-out R², bandgap, n_train 80 (shipped non-ARD RBF on descriptors) | **0.502** |
| Held-out R², `m_star`, same | **−0.016** — and −0.85 to +0.10 across every kernel × representation tried |
| Feasible materials inside the bandgap window | 14; **7 of them rejected by m\* ≥ 4** |
| Posterior at a 50-observation state: sd observed vs unobserved | 0.045 vs 1.023 (reverts to the prior off the training set) |
| Range probability, best observed vs best unobserved | (0.99, 0.95) vs (0.19, 0.09) — all 193 unobserved dominated |
| qEHVI over unobserved materials at that state | **exactly zero at 100%**; objective deterministic in x (spread 2e-16) |
| C3 in-band filter at that state | passes 2 of 240 materials, **both already observed** |
| Labeled rows after a 30-point initial design (10 seeds) | dense 8–18, restricted 4–12, **full 2–5** |
| Initial designs needing the `min_labeled` extension | only on `full`, 1 seed in 10 (60 oracle calls) |
| Hits already present in the initial design (10 seeds, summed) | dense 10, restricted 3, full 1 |

## The mechanism under test

The other two suites answer *why* the method works and *whether it generalizes*.
This one answers the question they left open, and it is the only one they left
open with a measured counter-example inside it.

`feasible_rate` is the share of a method's proposals that are manufacturable at
all. Across 2,424 cells:

| task | infeasible | Random | TRMC-BO full | Point-target | distinct: TRMC vs PT |
| --- | ---: | ---: | ---: | ---: | --- |
| MatFormBench L3-1 | 24% | 0.764 | 0.867 | 0.889 | **8.33** vs 3.40 |
| MatFormBench L4-1 | 41% | 0.588 | **0.634** | 0.804 | **7.70** vs 3.63 |
| MatFormBench L5-4 | 69% | 0.314 | **0.313** | 0.329 | 1.07 vs **1.23** |

C1, C2 and C3 all aim at the outcome window and supply no signal on the
manufacturability axis, so TRMC-BO's proposals are feasible at essentially the
random-search rate while point-target and TB sit above it. The diversity advantage
dominates at 24–41% infeasible and reverses by 69% — **on one task**, and nothing
else in the world has tested it. Olympus could not: its usable mixture tasks are
99.8–100% feasible.

HOIP's ladder brackets that crossing inside one chemistry, with the valid set held
exactly fixed so the ceiling on the primary metric does not move along the axis.

## Predictions

**P1 — feasibility blindness is an all-task fact, not an L5-4 fact.**
`M5_full`'s `feasible_rate` is at or below `M0_random`'s on **all three**
catalogues, while `M2_target_distance` and `M6_range_aware_tb` sit above both.
Falsified if TRMC-BO's feasible rate exceeds random's by more than one seed-level
standard deviation on any rung — which would mean the range objective does carry
some manufacturability signal and the MatFormBench reading was task-specific.

**P2 — the distinct-design advantage falls monotonically with infeasibility, and
reverses.** `distinct(M5_full) − distinct(M2_target_distance)` is positive on
`dense` (60.0%), smaller on `restricted` (73.5%), and ≤ 0 on `full` (91.3%). If
L5-4's crossing near 69% generalizes, the sign changes between `dense` and
`restricted`. Falsified if the margin is non-monotone, or if it stays positive on
`full` — the latter would mean 69% was not a boundary but a property of that one
MatFormBench task, and the paper would have to say so.

**P3 — REVISED before the sweep, and the original is kept.** As first written it
read:

> *no acquisition saturation, so C1 alone beats random.* HOIP has m = 2 and takes
> the qEHVI path, so the single-output collapse measured on Olympus cannot occur.
> `M3_range_prob` therefore scores above `M0_random` on distinct designs on every
> rung.

The mechanism measurement above — run after that sentence was written and **before
any campaign** — shows the premise is wrong. A multi-output collapse does occur,
by a different route than Olympus's hinge: the surrogate reverts to its prior off
the training set, so an observed in-window material scores range probability
(0.99, 0.95) while nothing unobserved exceeds (0.19, 0.09), the observed front
dominates the entire catalogue, and qEHVI is identically zero everywhere else. It
is not the hinge and not m = 1; it is a weak surrogate, and `m_star` is the reason
the surrogate is weak.

So the revised prediction is the opposite: **`M3_range_prob` lands at
`M0_random`'s level on all three rungs**, because with a flat acquisition
`optimize_acqf` returns its initial condition and M3's restarts are uniform draws
from the catalogue — which is random search. Falsified if M3 separates from random
on any rung.

Revising a registered prediction is only legitimate when the revision is forced by
a *mechanism* measurement rather than by an outcome, when the original is left
standing, and when it happens before the results exist. All three hold here.

**P4 — C2 needs C3 on a sparse catalogue.** `M4` (C1+C2, unfiltered restarts) is
**not** better than `M3` here, and may be worse. The mechanism is specific and
checkable: C2 attaches the posterior-mean band as a hard nonlinear constraint, and
BoTorch rejects any initial condition that violates it. With the in-band set a
vanishing fraction of a finite catalogue, essentially no unfiltered restart
qualifies, every initial condition is rejected, and the shipped code falls back to
soft range handling — so C2 is discarded before it acts. `M5` (C1+C2+C3) does not
have this problem because C3 selects in-band restarts in the first place. Expect
`M4 ≈ M3 < M5` rather than the monotone `M3 < M4 < M5` ladder the other suites
show. *The protocol probe already showed the rejection message on `dense`; the
prediction is about whether it costs M4 measurable performance across ten seeds,
which the probe cannot say.* Falsified if M4 separates cleanly above M3.

Given the revised P3, the expected ordering on HOIP is not a ladder at all:
`M0 ≈ M3 ≈ M4`, with `M5` and `M6` *below* them, because C3 and the ball
acquisition both concentrate restarts onto already-observed materials and the loop
then spends the evaluation on a random design. A method can be worse than random
here only by wasting its restarts, which is what that would show.

**P5 — the §4.2 product replicates its MatFormBench behaviour on a second
multi-output suite.** `M5p_full_product` scores a higher hit rate and *fewer*
distinct designs than `M5_full`. On MatFormBench the specified product collapsed
27 qualifying evaluations into 1 distinct design on L4-1. HOIP is the only other
multi-output suite, so this is the only available replication. Falsified if the
product matches or beats the shipped hypervolume path on distinct designs, which
would make the §4.2 correction a one-suite result.

**P6 — random fallback rises with campaign coverage, not with method quality.**
The loop spends an evaluation on a random design when every retry re-proposes an
already-evaluated one. On a finite catalogue that is expected and is applied
identically to every method, so `random_fallback_rate` should track the coverage
ordering (dense 33.3% > restricted 19.6% > full 6.3%) rather than separating
methods. Falsified if one method's fallback rate is far above the others' on the
same rung — which would mean the comparison on that rung is measuring the fallback
policy rather than the acquisition, and the result would have to be withheld.

This one needs care in the *reading*, not only in the result, and the reason is
worth stating before the numbers exist. The fallback rate varies systematically
along the very axis under test — it is highest on `dense`, which is the rung where
TRMC-BO is predicted to lead by the most. The single-seed probe already shows
0.26–0.46 there. So a decline in the margin across the ladder could in principle be
produced by the coverage gradient rather than by infeasibility. Two things guard
against reading it wrongly:

* the gradient runs **against** P2 — it suppresses the margin where the margin is
  predicted to be largest, so it cannot manufacture the predicted decline, only
  hide one;
* `distinct` will additionally be reported over **non-fallback evaluations only**,
  reconstructed from the per-iteration `cnt_random_fallback` column in
  `trace.parquet`. If the two readings disagree in direction, the fallback policy
  is what the rung measured and P2 is unanswerable on it.

The policy itself is left exactly as the other two suites ran it. Swapping the
random fallback for the method's next-best candidate would sharpen HOIP — every
model-based method inherits `ProposedMethod.propose_pool`, so the ranked
alternative is already there — but duplicates fired in 33 of 1,080 Olympus cells
(mean fallback 0.0067, max 0.64) and 7 of 1,344 MatFormBench cells, so changing it
would make HOIP's protocol differ from the two finished suites and would invalidate
those cells. Changing a protocol mid-paper in the direction that sharpens an effect
one has predicted is precisely what should not be done.

## What would make this suite unable to answer its question

Declared in advance so that neither outcome can be presented as a finding after
the fact.

* **Everything lands at zero on `full`.** Valid density there is 0.55%, so random
  search expects ~0.44 hits in 80 evaluations. If every method finds nothing on
  `full` across ten seeds, that rung measures nothing and P2 rests on two points,
  not three. The result is still reportable — as the observation that at 91%
  infeasible a 50-evaluation budget is not enough for any method — but it is not a
  measurement of the crossing.
* **`random_fallback_rate` dominates.** See P6.
* **The 14-D descriptor GP is too weak on ~3 labeled rows.** On `full`, 30 initial
  draws label about 3 materials. If every model-based method sits at random-search
  level on that rung, the rung reports the surrogate rather than the acquisition.
  **This one has already been measured, and it is not a risk but a fact — on every
  rung, not only `full`.** `m_star` is not predictable from composition: held-out
  R² is −0.85 to +0.10 across descriptors and one-hot, RBF, ARD and Matérn, raw and
  log. Bandgap is (0.50 shipped, 0.78 with ARD), so HOIP is a *half*-learnable
  problem — and the unlearnable half binds, rejecting 7 of the 14 feasible
  materials that clear the bandgap window.

  The consequence has to be stated plainly rather than absorbed: **this suite
  cannot locate the crossing P2 asks about**, because on `dense` — the rung where
  TRMC-BO is supposed to lead by the most — there is no advantage to decline. The
  feasibility question the suite exists for survives intact, since `feasible_rate`
  against random does not depend on the surrogate working, and the mechanism above
  turns the blindness from a correlation into something with a measured cause. But
  P2's ladder reading is void unless something separates from random on `dense`.

  The plan's declared fallback for this case was a PCA-reduced descriptor set. It
  **does not apply**: dimension is not the binding constraint, `m_star`'s
  unlearnability is, and no projection of the inputs fixes an output that carries
  no signal. Switching to ARD or to the one-hot representation would raise the
  bandgap fit and leave `m_star` where it is, so neither is a fix either — and the
  non-ARD RBF is the shipped production surrogate, which is what the ablation is
  supposed to be about.

## The single-seed protocol probe

Seed 0, `n_init` 30, `budget` 50, `num_restarts` 128, uncontended (one core of
24). **Not evidence for any prediction above**; recorded because it was run before
this document and because two of its observations shaped P4 and P6.

Wall-clock is from an uncontended run and is therefore optimistic: the sweep runs
22 workers, and MatFormBench and Olympus both measured per-cell cost under
contention. Treat the sweep's cost as unmeasured until it is measured under
contention.

Stopped after `dense` plus one `restricted` row: with a 10-seed sweep about to run,
further single-seed cells were redundant.

| task | method | s/cell | hits | distinct | feasible | duplicate | random fallback |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | M0_random | 0.0 | 3 | 3 | 0.46 | 0.32 | 0.00 |
| dense | M2_target_distance | 151.8 | 4 | 4 | 0.50 | 2.34 | 0.42 |
| dense | M3_range_prob | 170.8 | 4 | 4 | 0.46 | 1.70 | 0.26 |
| dense | M4_range_prob_constraints | 141.0 | 4 | 3 | 0.46 | 2.44 | 0.46 |
| dense | M5_full | 432.3 | 3 | 3 | 0.36 | 4.00 | **1.00** |
| dense | M6_range_aware_tb | 183.5 | 3 | 3 | 0.38 | 3.92 | **0.96** |
| restricted | M0_random | 0.0 | 0 | 0 | 0.20 | 0.14 | 0.00 |

`duplicate` is per loop iteration, so 4.00 means all four attempts collapsed onto an
already-evaluated material every single iteration — which is what a 1.00 fallback
rate means: `M5_full` was random search for the entire loop, and `M6` nearly so.

Nothing here separates from `M0_random` by more than one design, which is what the
revised P3 says to expect. Ten seeds will say whether that is the result or the
seed.

Cost: the C3-filtered variants (`M5`, `M5p`, `M5q`) run ~2.5× the others because
`_collect_feasible_samples` scores 5,120 catalogue draws per iteration against the
posterior-mean band. Projected sweep cost is therefore ~22 CPU-hours for 360 cells
— **uncontended**, so the sweep's own `wall_clock_per_iter` is the number to quote,
not this one.

## Outcome

Reported in full in `results_issue_hoip.md`, written after the 360-cell sweep
completed (0 errors). Verdict per prediction:

| # | verdict |
| --- | --- |
| P1 | falsified as literally stated (M5_full's delta is always slightly positive, not at-or-below random) -- but the reframe holds: every method's delta sits in a narrow band, so the blindness is suite-wide, not TRMC-BO-specific |
| P2 | undecidable -- there was no advantage to decline in the first place |
| P3 (revised) | confirmed -- M3 lands at random's level on all three rungs |
| P4 | partially confirmed -- the predicted ordering holds on `dense`; fallback noise dominates on `restricted`/`full` |
| P5 | not observed -- the product variants do not reproduce MatFormBench's hit-rate-up/distinct-down pattern on this suite |
| P6 | confirmed, in its strongest form -- the non-fallback recount in §3.2 of `results_issue_hoip.md` erases most of what looked like a win |

The single most consequential result was not predicted as a numbered item: with
random-fallback excluded, several of the "wins" over random search in the raw
table (M5_full and the TB variants on `dense`) go to exactly zero. What the raw
distinct-designs table measures on this suite is, for a large share of its cells,
the loop's fallback policy rather than the acquisition.