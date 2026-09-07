# C4 (unknown input feasibility) — predictions registered before the sweep

Written and frozen **before** any `M10`/`M11`/`M12` cell was run. `eval/results/`
contained zero cells for the three C4 variants when this file reached its final
state; the component itself (`src/utils/feasibility.py`, the `ProposedSpec`
fields in `eval/methods/proposed.py`, the registry entries and the two config
blocks) was already written and merged, so what is being pre-registered here is
the *outcome*, not the code.

Same reason as `predictions_hoip.md`: on MatFormBench the primary metric was
promoted from hit rate to distinct designs *after* seeing the data. That was
defensible and was disclosed, but it cannot be repeated. C4 is the component most
likely to change a headline — it is the only one that can move HOIP's sign — so
the direction of every expected effect is fixed in advance, together with what we
will write in each case.

---

## 1. What is already measured, not predicted

### 1.1 The hole C4 exists to fill

C1 (range probability), C2 (two-sided posterior-mean constraints) and C3
(mean-filtered restarts) are all statements about the **outcome** space. None of
them can learn input feasibility, and the reason is structural, not incidental:
`initialize_model` fits output `j` only on rows where `mask[:, j]` is True, so an
all-NaN (infeasible) observation is dropped from every GP. Nothing downstream
ever sees a failure.

Measured `feasible_rate`, MatFormBench range-adapted, 1,344 cells:

| method | L1-1 | L2-1 | L3-1 | L4-1 | L5-4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| M0_random | 1.000 | 1.000 | 0.764 | 0.588 | 0.314 |
| M2_target_distance | 1.000 | 1.000 | 0.889 | **0.804** | 0.329 |
| M3_range_prob | 1.000 | 1.000 | 0.811 | 0.601 | 0.366 |
| M4_range_prob_constraints | 1.000 | 1.000 | 0.821 | 0.603 | 0.333 |
| M5_full | 1.000 | 1.000 | 0.867 | 0.634 | 0.313 |

L1-1/L2-1 are fully feasible and carry no information. On L4-1 and L5-4 the
proposed method sits at 0.634 / 0.313 against random's 0.588 / 0.314 — that is
the hole.

### 1.2 HOIP's two failure causes, and what C4 touches

`results_issue_hoip.md` attributes HOIP's negative result to two *separate*
mechanisms:

1. `m_star` is unlearnable from composition (held-out R² −0.85 to +0.10 across
   every kernel × representation tried), so at a 50-observation state qEHVI is
   **exactly zero at 100%** of unobserved materials.
2. C3's in-band restart filter therefore starves — 27.7% / 19.1% / 10.4% of
   requested restarts survive across the three rungs — and the loop fills the
   iteration with a random design.

### 1.3 Fallback rates, measured across all three suites

Aggregated from every `metrics.json` on disk (1,344 + 1,080 + 360 cells):

| suite | `random_fallback_rate`, all methods |
| --- | --- |
| MatFormBench range_adapted | **≡ 0.000**, except `M8_comboo` 0.010 |
| Olympus branin | **≡ 0.000** |
| Olympus emulator | **≡ 0.000**, except `M1_qehvi_extremum` 0.080 |
| HOIP catalogue | **0.000 – 0.992** (see below) |

HOIP per method (mean over 10 seeds):

| method | dense | restricted | full | dup /4 | feasible_rate (dense/restr./full) |
| --- | ---: | ---: | ---: | ---: | --- |
| M0_random | 0.002 | 0.000 | 0.000 | 0.25 | 0.388 / 0.248 / 0.106 |
| M3_range_prob | 0.344 | 0.134 | 0.484 | 1.92 | 0.408 / 0.252 / 0.114 |
| M4_range_prob_constraints | 0.560 | 0.192 | 0.438 | 2.65 | 0.434 / 0.278 / 0.112 |
| **M5_full** | **0.918** | **0.722** | **0.770** | 3.72 | 0.418 / 0.272 / 0.118 |
| M6_range_aware_tb | 0.918 | 0.982 | 0.836 | 3.70 | 0.422 / 0.260 / 0.102 |
| M8_comboo | 0.582 | 0.450 | 0.500 | 2.72 | 0.466 / 0.344 / 0.124 |
| **M9_anubis** | **0.000** | **0.000** | **0.000** | **0.00** | 0.320 / 0.174 / 0.080 |

**`M9_anubis` is the only model-based method on HOIP with zero fallback, and it
is also the only one with zero duplicate proposals.** It differs from every other
method in one way: it draws candidates from `restart_pool`, the enumerated
catalogue, rather than by continuous multistart. That is direct evidence that
what fixes HOIP's starvation is the **restart pool**, not the classifier — and
`M10`/`M11`/`M12` take the classifier without the pool.

### 1.4 A confound in `feasible_rate` on HOIP — registered before it can be exploited

`M9_anubis` has the **lowest** `feasible_rate` of any method on all three rungs
(0.320 / 0.174 / 0.080 against random's 0.388 / 0.248 / 0.106), while having the
*only* clean exploration record (dup 0.00, fallback 0.000).

The explanation is mechanical. HOIP's oracle is a deterministic lookup, and every
other method re-proposes materials it has already evaluated (dup 1.9–4.0 out of a
maximum of 4). A re-proposed material that was evaluated successfully is
*counted as feasible again*. So a method that churns on known-good points scores
a high `feasible_rate` without discovering anything, and a method that keeps
opening fresh catalogue rows is scored against HOIP's 60–91% base infeasibility
rate.

**This matters for C4 specifically**: C4a ranks by `P(feasible|x) · acq(x)`, and
`P(feasible|x)` is highest exactly where feasible observations already are. The
cheapest way for C4a to raise `feasible_rate` is to raise the duplicate rate.

> **Registered rule.** On HOIP, `feasible_rate` is not read on its own. Any C4
> claim about manufacturability must be accompanied by `duplicate_proposal_rate`,
> and a rise in `feasible_rate` that comes with a rise in duplicates is reported
> as **no effect**, not as an improvement.

---

## 2. Predictions

### C1p — C4a raises `feasible_rate` where there is room

`M10_full_feasw` beats `M5_full` on `feasible_rate` on MatFormBench **L3-1, L4-1
and L5-4** (L1-1/L2-1 are at 1.000 and cannot move). Confidence: moderate. This
is the one thing the component is built to do and the one axis on which nothing
currently competes.

*Falsified if* `M10` is within ±0.02 of `M5_full` on all three, or below it on two
of them.

### C2p — C4b makes the starvation worse, not better

`M11_full_feasf`'s `random_fallback_rate` is **higher** than `M5_full`'s on all
three HOIP rungs, and its `cnt_restart_in_band / cnt_restart_target` ratio is
**lower**. Confidence: high. C4b ANDs a second predicate onto a restart pool that
already passes only 10–28% of requested starts (`compose_predicates`), and
`feasibility_start_predicate`'s `tau = 0.1` was deliberately set low precisely
because of this — a low threshold reduces the damage but cannot remove it.

*Falsified if* `M11`'s fallback is at or below `M5_full`'s on two of three rungs.

### C3p — HOIP's negative result survives C4

None of `M10`/`M11`/`M12` exceeds `M0_random` on **distinct designs** on more than
one of the three rungs, and none exceeds it on the **non-fallback** count on any
rung. Confidence: moderate-to-high. C4 addresses input feasibility; HOIP's
collapse is driven by an unlearnable *outcome* property (`m_star`, R² ≈ 0), which
flattens the acquisition regardless of how candidates are ranked for
manufacturability.

Note the interaction: at HOIP's flat-acquisition state, `feasibility_weighted_scores`
will hit its **flat guard** (`np.ptp(base_acq) == 0`) and fall back to ranking by
`P(feasible|x)` **alone**. So on HOIP, C4a is not "feasibility-weighted range
search"; for many iterations it is a pure feasibility search. That is worth
measuring and is *not* a bug — but it means a HOIP win for `M10` would be a win
for feasibility search, not for the C1–C4 combination, and must be described that
way.

*Falsified if* any C4 variant beats random on distinct designs on ≥2 rungs **and**
survives the non-fallback recount.

### C4p — the two axes stay orthogonal on MatFormBench

On MatFormBench, `M10`/`M11`/`M12` do not change **distinct designs** relative to
`M5_full` by more than the seed standard deviation on any (task, width) cell,
while C1p's `feasible_rate` effect appears. Confidence: moderate. The components
are claimed to be orthogonal; this is the test of that claim, and it is the
cleanest place to run it because MatFormBench's fallback rate is identically zero,
so nothing is confounded by fallback policy.

*Falsified if* the C4 variants move distinct designs systematically in either
direction — which would mean the axes are not orthogonal and §4's framing needs
rewriting.

### C5p — the recount rule, stated before it is needed

If C3p is falsified — if C4 lifts HOIP's distinct designs above random — the
result is **not reported** until it has been recounted excluding random-fallback
evaluations. HOIP §3.2 is exactly this trap: `M5_full`'s apparent 1.00 distinct
designs on `dense` became 0.00 on the non-fallback count, and `M6`/`M6b` did the
same. A C4 variant with fallback still at 0.5–0.9 is under the same suspicion by
default.

### C6p — the diagnosis of what would actually fix HOIP

If C2p holds (C4b worsens starvation) and C3p holds (HOIP stays negative), then
the measured explanation for `M9_anubis`'s fallback of 0.000 is its enumerated
`restart_pool`, and the honest conclusion is that **HOIP's failure is a restart-
generation problem, not a candidate-ranking problem.** A pool-based restart
fallback ("C5") would then be the indicated fix. It is deliberately *not* being
run in this round, so that this prediction is tested rather than assumed.

---

## 3. What gets written in each outcome

Fixed now, so the narrative is not adjusted after the fact.

| outcome | C4's place in the paper | HOIP section | knock-on |
| --- | --- | --- | --- |
| **H1** — C1p also fails; C4 does nothing anywhere | one paragraph in §4 plus an appendix ablation | unchanged: scope boundary | none |
| **H2** — C1p holds, C3p holds (expected) | promoted to a fourth component in §4 with its own results table | sharpened: "the feasibility axis is separable and addressable; outcome diversity still requires a learnable target property" | §4.7's overclaim is repaired; Limitations rewritten |
| **H3** — C3p falsified and survives the C5p recount | promoted to a main contribution; §4 restructured as a C1–C4 ladder | flips to a positive result | non-fallback distinct becomes a permanent column beside the primary metric in §5 |

In **H2** and **H3**, §5 must additionally state the §1.4 confound and report
`duplicate_proposal_rate` beside `feasible_rate` for every HOIP table.

---

## 4. What is being run

No config changes. Both configs already list `M10_full_feasw`, `M11_full_feasf`
and `M12_full_feas`; `run_sweep` skips completed cells, so re-running fills only
the new ones.

```bash
# MatFormBench: 3 methods x 5 tasks x 3 widths x 10 seeds = 450 cells
PYTHONPATH=core:. .venv-eval/bin/python -m eval.scripts.run_sweep \
    --config eval/configs/matformbench_range_adapted.yaml --workers 22

# HOIP: 3 methods x 3 tasks x 1 width x 10 seeds = 90 cells
PYTHONPATH=core:. .venv-eval/bin/python -m eval.scripts.run_sweep \
    --config eval/configs/hoip_catalogue.yaml --workers 22
```

Protocol is unchanged from the completed sweeps — `n_init` 30, budget 50, `q` 1,
`num_restarts` 128, 10 seeds, identical initial designs per seed across methods.
C4 reads no field of `target_info` that `M5_full` does not; its only additional
input is the feasibility label derived from the observation mask
(`resolve_feasibility_labels`), which is the oracle's own NaN pattern and is
available to every method. `eval/scripts/audit_leakage.py` is re-run afterwards to
assert that.
