"""Assemble the result tables for one suite/protocol.

    PYTHONPATH=core:. .venv-eval/bin/python -m eval.scripts.report \
        --suite olympus --protocol branin

Suite-neutral, because the MatFormBench tables were assembled by hand and the
same tables are now wanted for two more suites. Everything here composes
`harness/aggregate` functions; nothing new is computed.

**Distinct designs lead, hit rate follows.** `hit_rate` counts evaluations that
landed in the window, so a method that converges on one basin and re-proposes
near it scores well while returning a single usable recipe -- on MatFormBench the
point-target baseline turned 11.01 hits into 2.86 distinct designs while
TRMC-BO's full variant kept 6.41 of 6.88. Both numbers answer real questions and
both are printed, but `distinct` is the primary metric and appears first.

`concentration` = distinct / valid is the mechanism column: near 1.0 means almost
every success was a different design; low means one basin was re-sampled.
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

from eval.benchmarks.registry import resolve
from eval.harness import aggregate as agg
from eval.harness.runner import RESULT_ROOT


def _section(title: str) -> None:
    print(f"\n\n{title}\n{'=' * len(title)}\n")


def main() -> None:
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 200)
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="olympus")
    ap.add_argument("--protocol", default="branin")
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--delta", type=float, default=agg.DEFAULT_DELTA)
    ap.add_argument("--deltas", type=float, nargs="*", default=[0.05, 0.1, 0.2, 0.4])
    ap.add_argument("--iterations", type=int, nargs="*", default=[1, 10, 20, 30, 50])
    ap.add_argument("--reference", default="M5_full",
                    help="method the paired comparison in section 1c scores against")
    args = ap.parse_args()

    suite = resolve(args.suite)
    task_ids = args.tasks or list(suite.default_tasks)
    kw = {"suite": args.suite, "protocol": args.protocol, "root": RESULT_ROOT}

    metrics = agg.load_runs(root=RESULT_ROOT, suite=args.suite, protocol=args.protocol)
    tasks = {t: suite.load_task(t) for t in task_ids}
    tasks = {t: task for t, task in tasks.items() if (metrics["task_id"] == t).any()}
    if not tasks:
        raise SystemExit(f"no results for {args.suite}/{args.protocol}")

    n_cells = len(metrics)
    print(f"{args.suite}/{args.protocol}: {n_cells} cells, "
          f"{metrics['task_id'].nunique()} task(s), {metrics['method'].nunique()} methods, "
          f"{metrics['seed'].nunique()} seeds")

    counts = agg.distinct_counts(tasks, delta=args.delta, **kw)

    _section(f"1. Main table -- distinct qualifying designs (delta={args.delta})")
    print(agg.render(agg.main_table(counts, metrics)))

    _section(f"1a. 95% bootstrap interval on the headline mean (resampled over seeds)")
    print(agg.render(agg.seed_bootstrap_ci(counts, value="distinct")))
    print("\nSeeds are the resampling unit: within a seed every method starts "
          "from the same\ninitial design, so cells sharing a seed are not "
          "independent draws.")

    # Not an optional diagnostic. On HOIP the loop's random-fallback safety net,
    # not the acquisition, produced most of what the model-based methods
    # "found" -- M5_full's 1.00 distinct designs on `dense` are 0.00 once those
    # iterations are removed. Printing both counts for every suite is what makes
    # "MatFormBench and Olympus are not that" checkable instead of asserted.
    _section(f"1b. How much of that came from the random-fallback safety net?")
    print(agg.render(agg.fallback_comparison(tasks, metrics=metrics,
                                             delta=args.delta, **kw)))
    print("\nIdentical columns mean the method never fell back, so the left "
          "column already\ndescribes its acquisition. Where they differ, "
          "`distinct_nonfb` is the honest one.")

    # A difference of means is not a claim about beating a method on the same
    # problem. Pairing is what turns it into one, and it is legitimate here
    # because within a seed every method starts from a byte-identical initial
    # design on a byte-identical frozen window.
    _section(f"1c. Paired against {args.reference}, within (task, width, seed)")
    try:
        print(agg.render(agg.paired_comparison(counts, reference=args.reference)))
        print("\n`win` counts cells where the reference recovered MORE distinct "
              "designs.\nTies are excluded from the two-sided sign test, which "
              "is why n_pairs is\nreported beside it: 8-1 over 45 cells is a "
              "weaker record than 8-1 over 9.")
    except KeyError as err:
        print(f"skipped: {err}")

    # An ablation row is only the configuration it is named after if the component
    # was actually switched on. botorch needs initial conditions that already
    # satisfy a nonlinear constraint, so C2 cannot apply to an out-of-band start;
    # when C3 is off, or on but starved, the optimizer proceeds without it and
    # nothing in the headline tables shows it.
    _section("1d. Was the window constraint actually in force?")
    ca = agg.constraint_activity_all(tasks, **kw)
    if ca.empty:
        print("no constraint-bearing method in these runs")
    else:
        print(agg.render(ca))
        print("\n`active` is the share of loop iterations on which C2 was really "
              "applied.\n`dropped` is the C3-off path (every start rejected, "
              "constraint removed and retried);\n`starved` is the C3-on path "
              "(no in-band start existed, so C2 was never attached).")

    _section("2. Does the ranking depend on delta?")
    print(agg.render(agg.delta_sweep(tasks, deltas=tuple(args.deltas), **kw)))

    _section("3. Average rank over (task, width) cells")
    for metric in ("hit_rate", "diversity_norm"):
        print(f"-- by {metric}")
        print(agg.render(agg.average_rank(metrics, metric=metric)))
        print()

    # A suite's own conditions, not a hardcoded triple: HOIP has one, `native`,
    # and reindexing to wide/medium/narrow silently produced an empty table.
    width_order = [w for w in suite.default_widths]
    if len(width_order) > 1:
        _section("4. Breakdown by window width")
        for value, name in (("distinct", "distinct designs"), ("valid", "hits")):
            pivot = counts.pivot_table(index="method", columns="width", values=value)
            pivot = pivot.reindex(columns=[w for w in width_order if w in pivot.columns])
            pivot.index = [agg.label(m) for m in pivot.index]
            print(f"-- {name}")
            print(pivot.round(2).to_string(), "\n")
    else:
        _section(f"4. Single condition ({width_order[0]}) -- no width axis on this suite")
        print("This suite supplies its own window and varies input feasibility "
              "instead;\nsee section 5 for the axis it does have.")

    # The manufacturability axis, reported against random search rather than in
    # isolation. C1/C2/C3 all aim at the outcome window and supply no signal on
    # whether a proposal can be made at all, so "is this method's feasible rate
    # above the rate you get by not modelling anything?" is the question, and an
    # absolute rate cannot answer it.
    _section("4b. Input feasibility, against random search")
    baseline = (metrics[metrics["method"] == "M0_random"]
                .groupby("task_id")["feasible_rate"].mean())
    feas = metrics.pivot_table(index="method", columns="task_id",
                               values="feasible_rate", aggfunc="mean")
    feas = feas.reindex(columns=[t for t in task_ids if t in feas.columns])
    if not baseline.empty:
        delta = feas.subtract(baseline.reindex(feas.columns), axis=1)
        out = feas.round(3).astype(str) + delta.map(lambda v: f" ({v:+.3f})")
    else:
        out = feas.round(3)
    out.index = [agg.label(m) for m in out.index]
    print("share of loop proposals that were input-feasible "
          "(difference from random search in brackets)")
    print(out.to_string())

    for column, title in (("random_fallback_rate",
                           "random fallback -- share of iterations the loop spent on a "
                           "random design"),
                          ("duplicate_proposal_rate",
                           "duplicate proposals per iteration (max 4: every retry "
                           "re-proposed an evaluated design)")):
        pivot = metrics.pivot_table(index="method", columns="task_id",
                                    values=column, aggfunc="mean")
        pivot = pivot.reindex(columns=[t for t in task_ids if t in pivot.columns])
        pivot.index = [agg.label(m) for m in pivot.index]
        print(f"\n-- {title}")
        print(pivot.round(3).to_string())

    # Read the three tables above together, not one at a time. On a deterministic
    # lookup oracle a re-proposed material that was evaluated successfully counts
    # as feasible again, so a method that churns on known-good points scores a
    # high feasible_rate without discovering anything. M9_anubis is the check on
    # this: zero duplicates and zero fallback, and the *lowest* feasible rate on
    # all three HOIP rungs.
    print("\nA feasible_rate above random that arrives together with a high "
          "duplicate rate is\nchurn on known-good designs, not "
          "manufacturability -- read the three tables jointly.")

    if len(tasks) > 1:
        _section("5. Breakdown by task -- distinct designs")
        pivot = counts.pivot_table(index="method", columns="task_id", values="distinct")
        pivot.index = [agg.label(m) for m in pivot.index]
        print(pivot.round(2).to_string())

    # For a 2-D task the valid set's connected components are known, so "did a
    # method spread across them or camp in one?" is answerable directly rather
    # than through a distance proxy. This is the §5.2 diversity claim in its own
    # terms.
    single = next(iter(tasks.values()))
    if len(tasks) == 1 and single.dim == 2:
        from eval.scripts.valid_region_geometry import assign_components, component_labels

        _section("5b. Connected components of the valid set that each method reached")
        oracle = suite.make_oracle(single)
        xcols = [f"x_{n}" for n in single.x_names]
        rows = []
        for width in ("wide", "medium", "narrow"):
            spec = suite.load_range_spec(single.task_id, width)
            labels = component_labels(oracle, single, spec, 1400)
            total = int(labels.max())
            designs = agg.valid_designs(single, width=width, **kw)
            if designs.empty:
                continue
            for method, g in designs.groupby("method"):
                comp = assign_components(labels, single, g[xcols].to_numpy(float))
                per_seed = [
                    len(set(comp[(g["seed"] == s).to_numpy()]) - {0})
                    for s in g["seed"].unique()
                ]
                rows.append({
                    "method": method, "width": width,
                    "of": total,
                    "reached": float(np.mean(per_seed)) if per_seed else 0.0,
                })
        if rows:
            frame = pd.DataFrame(rows)
            pivot = frame.pivot_table(index="method", columns="width", values="reached")
            pivot = pivot.reindex(columns=[w for w in ("wide", "medium", "narrow")
                                           if w in pivot.columns])
            pivot.index = [agg.label(m) for m in pivot.index]
            available = frame.groupby("width")["of"].first().to_dict()
            print("mean number of distinct components a seed reached "
                  f"(available: {available})")
            print(pivot.round(2).to_string())

    _section("6. Discovery curves -- cumulative distinct-agnostic hits")
    curves = agg.discovery_curves(suite=args.suite, protocol=args.protocol, root=RESULT_ROOT)
    at = curves[curves["iteration"].isin(args.iterations)]
    pivot = at.pivot_table(index="method_label", columns="iteration", values="mean")
    print(pivot.round(2).to_string())

    _section("7. Optimizer health -- these must be near zero to trust the rest")
    health = metrics.groupby("method")[
        ["acqf_total_failure_rate", "random_fallback_rate", "duplicate_proposal_rate",
         "feasible_rate"]
    ].mean()
    health.index = [agg.label(m) for m in health.index]
    print(health.round(3).to_string())


if __name__ == "__main__":
    main()
