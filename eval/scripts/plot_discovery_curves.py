"""Cumulative *distinct* qualifying designs against iteration, per method.

`aggregate.discovery_curves` plots cumulative hits, which is the metric this
project argues against: a method that re-proposes one basin keeps accumulating
hits and its curve keeps climbing. Counting distinct designs instead makes the
difference visible as a shape rather than as a ratio -- a concentrating method
flattens once it has found its basin, while a method that keeps opening new
regions does not.

The distinct flag is computed greedily over a run's qualifying evaluations in
proposal order, so a design counts at the iteration it was first proposed and the
curve is a genuine cumulative count rather than a re-ordering of the final set.

    PYTHONPATH=core:. .venv-eval/bin/python -m eval.scripts.plot_discovery_curves \
        --suite olympus --protocol branin --out paper/figures/curves_branin.png
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from eval.benchmarks.registry import resolve  # noqa: E402
from eval.harness.aggregate import (  # noqa: E402
    DEFAULT_DELTA,
    DISPLAY_ORDER,
    _greedy_distinct,
    _normalize_designs,
    label,
    valid_designs,
)

#: One line per panel would be unreadable with every registered method, and the
#: argument only needs the contrast: the two baselines that win on hit rate, the
#: ablation ladder, and random search as the floor.
DEFAULT_METHODS = [
    "M0_random",
    "M2_target_distance",
    "M6b_range_aware_tb_inscribed",
    "M3_range_prob",
    "M13_range_prob_starts",
    "M16_constraints_starts",
    "M5_full",
]

#: `M4_range_prob_constraints` is deliberately absent. It is the C2-on/C3-off
#: cell, and the optimizer cannot apply a nonlinear constraint to a start that
#: does not already satisfy it -- so on that arm the constraint is dropped and the
#: iteration retried without it on 9-59% of iterations depending on the suite
#: (`aggregate.constraint_activity`). Plotting it beside the others would label a
#: curve with a configuration it did not run. The valid ablation axis is the
#: optimizer treatment at three levels: none, C3, C2+C3.


def curves(task, methods, delta, budget, **kw) -> dict[str, np.ndarray]:
    """method -> mean cumulative distinct count, shape (budget,)."""
    df = valid_designs(task, **kw)
    xcols = [f"x_{n}" for n in task.x_names]
    out: dict[str, list[np.ndarray]] = {m: [] for m in methods}
    cells = df.attrs.get("cells", [])
    seen = {(w, m, s) for _, w, m, s in cells}
    if not df.empty:
        for (w, m, s), g in df.groupby(["width", "method", "seed"]):
            if m not in out:
                continue
            g = g.sort_values("iteration")
            Z = _normalize_designs(g[xcols].to_numpy(float), task)
            keep = _greedy_distinct(Z, delta)
            it = g["iteration"].to_numpy()[keep]
            step = np.zeros(budget)
            for i in it:
                if 1 <= i <= budget:
                    step[int(i) - 1] += 1
            out[m].append(np.cumsum(step))
    # A cell that found nothing contributes a flat zero curve. Dropping it would
    # divide by the number of *successful* cells and quietly favour whichever
    # method fails most often.
    for m in methods:
        n_cells = sum(1 for _, w, mm, s in cells if mm == m)
        while len(out[m]) < n_cells:
            out[m].append(np.zeros(budget))
    return {m: np.mean(v, axis=0) for m, v in out.items() if v}


def main() -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="olympus")
    ap.add_argument("--protocol", default="branin")
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--width", default=None)
    ap.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    ap.add_argument("--delta", type=float, default=DEFAULT_DELTA)
    ap.add_argument("--budget", type=int, default=50)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    suite = resolve(args.suite)
    task_ids = args.tasks or list(suite.default_tasks)
    kw = {"suite": args.suite, "protocol": args.protocol,
          "width": args.width}

    acc: dict[str, list[np.ndarray]] = {}
    for tid in task_ids:
        task = suite.load_task(tid)
        for m, c in curves(task, args.methods, args.delta, args.budget, **kw).items():
            acc.setdefault(m, []).append(c)
    means = {m: np.mean(v, axis=0) for m, v in acc.items()}
    if not means:
        raise SystemExit("no cells found")

    order = [m for m in DISPLAY_ORDER if m in means]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    styles = {
        "M0_random": dict(color="0.55", ls=":", lw=1.6),
        "M2_target_distance": dict(color="#c2571a", ls="--", lw=1.8),
        "M6b_range_aware_tb_inscribed": dict(color="#7b4fa8", ls="-.", lw=1.8),
        "M3_range_prob": dict(color="#8fbfe0", ls="-", lw=1.4),
        "M4_range_prob_constraints": dict(color="#3d84b8", ls="-", lw=1.8),
        "M13_range_prob_starts": dict(color="#3d84b8", ls="-", lw=1.8),
        "M16_constraints_starts": dict(color="#2a9d8f", ls="-", lw=1.8),
        "M5_full": dict(color="#0b3c5d", ls="-", lw=2.6),
    }
    x = np.arange(1, args.budget + 1)
    for m in order:
        ax.plot(x, means[m][: args.budget], label=label(m),
                **styles.get(m, dict(lw=1.5)))
    ax.set_xlabel("evaluation")
    ax.set_ylabel(f"cumulative distinct qualifying designs ($\\delta={args.delta}$)")
    ax.set_xlim(1, args.budget)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25, lw=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()

    out = args.out or f"curves_{args.suite}_{args.protocol}.png"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")
    for m in order:
        print(f"  {label(m):42s} it10={means[m][9]:5.2f}  it30={means[m][29]:5.2f}  "
              f"it50={means[m][args.budget - 1]:5.2f}")


if __name__ == "__main__":
    main()
