"""Multi-task version of `plot_discovery_curves`: one panel per task.

`plot_discovery_curves` pools every task passed to it into a single mean
curve, which is the right call for the main-text branin figure but hides
whether the saturation/reversal pattern actually holds task-by-task on a
suite with several tasks (Olympus's three mixture emulator tasks, HOIP's
three catalogues). This script draws one panel per task instead, sharing one
legend and the same method styling as the single-task figure.

    PYTHONPATH=core:. .venv-eval/bin/python -m eval.scripts.plot_discovery_curves_grid \
        --suite olympus --protocol emulator --tasks pce10 thin_film wf3 \
        --out paper/figures/curves_olympus_emulator.png

    PYTHONPATH=core:. .venv-eval/bin/python -m eval.scripts.plot_discovery_curves_grid \
        --suite hoip --protocol catalogue --tasks dense full restricted \
        --width native --out paper/figures/curves_hoip.png
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
from eval.harness.aggregate import DEFAULT_DELTA, DISPLAY_ORDER, label  # noqa: E402
from eval.scripts.plot_discovery_curves import DEFAULT_METHODS, curves  # noqa: E402

STYLES = {
    "M0_random": dict(color="0.55", ls=":", lw=1.6),
    "M2_target_distance": dict(color="#c2571a", ls="--", lw=1.8),
    "M6b_range_aware_tb_inscribed": dict(color="#7b4fa8", ls="-.", lw=1.8),
    "M3_range_prob": dict(color="#8fbfe0", ls="-", lw=1.4),
    "M4_range_prob_constraints": dict(color="#3d84b8", ls="-", lw=1.8),
    "M13_range_prob_starts": dict(color="#3d84b8", ls="-", lw=1.8),
    "M16_constraints_starts": dict(color="#2a9d8f", ls="-", lw=1.8),
    "M5_full": dict(color="#0b3c5d", ls="-", lw=2.6),
}


def main() -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="olympus")
    ap.add_argument("--protocol", default="emulator")
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--width", default=None)
    ap.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    ap.add_argument("--delta", type=float, default=DEFAULT_DELTA)
    ap.add_argument("--budget", type=int, default=50)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    suite = resolve(args.suite)
    task_ids = args.tasks or list(suite.default_tasks)
    kw = {"suite": args.suite, "protocol": args.protocol, "width": args.width}

    per_task: dict[str, dict[str, np.ndarray]] = {}
    for tid in task_ids:
        task = suite.load_task(tid)
        per_task[tid] = curves(task, args.methods, args.delta, args.budget, **kw)

    order = [m for m in DISPLAY_ORDER
             if any(m in c for c in per_task.values())]

    fig, axes = plt.subplots(1, len(task_ids), figsize=(4.6 * len(task_ids), 4.0),
                             sharey=True)
    axes = np.atleast_1d(axes)
    x = np.arange(1, args.budget + 1)

    for ax, tid in zip(axes, task_ids):
        means = per_task[tid]
        for m in order:
            if m not in means:
                continue
            ax.plot(x, means[m][: args.budget], label=label(m),
                    **STYLES.get(m, dict(lw=1.5)))
        ax.set_title(tid, fontsize=10)
        ax.set_xlabel("evaluation")
        ax.set_xlim(1, args.budget)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25, lw=0.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    axes[0].set_ylabel(f"cumulative distinct qualifying designs ($\\delta={args.delta}$)")
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, frameon=False, fontsize=8, loc="lower center",
              ncol=min(4, len(handles)), bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.10, 1, 1))

    out = args.out or f"curves_{args.suite}_{args.protocol}_grid.png"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")
    for tid in task_ids:
        means = per_task[tid]
        print(f"[{tid}]")
        for m in order:
            if m not in means:
                continue
            print(f"  {label(m):42s} it10={means[m][9]:5.2f}  it30={means[m][29]:5.2f}  "
                  f"it50={means[m][args.budget - 1]:5.2f}")


if __name__ == "__main__":
    main()
