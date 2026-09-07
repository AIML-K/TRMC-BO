"""Multi-task version of `plot_distinct_designs`: one row per task.

The single-task figure makes the hits-vs-distinct point on one representative
task (L3-1 in the paper). This script reuses the same frame-building and
panel-drawing code to ask the next question: does the same concentration
pattern hold on the other tasks, or was L3-1 special? One row per task, same
four method columns, so the panels are directly comparable across rows.

    PYTHONPATH=core:. .venv-eval/bin/python -m eval.scripts.plot_distinct_designs_grid \
        --suite matformbench --protocol range_adapted \
        --tasks L1-1 L2-1 L3-1 L4-1 L5-4 --width medium \
        --out paper/figures/distinct_all_tasks.png
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
from eval.harness.aggregate import DEFAULT_DELTA, DISPLAY_ORDER, valid_designs  # noqa: E402
from eval.scripts.plot_distinct_designs import (  # noqa: E402
    DEFAULT_PANELS,
    build_frame,
    draw_panel,
)


def main() -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="matformbench")
    ap.add_argument("--protocol", default="range_adapted")
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--width", default="medium")
    ap.add_argument("--methods", nargs="*", default=DEFAULT_PANELS)
    ap.add_argument("--delta", type=float, default=DEFAULT_DELTA)
    ap.add_argument("--background", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    suite = resolve(args.suite)
    task_ids = args.tasks or list(suite.default_tasks)
    panels = [m for m in DISPLAY_ORDER if m in args.methods]

    nrow, ncol = len(task_ids), len(panels)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.4 * nrow),
                             sharex="row", sharey="row")
    axes = np.atleast_2d(axes)
    if axes.shape != (nrow, ncol):
        axes = axes.reshape(nrow, ncol)

    for row, tid in enumerate(task_ids):
        task = suite.load_task(tid)
        pca, Zbg, feasible, valid, _ = build_frame(
            suite, task, args.width, args.background, args.seed
        )
        Pbg = pca.transform(Zbg)
        designs = valid_designs(task, suite=args.suite, protocol=args.protocol,
                                width=args.width)
        cells_per_method: dict[str, int] = {}
        for _, _w, _m, _s in designs.attrs.get("cells", []):
            cells_per_method[_m] = cells_per_method.get(_m, 0) + 1
        xcols = [f"x_{n}" for n in task.x_names]

        for col, method in enumerate(panels):
            ax = axes[row, col]
            draw_panel(ax, method, pca=pca, Pbg=Pbg, feasible=feasible,
                       designs=designs, xcols=xcols, task=task,
                       delta=args.delta, cells_per_method=cells_per_method)
            if col == 0:
                ax.set_ylabel(f"{tid}  ({task.dim}D)", fontsize=10, fontweight="bold")

    handles, labels_ = axes[0, 0].get_legend_handles_labels()
    leg = fig.legend(handles, labels_, loc="lower center", ncol=3, frameon=False,
                     fontsize=10, markerscale=2.2, handletextpad=0.5)
    for h in leg.legend_handles:
        h.set_alpha(1.0)

    fig.suptitle(
        f"Where the qualifying formulations landed, across all {nrow} "
        f"MatFormBench tasks ({args.width} width, PC1-PC2 of the in-window "
        "region per task; counts pool all seeds, per run)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0.02, 0.04, 1, 0.96))

    out = Path(args.out) if args.out else Path("paper/figures/distinct_all_tasks.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
