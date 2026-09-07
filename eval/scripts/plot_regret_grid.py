"""Multi-task version of `plot_regret`: one (violation, coverage) row per task.

`plot_regret` pools every task it is given into one mean pair of curves. This
script instead draws one row per task so the branin-only reversal in the
main-text figure can be checked task-by-task on the other Olympus mixture
tasks. Requires frozen delta-separated capacity for each task in
`eval/specs/capacity.json` (see `valid_region_geometry.py --json`); a task
missing capacity prints a warning and its coverage panel is left empty.

    PYTHONPATH=core:. .venv-eval/bin/python -m eval.scripts.valid_region_geometry \
        --suite olympus --tasks pce10 thin_film wf3 --json eval/specs/capacity.json

    PYTHONPATH=core:. .venv-eval/bin/python -m eval.scripts.plot_regret_grid \
        --suite olympus --protocol emulator --tasks pce10 thin_film wf3 \
        --out paper/figures/regret_olympus_emulator.png
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
from eval.scripts.plot_regret import STYLES, load_capacity, violation_regret  # noqa: E402


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
    widths = [args.width] if args.width else list(suite.default_widths)

    v_per_task: dict[str, dict[str, np.ndarray]] = {}
    c_per_task: dict[str, dict[str, np.ndarray]] = {}
    bounded = False

    for tid in task_ids:
        task = suite.load_task(tid)
        v_per_task[tid] = violation_regret(task, args.methods, args.budget,
                                            suite=args.suite,
                                            protocol=args.protocol,
                                            width=args.width)
        cover: dict[str, list[np.ndarray]] = {}
        for w in widths:
            cap = load_capacity(args.suite, tid, w, args.delta)
            if cap is None:
                print(f"[warn] no frozen capacity for {args.suite}/{tid}/{w}; "
                      f"run valid_region_geometry --json eval/specs/capacity.json")
                continue
            if cap["is_bound"]:
                bounded = True
            per = curves(task, args.methods, args.delta, args.budget,
                         suite=args.suite, protocol=args.protocol, width=w)
            for m, c in per.items():
                cover.setdefault(m, []).append(
                    1.0 - np.clip(c / max(cap["capacity"], 1), 0.0, 1.0)
                )
        c_per_task[tid] = {m: np.mean(v, axis=0) for m, v in cover.items()}

    order = [m for m in DISPLAY_ORDER if any(m in v for v in v_per_task.values())]

    nrow = len(task_ids)
    fig, axes = plt.subplots(nrow, 2, figsize=(9.6, 3.6 * nrow))
    axes = np.atleast_2d(axes)
    x = np.arange(1, args.budget + 1)

    for row, tid in enumerate(task_ids):
        v_mean, c_mean = v_per_task[tid], c_per_task[tid]
        ax_v, ax_c = axes[row, 0], axes[row, 1]
        for m in order:
            if m in v_mean:
                ax_v.plot(x, v_mean[m][: args.budget], label=label(m),
                          **STYLES.get(m, dict(lw=1.5)))
            if m in c_mean:
                ax_c.plot(x, c_mean[m][: args.budget], label=label(m),
                          **STYLES.get(m, dict(lw=1.5)))
        ax_v.set_ylabel(f"{tid}\nviolation regret", fontsize=9)
        ax_v.set_yscale("symlog", linthresh=1e-3)
        ax_c.set_ylabel("coverage regret", fontsize=9)
        ax_c.set_ylim(0, 1)
        if row == 0:
            ax_v.set_title("(a) how fast a method gets into spec", fontsize=9)
            ax_c.set_title("(b) share of the acceptable set returned", fontsize=9)
        for ax in (ax_v, ax_c):
            ax.set_xlabel("evaluation")
            ax.set_xlim(1, args.budget)
            ax.grid(alpha=0.25, lw=0.5)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)

    handles, labels_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_, frameon=False, fontsize=8, loc="lower center",
              ncol=min(4, len(handles)), bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1))

    out = args.out or f"regret_{args.suite}_{args.protocol}_grid.png"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")
    if bounded:
        print("[note] at least one capacity is a lower bound (greedy scan cap), "
              "so a coverage panel understates regret there")
    for tid in task_ids:
        v_mean, c_mean = v_per_task[tid], c_per_task[tid]
        print(f"[{tid}]")
        for m in order:
            if m not in v_mean:
                continue
            vr = v_mean[m][args.budget - 1]
            cr = c_mean[m][args.budget - 1] if m in c_mean else float("nan")
            print(f"  {label(m):42s} violation@50={vr:8.4f}  coverage@50={cr:5.3f}")


if __name__ == "__main__":
    main()
