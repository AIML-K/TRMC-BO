"""Two regret curves for a target-range campaign, side by side.

BO papers are read with regret curves, and this one has none. The reason is real
rather than an oversight: simple regret is ``f(x*) - max_t f(x_t)``, and this
problem has no ``f(x*)``. Every design inside the window satisfies the
specification, a point at the centre is not better than one near an edge, and the
goal names a set rather than a point. Importing extremum regret unmodified would
commit exactly the reduction this paper argues against.

So we define the two analogues that *are* well posed here and plot them together,
because the contrast between them is the paper's first claim:

    violation regret   min over evaluations so far of the normalized distance to
                       the window, ``(1/m) sum_k max(0, L_k - y_k, y_k - U_k) /
                       (U_k - L_k)``. This is the closest thing to classical
                       simple regret: it is monotone, it reaches zero exactly
                       when the first qualifying design is found, and it says how
                       fast a method gets *into* spec.

    coverage regret    ``1 - distinct_t / C_delta``, where ``C_delta`` is the
                       measured delta-separated capacity of the valid region
                       (frozen by `eval/scripts/valid_region_geometry.py --json`).
                       It says how much of the acceptable set the campaign has
                       actually returned.

A concentrating method drives the left panel to zero fastest and then lies flat
near the top of the right one. Same runs, two axes. Cf. Kim et al., *Beyond
Regrets: Geometric Metrics for Bayesian Optimization*, which makes the general
version of this argument: evaluation-value metrics cannot tell whether one
solution was found or many.

    PYTHONPATH=core:. .venv-eval/bin/python -m eval.scripts.plot_regret \
        --suite olympus --protocol branin --out paper/figures/regret_branin.png
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from eval.benchmarks.registry import resolve  # noqa: E402
from eval.harness.aggregate import (  # noqa: E402
    DEFAULT_DELTA,
    DISPLAY_ORDER,
    RESULT_ROOT,
    label,
    load_runs,
)
from eval.scripts.plot_discovery_curves import DEFAULT_METHODS, curves  # noqa: E402

CAPACITY_PATH = Path("eval/specs/capacity.json")

#: Shared with `plot_discovery_curves` so the two figures read as one system.
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


def violation_regret(
    task, methods, budget, root: Path = RESULT_ROOT, suite: str = "olympus",
    protocol: str = "branin", width: str | None = None, max_seed: int | None = 10,
) -> dict[str, np.ndarray]:
    """method -> mean running-min normalized distance to the window, (budget,).

    The per-iteration ``violation_total`` column is already width-normalized and
    averaged over properties (`eval/benchmarks/base.py`), so the only work here is
    the running minimum -- and one correction that matters.

    **Failed evaluations must be masked.** `RangeSpec.violations` wraps its clip
    in ``nan_to_num``, so a design that returned no value at all scores a
    violation of exactly 0.0 -- the same as a perfect hit. Taking a cummin over
    the raw column therefore hands a regret of zero to any method whose
    evaluation crashed, which on the catalogue suite is most of them. We drop
    rows that are not marked feasible or whose outputs are missing, and carry the
    previous best forward instead.
    """
    base = Path(root) / suite / protocol / task.task_id
    ycols = [f"y_true_{n}" for n in task.y_names]
    out: dict[str, list[np.ndarray]] = {m: [] for m in methods}

    for path in sorted(base.glob("*/*/*/trace.parquet")):
        width_dir, method, seed_dir = path.parts[-4], path.parts[-3], path.parts[-2]
        if method not in out:
            continue
        if width and width_dir != width:
            continue
        seed = int(seed_dir.replace("seed", ""))
        if max_seed is not None and seed >= max_seed:
            continue
        trace = pd.read_parquet(path)
        loop = trace[~trace["is_init"]].sort_values("iteration")
        if loop.empty:
            continue

        v = loop["violation_total"].to_numpy(float).copy()
        # The mask: a row only carries a meaningful violation if the design was
        # actually evaluated. `feasible` is the input-space verdict; a NaN in any
        # true output means the oracle returned nothing for that property.
        usable = np.ones(len(loop), dtype=bool)
        if "feasible" in loop.columns:
            usable &= loop["feasible"].fillna(False).to_numpy(bool)
        present = [c for c in ycols if c in loop.columns]
        if present:
            usable &= np.isfinite(loop[present].to_numpy(float)).all(axis=1)
        v[~usable] = np.inf

        run = np.minimum.accumulate(v)
        curve = np.full(budget, np.nan)
        it = loop["iteration"].to_numpy(int)
        for k, i in enumerate(it):
            if 1 <= i <= budget:
                curve[i - 1] = run[k]
        # Iterations the trace skipped inherit the running best.
        curve = pd.Series(curve).ffill().to_numpy()
        out[method].append(curve)

    return {m: np.nanmean(np.vstack(v), axis=0) for m, v in out.items() if v}


def load_capacity(suite: str, task_id: str, width: str, delta: float) -> dict | None:
    if not CAPACITY_PATH.exists():
        return None
    data = json.loads(CAPACITY_PATH.read_text(encoding="utf-8"))
    entry = data.get(f"{suite}/{task_id}/{width}")
    if entry is None or abs(entry.get("delta", delta) - delta) > 1e-12:
        return None
    return entry


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
    # Default to the tasks this protocol actually has on disk, not the suite's
    # full list: `olympus` spans two protocols with disjoint task sets, and
    # asking for the other one's tasks only produces missing-capacity warnings.
    if args.tasks:
        task_ids = args.tasks
    else:
        runs = load_runs(suite=args.suite, protocol=args.protocol)
        if runs.empty:
            raise SystemExit(f"no runs under {args.suite}/{args.protocol}")
        task_ids = sorted(runs["task_id"].unique())
    widths = [args.width] if args.width else list(suite.default_widths)
    kw = {"suite": args.suite, "protocol": args.protocol, "width": args.width}

    viol: dict[str, list[np.ndarray]] = {}
    cover: dict[str, list[np.ndarray]] = {}
    bounded = False

    for tid in task_ids:
        task = suite.load_task(tid)
        for m, c in violation_regret(task, args.methods, args.budget, **kw).items():
            viol.setdefault(m, []).append(c)

        # Coverage needs a per-(task, width) denominator, so it is accumulated one
        # width at a time even when the panel pools them.
        for w in widths:
            cap = load_capacity(args.suite, tid, w, args.delta)
            if cap is None:
                print(f"[warn] no frozen capacity for {args.suite}/{tid}/{w}; "
                      f"run valid_region_geometry --json {CAPACITY_PATH}")
                continue
            if cap["is_bound"]:
                bounded = True
            per = curves(task, args.methods, args.delta, args.budget,
                         suite=args.suite, protocol=args.protocol, width=w)
            for m, c in per.items():
                cover.setdefault(m, []).append(
                    1.0 - np.clip(c / max(cap["capacity"], 1), 0.0, 1.0)
                )

    if not viol:
        raise SystemExit("no cells found")

    v_mean = {m: np.mean(v, axis=0) for m, v in viol.items()}
    c_mean = {m: np.mean(v, axis=0) for m, v in cover.items()}
    order = [m for m in DISPLAY_ORDER if m in v_mean]

    x = np.arange(1, args.budget + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))

    for m in order:
        axes[0].plot(x, v_mean[m][: args.budget], label=label(m),
                     **STYLES.get(m, dict(lw=1.5)))
        if m in c_mean:
            axes[1].plot(x, c_mean[m][: args.budget], label=label(m),
                         **STYLES.get(m, dict(lw=1.5)))

    axes[0].set_ylabel("violation regret (window widths)")
    axes[0].set_yscale("symlog", linthresh=1e-3)
    axes[0].set_title("(a) how fast a method gets into spec\n"
                      "(best distance to the window so far)", fontsize=9)
    axes[1].set_ylabel("coverage regret  $1-$ distinct$/C_\\delta$")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("(b) how much of the acceptable set it returns\n"
                      "(share of the $\\delta$-separated capacity still missing)", fontsize=9)

    for ax in axes:
        ax.set_xlabel("evaluation")
        ax.set_xlim(1, args.budget)
        ax.grid(alpha=0.25, lw=0.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    # One legend under both panels rather than inside either: the point of the
    # figure is that a line's position swaps between them, which is hard to read
    # if a legend box is covering one of the two.
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, frameon=False, fontsize=8,
               loc="lower center", ncol=min(3, len(handles)),
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.10, 1, 1))

    out = args.out or f"regret_{args.suite}_{args.protocol}.png"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")
    if bounded:
        print("[note] at least one capacity is a lower bound (greedy scan cap), "
              "so panel (b) understates regret there")
    for m in order:
        vr = v_mean[m][args.budget - 1]
        cr = c_mean[m][args.budget - 1] if m in c_mean else float("nan")
        print(f"  {label(m):42s} violation@50={vr:8.4f}  coverage@50={cr:5.3f}")


if __name__ == "__main__":
    main()
