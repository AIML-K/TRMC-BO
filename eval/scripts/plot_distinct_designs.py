"""Plot where each method's qualifying formulations actually landed.

The point of the figure is the difference between *hits* and *distinct designs*.
Two methods can return the same number of in-window evaluations while one has
converged onto a single basin and the other has found several separated recipes.
A table shows that as a ratio; a scatter shows it at a glance.

Design spaces here are 5-15 dimensional, so the panels project onto two
principal components fitted on the **valid region of the calibration sample** --
not on the proposals. Fitting on the proposals would give each method its own
axes and make the panels incomparable; fitting on the valid region keeps one
shared frame and puts the axes where the target actually is.

    PYTHONPATH=core:. .venv-eval/bin/python -m eval.scripts.plot_distinct_designs \
        --task L1-1 --width medium
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from eval.benchmarks import calibration  # noqa: E402
from eval.benchmarks.registry import resolve  # noqa: E402
from eval.harness.aggregate import (  # noqa: E402
    DEFAULT_DELTA,
    DISPLAY_ORDER,
    _normalize_designs,
    label,
    valid_designs,
)

OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "figures"

#: Panels to draw, and in what order. Four panels for the compact main-text
#: figure: random search as the floor, the two concentrating baselines
#: (point-target and the tolerance ball) that win hit rate, and the full method
#: that wins diversity -- the reversal made visible as geometry rather than as a
#: ratio. Standard BO is omitted: it finds ~0.2 designs per run, so its panel is
#: empty and says nothing the table does not already say. The C1-only and
#: C1+C2 ablation panels (`M3_range_prob`, `M4_range_prob_constraints`) that
#: were here previously belong to the ablation story, not the reversal story,
#: and `M4_range_prob_constraints` additionally names a configuration that did
#: not run as advertised (see `eval/scripts/emit_tables.py`'s `FACTORIAL_ROWS`
#: note) -- pass `--methods` explicitly to reproduce the old 6-panel figure.
DEFAULT_PANELS = [
    "M0_random",
    "M2_target_distance",
    "M6b_range_aware_tb_inscribed",
    "M5_full",
]


def build_frame(suite, task, width: str, n_background: int, seed: int):
    """Shared 2-D frame: PCA fitted on the valid part of a uniform sample.

    Fitted on the in-window subset even though the plotted background is every
    feasible design: the axes should be oriented by the region the methods are
    aiming at, not by the bulk of the domain.
    """
    oracle = suite.make_oracle(task)
    spec = suite.load_range_spec(task.task_id, width)
    X = calibration.sample_design_space(task, n_background, seed=seed)
    obs = oracle.truth(X)
    ok = obs.feasible & np.isfinite(obs.Y).all(axis=1)
    valid = ok & spec.hits(obs.Y, task.y_names)

    Z = _normalize_designs(X, task)
    if valid.sum() < 3:
        raise SystemExit(
            f"{task.task_id}/{width}: only {int(valid.sum())} valid designs in "
            f"{n_background} samples — raise --background"
        )
    pca = PCA(n_components=2).fit(Z[valid])
    return pca, Z, ok, valid, spec


def draw_panel(ax, method: str, *, pca, Pbg, feasible, designs, xcols, task,
               delta: float, cells_per_method: dict[str, int]) -> None:
    """Draw one method's panel (feasible background + duplicate/distinct hits).

    Shared by `main()` (single-task figure) and `plot_distinct_designs_grid`
    (multi-task figure) so the two figures read as one system rather than as
    independently-tuned plots.
    """
    # Three layers only, drawn light-to-dark so the found designs read on top.
    ax.scatter(Pbg[feasible, 0], Pbg[feasible, 1], s=1.2, c="#8ecae6",
               alpha=0.30, linewidths=0, rasterized=True,
               label="feasible design")

    g = designs[designs["method"] == method]
    n_valid = n_distinct = 0
    if not g.empty:
        P = pca.transform(_normalize_designs(g[xcols].to_numpy(float), task))
        dup, dis = ~g["distinct"].to_numpy(), g["distinct"].to_numpy()
        # Duplicates are drawn larger than the distinct markers on purpose:
        # a pile-up then reads as a grey halo around the black square instead
        # of hiding underneath it, which is the whole point of the figure.
        ax.scatter(P[dup, 0], P[dup, 1], s=42, c="#a3a3a3", marker="o",
                   linewidths=0, alpha=0.75, label="hit, near-duplicate")
        ax.scatter(P[dis, 0], P[dis, 1], s=30, c="black", marker="s",
                   edgecolors="white", linewidths=0.7,
                   label=f"distinct hit (> {delta})")
        n_valid, n_distinct = len(g), int(dis.sum())

    # Count the cells that RAN, not the ones that hit. `designs` holds only
    # in-window rows, so a seed that found nothing contributes no row and
    # `g["seed"].nunique()` silently drops it from the denominator -- which
    # inflated the baselines' per-run counts (point-target read 17.8 hits
    # over 9 hitting seeds where 10 ran, i.e. 16.0) and made this figure
    # disagree with `tab_main`. `plot_discovery_curves` pads the same way.
    seeds = max(cells_per_method.get(method, 0), 1)
    conc = f"{100 * n_distinct / n_valid:.0f}%" if n_valid else "-"
    ax.set_title(
        f"{label(method)}\n"
        f"{n_valid / seeds:.1f} hits  →  {n_distinct / seeds:.1f} distinct "
        f"per run  ({conc})",
        fontsize=9.5,
    )
    ax.tick_params(labelsize=7, length=2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.12, linewidth=0.5)
    ax.set_axisbelow(True)


def main() -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="matformbench")
    ap.add_argument("--protocol", default="range_adapted")
    ap.add_argument("--task", default="L1-1")
    ap.add_argument("--width", default="medium")
    ap.add_argument("--methods", nargs="*", default=DEFAULT_PANELS)
    ap.add_argument("--delta", type=float, default=DEFAULT_DELTA)
    ap.add_argument("--background", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    suite = resolve(args.suite)
    task = suite.load_task(args.task)
    pca, Zbg, feasible, valid, _ = build_frame(
        suite, task, args.width, args.background, args.seed
    )
    Pbg = pca.transform(Zbg)

    designs = valid_designs(task, suite=args.suite, protocol=args.protocol,
                            width=args.width)
    # `attrs["cells"]` lists every (task, width, method, seed) that ran, hits or
    # not; it is the only record here of the cells a method found nothing in.
    cells_per_method: dict[str, int] = {}
    for _, _w, _m, _s in designs.attrs.get("cells", []):
        cells_per_method[_m] = cells_per_method.get(_m, 0) + 1
    xcols = [f"x_{n}" for n in task.x_names]

    panels = [m for m in DISPLAY_ORDER if m in args.methods]
    # 2 columns fits the default 4-panel set as a clean 2x2; 3 left two axes
    # blank for that count and only suited the old 6-panel default.
    ncol = 2 if len(panels) <= 4 else 3
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.1 * ncol, 3.9 * nrow),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, method in zip(axes, panels):
        draw_panel(ax, method, pca=pca, Pbg=Pbg, feasible=feasible,
                   designs=designs, xcols=xcols, task=task, delta=args.delta,
                   cells_per_method=cells_per_method)

    for ax in axes[len(panels):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
                     fontsize=10, markerscale=2.2, handletextpad=0.5)
    for h in leg.legend_handles:
        h.set_alpha(1.0)
    n_seeds = max(cells_per_method.values(), default=0)
    ev = 100 * pca.explained_variance_ratio_.sum()
    # Font size and wrapping scale with panel count: a 2-column figure (the
    # 4-panel default) is ~2/3 the width of the old 3-column one, so the same
    # long suptitle clips at the sides unless it shrinks with it.
    fig.suptitle(
        f"{task.task_id} / {args.width}, {task.dim}D projected on PC1-PC2 "
        f"({ev:.0f}% variance) — {n_seeds} seeds pooled, counts per run"
        if ncol <= 2 else
        f"{task.task_id} / {args.width} — where the qualifying formulations landed\n"
        f"{task.dim}D design space projected on PC1-PC2 of the in-window region "
        f"({ev:.0f}% of its variance); markers pool all {n_seeds} seeds, counts are per run",
        fontsize=9.5 if ncol <= 2 else 11,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else OUT_DIR / f"distinct_{args.task}_{args.width}.png"
    fig.savefig(out, dpi=170)
    print(f"wrote {out}")
    var = pca.explained_variance_ratio_
    print(f"  PC1+PC2 explain {100 * var.sum():.0f}% of the valid region's variance")


if __name__ == "__main__":
    main()
