"""What the valid region *is*, before any method is run.

    PYTHONPATH=core:. .venv-eval/bin/python -m eval.scripts.valid_region_geometry \
        --suite olympus

Two properties of a frozen window, both of which the results have to be read
against:

**Connected components** (2-D tasks only, by flood fill on a grid). The §5.2
diversity argument is that a window near the optimum of a multi-modal surface
splits the valid region into disconnected pieces, so recovering diverse designs
requires finding several of them. Branin has three global minima, so the argument
*predicts* three components -- but the calibration rule keeps the native
threshold as the upper bound and tightens downward, which places the window as a
level-set shell just inside the specification rather than a blob at the optimum.
Whether that shell separates is a fact about the frozen spec, not an assumption,
and a width where it does not separate demonstrates something different from one
where it does.

Measured on Branin, and the resolution matters: at grid=700 the narrow window
reports 75 components, but at 1400 and 2800 it is stably 3. The 72 extras were
the grid slicing a thin level-set shell, not real structure -- hence the default
of 1400 and this note. The converged counts are 4 / 3 / 3 for wide / medium /
narrow; wide's fourth component is genuine (it scales with grid area, ~0.2% of
the valid set) rather than an artifact. Medium is the cleanest demonstration:
three components of comparable size.

**delta-separated capacity.** A greedy count of how many mutually delta-apart
valid designs *exist*, which is an upper bound on the primary metric (distinct
qualifying designs at delta = 0.1). Without it a method returning 6 distinct
designs cannot be read: 6 out of a possible 8 and 6 out of a possible 400 are
different results. Computed on a uniform sample, so it is a lower bound on the
true capacity and is reported as such -- except on a suite whose oracle offers
`enumerate_designs`, where the whole design space is scored and the count is
exact. HOIP is such a suite, and there its three tasks share a valid set by
construction, so the capacity is not merely known but identical across them.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from eval.benchmarks import calibration
from eval.benchmarks.registry import resolve
from eval.harness.aggregate import DEFAULT_DELTA


def _greedy_capacity(Z: np.ndarray, delta: float, cap: int = 4000) -> int:
    """Greedy count of mutually `delta`-apart rows, in normalized coordinates.

    Same greedy rule `aggregate._greedy_distinct` applies to proposals, so the
    number is comparable to what a method is credited with. Greedy is not the
    maximum packing, but it is the same estimator on both sides.
    """
    kept: list[np.ndarray] = []
    for z in Z[:cap]:
        if all(np.linalg.norm(z - k) > delta for k in kept):
            kept.append(z)
    return len(kept)


def component_labels(oracle, task, spec, grid: int) -> np.ndarray:
    """Flood-fill the valid set on a `grid x grid` mesh; 4-connectivity.

    Returns a label grid: 0 for invalid, 1..k for the components ordered largest
    first, so a label is stable across callers. `assign_components` maps designs
    onto it, which is how "how many components did this method touch?" is
    answered without re-deriving the geometry.
    """
    axis = (np.arange(grid) + 0.5) / grid
    lo, hi = task.bounds
    P = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape(-1, 2)
    Y = oracle.truth(lo + P * (hi - lo)).Y
    mask = spec.hits(Y, task.y_names).reshape(grid, grid)

    labels = np.zeros((grid, grid), dtype=int)
    blobs = []
    for start in zip(*np.nonzero(mask)):
        if labels[start]:
            continue
        stack, cells = [start], []
        labels[start] = -1
        while stack:
            i, j = stack.pop()
            cells.append((i, j))
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                a, b = i + di, j + dj
                if 0 <= a < grid and 0 <= b < grid and mask[a, b] and not labels[a, b]:
                    labels[a, b] = -1
                    stack.append((a, b))
        blobs.append(cells)

    for new_id, cells in enumerate(sorted(blobs, key=len, reverse=True), start=1):
        for i, j in cells:
            labels[i, j] = new_id
    return labels


def assign_components(labels: np.ndarray, task, X: np.ndarray) -> np.ndarray:
    """Component id of each design (0 = outside the valid set on this mesh)."""
    grid = labels.shape[0]
    lo, hi = task.bounds
    span = np.where(hi - lo == 0, 1.0, hi - lo)
    idx = np.clip(((np.asarray(X, dtype=float) - lo) / span * grid).astype(int), 0, grid - 1)
    return labels[idx[:, 0], idx[:, 1]]


def _components_2d(oracle, task, spec, grid: int) -> tuple[int, float, list[int]]:
    labels = component_labels(oracle, task, spec, grid)
    sizes = [int((labels == i).sum()) for i in range(1, labels.max() + 1)]
    return len(sizes), float((labels > 0).mean()), sizes


def main() -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="olympus")
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--widths", nargs="*", default=None,
                    help="default: the suite's own conditions (HOIP has only 'native')")
    ap.add_argument("--sample", type=int, default=400_000)
    ap.add_argument("--grid", type=int, default=1400)
    ap.add_argument("--delta", type=float, default=DEFAULT_DELTA)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--json", default=None,
                    help="also write the capacities here, keyed (suite, task, width). "
                         "Merges into an existing file rather than replacing it, so "
                         "one file can hold every suite.")
    args = ap.parse_args()

    suite = resolve(args.suite)
    task_ids = args.tasks or list(suite.default_tasks)
    widths = args.widths or list(suite.default_widths)

    # Whether the design space is sampled or enumerated is not a detail: it
    # decides whether `capacity` is a lower bound or the answer.
    probe = resolve(args.suite).make_oracle(suite.load_task(task_ids[0]))
    exact_suite = hasattr(probe, "enumerate_designs")
    measure = ("design space enumerated (capacity is exact)" if exact_suite
               else f"uniform sample = {args.sample:,}")
    print(f"delta = {args.delta}, {measure}, grid = {args.grid}^2\n")
    head = (f"{'task':16s} {'width':7s} {'valid%':>7s} {'n_valid':>8s} "
            f"{'capacity':>9s} {'components':>11s}  largest components")
    print(head)
    print("-" * len(head))

    frozen: dict = {}
    for task_id in task_ids:
        task = suite.load_task(task_id)
        oracle = suite.make_oracle(task)
        # A catalogue benchmark can answer this exactly: its design space is a
        # finite set, so the capacity below is the true greedy count rather than
        # a lower bound from a uniform sample. Absent on the continuous suites.
        enumerate_designs = getattr(oracle, "enumerate_designs", None)
        exact = enumerate_designs is not None
        X = (enumerate_designs() if exact
             else calibration.sample_design_space(task, args.sample, seed=args.seed))
        obs = oracle.truth(X)
        feasible = obs.feasible & np.isfinite(obs.Y).all(axis=1)

        for width in widths:
            spec = suite.load_range_spec(task_id, width)
            valid = feasible & spec.hits(obs.Y, task.y_names)

            lo, hi = task.bounds
            span = np.where(hi - lo == 0, 1.0, hi - lo)
            Z = (X[valid] - lo) / span
            capacity = _greedy_capacity(Z, args.delta) if valid.any() else 0

            if task.dim == 2:
                n_comp, frac, sizes = _components_2d(oracle, task, spec, args.grid)
                comp = f"{n_comp:>11d}"
                extra = "  " + ", ".join(str(s) for s in sizes[:5])
                extra += f"   (grid valid {frac:.2%})"
            else:
                comp = f"{'n/a (dim>2)':>11s}"
                extra = ""

            # Only a sampled capacity needs a caveat; the header already says
            # when the space was enumerated instead.
            note = ("" if exact else
                    "  >= capacity is a lower bound (uniform sample)" if capacity >= 3900 else "")
            print(f"{task_id:16s} {width:7s} {valid.mean():6.2%} {int(valid.sum()):8d} "
                  f"{capacity:9d} {comp}{extra}{note}")

            # `is_bound` is the field consumers must respect: a sampled capacity
            # that hit the greedy scan cap is a floor, and dividing by it would
            # overstate coverage. `_greedy_capacity` caps its scan at 4000.
            frozen[f"{args.suite}/{task_id}/{width}"] = {
                "suite": args.suite, "task_id": task_id, "width": width,
                "delta": args.delta, "capacity": int(capacity),
                "n_valid": int(valid.sum()),
                "exact": bool(exact),
                "is_bound": bool(not exact and capacity >= 3900),
                "measure": ("enumerated" if exact else f"uniform-{args.sample}"),
            }

    if args.json:
        out = Path(args.json)
        # Merge: each suite is measured by its own invocation (they need
        # different samplers and take very different times), and a plain
        # overwrite would silently drop the ones not in this run.
        existing = {}
        if out.exists():
            existing = json.loads(out.read_text(encoding="utf-8"))
        existing.update(frozen)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
        print(f"\nwrote {len(frozen)} capacities to {out} "
              f"({len(existing)} total in file)")


if __name__ == "__main__":
    main()
