"""Freeze the HOIP target windows.

    PYTHONPATH=core:. .venv-eval/bin/python -m eval.scripts.freeze_hoip_specs
    #   --check   re-derive and compare against the committed specs, writing nothing

This is HOIP's counterpart of `scripts/calibrate_ranges.py`, and it is much
shorter for one reason: **there is nothing to calibrate.** MatFormBench supplied
one-sided thresholds and Olympus supplied only `minimize`, so on both suites a
two-sided window had to be constructed by bisecting a quantile until a target
valid fraction was hit. HOIP supplies the window outright -- upstream's success
test is `abs(bandgap - 1.25) < 0.5 and m_star < 4.` -- so the window is written
down, not derived, and the achieved valid fraction is whatever it is.

What still has to be produced is the `calibration` record every downstream
criterion reads. `metrics.run_metrics` computes both ball criteria
unconditionally and raises without `y_reference_center` / `y_reference_scale`,
and the tolerance-ball baselines place their target at `y_valid_centroid`. All
three are computed **by exact enumeration** over the catalogue's feasible
materials -- the design space is finite, so unlike every other suite there is no
sampling error and no reference-measure choice to defend.

Robust statistics, for the reason recorded in `benchmarks/base.RangeSpec`:
`m_star` spans 0.15 to 364 over 109 measured materials, so its standard deviation
is set by a handful of blow-ups. IQR/1.349 is what stays comparable.
"""

from __future__ import annotations

import argparse

import numpy as np

from eval.benchmarks import calibration
from eval.benchmarks.base import RangeSpec
from eval.benchmarks.hoip import data as hd
from eval.benchmarks.hoip import ranges as hranges
from eval.benchmarks.hoip import tasks as htasks

WIDTH = "native"


def _finite(column: np.ndarray) -> np.ndarray:
    return column[np.isfinite(column)]


def build_spec(task_id: str) -> RangeSpec:
    task = htasks.load_task(task_id)
    keys = htasks.materials(task_id)
    responses = hd.response_map()

    feasible_keys = [k for k in keys if k in responses]
    Y = np.asarray([responses[k] for k in feasible_keys], dtype=float)

    gap_lo, gap_hi = hd.NATIVE_WINDOW["bandgap"]
    lower = {"bandgap": float(gap_lo), "m_star": float(htasks.M_STAR_FLOOR)}
    upper = {"bandgap": float(gap_hi), "m_star": float(hd.NATIVE_WINDOW["m_star"][1])}

    valid = np.ones(len(Y), dtype=bool)
    for j, name in enumerate(task.y_names):
        valid &= (Y[:, j] >= lower[name]) & (Y[:, j] <= upper[name])

    marginals = {
        name: float(np.nanmean((Y[:, j] >= lower[name]) & (Y[:, j] <= upper[name])))
        for j, name in enumerate(task.y_names)
    }
    centroid = {
        name: (float(np.median(_finite(Y[valid, j]))) if valid.any() and _finite(Y[valid, j]).size
               else 0.5 * (lower[name] + upper[name]))
        for j, name in enumerate(task.y_names)
    }

    n_all = len(keys)
    n_feasible = len(feasible_keys)
    achieved = float(valid.sum()) / n_feasible

    return RangeSpec(
        lower=lower,
        upper=upper,
        width_label=WIDTH,
        calibration={
            "rule": (
                "benchmark-supplied window, not calibrated: upstream's success "
                "test abs(bandgap - 1.25) < 0.5 and m_star < 4. m_star's lower "
                "bound is the physical floor 0, not a fitted quantile."
            ),
            "constructed": False,
            "target_fraction_of_feasible": None,
            "achieved_fraction_of_feasible": achieved,
            "achieved_fraction_of_all": float(valid.sum()) / n_all,
            "alpha": None,
            # Both properties are capped by the specification, so the "native"
            # one-sided reading M1 gets is the joint cap rate.
            "native_joint_rate_of_feasible": float(
                np.mean((Y[:, 0] <= upper["bandgap"]) & (Y[:, 1] <= upper["m_star"]))
            ),
            "input_feasible_fraction": n_feasible / n_all,
            # Exact: the catalogue is enumerated, not sampled.
            "n_samples": int(n_all),
            "n_input_feasible": int(n_feasible),
            "n_valid": int(valid.sum()),
            "exact_enumeration": True,
            "marginal_valid_fraction": marginals,
            "y_valid_centroid": centroid,
            "y_reference_center": {
                name: float(np.median(_finite(Y[:, j])))
                for j, name in enumerate(task.y_names)
            },
            "y_reference_scale": {
                name: float(max(calibration._robust_scale(_finite(Y[:, j])), 1e-12))
                for j, name in enumerate(task.y_names)
            },
            "y_reference_mean": {
                name: float(np.mean(_finite(Y[:, j]))) for j, name in enumerate(task.y_names)
            },
            "y_reference_std": {
                name: float(max(np.std(_finite(Y[:, j])), 1e-12))
                for j, name in enumerate(task.y_names)
            },
            "n_censored_m_star": int(np.isnan(Y[:, 1]).sum()),
            "seed": None,
        },
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    task_ids = args.tasks or list(htasks.SELECTED_TASKS)
    header = (f"{'task':12s} {'feasible':>9s} {'valid':>6s} {'of_feasible':>12s} "
              f"{'of_all':>8s}  window")
    print(header)
    print("-" * len(header))

    mismatches = []
    for task_id in task_ids:
        spec = build_spec(task_id)
        cal = spec.calibration
        window = "  ".join(
            f"{n}:[{spec.lower[n]:g}, {spec.upper[n]:g}]" for n in spec.lower
        )
        print(f"{task_id:12s} {cal['n_input_feasible']:9d} {cal['n_valid']:6d} "
              f"{cal['achieved_fraction_of_feasible']:11.3%} "
              f"{cal['achieved_fraction_of_all']:7.3%}  {window}")
        if args.check:
            committed = hranges.load(task_id, WIDTH)
            if (committed.lower, committed.upper) != (spec.lower, spec.upper):
                mismatches.append(f"{task_id}/{WIDTH}")
        else:
            hranges.save(task_id, spec)

    if args.check:
        print("\nspecs match the committed windows" if not mismatches
              else f"\nMISMATCH: {mismatches}")
        raise SystemExit(1 if mismatches else 0)
    print(f"\nwrote {len(task_ids)} specs to {calibration.spec_dir('hoip')}")


if __name__ == "__main__":
    main()
