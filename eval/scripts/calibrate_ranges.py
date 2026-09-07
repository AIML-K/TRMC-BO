"""Freeze the range-adapted target windows (paper §5.1).

Run once per suite. The resulting JSON under `eval/specs/<suite>/` is committed
and every sweep reads it, so all methods and seeds see identical targets.

    PYTHONPATH=core:. .venv-eval/bin/python -m eval.scripts.calibrate_ranges \
        --suite matformbench
    PYTHONPATH=core:. .venv-eval/bin/python -m eval.scripts.calibrate_ranges \
        --suite olympus

`--check` recalibrates and compares against what is already committed without
writing, which is how a refactor of the calibration code is shown not to have
moved any window.
"""

from __future__ import annotations

import argparse

from eval.benchmarks import calibration
from eval.benchmarks.registry import resolve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="matformbench")
    ap.add_argument("--tasks", nargs="*", default=None, help="default: the suite's own selection")
    ap.add_argument("--n-samples", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=20260820)
    ap.add_argument(
        "--check",
        action="store_true",
        help="compare against the committed specs instead of overwriting them",
    )
    args = ap.parse_args()

    suite = resolve(args.suite)
    task_ids = args.tasks or list(suite.default_tasks)

    header = f"{'task':16s} {'width':7s} {'alpha':>7s} {'achieved':>9s} {'of_all':>8s}  window"
    print(header)
    print("-" * len(header))

    mismatches = []
    for task_id in task_ids:
        task = suite.load_task(task_id)
        oracle = suite.make_oracle(task)
        # One scoring pass per task, shared by all three difficulty levels.
        reference = calibration.reference_sample(task, oracle, args.n_samples, args.seed)
        for width in calibration.DIFFICULTY:
            spec = calibration.calibrate(
                task, oracle, width, n_samples=args.n_samples, seed=args.seed,
                reference=reference,
            )
            if args.check:
                committed = suite.load_range_spec(task_id, width)
                if (committed.lower, committed.upper) != (spec.lower, spec.upper):
                    mismatches.append(f"{task_id}/{width}")
            else:
                calibration.save(args.suite, task_id, spec)

            c = spec.calibration
            window = " ".join(
                f"{n}:[{spec.lower[n]:.4g},{spec.upper[n]:.4g}]" for n in task.y_names
            )
            print(
                f"{task_id:16s} {width:7s} {c['alpha']:7.4f} "
                f"{c['achieved_fraction_of_feasible']:8.2%} "
                f"{c['achieved_fraction_of_all']:7.2%}  {window}"
            )

    if args.check:
        if mismatches:
            print(f"\n{len(mismatches)} window(s) differ from the committed specs: "
                  f"{', '.join(mismatches)}")
            raise SystemExit(1)
        print(f"\nall {len(task_ids) * len(calibration.DIFFICULTY)} windows match the committed specs")


if __name__ == "__main__":
    main()
