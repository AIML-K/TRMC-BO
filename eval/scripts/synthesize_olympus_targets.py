"""Freeze Olympus's synthesized specification caps (milestone O4, first half).

    PYTHONPATH=core:. .venv-eval/bin/python -m eval.scripts.synthesize_olympus_targets

Olympus datasets carry only `default_goal: minimize`, so a two-sided
specification has to be constructed. This writes the one-sided half of it --
`tau = q25(response)` under the uniform reference measure -- to
`eval/specs/olympus/native_targets.json`, which is **committed**.
`calibrate_ranges --suite olympus` then places the lower bounds.

Two properties make this worth a separate frozen artifact rather than a runtime
computation:

* tau needs oracle evaluations, and recomputing it in each of 810 cells would pay
  that cost 810 times.
* A task whose definition is recomputed is a task that can drift. Freezing tau
  makes the task identical across methods, seeds and machines, in the same way
  and for the same reason as freezing the windows.

`--check` recomputes without writing and reports any disagreement, which is how a
change to the sampling or the oracle is shown not to have moved a task.
"""

from __future__ import annotations

import argparse

import numpy as np

from eval.benchmarks import calibration
from eval.benchmarks.olympus import tasks as otasks
from eval.benchmarks.olympus.oracle import make_oracle


def measure(task_id: str, n_samples: int, seed: int) -> dict:
    """Response distribution of the input-feasible part of a uniform sample."""
    task = otasks.provisional_task(task_id)
    oracle = make_oracle(task)
    X = calibration.sample_design_space(task, n_samples, seed=seed)
    obs = oracle.truth(X)

    feasible = obs.feasible & np.isfinite(obs.Y).all(axis=1)
    if not feasible.any():
        raise RuntimeError(f"{task_id}: no input-feasible design in {n_samples} draws")
    y = obs.Y[feasible, 0]

    goal = task.meta.get("default_goal", "minimize")
    quantile = otasks.native_quantile(goal)
    tau = float(np.quantile(y, quantile))
    satisfying = (y <= tau) if goal == "minimize" else (y >= tau)
    return {
        "tau": tau,
        "goal": goal,
        "quantile": quantile,
        "n_samples": int(n_samples),
        "n_input_feasible": int(feasible.sum()),
        "input_feasible_fraction": float(feasible.mean()),
        "seed": int(seed),
        # Enough of the distribution to read a spec without re-running anything.
        "response_min": float(y.min()),
        "response_median": float(np.median(y)),
        "response_max": float(y.max()),
        "fraction_satisfying_native": float(satisfying.mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=list(otasks.SELECTED_TASKS))
    ap.add_argument("--n-samples", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=20260821)
    ap.add_argument("--check", action="store_true",
                    help="recompute and compare against the frozen file")
    args = ap.parse_args()

    measured = {t: measure(t, args.n_samples, args.seed) for t in args.tasks}

    header = (f"{'task':16s} {'tau':>10s} {'P(native)':>10s} {'feasible':>9s} "
              f"{'min':>10s} {'median':>10s} {'max':>10s}")
    print(header)
    print("-" * len(header))
    for task_id, m in measured.items():
        print(f"{task_id:16s} {m['tau']:10.5g} {m['fraction_satisfying_native']:9.2%} "
              f"{m['input_feasible_fraction']:8.2%} {m['response_min']:10.5g} "
              f"{m['response_median']:10.5g} {m['response_max']:10.5g}")

    if args.check:
        frozen = otasks.load_native_targets()["tasks"]
        bad = [t for t, m in measured.items()
               if t not in frozen or not np.isclose(frozen[t]["tau"], m["tau"], rtol=0, atol=1e-12)]
        if bad:
            print(f"\n{len(bad)} task(s) disagree with the frozen file: {', '.join(bad)}")
            raise SystemExit(1)
        print(f"\nall {len(measured)} synthesized caps match the frozen file")
        return

    path = otasks.save_native_targets({
        "rule": (f"tau at the q{int(otasks.NATIVE_SHARE * 100)} tail of the response "
                 f"(q25 for minimize, q75 for maximize) over a uniform "
                 "sample of input-feasible designs; the specification is CONSTRUCTED, "
                 "not supplied by Olympus"
                 ),
        "why_q25": (
            "With one output the calibration is analytic: valid fraction = "
            "P(native) * alpha. Placing tau at the target quantile would force "
            "alpha=1 for the widest condition, collapsing its lower bound onto the "
            "sample minimum and making it one-sided. q25 gives alpha ~ 0.40/0.12/0.04."
        ),
        "tasks": measured,
    })
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
