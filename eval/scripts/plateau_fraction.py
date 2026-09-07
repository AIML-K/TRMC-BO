"""How much of the design space the single-output C1 acquisition cannot see.

    PYTHONPATH=core:. .venv-eval/bin/python -m eval.scripts.plateau_fraction \
        --suite olympus --task branin --width medium

With one output, C1 runs through the shipped `CDFRangeObjective` + qEI path. That
objective is a deterministic function of `x`, so the expectation over posterior
samples has nothing to average and qEI collapses to its own integrand:

    A_C1(x) = max(0, P(L <= f(x) <= U) - best_f)

Everything with `P <= best_f` is therefore tied at *exactly* zero -- no value, no
gradient. This script measures how large that region is, per iteration, by
refitting the surrogate on the prefix of a completed run's trace and evaluating
`P` on a large uniform sample. `benchmarks/olympus/tasks.py` and
`predictions_olympus.md` explain why it matters.

**This is a mechanism measurement, not evidence for a campaign-level claim.** The
lesson is in `results_issue_matformbench.md`: a one-step acquisition-ranking
diagnostic said the §4.2 product aggregation was far better than the shipped path
(AUC 0.641 vs 0.542), and campaigns reversed it -- the product raises hit rate and
destroys diversity. A diagnostic must be read against the endpoint it measures. So
this quantifies the plateau's *size*; whether the plateau helps or hurts is
settled only by the sweep.

Reads a finished cell rather than running its own campaign, so the numbers refer
to states the method actually visited.
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
import torch

from eval.benchmarks import calibration
from eval.benchmarks.registry import resolve
from eval.harness.runner import RESULT_ROOT


def range_probability(model, Z: torch.Tensor, lower: float, upper: float) -> np.ndarray:
    """`P(lower <= f(x) <= upper)` in the model's standardized output space."""
    with torch.no_grad():
        posterior = model.posterior(Z)
        mean = posterior.mean[..., 0]
        std = posterior.variance[..., 0].clamp_min(1e-12).sqrt()
        normal = torch.distributions.Normal(torch.zeros_like(mean), torch.ones_like(mean))
        return (normal.cdf((upper - mean) / std) - normal.cdf((lower - mean) / std)).numpy()


def main() -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="olympus")
    ap.add_argument("--protocol", default="branin")
    ap.add_argument("--task", default="branin")
    ap.add_argument("--width", default="medium")
    ap.add_argument("--method", default="M3_range_prob")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--probe", type=int, default=20_000)
    ap.add_argument("--every", type=int, default=5)
    args = ap.parse_args()

    from bayesian_optimization import initialize_model
    from botorch import fit_gpytorch_mll
    from botorch.utils.transforms import normalize

    suite = resolve(args.suite)
    task = suite.load_task(args.task)
    spec = suite.load_range_spec(args.task, args.width)
    (y_name,) = task.y_names

    cell = (RESULT_ROOT / args.suite / args.protocol / args.task / args.width
            / args.method / f"seed{args.seed:03d}")
    trace = pd.read_parquet(cell / "trace.parquet")
    X = trace[[f"x_{n}" for n in task.x_names]].to_numpy(float)
    Y = trace[[f"y_obs_{y_name}"]].to_numpy(float)
    n_init = int(trace["is_init"].sum())

    bounds = torch.as_tensor(task.bounds, dtype=torch.double)
    probe = calibration.sample_design_space(task, args.probe, seed=99)
    probe_t = normalize(torch.as_tensor(probe, dtype=torch.double), bounds)

    print(f"{args.suite}/{args.protocol} {args.task}/{args.width} {args.method} "
          f"seed{args.seed}: {len(X)} rows, {n_init} initial")
    print(f"probe: {args.probe:,} uniform designs\n")
    head = (f"{'iter':>5}{'n_obs':>7}{'best_f':>9}{'plateau %':>11}"
            f"{'P max':>9}{'P mean':>9}{'valid % of probe':>18}")
    print(head)
    print("-" * len(head))

    for stop in range(n_init, len(X) + 1, args.every):
        Xi, Yi = X[:stop], Y[:stop]
        labeled = np.isfinite(Yi).all(axis=1)
        if labeled.sum() < 2:
            continue
        Xt = normalize(torch.as_tensor(Xi[labeled], dtype=torch.double), bounds)
        Yt = torch.as_tensor(Yi[labeled], dtype=torch.double)
        mean, std = Yt.mean(0), Yt.std(0).clamp_min(1e-9)
        Yn = (Yt - mean) / std

        mll, model = initialize_model(Xt, Yn, [], torch.ones_like(Yn, dtype=torch.bool))
        fit_gpytorch_mll(mll)

        lo = (float(spec.lower[y_name]) - float(mean)) / float(std)
        hi = (float(spec.upper[y_name]) - float(mean)) / float(std)

        p_train = range_probability(model, Xt, lo, hi)
        p_probe = range_probability(model, probe_t, lo, hi)
        best_f = float(p_train.max())
        plateau = float((p_probe <= best_f).mean())

        # How much of the probe is genuinely in-window, for contrast: a large
        # plateau is expected when little of the space qualifies.
        truth = suite.make_oracle(task).truth(probe)
        valid = float(spec.hits(truth.Y, task.y_names).mean())

        print(f"{stop - n_init:5d}{stop:7d}{best_f:9.4f}{plateau * 100:10.2f}%"
              f"{p_probe.max():9.4f}{p_probe.mean():9.4f}{valid * 100:17.2f}%")


if __name__ == "__main__":
    main()
