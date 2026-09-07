"""How much of the design space the range acquisition cannot see, at any m.

    PYTHONPATH=core:. .venv-eval/bin/python -m eval.scripts.acquisition_plateau \
        --suite hoip --task dense

`scripts/plateau_fraction.py` measures this for a single output, where the
mechanism is a hinge: C1 runs `qEI(CDFRangeObjective)`, the objective is a
deterministic function of `x`, so qEI collapses to `max(0, P(L<=f<=U) - best_f)`
and everything below the incumbent is tied at exactly zero. That script hardcodes
`mean[..., 0]` and a scalar window, so it cannot run on a multi-output task.

This one is suite- and m-agnostic, and it exists because the multi-output case
turned out **not** to be safe. Two things have to hold for a multi-output plateau,
and both are measured here rather than argued:

1. **the objective is deterministic in x** -- then qEHVI's average over posterior
   samples has nothing to average, and each candidate contributes the hypervolume
   improvement of a single point;
2. **the observed front dominates the rest of the space** -- then that improvement
   is exactly zero everywhere else.

(2) is where the noise model enters. At an observed design a *noiseless* oracle
collapses the posterior variance, so `P(L <= f <= U)` there goes to 1 in every
coordinate while unobserved designs stay diffuse and score far lower. Nothing
unobserved can then dominate an observed in-window point. Where observations are
noisy the posterior does not collapse and the front stays reachable.

So the question this answers is not "does the acquisition plateau" but "on which
suites, and why" -- and the answer belongs next to any claim about what C1
contributes. Read it against its endpoint: this is a one-state mechanism
measurement, and `results_issue_matformbench.md` records what happens when a
one-step diagnostic is mistaken for a campaign result.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import warnings

import numpy as np
import torch

import bayesian_optimization as bo
from eval.benchmarks import calibration
from eval.benchmarks.registry import resolve
from eval.harness.adapter import BenchmarkVariables
from eval.harness.loop import CampaignConfig, build_initial_design
from eval.methods import make as make_method


def _capture_acquisition(task, spec, X, Y, method_name, restarts, restart_pool):
    """Run one proposal and keep the acquisition object it built."""
    captured: dict = {}
    original = bo.optimize_acqf_on_filtered_points

    def spy(acq_function, optimize_config, dataloader, cat_dims, dis_dims,
            range_spec=None, **kwargs):
        captured["acqf"] = acq_function
        captured["range_spec"] = range_spec
        return original(acq_function, optimize_config, dataloader, cat_dims,
                        dis_dims, range_spec=range_spec, **kwargs)

    method = make_method(method_name, restarts)
    variables = BenchmarkVariables(
        task, spec, X, Y, seed=0,
        objective_mode=getattr(method, "objective_mode", "range"),
        restart_pool=restart_pool,
    )
    bo.optimize_acqf_on_filtered_points = spy
    try:
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            method.propose(variables, q=1, seed=0)
    finally:
        bo.optimize_acqf_on_filtered_points = original
    return captured, variables


def _dominated(front: np.ndarray, rows: np.ndarray) -> int:
    """How many of `rows` some row of `front` weakly dominates and strictly beats."""
    count = 0
    for row in rows:
        if np.any(np.all(front >= row, axis=1) & np.any(front > row, axis=1)):
            count += 1
    return count


def main() -> None:
    warnings.filterwarnings("ignore")
    torch.set_num_threads(2)

    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="hoip")
    ap.add_argument("--task", default=None)
    ap.add_argument("--width", default=None)
    ap.add_argument("--method", default="M3_range_prob")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-init", type=int, default=30)
    # Mid-campaign rather than at the initial design: a plateau that only appears
    # once something in-window has been seen is the interesting case.
    ap.add_argument("--extra", type=int, default=20,
                    help="observations drawn beyond the initial design")
    ap.add_argument("--restarts", type=int, default=16)
    ap.add_argument("--probe", type=int, default=20_000,
                    help="design-space sample; ignored when the space is enumerable")
    args = ap.parse_args()

    suite = resolve(args.suite)
    task_id = args.task or suite.default_tasks[0]
    width = args.width or suite.default_widths[0]
    task = suite.load_task(task_id)
    spec = suite.load_range_spec(task_id, width)
    oracle = suite.make_oracle(task)

    obs, _spent = build_initial_design(
        oracle, CampaignConfig(n_init=args.n_init, budget=0), args.seed
    )
    X, Y = obs.X, obs.Y
    if args.extra:
        more = oracle.observe(oracle.initial_design(args.extra, seed=args.seed + 7_000))
        X, Y = np.vstack([X, more.X]), np.vstack([Y, more.Y])

    enumerate_designs = getattr(oracle, "enumerate_designs", None)
    exact = enumerate_designs is not None
    probe = (enumerate_designs() if exact
             else calibration.sample_design_space(task, args.probe, seed=args.seed + 99))

    captured, variables = _capture_acquisition(
        task, spec, X, Y, args.method, args.restarts,
        enumerate_designs() if exact else None,
    )
    acqf = captured["acqf"]

    Z = torch.as_tensor(
        np.clip(variables.to_normalized(probe), 0.0, 1.0), dtype=torch.double
    ).unsqueeze(1)

    labeled = int(np.isfinite(Y).all(axis=1).sum())
    print(f"suite={args.suite} task={task_id} width={width} method={args.method}")
    print(f"observations={len(X)} labeled={labeled} m={len(task.y_names)} "
          f"probe={'enumerated ' if exact else 'sampled '}{len(probe)}")
    print(f"acquisition={type(acqf).__name__} "
          f"objective={type(getattr(acqf, 'objective', None)).__name__}")

    with torch.no_grad():
        posterior = acqf.model.posterior(Z)
        draws = posterior.rsample(torch.Size([8]))
        objective = getattr(acqf, "objective", None)
        if objective is not None:
            values = objective(draws, X=Z).numpy()
            spread = float(values.std(axis=0).max())
            print(f"\nobjective spread across 8 posterior draws: {spread:.3e} "
                  f"-> deterministic in x: {spread < 1e-12}")
            obj = objective(posterior.mean.unsqueeze(0), X=Z).numpy()
            obj = obj.squeeze(0).squeeze(1)
        else:
            obj = None
            print("\nacquisition exposes no MC objective (tolerance ball); "
                  "skipping the determinism check")

        acq = torch.cat([acqf(Z[i:i + 256]) for i in range(0, len(Z), 256)]).numpy()

    observed = {tuple(np.round(row, 10)) for row in X}
    is_obs = np.array([tuple(np.round(row, 10)) in observed for row in probe])

    print(f"\nacquisition over the design space")
    print(f"  exactly zero at {np.mean(acq <= 0):7.2%} of probe points "
          f"(max {acq.max():.3e})")
    if is_obs.any():
        print(f"  among the {int((~is_obs).sum())} unobserved: zero at "
              f"{np.mean(acq[~is_obs] <= 0):7.2%} (max {acq[~is_obs].max():.3e})")

    if obj is not None and obj.ndim == 2:
        print(f"\nrange probability per output")
        for j, name in enumerate(task.y_names):
            print(f"  {name:10s} median {np.median(obj[:, j]):.4f}  max {obj[:, j].max():.4f}")
        if is_obs.any() and (~is_obs).any():
            print(f"  best observed   {np.round(obj[is_obs].max(axis=0), 4)}")
            print(f"  best unobserved {np.round(obj[~is_obs].max(axis=0), 4)}")
            n_dom = _dominated(obj[is_obs], obj[~is_obs])
            print(f"  unobserved points dominated by an observed one: "
                  f"{n_dom} of {int((~is_obs).sum())} ({n_dom / max((~is_obs).sum(), 1):.1%})")

    # Where the noise model shows up: a noiseless oracle collapses the posterior
    # at an observed design, which is what sends its range probability to 1.
    with torch.no_grad():
        posterior = acqf.model.posterior(Z)
        mu = posterior.mean.squeeze(1).numpy()
        sd = posterior.variance.clamp_min(0).sqrt().squeeze(1).numpy()
    if is_obs.any() and (~is_obs).any():
        print("\nposterior in the standardized space the window is applied in")
        print(f"  {'output':10s} {'group':11s} {'mean med':>9s} {'mean min':>9s} "
              f"{'mean max':>9s} {'sd med':>8s} {'sd min':>8s} {'sd max':>8s}")
        for j, name in enumerate(task.y_names):
            for label, mask in (("observed", is_obs), ("unobserved", ~is_obs)):
                print(f"  {name:10s} {label:11s} {np.median(mu[mask, j]):9.4f} "
                      f"{mu[mask, j].min():9.4f} {mu[mask, j].max():9.4f} "
                      f"{np.median(sd[mask, j]):8.4f} {sd[mask, j].min():8.4f} "
                      f"{sd[mask, j].max():8.4f}")
        print("  A posterior that reverts to the prior at every probe point -- sd flat "
              "and\n  mean flat -- means the surrogate learned nothing, which is a "
              "different\n  failure from a plateau and has a different fix.")
    else:
        print("\nno probe point coincides with an observation "
              "(continuous space) -- posterior-collapse check not applicable")


if __name__ == "__main__":
    main()
