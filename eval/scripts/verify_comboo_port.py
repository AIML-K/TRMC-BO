"""Check that the COMBOO transcription reproduces the authors' own result.

COMBOO was copied out of `experiments_COMBOO/branin-currin_COMBOO_exp.ipynb`
(the public repo ships no library). A mis-transcribed baseline would lose for
reasons that have nothing to do with the method, so the port is only usable once
it reproduces their setting: the loop should run to the end without declaring
infeasibility, with hypervolume above the reference rising.

Measured when the port was accepted (3 seeds x 30 iterations):

    seed 83810  30 iters  0 declarations  HV 6.90 -> 65.66
    seed 14592  30 iters  0 declarations  HV 0.00 -> 61.81
    seed  3278  30 iters  0 declarations  HV 0.00 -> 61.79

Their one-sided constraint form is used here (`upper=None`), matching the paper;
the two-sided window form is the adaptation used on our own benchmarks.

    PYTHONPATH=core:. .venv-eval/bin/python -m eval.scripts.verify_comboo_port
"""

from __future__ import annotations

import math
import os
import warnings

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import torch  # noqa: E402

torch.set_num_threads(1)
warnings.filterwarnings("ignore")

from botorch.fit import fit_gpytorch_mll  # noqa: E402
from botorch.models import ModelListGP, SingleTaskGP  # noqa: E402
from botorch.models.transforms.outcome import Standardize  # noqa: E402
from botorch.optim import optimize_acqf  # noqa: E402
from botorch.test_functions.multi_objective import BraninCurrin  # noqa: E402
from botorch.utils.sampling import draw_sobol_samples  # noqa: E402
from botorch.utils.transforms import normalize, unnormalize  # noqa: E402
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood  # noqa: E402

from eval.methods.comboo import (  # noqa: E402
    AuxiliaryUCB,
    HyperVolumeScalarizedUCB,
    _sample_theta,
    optimistic_window_constraints,
)

SEEDS = [83810, 14592, 3278]
N_ITER = 30
NOISE = 0.05
THRESHOLDS = torch.tensor([-20.0, -6.0], dtype=torch.double)  # their a, b


def dominated_hypervolume_2d(Y: torch.Tensor, ref: torch.Tensor) -> float:
    """Area dominated above `ref`, by sweep. Matches their toolkits/metrics.HV."""
    P = Y[(Y > ref).all(dim=-1)]
    if P.numel() == 0:
        return 0.0
    P = P[P[:, 0].argsort(descending=True)]
    total, y_max = 0.0, ref[1].item()
    for x, y in P.tolist():
        if y > y_max:
            total += (x - ref[0].item()) * (y - y_max)
            y_max = y
    return total


def main() -> None:
    test_f = BraninCurrin(negate=True).to(torch.double)
    bounds = test_f.bounds.to(torch.double)
    unit = torch.stack([torch.zeros(2, dtype=torch.double), torch.ones(2, dtype=torch.double)])
    m = 2

    print(f"{'seed':>7s} {'iters':>6s} {'declared':>9s} {'HV start':>9s} {'HV end':>9s}")
    failures = 0
    for seed in SEEDS:
        torch.manual_seed(seed)
        train_X = draw_sobol_samples(bounds=bounds, n=10, q=1).squeeze(1).to(torch.double)
        train_Y = test_f(train_X) + NOISE * torch.randn(10, m, dtype=torch.double)
        train_X = normalize(train_X, bounds)
        hv_start = dominated_hypervolume_2d(train_Y, THRESHOLDS)

        gen = torch.Generator().manual_seed(seed)
        declared, done = 0, 0
        for t in range(N_ITER):
            models = [
                SingleTaskGP(
                    train_X=train_X,
                    train_Y=train_Y[:, i : i + 1],
                    outcome_transform=Standardize(m=1),
                    train_Yvar=torch.full((train_X.shape[0], 1), NOISE**2, dtype=torch.double),
                )
                for i in range(m)
            ]
            model = ModelListGP(*models)
            fit_gpytorch_mll(SumMarginalLogLikelihood(model.likelihood, model))

            theta = _sample_theta(m, gen).clamp_min(1e-6)
            beta = 0.2 * 2 * math.log(4 * (t + 1))
            beta_const = 0.2 * 2 * math.log(2 * (t + 1) * 2)

            aux, aux_value = optimize_acqf(
                acq_function=AuxiliaryUCB(
                    model=model, beta=torch.tensor(beta_const), ref=THRESHOLDS
                ),
                bounds=unit, q=1, num_restarts=20, raw_samples=20,
            )
            if aux_value.item() < 0:
                declared += 1
                break

            try:
                candidate, _ = optimize_acqf(
                    acq_function=HyperVolumeScalarizedUCB(
                        model=model, beta=torch.tensor(beta), theta=theta, ref=THRESHOLDS
                    ),
                    bounds=unit, q=1, num_restarts=1,
                    batch_initial_conditions=aux.view(1, 1, 2),
                    nonlinear_inequality_constraints=optimistic_window_constraints(
                        model, beta_const, THRESHOLDS  # one-sided, as published
                    ),
                    options={"batch_limit": 1, "maxiter": 500},
                )
            except (ValueError, RuntimeError):
                candidate = aux

            train_X = torch.cat([train_X, candidate], dim=0)
            train_Y = torch.cat(
                [train_Y,
                 test_f(unnormalize(candidate, bounds=bounds))
                 + NOISE * torch.randn(1, m, dtype=torch.double)],
                dim=0,
            )
            done += 1

        hv_end = dominated_hypervolume_2d(train_Y, THRESHOLDS)
        print(f"{seed:>7d} {done:>6d} {declared:>9d} {hv_start:>9.2f} {hv_end:>9.2f}")
        if declared or done < N_ITER or hv_end <= hv_start:
            failures += 1

    print()
    if failures:
        print(f"FAIL: {failures}/{len(SEEDS)} seeds did not reproduce the authors' behaviour.")
        raise SystemExit(1)
    print("OK: the port reproduces the authors' branin-currin result.")


if __name__ == "__main__":
    main()
