"""Check the SCBO trust-region transcription against the botorch tutorial's own
constrained test problem.

`eval/methods/scbo.py`'s `TurboState` / `update_tr_length` / `update_state` /
`get_best_index_for_batch` / `_box_candidates` are transcribed out of the
`scalable_constrained_bo` tutorial notebook (botorch ships no importable SCBO
library, only `botorch.generation.sampling.ConstrainedMaxPosteriorSampling`).
This script exercises exactly those ported pieces -- unmodified, imported
directly from `scbo.py` -- on the tutorial's own 10-D Ackley problem with two
independent black-box constraints, run through botorch's real
`ConstrainedMaxPosteriorSampling` and three *independent* GPs (objective, c1,
c2), i.e. the tutorial's own literal formulation rather than our single-shared-
-GP window adaptation. This is the same strategy `verify_comboo_port.py` uses:
validate the transcription against the paper's own setting first; the
project-specific adaptation (the window-to-constraint mapping, the shared
posterior draw) is checked separately, by direct formula, in `test_scbo.py`.

Problem (tutorial): `Ackley(dim=10, negate=True)` on `[-5, 10]^10`, subject to
    c1(x) = sum(x)      <= 0
    c2(x) = ||x||_2 - 5 <= 0
Both are satisfied with room to spare at the origin, where Ackley(negate=True)
attains its maximum of 0 -- so "found a good feasible point" has a concrete
target: the best feasible objective should end up well above the plateau most
of the domain sits on (Ackley is famously flat and near -20 almost everywhere
away from the optimum) and not stall at -inf (which would mean SCBO never
found *any* feasible point).

What "reproducing the tutorial" means here, quantitatively (checked, not just
described):

    1. `TurboState.failure_tolerance` matches the closed-form formula exactly
       (ceil(max(4/batch_size, dim/batch_size)) = ceil(max(1.0, 2.5)) = 3 for
       dim=10, batch_size=4) -- a deterministic, seed-independent check.
    2. The shrink path fires and the restart trigger fires: `state.length`
       drops below the 0.8 default, never exceeds `length_max`, and
       `restart_triggered` becomes true within the iteration budget (the
       tutorial's own loop condition, `while not state.restart_triggered`) --
       which by construction means `length` has just dropped below
       `length_min`, the trigger condition itself.
    3. The best *feasible* objective value found reaches above -5.0 (Ackley
       is near its floor of about -20 almost everywhere except a narrowing
       cone around the origin; -5.0 requires real progress, not luck).

    The grow path (`length = min(2*length, length_max)` after 10 *consecutive*
    successes) is not exercised end-to-end here: with `failure_tolerance = 3`
    versus `success_tolerance = 10`, a single bad batch is far more likely on
    10-D Ackley than a ten-long clean streak, so every seed below shrinks
    monotonically to a restart without ever doubling -- a real, expected
    property of this problem's difficulty relative to the batch size, not an
    artifact of the port. The doubling arithmetic itself (and the halving
    arithmetic, and the restart threshold) is checked exactly, by hand-computed
    example, in `test_scbo.py::test_length_doubles_on_success_streak_and_caps_at_length_max`.

Measured when the port was accepted (3 seeds x up to 60 iterations, batch_size=4,
20-point Sobol initial design, N_CANDIDATES=1000):

    seed 0  length 0.8000 -> 0.0063  restart True   best_feasible -0.16  halvings 7
    seed 1  length 0.8000 -> 0.0063  restart True   best_feasible -0.13  halvings 7
    seed 2  length 0.8000 -> 0.0063  restart True   best_feasible -0.15  halvings 7

    PYTHONPATH=core:. .venv-eval/bin/python -m eval.scripts.verify_scbo_port
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
from botorch.generation.sampling import ConstrainedMaxPosteriorSampling  # noqa: E402
from botorch.models import ModelListGP, SingleTaskGP  # noqa: E402
from botorch.models.transforms.outcome import Standardize  # noqa: E402
from botorch.test_functions.synthetic import Ackley  # noqa: E402
from botorch.utils.transforms import unnormalize  # noqa: E402
from torch.quasirandom import SobolEngine  # noqa: E402
from gpytorch.constraints import Interval  # noqa: E402
from gpytorch.kernels import MaternKernel, ScaleKernel  # noqa: E402
from gpytorch.likelihoods import GaussianLikelihood  # noqa: E402
from gpytorch.mlls import ExactMarginalLogLikelihood  # noqa: E402

from eval.methods.scbo import (  # noqa: E402
    TurboState,
    _box_candidates,
    get_best_index_for_batch,
    update_state,
)

SEEDS = [0, 1, 2]
DIM = 10
N_INIT = 20
N_ITER = 60
BATCH_SIZE = 4
N_CANDIDATES = 1000

BOUNDS = torch.stack([torch.full((DIM,), -5.0), torch.full((DIM,), 10.0)]).to(torch.double)


_ACKLEY = Ackley(dim=DIM, negate=True).to(torch.double)


def eval_objective(x_norm: torch.Tensor) -> float:
    """`x_norm` lives in [0, 1]^DIM, as `train_X` does throughout this script
    (the tutorial's own convention) -- unnormalize before touching the raw
    Ackley domain [-5, 10]^DIM."""
    return float(_ACKLEY(unnormalize(x_norm, BOUNDS)))


def eval_c1(x_norm: torch.Tensor) -> float:
    return float(unnormalize(x_norm, BOUNDS).sum())


def eval_c2(x_norm: torch.Tensor) -> float:
    return float(torch.norm(unnormalize(x_norm, BOUNDS), p=2) - 5.0)


def get_fitted_model(X: torch.Tensor, Y: torch.Tensor):
    """Tutorial's `get_fitted_model`: Matern 5/2, ARD, tight noise/lengthscale
    constraints -- used identically for the objective and each constraint."""
    likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
    covar_module = ScaleKernel(
        MaternKernel(nu=2.5, ard_num_dims=DIM, lengthscale_constraint=Interval(0.005, 4.0))
    )
    model = SingleTaskGP(
        X, Y, covar_module=covar_module, likelihood=likelihood,
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


def main() -> None:
    # Deterministic check, independent of any run: the closed-form formula.
    tolerance = TurboState(dim=DIM, batch_size=BATCH_SIZE).failure_tolerance
    expected = math.ceil(max(4.0 / BATCH_SIZE, DIM / BATCH_SIZE))
    print(f"failure_tolerance: got {tolerance}, expected {expected}")
    if tolerance != expected:
        print("FAIL: failure_tolerance formula does not match the tutorial.")
        raise SystemExit(1)

    header = (
        f"{'seed':>5s} {'len0->lenN':>14s} {'min':>7s} {'max':>7s} "
        f"{'best_feas':>10s} {'restart':>9s} {'halvings':>10s}"
    )
    print()
    print(header)
    failures = 0
    for seed in SEEDS:
        torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(seed)

        # `train_X` is normalized to [0, 1]^DIM throughout, matching the
        # tutorial: `_box_candidates` (imported unmodified from scbo.py)
        # operates in normalized space, and `eval_objective`/`eval_c1`/`eval_c2`
        # unnormalize internally before touching the raw Ackley domain.
        init_sobol = SobolEngine(dimension=DIM, scramble=True, seed=seed)
        train_X = init_sobol.draw(N_INIT).to(torch.double)
        train_Y = torch.tensor([[eval_objective(x)] for x in train_X], dtype=torch.double)
        C1 = torch.tensor([[eval_c1(x)] for x in train_X], dtype=torch.double)
        C2 = torch.tensor([[eval_c2(x)] for x in train_X], dtype=torch.double)

        state = TurboState(dim=DIM, batch_size=BATCH_SIZE)
        length_min, length_max = state.length, state.length
        halvings, doublings = 0, 0

        for _ in range(N_ITER):
            if state.restart_triggered:
                break
            length_before = state.length

            model = get_fitted_model(train_X, train_Y)
            c1_model = get_fitted_model(train_X, C1)
            c2_model = get_fitted_model(train_X, C2)

            best_ind = get_best_index_for_batch(
                Y=train_Y.squeeze(-1), C=torch.cat([C1, C2], dim=-1)
            )
            x_center = train_X[best_ind]  # already normalized
            X_cand = _box_candidates(
                x_center=x_center, length=state.length, dim=DIM,
                n_candidates=N_CANDIDATES, generator=generator,
            )  # normalized [0, 1]^DIM, exactly what the GPs were fit on

            sampler = ConstrainedMaxPosteriorSampling(
                model=model, constraint_model=ModelListGP(c1_model, c2_model),
                replacement=False,
            )
            with torch.no_grad():
                X_next = sampler(X_cand, num_samples=BATCH_SIZE)

            Y_next = torch.tensor([[eval_objective(x)] for x in X_next], dtype=torch.double)
            C1_next = torch.tensor([[eval_c1(x)] for x in X_next], dtype=torch.double)
            C2_next = torch.tensor([[eval_c2(x)] for x in X_next], dtype=torch.double)

            state = update_state(
                state, Y_next=Y_next.squeeze(-1), C_next=torch.cat([C1_next, C2_next], dim=-1)
            )
            if state.length < length_before:
                halvings += 1
            elif state.length > length_before:
                doublings += 1
            length_min = min(length_min, state.length)
            length_max = max(length_max, state.length)

            train_X = torch.cat([train_X, X_next], dim=0)
            train_Y = torch.cat([train_Y, Y_next], dim=0)
            C1 = torch.cat([C1, C1_next], dim=0)
            C2 = torch.cat([C2, C2_next], dim=0)

        is_feasible = (C1 <= 0).squeeze(-1) & (C2 <= 0).squeeze(-1)
        best_feasible = (
            float(train_Y[is_feasible].max()) if bool(is_feasible.any()) else -float("inf")
        )

        print(
            f"{seed:>5d} {f'{0.8:.4f}->{state.length:.4f}':>14s} "
            f"{length_min:>7.4f} {length_max:>7.4f} {best_feasible:>10.2f} "
            f"{str(state.restart_triggered):>9s} {halvings:>10d}"
        )

        ok = (
            length_min < 0.8 - 1e-9      # the shrink path fired
            # `length` legitimately ends up *below* `length_min` once restart
            # triggers -- that crossing is the trigger condition itself, so it
            # is not treated as an out-of-range violation. It never exceeds
            # `length_max`, which the growth path (unexercised here, see
            # module docstring) would be the only way to threaten.
            and length_max <= state.length_max + 1e-12
            and state.restart_triggered                  # shrinking eventually restarts
            and best_feasible > -5.0                      # real progress on Ackley
            and halvings >= 1
        )
        if not ok:
            failures += 1

    print()
    if failures:
        print(f"FAIL: {failures}/{len(SEEDS)} seeds did not reproduce the expected behaviour.")
        raise SystemExit(1)
    print("OK: the trust-region port reproduces the tutorial's constrained-Ackley behaviour.")


if __name__ == "__main__":
    main()
