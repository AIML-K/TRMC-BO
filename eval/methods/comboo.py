"""COMBOO baseline — constrained multi-objective BO via optimistic constraints (§6.5).

Li, Zhang, Liu & Chen, *Constrained Multi-objective Bayesian Optimization through
Optimistic Constraints Estimation*, AISTATS 2025 (arXiv 2411.03641).

## Why this baseline exists

It represents the structure the paper argues against (§2.2): constrained BO
**subordinates range satisfaction to a separate optimization objective**. Here
the window becomes a constraint while the acquisition maximizes the raw
properties. Our method instead makes range satisfaction the primary search
signal. That contrast is the §10.2 argument, so the baseline has to be a faithful
COMBOO rather than a strawman.

## Why it is ported rather than installed

The public repository is experiment code, not a library. `toolkits/` (8 KB) holds
kernels, a design helper and metrics -- no algorithm; the algorithm is inline in
`experiments_COMBOO/*.ipynb`. Its pins (torch 2.3.0, botorch 0.11.3,
gpytorch 1.12, plus GPy/GPyOpt/gauche/rdkit) also conflict with this project's
stack. So the algorithm is transcribed from `branin-currin_COMBOO_exp.ipynb`
onto our surrogate, and `tests/test_comboo.py` checks the transcription against
values computed directly from their formulas.

## The algorithm, as implemented there

Per iteration `t` (0-based):

    beta        = 0.2 * 2 * log(4 * (t + 1))          objective UCB width
    beta_const  = 0.2 * 2 * log(2 * (t + 1) * 2)      constraint UCB width
    theta       ~ uniform on the positive orthant of the unit m-sphere

    1. Auxiliary acquisition (their Algo. 2):
           A_aux(x) = min_i ( mean_i + beta_const * sd_i - ref_i )
       Maximize it. If the optimum is < 0, no design has every optimistic bound
       above its threshold and the original code declares infeasibility and stops.

    2. Main acquisition, seeded from the auxiliary optimum:
           A(x) = min_i ( max(0, (mean_i + beta * sd_i - ref_i) / theta_i) )^m
       maximized subject to `mean_i + beta_const * sd_i >= ref_i` for every i,
       supplied as BoTorch nonlinear inequality constraints.

## Mapping fix — hypervolume reference separated from the constraint threshold

`results_issue_matformbench.md` withheld the first 90-cell run because `ref`
(the point "improvement" is measured above, in both `AuxiliaryUCB` and
`HyperVolumeScalarizedUCB`) was set equal to the window's lower edge -- the same
value also used as the constraint threshold in `optimistic_window_constraints`.
Their own code uses `ref` as a loose performance floor, well below their
constraint threshold; collapsing the two here made "clear the reference" nearly
as hard as "clear the constraint", leaving almost no positive signal to climb
(+0.03 measured on L1-1/medium).

The fix: `ref` is now a low quantile (`HV_REF_QUANTILE = 0.1`) of the outputs
observed so far, in the same standardized/negated model space the window lives
in -- used only inside `HyperVolumeScalarizedUCB` (stage 2's main acquisition).
`lower`/`upper` continue to define the constraint threshold unchanged, in
`optimistic_window_constraints`.

**That separation surfaced a second, previously-masked bug.** Stage 1's probe
seeds stage 2's `optimize_acqf` with its optimum (`batch_initial_conditions`),
which BoTorch requires to already satisfy stage 2's nonlinear constraints. As
long as `ref == lower`, the one-sided `AuxiliaryUCB` probe (`min_i(UCB_i -
ref_i)`) happened to also check stage 2's lower-side constraint, so its optimum
was usually a valid seed. Once `ref` became a looser quantile, `AuxiliaryUCB`
kept probing against that loose `ref` -- clearing it is easy, but says nothing
about clearing the *actual* window's lower bound, and `AuxiliaryUCB` never
checked the upper bound at all, one-sided as published. The result: stage 2's
`optimize_acqf` raised `batch_initial_conditions must satisfy the non-linear
inequality constraints` on nearly every call (measured ~98-100% of iterations
on L1-1/L2-1, ~67% on L3-1), silently falling back to the probe's own point
every time -- so COMBOO was almost never actually running its main acquisition,
just proposing the probe's optimum dressed up as a full run. `TwoSidedAuxiliaryUCB`
fixes this: the probe now checks both sides of the real window (`lower`,
`upper`), so its optimum is a valid stage-2 seed again, independent of `ref`.

Re-measure after this change before reporting: if COMBOO still underperforms,
that is now a result rather than an artifact of the mapping.

## Three adaptations, all deliberate

**Two-sided windows.** Their constraint form is one-sided (`UCB >= threshold`).
A window needs both sides, and "optimistic" means the widest reading the
posterior allows, so each output contributes

    mean + beta_const * sd >= L        (optimistic about clearing the floor)
    mean - beta_const * sd <= U        (optimistic about staying under the cap)

**No early stop.** Their loop breaks out when infeasibility is declared. A
campaign here must spend every evaluation or the budget stops being comparable
across methods, so the declaration is recorded (`comboo_declared_infeasible`) and
the auxiliary optimum -- the most optimistic point available -- is proposed
instead.

**No observations yet for some output.** COMBOO needs a posterior for *every*
output simultaneously (both the probe and the main acquisition take a joint
min over all of them), so a column that is still all-NaN this early in a
campaign (plausible on a sparse-feasibility benchmark like HOIP) leaves nothing
to fit a GP to for that output -- `fit_gpytorch_mll` would otherwise crash on
the resulting 0-row model. Treated like the infeasibility-declaration case: a
linear-constraint-respecting point is proposed instead
(`comboo_no_observations_for_output:<names>` records which), rather than
crashing the campaign.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition import AnalyticAcquisitionFunction
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize, t_batch_mode_transform
from torch import Tensor

from bayesian_optimization import initialize_model, negate_y, standardize

NAME = "M8_comboo"

#: Quantile of the observed (standardized, negated) outputs used as the
#: hypervolume/UCB reference point. Kept well below the constraint threshold
#: (`lower`) so "improvement above the reference" stays a meaningful signal
#: instead of coinciding with "clears the constraint" -- see the module
#: docstring's "Mapping fix" section.
HV_REF_QUANTILE = 0.1


class HyperVolumeScalarizedUCB(AnalyticAcquisitionFunction):
    """``min_i ( max(0, (UCB_i - ref_i) / theta_i) )^m`` — their main acquisition."""

    def __init__(self, model, beta: Tensor, theta: Tensor, ref: Tensor):
        super(AnalyticAcquisitionFunction, self).__init__(model)
        self.register_buffer("beta", torch.as_tensor(beta))
        self.register_buffer("theta", torch.as_tensor(theta))
        self.register_buffer("ref", torch.as_tensor(ref))

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        posterior = self.model.posterior(X)
        mean = posterior.mean.squeeze(dim=-2)
        sd = posterior.variance.squeeze(dim=-2).clamp_min(1e-12).sqrt()
        m = mean.shape[-1]
        u_t = mean + self.beta.to(X) * sd - self.ref.to(X)
        scaled = torch.clamp(u_t / self.theta.to(X), min=0.0) ** m
        return scaled.min(dim=-1).values


class AuxiliaryUCB(AnalyticAcquisitionFunction):
    """``min_i (UCB_i - ref_i)`` — their Algo. 2 feasibility probe."""

    def __init__(self, model, beta: Tensor, ref: Tensor):
        super(AnalyticAcquisitionFunction, self).__init__(model)
        self.register_buffer("beta", torch.as_tensor(beta))
        self.register_buffer("ref", torch.as_tensor(ref))

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        posterior = self.model.posterior(X)
        mean = posterior.mean.squeeze(dim=-2)
        sd = posterior.variance.squeeze(dim=-2).clamp_min(1e-12).sqrt()
        return (mean + self.beta.to(X) * sd - self.ref.to(X)).min(dim=-1).values


class TwoSidedAuxiliaryUCB(AnalyticAcquisitionFunction):
    """``min`` over *both* sides of every output's window -- the feasibility
    probe `AuxiliaryUCB` cannot express, and the seed-validity fix this class
    exists for.

    `AuxiliaryUCB` (their published Algo. 2) is one-sided: `min_i(UCB_i -
    ref_i)`. In the one-sided setting that is exactly the same expression stage
    2's constraint checks, so the probe's optimum is automatically a valid seed
    for the constrained solve. Our two-sided window needs *two* per-output
    checks -- `mean + beta*sd - lower >= 0` and `upper - (mean - beta*sd) >= 0`
    -- and `AuxiliaryUCB` only ever tests the first. Seeding stage 2's
    `optimize_acqf` with a point that never had its *upper* side checked made
    `batch_initial_conditions` fail the "must satisfy the non-linear inequality
    constraints" precondition on nearly every iteration once the hypervolume
    reference stopped coinciding with `lower` (see the module docstring's
    "Mapping fix" section) -- `optimize_acqf` then raised, and `propose()`'s
    `except` silently returned the probe's point on every one of those calls,
    which is most of them. This class's optimum satisfies every component of
    `optimistic_window_constraints(model, beta_const, lower, upper)` whenever
    its value is >= 0, which is what a valid seed requires.
    """

    def __init__(self, model, beta: Tensor, lower: Tensor, upper: Tensor):
        super(AnalyticAcquisitionFunction, self).__init__(model)
        self.register_buffer("beta", torch.as_tensor(beta))
        self.register_buffer("lower", torch.as_tensor(lower))
        self.register_buffer("upper", torch.as_tensor(upper))

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        posterior = self.model.posterior(X)
        mean = posterior.mean.squeeze(dim=-2)
        sd = posterior.variance.squeeze(dim=-2).clamp_min(1e-12).sqrt()
        beta = self.beta.to(X)
        u_lower = mean + beta * sd - self.lower.to(X)
        u_upper = self.upper.to(X) - (mean - beta * sd)
        return torch.cat([u_lower, u_upper], dim=-1).min(dim=-1).values


def optimistic_window_constraints(
    model, beta_const: float, lower: Tensor, upper: Tensor | None = None
):
    """Optimistic constraints in BoTorch's ``callable(X) >= 0`` form.

    With ``upper=None`` this is exactly their published one-sided form
    (``UCB >= threshold``), which is what the fidelity check against their
    branin-currin setup needs. Passing an upper bound adds the second half of the
    window. Both sides use ``beta_const`` -- the constraint confidence width, not
    the objective one.
    """
    def bound(X: Tensor, index: int, side: str) -> Tensor:
        posterior = model.posterior(X.unsqueeze(0))
        mean = posterior.mean[..., index].reshape(-1)
        sd = posterior.variance[..., index].clamp_min(1e-12).sqrt().reshape(-1)
        if side == "lower":                       # mean + b*sd >= L
            return mean + beta_const * sd - lower[index].to(mean)
        return upper[index].to(mean) - (mean - beta_const * sd)   # mean - b*sd <= U

    constraints = []
    for i in range(lower.numel()):
        constraints.append((lambda X, i=i: bound(X, i, "lower")[0], True))
        if upper is not None:
            constraints.append((lambda X, i=i: bound(X, i, "upper")[0], True))
    return constraints


def _sample_theta(m: int, generator: torch.Generator) -> Tensor:
    """Uniform on the positive orthant of the unit m-sphere (their helper)."""
    v = torch.randn(m, generator=generator, dtype=torch.double)
    return (v / v.norm()).abs()


class COMBOOMethod:
    """Runs COMBOO on this project's surrogate so the comparison isolates the algorithm."""

    name = NAME
    #: Objectives are the benchmark's own directions; the window enters as a
    #: constraint. That split *is* the baseline's defining characteristic.
    objective_mode = "native_extremum"

    def __init__(self, num_restarts: int = 128, aux_restarts: int | None = None):
        self.num_restarts = num_restarts
        # The probe gets the *full* restart budget, not a fraction of it.
        #
        # An earlier version used num_restarts // 4 and produced spurious
        # infeasibility declarations: measured on L1-1/medium the auxiliary
        # optimum is only +0.03, so a thin restart set frequently misses it and
        # returns a negative value. That fabricates an infeasibility the method
        # never actually claimed. With the full budget the same landscape reports
        # feasible at every iteration. Under-resourcing the feasibility test is
        # the one place a stingy budget invents a result rather than degrading one.
        self.aux_restarts = aux_restarts or num_restarts
        self._iteration = 0

    # -- window mapping ----------------------------------------------------

    @staticmethod
    def _window_in_model_space(variables, y_mean, y_std) -> tuple[Tensor, Tensor]:
        """Raw window -> the standardized, direction-flipped space the GP works in.

        `negate_y` flips outputs whose objective is 'min' so everything is a
        maximization. A window flips with them: ``[L, U] -> [-U_s, -L_s]``.
        Getting this backwards would silently hand COMBOO an inverted target.
        """
        spec = variables.range_spec
        lower, upper = [], []
        for i, key in enumerate(variables.y_key):
            lb = (float(spec.lower[key]) - float(y_mean[i])) / float(y_std[i])
            ub = (float(spec.upper[key]) - float(y_mean[i])) / float(y_std[i])
            if variables.target_info[key]["obj"] == "min":
                lb, ub = -ub, -lb
            lower.append(lb)
            upper.append(ub)
        kw = {"dtype": torch.double}
        return torch.tensor(lower, **kw), torch.tensor(upper, **kw)

    @staticmethod
    def _hv_reference(train_y_std: Tensor) -> Tensor:
        """Low quantile of the observed outputs, per-output, NaN-aware.

        Kept as its own method (mirroring `_window_in_model_space`) so the
        separation from the constraint threshold is directly testable rather
        than only visible inside the full `propose()` pipeline.
        """
        return torch.nanquantile(train_y_std, HV_REF_QUANTILE, dim=0)

    # -- Method protocol ---------------------------------------------------

    def propose(self, variables, q: int, seed: int, recorder=None) -> np.ndarray:
        if len(variables.y_key) < 2:
            raise ValueError(
                "COMBOO is a constrained *multi-objective* method; it needs at least "
                f"two outputs but this task has {len(variables.y_key)}."
            )
        if q != 1:
            # Both internal `optimize_acqf` calls below are hardcoded to q=1
            # (the ported algorithm proposes one point per iteration); nothing
            # currently asks for q != 1, but silently reshaping a (1, d)
            # candidate into (q, d) for q > 1 would crash with an opaque numpy
            # error instead of an honest one.
            raise ValueError(f"COMBOO only supports q=1 (a batch of size {q} was requested)")

        t = self._iteration
        self._iteration += 1

        bounds_x = variables.item_bound
        train_x = normalize(variables.train_x, bounds_x)

        empty_outputs = ~variables.continuous_value_mask.any(dim=0)
        if bool(empty_outputs.any()):
            # At least one output has zero valid observations so far (every
            # row NaN) -- `initialize_model` would build a 0-row SingleTaskGP
            # for it and `fit_gpytorch_mll` crashes on that (a degenerate
            # reshape, not a graceful failure). COMBOO structurally needs a
            # posterior for *every* output simultaneously (both stage-1's probe
            # and stage-2's acquisition take a joint min over all of them), so
            # there is no meaningful partial fit to fall back to -- this is the
            # same "cannot even attempt the real algorithm yet" situation the
            # infeasibility-declaration path already handles, so it gets the
            # same treatment: propose a linear-constraint-respecting point and
            # note why, rather than crash the whole campaign.
            if recorder is not None:
                missing = [variables.y_key[i] for i in torch.where(empty_outputs)[0].tolist()]
                recorder.note("comboo_no_observations_for_output:" + ",".join(missing))
            d = variables.task.dim
            fallback = variables.sample_raw_candidates(1, variables, {})
            return variables.to_raw(
                fallback.detach().cpu().numpy()
            ).reshape(q, d)[:q]

        train_y_std, y_mean, y_std = standardize(
            variables.train_y, variables.continuous_value_mask
        )
        train_y_std = negate_y(train_y_std, variables.y_key, variables.target_info)

        mll, model = initialize_model(
            train_x, train_y_std, [], variables.continuous_value_mask
        )
        fit_gpytorch_mll(mll)

        lower, upper = self._window_in_model_space(variables, y_mean, y_std)
        # Hypervolume/UCB reference: a low quantile of what has actually been
        # observed, kept distinct from the constraint threshold (`lower`). Using
        # `lower` itself here (the original transcription) made "improvement
        # above the reference" nearly synonymous with "clears the constraint" --
        # see the module docstring's "Mapping fix" section.
        ref = self._hv_reference(train_y_std)

        m = len(variables.y_key)
        beta = 0.2 * 2 * math.log(4 * (t + 1))
        beta_const = 0.2 * 2 * math.log(2 * (t + 1) * 2)

        generator = torch.Generator().manual_seed(seed * 7919 + t)
        theta = _sample_theta(m, generator).clamp_min(1e-6)

        d = variables.task.dim
        unit = torch.stack([torch.zeros(d, dtype=torch.double),
                            torch.ones(d, dtype=torch.double)])
        linear = variables.linear_constraints()

        # -- stage 1: optimistic feasibility probe -------------------------
        # Two-sided, not `AuxiliaryUCB`: the probe's optimum is used as stage
        # 2's `batch_initial_conditions`, which BoTorch requires to already
        # satisfy `optimistic_window_constraints(model, beta_const, lower,
        # upper)` -- both sides, every output. `AuxiliaryUCB` only ever checks
        # one side against `ref`, which stopped being `lower` once the
        # hypervolume reference was separated from the constraint threshold
        # (see `TwoSidedAuxiliaryUCB`'s docstring for what broke and why).
        aux, aux_value = optimize_acqf(
            acq_function=TwoSidedAuxiliaryUCB(
                model=model, beta=torch.tensor(beta_const), lower=lower, upper=upper
            ),
            bounds=unit,
            q=1,
            num_restarts=self.aux_restarts,
            raw_samples=max(256, self.aux_restarts * 8),
            equality_constraints=linear["equality_constraints"],
            inequality_constraints=linear["inequality_constraints"],
        )

        declared_infeasible = bool(aux_value.item() < 0)
        if declared_infeasible:
            # Their loop stops here. We cannot, so record it and propose the
            # most-optimistic point rather than skipping the evaluation.
            if recorder is not None:
                recorder.note("comboo_declared_infeasible")
            return variables.to_raw(aux.detach().cpu().numpy()).reshape(q, d)[:q]

        # -- stage 2: scalarized UCB inside the optimistic feasible set -----
        try:
            candidate, _ = optimize_acqf(
                acq_function=HyperVolumeScalarizedUCB(
                    model=model, beta=torch.tensor(beta), theta=theta, ref=ref
                ),
                bounds=unit,
                q=1,
                num_restarts=1,
                batch_initial_conditions=aux.view(1, 1, d),
                nonlinear_inequality_constraints=optimistic_window_constraints(
                    model, beta_const, lower, upper
                ),
                equality_constraints=linear["equality_constraints"],
                inequality_constraints=linear["inequality_constraints"],
                options={"batch_limit": 1, "maxiter": 500},
            )
        except (ValueError, RuntimeError):
            # The constrained solve can reject its own seed at the boundary.
            # Falling back to the probe's optimum keeps the budget aligned.
            if recorder is not None:
                recorder.note("comboo_constrained_solve_failed")
            candidate = aux

        return variables.to_raw(candidate.detach().cpu().numpy()).reshape(q, d)[:q]


def make(_name: str = NAME, num_restarts: int = 128) -> COMBOOMethod:
    return COMBOOMethod(num_restarts=num_restarts)
