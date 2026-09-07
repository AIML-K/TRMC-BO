"""SCBO baseline — Scalable Constrained Bayesian Optimization.

Eriksson & Poloczek, *Scalable Constrained Bayesian Optimization*, AISTATS 2021
(arXiv:2002.08526). A TuRBO-style trust-region method: it fits a GP over a
single scalar objective plus one GP per black-box inequality constraint, draws
Thompson samples inside a shrinking/growing trust region, and selects the
sample that maximizes the objective among the feasible draws (or, if none are
feasible, the one with least total constraint violation).

## Why it is ported rather than installed

BoTorch 0.15.0 (already pinned by this project) ships the primitive SCBO needs
verbatim: `botorch.generation.sampling.ConstrainedMaxPosteriorSampling`. What it
does *not* ship is a runnable method -- the trust-region bookkeeping
(`TurboState`, `update_tr_length`, `generate_batch`) lives only in the
`scalable_constrained_bo` tutorial notebook (botorch.org / the botorch GitHub
`tutorials/scalable_constrained_bo` tree), which is not importable. So the
tutorial's `TurboState`/`update_tr_length`/`update_state`/
`get_best_index_for_batch`/`generate_batch` are transcribed here verbatim (down
to the exact formulas), the same way `comboo.py` transcribes
`branin-currin_COMBOO_exp.ipynb`, and `eval/scripts/verify_scbo_port.py`
exercises the transcribed pieces against one of the tutorial's own constrained
test problems before trusting them on the benchmark suites.

Reference formulas (tutorial `ScboState`/`update_tr_length`/`update_state`):

    length_min = 0.5 ** 7                 length_max = 1.6
    failure_tolerance = ceil(max(4 / batch_size, dim / batch_size))
    success_tolerance = 10                 (fixed)

    on success_counter == success_tolerance:
        length = min(2 * length, length_max);  success_counter = 0
    on failure_counter == failure_tolerance:
        length = length / 2;                   failure_counter = 0
    restart_triggered = length < length_min

    best_ind = get_best_index_for_batch(Y, C)   # feasible: argmax Y; else argmin
                                                  # total violation of C
    y_next, c_next = Y[best_ind], C[best_ind]
    if (c_next <= 0).all():
        improved = y_next > best_value + 1e-3 * |best_value|
        if improved or any(best_constraint_values > 0):
            success; best_value, best_constraint_values <- y_next, c_next
        else: failure
    else:
        if total_violation(c_next) < total_violation(best_constraint_values):
            success; best_value, best_constraint_values <- y_next, c_next
        else: failure

## Scope: single-output tasks only

Vanilla SCBO optimizes one scalar objective under separately observed
black-box constraints. In this repo only Olympus (`branin`, `pce10`,
`oer_plate_3496`) is single-output; MatFormBench (m=3) and HOIP (m=2) are
structurally the wrong shape for it. Mirroring COMBOO's own guard (inverted:
COMBOO needs m>=2 and raises below that; SCBO needs m==1 and raises above it),
`propose()` raises a `ValueError` rather than silently degenerating on a
multi-output task.

## Adaptation — a two-sided window is not SCBO's native problem

Olympus tasks specify a target window `[L, U]` on the single output, not
"maximize y subject to separately observed constraints." The window is turned
into SCBO's shape as follows:

**Constraints**, both read off the *same* GP posterior over `y` (there is only
one surrogate here, unlike SCBO's usual independent constraint models -- a
deliberate simplification, since in a range-targeting problem `y` is
simultaneously the objective's basis and both constraints' basis, so fitting
three independent GPs on the one property actually observed would be a
redundant, potentially inconsistent use of the same information):

    c1(x) = L - y(x) <= 0      (y(x) >= L, BoTorch's "constraint satisfied" sign)
    c2(x) = y(x) - U <= 0      (y(x) <= U)

**Objective**, for the trust region's success/failure bookkeeping and for
ranking feasible Thompson draws: the same idea as `standard_bo.py`'s
`NegativeTargetDistanceObjective` -- negative squared distance from `y` to the
window midpoint, scaled by the half-width:

    obj(x) = -((y(x) - mid) / half) ** 2,   mid = (L+U)/2,  half = (U-L)/2

This gives SCBO a genuine scalar "how much better" signal distinct from the
constraints (a pure feasibility problem has none, and the trust region's
grow/shrink logic needs one), while still being anchored to the window rather
than an unrelated extremum.

All three quantities (`obj`, `c1`, `c2`) are computed from a *single* posterior
draw of `y` at each candidate -- not three separate `model.posterior()` calls
-- so they are perfectly consistent realizations of the same latent function
rather than merely trained on consistent data. Bookkeeping and Thompson-sample
scoring both work in **raw output units**, not the per-iteration standardized
model space: `L`, `U`, `mid`, `half` are fixed by the frozen window, so a
posterior draw is un-standardized back to raw before `obj`/`c1`/`c2` are
computed. This keeps `TurboState.best_value`/`best_constraint_values`
comparable across iterations even though the GP's own standardization
(`y_mean`, `y_std`) drifts slightly as more data arrives -- comparing values
expressed in a shifting scale would corrupt the success/failure counters that
drive the trust region.

**Selection.** Thompson-sampled candidates are scored by the exact rule
`botorch.generation.sampling.ConstrainedMaxPosteriorSampling` implements
(`_convert_samples_to_scores`, doi.org/10.48550/arXiv.2002.08526): feasible
draws are ranked by the sampled objective; if a draw has no feasible candidate
in the pool, it is ranked by total constraint violation instead. Because our
constraints and objective share one posterior sample rather than living in
independent constraint-model objects, that class itself cannot be
instantiated directly here -- the scoring logic is reimplemented on our
sample tensor (`_thompson_select` below), but the final top-k-without-
-replacement selection is delegated to `MaxPosteriorSampling.maximize_samples`
(the *same*, unmodified botorch method `ConstrainedMaxPosteriorSampling`
itself calls) so the batch-deduplication mechanics are not reinvented either.

## Adaptation — simplex tasks (`pce10`, 4-D; `oer_plate_3496`, 6-D)

These carry an equality constraint (`variables.linear_constraints()`):
components sum to 1. The tutorial's candidate generator perturbs an
axis-aligned box and has no notion of a simplex. Rather than inventing a
projection scheme, the candidate pool on these tasks is drawn from
`variables.sample_raw_candidates()` -- the same Dirichlet-based,
floor-respecting sampler the rest of the harness uses for simplex restarts --
and then interpolated toward the current trust-region center:

    candidate = (1 - alpha) * center + alpha * fresh_dirichlet_draw
    alpha = clip(length, 0, 1)

A convex combination of two points that each satisfy the simplex's equality
(linear) and per-component floor (also linear, one-sided) constraints
satisfies them too, so no post-hoc projection or rounding is needed; an
assertion after sampling checks every candidate still sums to 1 within
tolerance, matching this repo's existing style (`make_samples_drs` in
`src/sampler.py`). `alpha` shrinking with the trust-region length concentrates
candidates near the incumbent exactly as the box perturbation's shrinking
window does; growing `length` (up to 1.0, where it saturates -- `length` can
reach 1.6) widens the draw back out toward an unconstrained fresh Dirichlet
sample. `branin` (plain 2-D box, no simplex) uses the tutorial's standard
Sobol-perturbation-in-a-box scheme directly.

## Adaptation — restart handling

SCBO's own loop just stops when `state.restart_triggered` fires (the tutorial
runs a single trust region to exhaustion). A campaign here must spend its
entire fixed evaluation budget for the comparison to stay apples-to-apples
across methods (the same reasoning as COMBOO's "no early stop" adaptation), so
a restart instead resets the trust-region bookkeeping (`length` back to 0.8,
counters cleared) and continues, without touching the accumulated GP training
data. `recorder.note("scbo_restart_triggered")` records that this happened
rather than silently absorbing it.

The trust-region *center* also gets a fresh random point (`_pending_reseed`,
via `variables.sample_raw_candidates`) for the one call right after a restart,
rather than the usual "best point observed so far". A restart means the
collapsed region found nothing better nearby; re-centering on the same
incumbent that caused the collapse (which the incumbent-tracking logic would
otherwise pick again, since one exploratory point rarely beats an established
best) would make the "restart" a no-op that just re-shrinks around the same
point. This measured empirically: restarts fire roughly once per 50-iteration
campaign on every Olympus task, so this is not a rare edge case.

## Adaptation — the initial design is not folded into the trust-region state

The `n_init` points evaluated before any method acts are not a batch SCBO
itself proposed via `generate_batch`, so `update_state` must not run on them --
the ported tutorial only ever calls it on batches its own candidate generator
produced. `propose()`'s first call therefore records `self._n_observed = n_obs`
directly (marking the initial design as "already accounted for") without a
bookkeeping verdict; only batches produced by this method's own `propose()`
calls afterward update `success_counter`/`failure_counter`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition.objective import IdentityMCObjective
from botorch.generation.sampling import MaxPosteriorSampling
from botorch.utils.transforms import normalize
from torch import Tensor
from torch.quasirandom import SobolEngine

from bayesian_optimization import initialize_model, standardize

NAME = "M7_scbo"

DTYPE = torch.double


# ---------------------------------------------------------------------------
# Ported verbatim from the botorch `scalable_constrained_bo` tutorial.
# ---------------------------------------------------------------------------


@dataclass
class TurboState:
    """The tutorial's `ScboState`. Renamed for this project; fields and the
    post-init `failure_tolerance` formula are unchanged."""

    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = field(init=False)
    success_counter: int = 0
    success_tolerance: int = 10
    best_value: float = -float("inf")
    best_constraint_values: Tensor = field(
        default_factory=lambda: torch.full((2,), float("inf"), dtype=DTYPE)
    )
    restart_triggered: bool = False

    def __post_init__(self) -> None:
        self.failure_tolerance = math.ceil(
            max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
        )


def update_tr_length(state: TurboState) -> TurboState:
    """Grow the trust region after a success streak, shrink after a failure
    streak, and flag a restart once it has shrunk past `length_min`."""
    if state.success_counter == state.success_tolerance:
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:
        state.length /= 2.0
        state.failure_counter = 0

    if state.length < state.length_min:
        state.restart_triggered = True
    return state


def get_best_index_for_batch(Y: Tensor, C: Tensor) -> Tensor:
    """Feasible candidates are ranked by objective value; if none in the batch
    are feasible, rank by total constraint violation instead (SCBO's own
    published selection rule, ported from the tutorial helper of the same
    name -- this is also what
    `botorch.generation.sampling.ConstrainedMaxPosteriorSampling` implements
    for a single posterior draw)."""
    is_feas = (C <= 0).all(dim=-1)
    if is_feas.any():
        score = Y.clone()
        score[~is_feas] = -float("inf")
        return score.argmax()
    return C.clamp(min=0).sum(dim=-1).argmin()


def update_state(state: TurboState, Y_next: Tensor, C_next: Tensor) -> TurboState:
    """Update success/failure counters and the trust-region length from one
    freshly evaluated batch (`Y_next`: (batch,), `C_next`: (batch, n_constraints)).
    """
    best_ind = get_best_index_for_batch(Y=Y_next, C=C_next)
    y_next, c_next = Y_next[best_ind], C_next[best_ind]

    if bool((c_next <= 0).all()):
        improvement_threshold = state.best_value + 1e-3 * math.fabs(state.best_value)
        if bool(y_next > improvement_threshold) or bool((state.best_constraint_values > 0).any()):
            state.success_counter += 1
            state.failure_counter = 0
            state.best_value = float(y_next)
            state.best_constraint_values = c_next
        else:
            state.success_counter = 0
            state.failure_counter += 1
    else:
        total_violation_next = c_next.clamp(min=0).sum(dim=-1)
        total_violation_center = state.best_constraint_values.clamp(min=0).sum(dim=-1)
        if bool(total_violation_next < total_violation_center):
            state.success_counter += 1
            state.failure_counter = 0
            state.best_value = float(y_next)
            state.best_constraint_values = c_next
        else:
            state.success_counter = 0
            state.failure_counter += 1

    return update_tr_length(state)


def _box_candidates(
    x_center: Tensor, length: float, dim: int, n_candidates: int, generator: torch.Generator
) -> Tensor:
    """Tutorial's `generate_batch` candidate pool for a plain box domain:
    Sobol points inside the trust region, with a per-candidate random subset
    of dimensions perturbed away from `x_center` (the rest held fixed at it).
    `x_center` and the return value are normalized to [0, 1]^dim.

    The tutorial's own fallback line -- forcing at least one perturbed
    dimension when the random mask picks none -- draws that dimension from
    `randint(0, dim - 1)`, so index `dim - 1` can never be the forced one. It
    is transcribed unchanged; it never actually fires for the tasks this
    method runs on; since `prob_perturb = min(20 / dim, 1.0) = 1.0` whenever
    `dim <= 20` (branin: 2, pce10: 4, oer_plate_3496: 6), the mask is always
    all-True and the fallback branch is dead code here.
    """
    tr_lb = torch.clamp(x_center - length / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + length / 2.0, 0.0, 1.0)

    seed = int(torch.randint(0, 2**31 - 1, (1,), generator=generator).item())
    sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
    pert = sobol.draw(n_candidates).to(dtype=DTYPE)
    pert = tr_lb + (tr_ub - tr_lb) * pert

    prob_perturb = min(20.0 / dim, 1.0)
    mask = torch.rand(n_candidates, dim, generator=generator, dtype=DTYPE) <= prob_perturb
    ind = torch.where(mask.sum(dim=1) == 0)[0]
    if len(ind) > 0:
        mask[ind, torch.randint(0, dim - 1, size=(len(ind),), generator=generator)] = True

    X_cand = x_center.expand(n_candidates, dim).clone()
    X_cand[mask] = pert[mask]
    return X_cand


def _simplex_candidates(
    variables, x_center: Tensor, length: float, n_candidates: int
) -> Tensor:
    """Simplex-respecting stand-in for `_box_candidates` (see module
    docstring's "simplex tasks" section). Returns normalized candidates."""
    fresh = variables.sample_raw_candidates(n_candidates, variables, {}).squeeze(1)
    alpha = float(min(max(length, 0.0), 1.0))
    center = x_center.unsqueeze(0).expand(n_candidates, -1)
    cand = (1.0 - alpha) * center + alpha * fresh.to(center)

    for group in variables.task.simplex_groups:
        idx = list(group.indices)
        total = cand[:, idx].sum(dim=-1)
        assert torch.allclose(total, torch.ones_like(total), atol=1e-6), (
            "simplex candidate does not sum to 1 in normalized space "
            f"(group indices {idx}, max deviation "
            f"{(total - 1.0).abs().max().item():.3e})"
        )
    return cand


def _thompson_select(
    model,
    X_cand: Tensor,
    batch_size: int,
    lower: float,
    upper: float,
    target_mid: float,
    target_half: float,
    y_mean: Tensor,
    y_std: Tensor,
) -> Tensor:
    """SCBO's constrained Thompson sampling, adapted so the objective and both
    constraints read off one shared posterior draw of `y` (see module
    docstring). Feasible draws are ranked by the window-distance objective;
    infeasible-only draws are ranked by total violation of `c1`/`c2` --
    exactly `ConstrainedMaxPosteriorSampling._convert_samples_to_scores`
    (doi.org/10.48550/arXiv.2002.08526), just computed from our own sample
    tensor instead of via that class's independent constraint-model posterior
    calls. The final without-replacement top-k selection over the batch is
    delegated to `MaxPosteriorSampling.maximize_samples`, unmodified -- the
    exact method `ConstrainedMaxPosteriorSampling` itself calls.
    """
    posterior = model.posterior(X_cand)
    with torch.no_grad():
        y_std_samples = posterior.rsample(sample_shape=torch.Size([batch_size])).squeeze(-1)
    y_raw_samples = y_std_samples * y_std.to(y_std_samples) + y_mean.to(y_std_samples)

    obj = -((y_raw_samples - target_mid) / target_half) ** 2
    c1 = lower - y_raw_samples
    c2 = y_raw_samples - upper
    is_feasible = (c1 <= 0) & (c2 <= 0)
    has_feasible = is_feasible.any(dim=-1)

    scores = obj.clone()
    scores[~is_feasible] = -float("inf")
    if not bool(has_feasible.all()):
        total_violation = c1.clamp(min=0) + c2.clamp(min=0)
        rows = ~has_feasible
        scores[rows] = -total_violation[rows]

    selector = MaxPosteriorSampling(model=model, objective=IdentityMCObjective(), replacement=False)
    return selector.maximize_samples(X_cand, scores.unsqueeze(-1), num_samples=batch_size)


class SCBOMethod:
    """Runs SCBO on this project's surrogate, window-adapted for a single
    two-sided target output. See module docstring for the full mapping."""

    name = NAME
    #: A raw two-sided window is exactly what `objective_mode="range"` exposes
    #: through `target_info[key]["lb"/"ub"]"` -- no min/max flipping needed,
    #: unlike COMBOO's `native_extremum` mapping.
    objective_mode = "range"

    def __init__(self, num_restarts: int = 128):
        self.num_restarts = num_restarts
        self._state: TurboState | None = None
        self._n_observed = 0
        self._iteration = 0
        # Set right after a restart fires; makes the *next* call's trust-region
        # center a fresh random point instead of the historical incumbent --
        # see the "trust-region center" section of `propose()` for why.
        self._pending_reseed = False

    # -- Method protocol -----------------------------------------------------

    def propose(self, variables, q: int, seed: int, recorder=None) -> np.ndarray:
        if len(variables.y_key) != 1:
            raise ValueError(
                "SCBO is a single-objective constrained method; it needs exactly "
                f"one output but this task has {len(variables.y_key)}."
            )

        key = variables.y_key[0]
        info = variables.target_info[key]
        lower, upper = float(info["lb"]), float(info["ub"])
        target_mid = 0.5 * (lower + upper)
        target_half = max(0.5 * (upper - lower), 1e-9)

        dim = variables.task.dim
        bounds_x = variables.item_bound

        mask = variables.continuous_value_mask[:, 0]
        train_x_raw = variables.train_x[mask]
        train_y_raw = variables.train_y[mask, 0]
        n_obs = int(train_x_raw.shape[0])

        train_x_norm = normalize(train_x_raw, bounds_x).clamp(0.0, 1.0)
        train_y_std, y_mean, y_std = standardize(train_y_raw.unsqueeze(-1))

        mll, model = initialize_model(
            train_x_norm, train_y_std, [], torch.ones_like(train_y_std, dtype=torch.bool)
        )
        fit_gpytorch_mll(mll)

        if self._state is None:
            self._state = TurboState(dim=dim, batch_size=q)
            # The initial design (n_init random/DoE points, evaluated before
            # any method acts) is not a batch SCBO itself proposed, so it must
            # not be folded into the trust-region bookkeeping -- the ported
            # tutorial never calls `update_state` on its initial batch either,
            # only on batches its own `generate_batch` produced. Treating the
            # whole initial design as a single success-or-failure verdict would
            # move `success_counter`/`failure_counter` by exactly 1 regardless
            # of how many initial points there were, corrupting the trust
            # region's state from iteration zero.
            self._n_observed = n_obs
        elif n_obs > self._n_observed:
            # -- fold newly observed rows into the trust-region bookkeeping --
            new_y = train_y_raw[self._n_observed : n_obs]
            self._n_observed = n_obs
            if new_y.numel() > 0:
                obj_new = -((new_y - target_mid) / target_half) ** 2
                c_new = torch.stack([lower - new_y, new_y - upper], dim=-1)
                self._state = update_state(self._state, obj_new, c_new)
                if self._state.restart_triggered:
                    if recorder is not None:
                        recorder.note("scbo_restart_triggered")
                    self._state = TurboState(dim=dim, batch_size=q)
                    self._n_observed = n_obs
                    self._pending_reseed = True

        # -- trust-region center -----------------------------------------
        if self._pending_reseed:
            # Standard TuRBO restart semantics: a collapsed trust region has
            # exhausted its local neighborhood, so the new region should start
            # from a fresh location, not re-center on the same incumbent that
            # caused the collapse (which `get_best_index_for_batch` below would
            # otherwise pick again, since one exploratory point rarely beats an
            # already-established incumbent -- re-centering on history would
            # make the "restart" a no-op that just re-shrinks around the same
            # point instead of exploring elsewhere).
            self._pending_reseed = False
            x_center = variables.sample_raw_candidates(1, variables, {})[0, 0]
        else:
            # Best point among everything observed so far.
            obj_all = -((train_y_raw - target_mid) / target_half) ** 2
            c_all = torch.stack([lower - train_y_raw, train_y_raw - upper], dim=-1)
            best_ind = get_best_index_for_batch(obj_all, c_all)
            x_center = train_x_norm[best_ind]

        n_candidates = min(5000, max(2000, 20 * self.num_restarts))
        generator = torch.Generator().manual_seed(seed * 7919 + self._iteration)
        self._iteration += 1

        if variables.task.has_simplex:
            X_cand = _simplex_candidates(variables, x_center, self._state.length, n_candidates)
        else:
            X_cand = _box_candidates(x_center, self._state.length, dim, n_candidates, generator)

        X_next = _thompson_select(
            model, X_cand, q, lower, upper, target_mid, target_half, y_mean, y_std
        )
        return variables.to_raw(X_next.detach().cpu().numpy()).reshape(q, dim)


def make(name: str = NAME, num_restarts: int = 128) -> SCBOMethod:
    return SCBOMethod(num_restarts=num_restarts)
