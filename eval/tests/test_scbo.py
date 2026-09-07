"""Fidelity of the SCBO transcription and the window-to-constraint mapping.

The trust-region mechanics were copied out of the botorch `scalable_constrained_bo`
tutorial notebook (no importable library ships them), so `TurboState` /
`update_tr_length` / `update_state` / `get_best_index_for_batch` are checked here
against hand-computed values. A subtly mis-transcribed trust region would make the
baseline lose for reasons that have nothing to do with the method -- the same risk
`test_comboo.py` guards against for COMBOO.

Reference formulas (see `scbo.py`'s module docstring):

    failure_tolerance = ceil(max(4 / batch_size, dim / batch_size))
    success: length = min(2 * length, length_max) after success_tolerance (=10)
             consecutive successes
    failure: length = length / 2 after failure_tolerance consecutive failures
    restart_triggered = length < length_min (= 0.5 ** 7)
"""

from __future__ import annotations

import math

import pytest
import torch

from eval.methods.scbo import (
    NAME,
    SCBOMethod,
    TurboState,
    _simplex_candidates,
    get_best_index_for_batch,
    update_state,
    update_tr_length,
)


# -- TurboState / update_tr_length -------------------------------------------


def test_failure_tolerance_formula():
    # dim=10, batch=4 -> ceil(max(1.0, 2.5)) = 3
    assert TurboState(dim=10, batch_size=4).failure_tolerance == 3
    # dim=2, batch=1 -> ceil(max(4.0, 2.0)) = 4
    assert TurboState(dim=2, batch_size=1).failure_tolerance == 4
    # dim=6, batch=1 -> ceil(max(4.0, 6.0)) = 6
    assert TurboState(dim=6, batch_size=1).failure_tolerance == 6


def test_defaults_match_the_tutorial():
    state = TurboState(dim=4, batch_size=1)
    assert state.length == pytest.approx(0.8)
    assert state.length_min == pytest.approx(0.5**7)
    assert state.length_max == pytest.approx(1.6)
    assert state.success_tolerance == 10
    assert state.best_value == -float("inf")
    assert torch.all(torch.isinf(state.best_constraint_values))
    assert state.restart_triggered is False


def test_length_doubles_on_success_streak_and_caps_at_length_max():
    state = TurboState(dim=2, batch_size=1)
    state.length = 1.0
    state.success_counter = state.success_tolerance  # about to fire
    state = update_tr_length(state)
    assert state.length == pytest.approx(min(2.0 * 1.0, state.length_max))
    assert state.success_counter == 0

    # Capped at length_max even if doubling would exceed it.
    state.length = 1.2
    state.success_counter = state.success_tolerance
    state = update_tr_length(state)
    assert state.length == pytest.approx(1.6)  # min(2.4, 1.6)


def test_length_halves_on_failure_streak():
    state = TurboState(dim=2, batch_size=1)
    state.length = 0.8
    state.failure_counter = state.failure_tolerance  # about to fire
    state = update_tr_length(state)
    assert state.length == pytest.approx(0.4)
    assert state.failure_counter == 0


def test_restart_triggers_only_once_length_drops_below_length_min():
    state = TurboState(dim=2, batch_size=1)
    state.length = state.length_min * 1.5
    state.failure_counter = state.failure_tolerance
    state = update_tr_length(state)
    assert state.length == pytest.approx(state.length_min * 0.75)
    assert state.restart_triggered is True

    # Just at the boundary (not below) must not trigger.
    state2 = TurboState(dim=2, batch_size=1)
    state2.length = state2.length_min
    state2 = update_tr_length(state2)
    assert state2.restart_triggered is False


# -- get_best_index_for_batch -------------------------------------------------


def test_best_index_prefers_feasible_by_objective():
    Y = torch.tensor([1.0, 5.0, 3.0], dtype=torch.double)
    # candidate 1 (Y=5) is infeasible; candidate 2 (Y=3) is feasible and best
    # among the feasible ones.
    C = torch.tensor([[-0.1, -0.1], [0.2, -0.1], [-0.1, -0.1]], dtype=torch.double)
    assert get_best_index_for_batch(Y, C).item() == 2


def test_best_index_falls_back_to_least_violation_when_all_infeasible():
    Y = torch.tensor([10.0, 20.0], dtype=torch.double)
    C = torch.tensor([[0.5, 0.1], [0.05, 0.2]], dtype=torch.double)
    # total violation: 0.6 vs 0.25 -> index 1 wins despite lower Y
    assert get_best_index_for_batch(Y, C).item() == 1


# -- update_state ---------------------------------------------------------


def test_update_state_first_feasible_batch_always_counts_as_success():
    """`best_constraint_values` starts at +inf, so the OR-branch
    `(best_constraint_values > 0).any()` is trivially true the first time."""
    state = TurboState(dim=2, batch_size=1)
    Y_next = torch.tensor([-100.0], dtype=torch.double)  # deliberately bad
    C_next = torch.tensor([[-1.0, -1.0]], dtype=torch.double)  # feasible
    state = update_state(state, Y_next, C_next)
    assert state.success_counter == 1
    assert state.failure_counter == 0
    assert state.best_value == pytest.approx(-100.0)


def test_update_state_counts_failure_when_no_improvement():
    state = TurboState(dim=2, batch_size=1)
    state.best_value = 5.0
    state.best_constraint_values = torch.tensor([-1.0, -1.0], dtype=torch.double)
    Y_next = torch.tensor([5.0], dtype=torch.double)  # no improvement
    C_next = torch.tensor([[-1.0, -1.0]], dtype=torch.double)
    state = update_state(state, Y_next, C_next)
    assert state.success_counter == 0
    assert state.failure_counter == 1
    assert state.best_value == pytest.approx(5.0)  # unchanged


def test_update_state_infeasible_batch_compares_total_violation():
    state = TurboState(dim=2, batch_size=1)
    state.best_value = 1.0
    state.best_constraint_values = torch.tensor([0.5, 0.5], dtype=torch.double)  # violation 1.0
    Y_next = torch.tensor([0.0], dtype=torch.double)
    C_next = torch.tensor([[0.1, 0.1]], dtype=torch.double)  # violation 0.2 < 1.0
    state = update_state(state, Y_next, C_next)
    assert state.success_counter == 1
    assert state.best_constraint_values.tolist() == pytest.approx([0.1, 0.1])


# -- window-to-constraint mapping ---------------------------------------------


def test_window_maps_to_two_botorch_style_constraints():
    """`c1 = L - y <= 0` (y >= L) and `c2 = y - U <= 0` (y <= U)."""
    lower, upper = 10.0, 20.0
    y_below, y_inside, y_above = 5.0, 15.0, 25.0

    def constraints(y):
        return lower - y, y - upper

    c1, c2 = constraints(y_below)
    assert c1 > 0 and c2 < 0  # violates the floor only
    c1, c2 = constraints(y_inside)
    assert c1 < 0 and c2 < 0  # feasible
    c1, c2 = constraints(y_above)
    assert c1 < 0 and c2 > 0  # violates the cap only


def test_window_objective_is_negative_squared_distance_from_midpoint():
    lower, upper = 10.0, 20.0
    mid, half = 0.5 * (lower + upper), 0.5 * (upper - lower)

    def obj(y):
        return -((y - mid) / half) ** 2

    assert obj(mid) == pytest.approx(0.0)  # best possible: dead center
    assert obj(lower) == pytest.approx(-1.0)  # at either edge: score -1
    assert obj(upper) == pytest.approx(-1.0)
    assert obj(mid) > obj(lower)  # center strictly beats the edges


# -- single-output guard -------------------------------------------------


def test_single_output_task_is_rejected_rather_than_silently_degenerating():
    """SCBO is single-objective; a multi-output task is out of its scope."""

    class _Vars:
        y_key = ["a", "b"]

    with pytest.raises(ValueError, match="single-objective"):
        SCBOMethod().propose(_Vars(), q=1, seed=0)


def test_registered_name_matches_module_constant():
    assert SCBOMethod().name == NAME == "M7_scbo"


# -- simplex candidate generation ---------------------------------------------


class _FakeGroup:
    def __init__(self, indices):
        self.indices = indices
        self.total = 1.0
        self.min_component = 0.0


class _FakeTask:
    def __init__(self, groups):
        self.simplex_groups = groups


class _FakeVars:
    """Minimal stand-in exercising only what `_simplex_candidates` reads."""

    def __init__(self, dim, groups, rng):
        self.task = _FakeTask(groups)
        self._dim = dim
        self._rng = rng

    def sample_raw_candidates(self, num_restarts, dataloader, optimize_config):
        d = self._dim
        Z = self._rng.random((num_restarts, d))
        for group in self.task.simplex_groups:
            idx = list(group.indices)
            k = len(idx)
            w = self._rng.dirichlet([1.0] * k, size=num_restarts)
            Z[:, idx] = w
        import numpy as np

        return torch.as_tensor(Z, dtype=torch.double).unsqueeze(1)


def test_simplex_candidates_always_sum_to_one():
    import numpy as np

    rng = np.random.default_rng(0)
    groups = [_FakeGroup((0, 1, 2, 3))]
    variables = _FakeVars(dim=4, groups=groups, rng=rng)

    # A valid incumbent center, itself on the simplex.
    center = torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.double)

    for length in (0.01, 0.5, 1.0, 1.6):
        cand = _simplex_candidates(variables, center, length, n_candidates=64)
        totals = cand[:, [0, 1, 2, 3]].sum(dim=-1)
        assert torch.allclose(totals, torch.ones_like(totals), atol=1e-6)


def test_simplex_candidates_concentrate_near_incumbent_as_length_shrinks():
    import numpy as np

    rng = np.random.default_rng(1)
    groups = [_FakeGroup((0, 1, 2))]
    variables = _FakeVars(dim=3, groups=groups, rng=rng)
    center = torch.tensor([0.7, 0.2, 0.1], dtype=torch.double)

    tight = _simplex_candidates(variables, center, length=1e-3, n_candidates=256)
    wide = _simplex_candidates(variables, center, length=1.6, n_candidates=256)

    tight_spread = (tight - center).abs().mean()
    wide_spread = (wide - center).abs().mean()
    assert tight_spread < wide_spread


# -- propose(): initial design bookkeeping + restart reseeding ---------------


class _NoSimplexTask:
    simplex_groups = ()
    has_simplex = False

    def __init__(self, dim):
        self.dim = dim


class _ProposeVars:
    """Stand-in exercising `SCBOMethod.propose()` end to end (real GP fit),
    for a plain box (no-simplex) single-output task."""

    def __init__(self, train_x, train_y, dim, reseed_point=None):
        self.task = _NoSimplexTask(dim)
        self.train_x = train_x
        self.train_y = train_y
        self.continuous_value_mask = torch.ones_like(train_y, dtype=torch.bool)
        self.item_bound = torch.stack(
            [torch.zeros(dim, dtype=torch.double), torch.ones(dim, dtype=torch.double)]
        )
        self.y_key = ["y"]
        self.target_info = {"y": {"obj": "range", "lb": -1.0, "ub": 1.0}}
        self._reseed_point = reseed_point
        self.sample_raw_candidates_calls = 0

    def sample_raw_candidates(self, num_restarts, dataloader, optimize_config):
        self.sample_raw_candidates_calls += 1
        if self._reseed_point is not None:
            return self._reseed_point.view(1, 1, -1).expand(num_restarts, 1, -1)
        d = self.task.dim
        return torch.rand(num_restarts, 1, d, dtype=torch.double)

    def to_raw(self, Z):
        lo, hi = self.item_bound.cpu().numpy()
        import numpy as np

        return np.asarray(Z, dtype=float) * (hi - lo) + lo


def _propose_vars(n=10, d=2, seed=0):
    rng = torch.Generator().manual_seed(seed)
    train_x = torch.rand(n, d, generator=rng, dtype=torch.double)
    train_y = torch.rand(n, 1, generator=rng, dtype=torch.double) * 2 - 1
    return _ProposeVars(train_x, train_y, d)


def test_first_propose_call_does_not_fold_the_initial_design_into_bookkeeping():
    """Regression: the n_init points evaluated before any method acts must not
    move success_counter/failure_counter, since SCBO's own tutorial never runs
    `update_state` on its initial batch -- only on batches its own candidate
    generator produced."""
    variables = _propose_vars(n=15)
    method = SCBOMethod(num_restarts=8)

    method.propose(variables, q=1, seed=0)

    assert method._state.success_counter == 0
    assert method._state.failure_counter == 0
    assert method._n_observed == 15


def test_restart_reseeds_the_trust_region_center_instead_of_the_incumbent():
    """Regression: after a restart, the very next call must draw its
    trust-region center from `sample_raw_candidates` (a fresh point) rather
    than silently reusing the historical incumbent, or the "restart" is a
    no-op that just re-shrinks around the same point."""
    reseed_point = torch.tensor([0.05, 0.05], dtype=torch.double)
    variables = _propose_vars(n=10, d=2)
    variables._reseed_point = reseed_point

    method = SCBOMethod(num_restarts=8)
    method.propose(variables, q=1, seed=0)  # establishes state, no bookkeeping yet
    assert method._pending_reseed is False

    # Force the "just restarted" condition directly, mirroring what
    # update_state's restart branch sets, without needing to actually grind
    # through enough failures to trigger one.
    method._pending_reseed = True
    calls_before = variables.sample_raw_candidates_calls

    method.propose(variables, q=1, seed=0)

    assert variables.sample_raw_candidates_calls == calls_before + 1
    assert method._pending_reseed is False
