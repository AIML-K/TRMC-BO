"""Fidelity of the COMBOO transcription.

The algorithm was copied out of a notebook, so the acquisitions are checked
against their formulas evaluated independently here. A baseline that is subtly
mis-transcribed is worse than no baseline: it would lose for reasons that have
nothing to do with the method.

Reference formulas, from `experiments_COMBOO/branin-currin_COMBOO_exp.ipynb`:

    beta       = 0.2 * 2 * log(4 * (t + 1))
    beta_const = 0.2 * 2 * log(2 * (t + 1) * 2)
    A_aux(x)   = min_i ( mu_i + beta_const * sd_i - ref_i )
    A(x)       = min_i ( max(0, (mu_i + beta * sd_i - ref_i) / theta_i) )^m
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from eval.methods.comboo import (
    AuxiliaryUCB,
    COMBOOMethod,
    HyperVolumeScalarizedUCB,
    TwoSidedAuxiliaryUCB,
    _sample_theta,
    optimistic_window_constraints,
)


class _Posterior:
    def __init__(self, mean, variance):
        self.mean = mean
        self.variance = variance


class _FixedModel(torch.nn.Module):
    """Fixed posterior so the acquisitions are testable without fitting a GP."""

    def __init__(self, mean, var):
        super().__init__()
        self._mean = torch.as_tensor(mean, dtype=torch.double)
        self._var = torch.as_tensor(var, dtype=torch.double)

    def posterior(self, X):
        shape = X.shape[:-1] + (self._mean.numel(),)
        return _Posterior(self._mean.expand(shape).clone(), self._var.expand(shape).clone())


MEAN = [1.0, -0.5, 0.25]
VAR = [0.04, 0.09, 0.01]
REF = [0.0, -1.0, 0.0]
X1 = torch.zeros(1, 1, 2, dtype=torch.double)


def test_beta_schedule_matches_the_paper_code():
    """The schedule is what makes the confidence widths shrink over iterations."""
    for t in range(6):
        assert 0.2 * 2 * math.log(4 * (t + 1)) == pytest.approx(0.4 * math.log(4 * t + 4))
        assert 0.2 * 2 * math.log(2 * (t + 1) * 2) == pytest.approx(0.4 * math.log(4 * t + 4))
    # The two happen to coincide in their code; assert it rather than assume it,
    # so a future edit that separates them is visible.
    for t in range(6):
        assert (0.2 * 2 * math.log(4 * (t + 1))
                == pytest.approx(0.2 * 2 * math.log(2 * (t + 1) * 2)))


def test_auxiliary_acquisition_matches_formula():
    beta_const = 0.4 * math.log(8.0)
    got = AuxiliaryUCB(
        model=_FixedModel(MEAN, VAR), beta=torch.tensor(beta_const),
        ref=torch.tensor(REF, dtype=torch.double),
    )(X1).item()

    mu, sd, ref = np.array(MEAN), np.sqrt(np.array(VAR)), np.array(REF)
    assert got == pytest.approx(float(np.min(mu + beta_const * sd - ref)))


def test_scalarized_ucb_matches_formula():
    beta = 0.4 * math.log(8.0)
    theta = np.array([0.6, 0.5, 0.62449979983])  # unit norm
    got = HyperVolumeScalarizedUCB(
        model=_FixedModel(MEAN, VAR), beta=torch.tensor(beta),
        theta=torch.tensor(theta, dtype=torch.double),
        ref=torch.tensor(REF, dtype=torch.double),
    )(X1).item()

    mu, sd, ref = np.array(MEAN), np.sqrt(np.array(VAR)), np.array(REF)
    u = mu + beta * sd - ref
    expected = float(np.min(np.clip(u / theta, 0.0, None) ** len(mu)))
    assert got == pytest.approx(expected)


def test_scalarized_ucb_is_zero_when_any_objective_is_below_reference():
    """The `max(0, .)` inside a `min` makes one lagging objective veto the point."""
    beta = 0.0
    mean = [1.0, -2.0, 0.25]   # second is below its ref of -1.0
    got = HyperVolumeScalarizedUCB(
        model=_FixedModel(mean, VAR), beta=torch.tensor(beta),
        theta=torch.ones(3, dtype=torch.double),
        ref=torch.tensor(REF, dtype=torch.double),
    )(X1).item()
    assert got == pytest.approx(0.0)


def test_auxiliary_goes_negative_when_no_output_can_clear_its_threshold():
    """This sign is the feasibility declaration; it must not be reversed."""
    aux = AuxiliaryUCB(
        model=_FixedModel([-5.0, -5.0, -5.0], [1e-9] * 3),
        beta=torch.tensor(0.1), ref=torch.tensor(REF, dtype=torch.double),
    )(X1).item()
    assert aux < 0


def test_theta_is_on_the_positive_unit_sphere():
    g = torch.Generator().manual_seed(0)
    for _ in range(20):
        theta = _sample_theta(3, g)
        assert theta.shape == (3,)
        assert torch.all(theta >= 0)
        assert theta.norm().item() == pytest.approx(1.0)


def test_window_constraints_are_two_sided_and_optimistic():
    """Two per output, and each uses the *widest* posterior reading."""
    lower = torch.tensor([0.0, 0.0], dtype=torch.double)
    upper = torch.tensor([1.0, 1.0], dtype=torch.double)
    beta_const = 2.0
    model = _FixedModel([0.5, 0.5], [0.25, 0.25])   # sd = 0.5

    cons = optimistic_window_constraints(model, beta_const, lower, upper)
    assert len(cons) == 4
    assert all(intra for _, intra in cons)

    X = torch.zeros(1, 2, dtype=torch.double)
    # lower side: mu + b*sd - L = 0.5 + 1.0 - 0 = 1.5
    assert cons[0][0](X).item() == pytest.approx(1.5)
    # upper side: U - (mu - b*sd) = 1 - (0.5 - 1.0) = 1.5
    assert cons[1][0](X).item() == pytest.approx(1.5)


def test_window_constraint_tightens_as_uncertainty_falls():
    """With no uncertainty the optimistic bound collapses onto the mean."""
    lower = torch.tensor([0.0], dtype=torch.double)
    upper = torch.tensor([1.0], dtype=torch.double)
    X = torch.zeros(1, 2, dtype=torch.double)

    wide = optimistic_window_constraints(_FixedModel([2.0], [1.0]), 2.0, lower, upper)
    tight = optimistic_window_constraints(_FixedModel([2.0], [1e-12]), 2.0, lower, upper)
    # mean 2.0 is above the cap of 1.0: optimism can excuse it, certainty cannot.
    assert wide[1][0](X).item() > 0
    assert tight[1][0](X).item() < 0


def test_window_flips_with_a_minimize_objective():
    """`negate_y` flips min outputs, so their window must flip too."""

    class _Vars:
        y_key = ["a", "b"]
        target_info = {"a": {"obj": "max"}, "b": {"obj": "min"}}

        class _Spec:
            lower = {"a": 10.0, "b": 100.0}
            upper = {"a": 20.0, "b": 200.0}

        range_spec = _Spec()

    lo, up = COMBOOMethod._window_in_model_space(_Vars(), y_mean=[0.0, 0.0], y_std=[1.0, 1.0])
    assert (lo[0].item(), up[0].item()) == pytest.approx((10.0, 20.0))
    # 'b' is minimized -> negated -> window becomes [-200, -100]
    assert (lo[1].item(), up[1].item()) == pytest.approx((-200.0, -100.0))


def test_hv_reference_is_not_the_constraint_threshold():
    """Regression for the withheld-results bug: `ref` must not collapse onto
    the window's lower edge, or "improvement above ref" becomes nearly
    synonymous with "clears the constraint" (see comboo.py's module docstring,
    "Mapping fix" section)."""
    torch.manual_seed(0)
    # Ten draws per output, standardized-ish scale; the window's lower edge
    # sits near the low tail (`lower ~= -1.5`), close to where the *old*
    # `ref = lower.clone()` would have put it.
    train_y_std = torch.randn(10, 2, dtype=torch.double)
    lower = torch.tensor([-1.5, -1.5], dtype=torch.double)

    ref = COMBOOMethod._hv_reference(train_y_std)

    assert ref.shape == (2,)
    # The 10th percentile of ten standard-normal draws is not exactly equal
    # to an arbitrary window edge -- this would only coincide by chance.
    assert not torch.allclose(ref, lower)


def test_hv_reference_ignores_nan_rows():
    """HOIP/MatFormBench can carry NaN rows for unobserved/infeasible outputs;
    the reference quantile must be computed over the observed values only."""
    train_y_std = torch.tensor(
        [[1.0, float("nan")], [2.0, 5.0], [3.0, 6.0], [4.0, 7.0]], dtype=torch.double
    )
    ref = COMBOOMethod._hv_reference(train_y_std)
    assert torch.isfinite(ref).all()


def test_two_sided_auxiliary_matches_formula():
    beta_const = 0.4 * math.log(8.0)
    lower = torch.tensor([0.0, -2.0], dtype=torch.double)
    upper = torch.tensor([2.0, 0.0], dtype=torch.double)
    model = _FixedModel([1.0, -0.5], [0.04, 0.09])

    got = TwoSidedAuxiliaryUCB(
        model=model, beta=torch.tensor(beta_const), lower=lower, upper=upper
    )(X1).item()

    mu, sd = np.array([1.0, -0.5]), np.sqrt(np.array([0.04, 0.09]))
    u_lower = mu + beta_const * sd - lower.numpy()
    u_upper = upper.numpy() - (mu - beta_const * sd)
    assert got == pytest.approx(float(np.min(np.concatenate([u_lower, u_upper]))))


def test_two_sided_auxiliary_is_a_valid_seed_for_the_window_constraints():
    """The whole point of `TwoSidedAuxiliaryUCB`: whenever its value is >= 0,
    every component `optimistic_window_constraints` builds for the SAME
    model/beta/lower/upper must also be >= 0 -- otherwise the probe's optimum
    is not a valid `batch_initial_conditions` for stage 2's constrained solve,
    which is exactly the bug this class fixes (see comboo.py's "Mapping fix"
    section)."""
    lower = torch.tensor([0.0, -2.0], dtype=torch.double)
    upper = torch.tensor([2.0, 0.0], dtype=torch.double)
    beta_const = 1.5

    for mean, var in [
        ([1.0, -1.0], [0.01, 0.01]),   # comfortably inside the window
        ([0.05, -1.95], [0.01, 0.01]),  # just barely inside
        ([1.9, -0.2], [0.5, 0.5]),      # near the edges, wide posterior
    ]:
        model = _FixedModel(mean, var)
        aux_value = TwoSidedAuxiliaryUCB(
            model=model, beta=torch.tensor(beta_const), lower=lower, upper=upper
        )(X1).item()
        cons = optimistic_window_constraints(model, beta_const, lower, upper)
        X = torch.zeros(1, 2, dtype=torch.double)
        con_values = [c(X).item() for c, _ in cons]
        if aux_value >= 0:
            assert all(v >= -1e-9 for v in con_values), (mean, var, con_values)


def test_two_sided_auxiliary_checks_the_upper_bound_unlike_auxiliary_ucb():
    """`AuxiliaryUCB` (one-sided, as published) is blind to a blown upper bound;
    `TwoSidedAuxiliaryUCB` must not be, or it cannot guard stage 2's seed."""
    lower = torch.tensor([0.0], dtype=torch.double)
    upper = torch.tensor([1.0], dtype=torch.double)
    # Mean is far above the upper bound with no uncertainty to excuse it.
    model = _FixedModel([5.0], [1e-9])

    one_sided = AuxiliaryUCB(
        model=model, beta=torch.tensor(0.1), ref=lower,
    )(X1).item()
    two_sided = TwoSidedAuxiliaryUCB(
        model=model, beta=torch.tensor(0.1), lower=lower, upper=upper,
    )(X1).item()

    assert one_sided > 0   # blind to the upper bound: reports "feasible"
    assert two_sided < 0   # catches it


def test_single_output_task_is_rejected_rather_than_silently_degenerating():
    """COMBOO is constrained *multi-objective*; one output is out of its scope."""

    class _Vars:
        y_key = ["only"]

    with pytest.raises(ValueError, match="multi-objective"):
        COMBOOMethod().propose(_Vars(), q=1, seed=0)


def test_batch_size_other_than_one_is_rejected():
    """Both internal `optimize_acqf` calls are hardcoded to q=1; a q=2 request
    must fail honestly rather than crash later on a reshape."""

    class _Vars:
        y_key = ["a", "b"]

    with pytest.raises(ValueError, match="q=1"):
        COMBOOMethod().propose(_Vars(), q=2, seed=0)


def test_zero_observations_for_one_output_falls_back_rather_than_crashing():
    """Regression: a column that is still all-NaN this early in a campaign
    (plausible on a sparse-feasibility benchmark) used to crash
    `fit_gpytorch_mll` on a 0-row sub-model. COMBOO cannot fit a posterior for
    that output at all, so it must fall back to a feasible raw proposal
    instead of taking down the whole campaign."""

    class _Vars:
        y_key = ["a", "b"]
        item_bound = torch.stack(
            [torch.zeros(2, dtype=torch.double), torch.ones(2, dtype=torch.double)]
        )
        train_x = torch.rand(5, 2, dtype=torch.double)
        # Output 'b' has zero valid observations; 'a' has some.
        continuous_value_mask = torch.tensor(
            [[True, False]] * 5, dtype=torch.bool
        )

        class _Task:
            dim = 2

        task = _Task()

        def sample_raw_candidates(self, num_restarts, dataloader, optimize_config):
            return torch.full((num_restarts, 1, 2), 0.5, dtype=torch.double)

        @staticmethod
        def to_raw(Z):
            return np.asarray(Z, dtype=float)

    out = COMBOOMethod().propose(_Vars(), q=1, seed=0)
    assert out.shape == (1, 2)
    assert np.allclose(out, 0.5)


def test_zero_observations_note_names_the_missing_outputs():
    class _Recorder:
        def __init__(self):
            self.notes = []

        def note(self, text):
            self.notes.append(text)

    class _Vars:
        y_key = ["a", "b"]
        item_bound = torch.stack(
            [torch.zeros(2, dtype=torch.double), torch.ones(2, dtype=torch.double)]
        )
        train_x = torch.rand(3, 2, dtype=torch.double)
        continuous_value_mask = torch.tensor([[False, True]] * 3, dtype=torch.bool)

        class _Task:
            dim = 2

        task = _Task()

        def sample_raw_candidates(self, num_restarts, dataloader, optimize_config):
            return torch.full((num_restarts, 1, 2), 0.5, dtype=torch.double)

        @staticmethod
        def to_raw(Z):
            return np.asarray(Z, dtype=float)

    recorder = _Recorder()
    COMBOOMethod().propose(_Vars(), q=1, seed=0, recorder=recorder)
    assert recorder.notes == ["comboo_no_observations_for_output:a"]
