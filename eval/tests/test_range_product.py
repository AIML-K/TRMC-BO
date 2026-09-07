"""Numerical validation of the §4.2 product acquisition.

The point of this variant is that a product cannot be satisfied by one output
alone, unlike the hypervolume the shipped path maximizes. That claim is only
worth making if the arithmetic is right, so the log-sum is checked against an
independent `scipy.stats.norm` reference and the joint behaviour is checked
against the property that motivated it.

The plateau test is deliberate, not a bug being enshrined: wrapping a
sample-independent objective in qEI makes every candidate below the incumbent tie
at exactly zero, and the comparison in `eval/scripts/acquisition_auc.py` reports
that variant at AUC 0.500 for exactly this reason. Pinning it here keeps the
explanation honest if someone later changes the wrapper.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.stats import norm

from eval import methods
from eval.methods.range_product import (
    JointRangeProbability,
    JointRangeProbabilityObjective,
    _log_normal_interval,
    _log_range_probability,
    _standardized_windows,
)


class _StubPosterior:
    def __init__(self, mean, variance):
        self.mean = mean
        self.variance = variance


def _stub(mean, std):
    m = torch.as_tensor(mean, dtype=torch.double)
    s = torch.as_tensor(std, dtype=torch.double)
    return _StubPosterior(m, s ** 2)


def test_log_range_probability_matches_scipy():
    mean = [[0.2, -0.5, 1.1]]
    std = [[0.4, 0.9, 0.3]]
    lower = torch.tensor([-0.5, -1.0, 0.5], dtype=torch.double)
    upper = torch.tensor([0.5, 0.0, 1.5], dtype=torch.double)

    got = _log_range_probability(
        _stub(mean, std), torch.tensor([0, 1, 2]), lower, upper
    )

    expect = sum(
        np.log(norm.cdf((u - m) / s) - norm.cdf((lo - m) / s))
        for m, s, lo, u in zip(mean[0], std[0], lower.tolist(), upper.tolist())
    )
    assert got.item() == pytest.approx(float(expect), abs=1e-10)


def test_one_hopeless_output_sinks_the_product():
    """The whole reason for preferring a product over a hypervolume.

    Candidate B is far better on output 0 and hopeless on output 1. Hypervolume
    over the probability vector would reward it; the product must not.
    """
    lower = torch.tensor([-1.0, -1.0], dtype=torch.double)
    upper = torch.tensor([1.0, 1.0], dtype=torch.double)
    idx = torch.tensor([0, 1])

    balanced = _log_range_probability(_stub([[0.0, 0.0]], [[0.5, 0.5]]), idx, lower, upper)
    lopsided = _log_range_probability(_stub([[0.0, 8.0]], [[0.1, 0.5]]), idx, lower, upper)

    assert lopsided.item() < balanced.item()
    # And the sum of the two probabilities *does* favour the lopsided candidate,
    # which is the failure mode being avoided rather than a hypothetical one.
    p_balanced = float(np.exp(balanced.item()))
    assert p_balanced > 0.5
    assert float(np.exp(lopsided.item())) < 1e-6


@pytest.mark.parametrize("a,b", [(-1.0, 1.0), (-3.0, -1.0), (0.5, 2.0), (-0.2, 0.0)])
def test_log_interval_agrees_with_the_naive_formula_where_it_is_accurate(a, b):
    """Inside a few standard deviations the naive difference is exact, so the
    stable version must reproduce it -- the tail handling must not cost accuracy
    in the regime that matters most of the time."""
    got = _log_normal_interval(
        torch.tensor([a], dtype=torch.double), torch.tensor([b], dtype=torch.double)
    )
    expect = np.log(norm.cdf(b) - norm.cdf(a))
    assert got.item() == pytest.approx(float(expect), rel=1e-12)


@pytest.mark.parametrize("a,b", [(-1.0, 2.0), (-45.0, -39.0), (39.0, 41.0), (0.0, 0.3)])
def test_log_interval_is_reflection_symmetric(a, b):
    """P(a <= Z <= b) == P(-b <= Z <= -a) for a standard normal.

    An independent invariant rather than a re-derivation: the two sides take
    different branches of the tail/straddle split, so agreement means the split
    itself is consistent.
    """
    t = lambda v: torch.tensor([v], dtype=torch.double)  # noqa: E731
    assert _log_normal_interval(t(a), t(b)).item() == pytest.approx(
        _log_normal_interval(t(-b), t(-a)).item(), rel=1e-12
    )


def test_log_interval_has_a_usable_gradient_deep_in_the_tail():
    """The reason the stable form is worth the code.

    `optimize_acqf` drives this with L-BFGS. A vanishing gradient 40 standard
    deviations out would leave a restart stranded exactly where the surrogate
    thinks nothing works yet, which is where the search starts.
    """
    mu = torch.tensor([40.0], dtype=torch.double, requires_grad=True)
    lower = torch.tensor([-1.0], dtype=torch.double)
    upper = torch.tensor([1.0], dtype=torch.double)
    _log_normal_interval(lower - mu, upper - mu).sum().backward()
    assert mu.grad is not None
    assert abs(mu.grad.item()) > 1.0


def test_log_sum_survives_probabilities_a_product_would_underflow():
    """Three windows at ~1e-120 each: the reason for summing logs."""
    mean = [[40.0, 40.0, 40.0]]
    std = [[1.0, 1.0, 1.0]]
    lower = torch.tensor([-1.0, -1.0, -1.0], dtype=torch.double)
    upper = torch.tensor([1.0, 1.0, 1.0], dtype=torch.double)

    got = _log_range_probability(_stub(mean, std), torch.tensor([0, 1, 2]), lower, upper)
    assert np.isfinite(got.item())
    # A plain float64 product of the three factors is exactly zero here.
    assert float(norm.cdf(1.0 - 40.0) - norm.cdf(-1.0 - 40.0)) ** 3 == 0.0
    # Still ordered against a slightly better candidate, which a product
    # bottoming out at 0.0 could not be.
    better = _log_range_probability(
        _stub([[39.0, 40.0, 40.0]], std), torch.tensor([0, 1, 2]), lower, upper
    )
    assert better.item() > got.item()


def test_standardized_windows_standardizes_with_the_model_scale():
    target_info = {
        "y1": {"obj": "range", "lb": 10.0, "ub": 20.0, "weight": 1.0},
        "y2": {"obj": "range", "lb": 0.0, "ub": 4.0, "weight": 1.0},
    }
    idx, lo, hi = _standardized_windows(
        ["y1", "y2"], target_info, y_mean=[15.0, 2.0], y_std=[5.0, 2.0]
    )
    assert idx == [0, 1]
    assert lo == pytest.approx([-1.0, -1.0])
    assert hi == pytest.approx([1.0, 1.0])


def test_non_range_target_is_rejected_rather_than_ignored():
    with pytest.raises(ValueError, match="no range target"):
        _standardized_windows(
            ["y1"], {"y1": {"obj": "max", "weight": 1.0}}, y_mean=[0.0], y_std=[1.0]
        )


class _StubModel:
    """Two independent unit-variance outputs centred on a fixed mean."""

    num_outputs = 2

    def __init__(self, center):
        self.center = torch.as_tensor(center, dtype=torch.double)

    def posterior(self, X):
        shape = X.shape[:-1] + (2,)
        mean = self.center.expand(shape).clone()
        return _StubPosterior(mean, torch.full(shape, 0.25, dtype=torch.double))


def test_joint_acquisition_shape_and_ordering():
    acqf = JointRangeProbability(
        model=_StubModel([0.0, 0.0]), index=[0, 1], lower=[-1.0, -1.0], upper=[1.0, 1.0]
    )
    X = torch.rand(7, 1, 3, dtype=torch.double)
    val = acqf(X)
    assert val.shape == (7,)

    far = JointRangeProbability(
        model=_StubModel([5.0, 5.0]), index=[0, 1], lower=[-1.0, -1.0], upper=[1.0, 1.0]
    )
    assert far(X).max().item() < val.min().item()


def test_qei_wrapped_objective_returns_the_product_itself():
    """Not a log, and not per-output: qEI must subtract a comparable incumbent."""
    obj = JointRangeProbabilityObjective(
        model=_StubModel([0.0, 0.0]), index=[0, 1], lower=[-1.0, -1.0], upper=[1.0, 1.0]
    )
    X = torch.rand(4, 1, 3, dtype=torch.double)
    samples = torch.rand(16, 4, 1, 2, dtype=torch.double)
    out = obj(samples, X)

    assert out.shape == (16, 4, 1)
    expected = float(norm.cdf(2.0) - norm.cdf(-2.0)) ** 2
    assert out.min().item() == pytest.approx(expected, abs=1e-9)
    assert out.max().item() == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize(
    "name,c2,c3",
    [
        ("M3p_range_prob_product", False, False),
        ("M3q_range_prob_product_ei", False, False),
        ("M5p_full_product", True, True),
        ("M5q_full_product_ei", True, True),
    ],
)
def test_registry_entries_carry_the_intended_components(name, c2, c3):
    m = methods.make(name, num_restarts=4)
    assert m.name == name
    assert m.spec.mean_range_constraints is c2
    assert m.spec.mean_filtered_starts is c3
    assert m.objective_mode == "range"


def test_c1_only_variants_do_not_attach_constraints():
    """The C1-only rows exist to let the acquisition decide alone.

    A smoke test found the C2/C3 variants proposing identical points under three
    different acquisitions, because the constraint set left almost no admissible
    volume. If these rows ever start attaching constraints, that separation --
    and the conclusion drawn from it -- silently disappears.
    """

    class _Vars:
        def linear_constraints(self):
            return {"equality_constraints": None, "inequality_constraints": None}

    cfg = methods.make("M3p_range_prob_product", 4).build_optimize_config(_Vars(), q=1)
    assert cfg["_use_mean_range_constraints"] is False
    assert cfg["_use_mean_filtered_starts"] is False
    assert cfg["_acqf_factory"] is not None

    full = methods.make("M5p_full_product", 4).build_optimize_config(_Vars(), q=1)
    assert full["_use_mean_range_constraints"] is True
    assert full["_use_mean_filtered_starts"] is True
