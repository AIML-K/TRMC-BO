"""Numerical validation of the Tolerance Ball acquisition.

TB's whole value over sampling-based set estimation is that it is closed-form, so
the closed form had better be right. It is checked twice, against independent
references: `scipy.stats.ncx2` for the CDF itself, and Monte Carlo through the
actual posterior for the acquisition as a whole.

The gradient test matters as much as the values: `optimize_acqf` drives this with
L-BFGS, and a silently-zero gradient would turn TB into random restart search
while still producing plausible-looking numbers.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.stats import ncx2

from eval.methods.range_aware_tb import (
    ToleranceBallAcquisition,
    box_to_ball,
    noncentral_chi2_cdf,
)


@pytest.mark.parametrize("df", [1, 3, 5])
@pytest.mark.parametrize("lam", [0.0, 0.5, 3.0, 25.0, 200.0])
@pytest.mark.parametrize("x_mult", [0.1, 0.5, 1.0, 2.0, 5.0])
def test_cdf_matches_scipy(df, lam, x_mult):
    """Includes large lambda, where a naive j=0 truncation returns ~0."""
    x = float(x_mult * (df + lam))
    got = noncentral_chi2_cdf(
        torch.tensor([x], dtype=torch.double), df, torch.tensor([lam], dtype=torch.double)
    )
    assert got.item() == pytest.approx(float(ncx2.cdf(x, df, lam)), abs=1e-8)


def test_cdf_is_monotone_in_x():
    x = torch.linspace(0.1, 60.0, 80, dtype=torch.double)
    lam = torch.full_like(x, 4.0)
    cdf = noncentral_chi2_cdf(x, 3, lam)
    assert torch.all(cdf[1:] - cdf[:-1] >= -1e-12)
    assert cdf[0] >= 0.0 and cdf[-1] <= 1.0


def test_cdf_decreases_with_noncentrality():
    """Pushing the mean away from the target can only lower the probability."""
    lam = torch.tensor([0.0, 1.0, 5.0, 20.0, 100.0], dtype=torch.double)
    x = torch.full_like(lam, 6.0)
    cdf = noncentral_chi2_cdf(x, 3, lam)
    assert torch.all(cdf[1:] - cdf[:-1] <= 1e-12)


def test_cdf_gradient_flows():
    x = torch.tensor([4.0], dtype=torch.double, requires_grad=True)
    lam = torch.tensor([2.0], dtype=torch.double, requires_grad=True)
    noncentral_chi2_cdf(x, 3, lam).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs() > 0
    assert lam.grad is not None and torch.isfinite(lam.grad).all() and lam.grad.abs() > 0


class _IsotropicPosterior:
    def __init__(self, mean, variance):
        self.mean = mean
        self.variance = variance


class _FakeModel(torch.nn.Module):
    """Fixed posterior, so the acquisition is testable without fitting a GP."""

    def __init__(self, mean, var):
        super().__init__()
        self._mean = mean
        self._var = var

    def posterior(self, X):
        shape = X.shape[:-1] + (self._mean.numel(),)
        return _IsotropicPosterior(
            self._mean.expand(shape).clone(), self._var.expand(shape).clone()
        )


def test_acquisition_matches_monte_carlo():
    torch.manual_seed(0)
    mean = torch.tensor([0.4, -0.2, 0.1], dtype=torch.double)
    var = torch.tensor([0.09, 0.09, 0.09], dtype=torch.double)  # isotropic
    target = torch.zeros(3, dtype=torch.double)
    radius = 0.6

    acqf = ToleranceBallAcquisition(
        model=_FakeModel(mean, var), target=target, radius=radius
    )
    got = acqf(torch.zeros(1, 1, 2, dtype=torch.double)).item()

    draws = torch.distributions.Normal(mean, var.sqrt()).sample((400_000,))
    mc = ((draws - target).pow(2).sum(-1) <= radius ** 2).double().mean().item()
    assert got == pytest.approx(mc, abs=3e-3)


def test_acquisition_is_highest_at_the_target():
    mean_on = torch.zeros(3, dtype=torch.double)
    mean_off = torch.tensor([1.5, 0.0, 0.0], dtype=torch.double)
    var = torch.full((3,), 0.04, dtype=torch.double)
    X = torch.zeros(1, 1, 2, dtype=torch.double)

    on = ToleranceBallAcquisition(_FakeModel(mean_on, var), torch.zeros(3, dtype=torch.double), 0.5)(X)
    off = ToleranceBallAcquisition(_FakeModel(mean_off, var), torch.zeros(3, dtype=torch.double), 0.5)(X)
    assert on.item() > off.item()


def _info(windows, ref_center, ref_scale, ball_center=None):
    """target_info as the adapter builds it, in the frozen scoring space."""
    out = {}
    for i, (name, (lb, ub)) in enumerate(windows.items()):
        out[name] = {
            "obj": "range", "lb": lb, "ub": ub, "weight": 1.0,
            "ref_center": ref_center[i], "ref_scale": ref_scale[i],
            "ball_center": (0.5 * (lb + ub)) if ball_center is None else ball_center[i],
        }
    return out


def test_ball_uses_the_frozen_scale_not_raw_units():
    """Radius must come from the tightest *scaled* window, not the raw one."""
    windows = {"y1": (60.0, 64.0), "y2": (280.0, 300.0)}  # raw widths 4 and 20
    # Scales chosen so the raw-narrower property is the scaled-wider one.
    info = _info(windows, ref_center=[62.0, 290.0], ref_scale=[1.0, 20.0])

    center, radius = box_to_ball(info, list(windows), "inscribed")
    assert center.tolist() == pytest.approx([0.0, 0.0])
    # scaled half-widths: y1 -> 2.0, y2 -> 0.5  => min is y2's
    assert radius == pytest.approx(0.5)


def test_inscribed_ball_never_exceeds_the_box():
    """Every point in the ball must satisfy the box, or the mapping is unfair."""
    windows = {"y1": (61.0, 67.0), "y2": (288.0, 315.0), "y3": (5.5, 6.0)}
    keys = list(windows)
    ref_center, ref_scale = [64.0, 300.0, 5.75], [2.0, 10.0, 0.25]
    info = _info(windows, ref_center, ref_scale)
    center, radius = box_to_ball(info, keys, "inscribed")

    lo = np.array([(windows[k][0] - ref_center[i]) / ref_scale[i] for i, k in enumerate(keys)])
    hi = np.array([(windows[k][1] - ref_center[i]) / ref_scale[i] for i, k in enumerate(keys)])
    c = center.numpy()
    # The ball reaches no further than the nearest face along any axis.
    assert np.all(c - radius >= lo - 1e-12)
    assert np.all(c + radius <= hi + 1e-12)


def test_inscribed_ball_centers_on_the_valid_centroid():
    """A box-centered ball misses the mass when the window is quantile-built."""
    windows = {"y1": (0.0, 10.0), "y2": (0.0, 10.0)}
    keys = list(windows)
    # Valid mass pressed against the lower face of y2, as on L4-1/L5-4.
    info = _info(windows, ref_center=[0.0, 0.0], ref_scale=[1.0, 1.0],
                 ball_center=[5.0, 1.5])
    center, radius = box_to_ball(info, keys, "inscribed")
    assert center.tolist() == pytest.approx([5.0, 1.5])
    # Radius is limited by the near face of y2, not by the box half-width.
    assert radius == pytest.approx(1.5)


def test_equal_volume_ball_matches_box_volume():
    """The point of this mapping is equal measure, so check the measure."""
    from math import gamma, pi

    windows = {"y1": (61.0, 67.0), "y2": (288.0, 315.0), "y3": (5.5, 6.0)}
    keys = list(windows)
    ref_center, ref_scale = [64.0, 300.0, 5.75], [2.0, 10.0, 0.25]
    info = _info(windows, ref_center, ref_scale)

    _, radius = box_to_ball(info, keys, "equal_volume")
    widths = [(windows[k][1] - windows[k][0]) / ref_scale[i] for i, k in enumerate(keys)]
    ball_volume = pi ** 1.5 * radius ** 3 / gamma(2.5)
    assert ball_volume == pytest.approx(float(np.prod(widths)), rel=1e-9)


def test_equal_volume_ball_is_larger_than_inscribed():
    """Otherwise it would not fix the coverage problem it exists to fix."""
    windows = {"y1": (61.0, 67.0), "y2": (288.0, 315.0), "y3": (5.5, 6.0)}
    keys = list(windows)
    info = _info(windows, ref_center=[64.0, 300.0, 5.75], ref_scale=[2.0, 10.0, 0.25])
    _, r_in = box_to_ball(info, keys, "inscribed")
    _, r_eq = box_to_ball(info, keys, "equal_volume")
    assert r_eq > r_in


def test_unknown_mapping_is_rejected():
    info = _info({"y1": (0.0, 1.0)}, ref_center=[0.5], ref_scale=[1.0])
    with pytest.raises(ValueError, match="unknown ball mapping"):
        box_to_ball(info, ["y1"], "circumscribed")


def test_to_frozen_rescales_the_posterior():
    """model space -> frozen space must transform mean and variance together."""
    mean = torch.tensor([1.0], dtype=torch.double)
    var = torch.tensor([0.25], dtype=torch.double)
    gain = torch.tensor([2.0], dtype=torch.double)
    offset = torch.tensor([-3.0], dtype=torch.double)
    X = torch.zeros(1, 1, 2, dtype=torch.double)

    # frozen mean = 2*1 - 3 = -1 ; frozen sd = 2*0.5 = 1
    acqf = ToleranceBallAcquisition(
        _FakeModel(mean, var), target=torch.tensor([-1.0], dtype=torch.double),
        radius=1.0, to_frozen=(gain, offset),
    )
    got = acqf(X).item()
    # P(|N(0,1)| <= 1) for a 1-D ball centered on the mean
    assert got == pytest.approx(float(ncx2.cdf(1.0, 1, 0.0)), abs=1e-9)
