"""Range-Aware BO baseline: the Tolerance Ball acquisition (paper §6.4).

From Jiang, Wu, Schroeder & Webb, *Range-Aware Bayesian Optimization for
Discovering Diverse Designs within Target Property Windows* (arXiv 2606.11574).
TB scores the posterior probability that a candidate lands inside a tolerance
ball around a target vector:

    A_TB(x) = P( ||f(x) - y_tgt||^2 <= eps^2 )

Under their isotropic predictive-variance approximation the K outputs share one
variance s^2(x), so

    ||f(x) - y_tgt||^2 / s^2(x)  ~  noncentral chi-square(K, lambda(x))
    lambda(x) = ||mu(x) - y_tgt||^2 / s^2(x)

    A_TB(x) = F_{K, lambda(x)}( eps^2 / s^2(x) )

## Why the CDF is reimplemented here

`scipy.stats.ncx2.cdf` carries no gradient, and the acquisition is optimized by
L-BFGS through `optimize_acqf`. So the CDF is built from the Poisson-mixture
identity

    F_{K,lambda}(x) = sum_j  Pois(j; lambda/2) * F_{K+2j}(x)

whose central chi-square terms are regularized incomplete gammas
(`torch.special.gammainc`) and therefore differentiable in `x` and `lambda`.
`tests/test_range_aware_tb.py` checks the result against scipy and against Monte
Carlo.

## Box vs. ball

Our method targets an axis-aligned box `[L_k, U_k]`; TB targets an L2 ball. They
are different sets, so a fair comparison has to state the mapping -- and the
choice turned out to matter a great deal.

Both mappings center the ball on the **valid centroid**, not the box center.
Windows are built from quantiles of the native-satisfying tail, so the valid mass
presses against one face: on L4-1/L5-4 the median valid design sits at 0.13-0.18
of the y2 window while the box center is nearly empty. Box-centered balls covered
under 1% of the valid mass there. Centroid centering also matches how the source
paper picks targets -- k-medoids in output space, at data-dense locations.

    equal_volume  ball whose volume equals the box's. Covers 83-93% of the box's
                  valid mass on every task, at the cost of spilling modestly
                  outside it. This is the headline variant.
    inscribed     largest ball from the centroid that stays wholly inside the box.
                  Containment is exact, but a ball inscribed in an elongated box
                  is small: 6-35% coverage. Kept as a sensitivity check.

Scoring is done on the box for every method alike; `hit_rate_ball` reports the
ball criterion alongside so the handicap stays visible. Because both the
acquisition and that metric work in the *frozen* scoring space, TB is scored
against exactly the ball it optimizes.

The surrogate is deliberately *not* switched to the paper's Matern 5/2: every
method here shares this project's RBF `SingleTaskGP`. Varying the kernel and the
acquisition together would confound the two.
"""

from __future__ import annotations

import math

import torch
from botorch.acquisition import AcquisitionFunction
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor

from eval.methods.proposed import ProposedMethod, ProposedSpec

#: Hard cap on Poisson-mixture terms. The window is centered on lambda/2 and
#: widened by ~10 standard deviations, so this only binds for absurd lambda --
#: where the probability is numerically zero anyway.
MAX_TERMS = 4096


def noncentral_chi2_cdf(x: Tensor, df: int, noncentrality: Tensor) -> Tensor:
    """Differentiable ``P(chi^2_df(lambda) <= x)``.

    Terms are summed over a window around ``lambda/2``, where the Poisson mass
    concentrates. Starting the sum at j=0 with a fixed term count silently
    returns ~0 for large lambda, since every retained weight underflows.
    """
    x = x.clamp_min(0.0)
    half_lambda = (noncentrality * 0.5).clamp_min(0.0)

    peak = float(half_lambda.max().detach().cpu())
    spread = 10.0 * math.sqrt(peak + 1.0) + 20.0
    j_lo = max(0, int(peak - spread))
    j_hi = min(j_lo + MAX_TERMS, int(peak + spread) + 1)
    j = torch.arange(j_lo, j_hi + 1, device=x.device, dtype=x.dtype)

    # log Pois(j; half_lambda), evaluated in log space to survive large lambda.
    hl = half_lambda.unsqueeze(-1).clamp_min(1e-300)
    log_w = -hl + j * torch.log(hl) - torch.lgamma(j + 1.0)

    # Central chi-square CDF with df + 2j degrees of freedom.
    a = 0.5 * (df + 2.0 * j)
    cdf = torch.special.gammainc(a, (0.5 * x).unsqueeze(-1))

    return (torch.exp(log_w) * cdf).sum(dim=-1).clamp(0.0, 1.0)


class ToleranceBallAcquisition(AcquisitionFunction):
    """TB acquisition over a subset of model outputs.

    ``target``/``radius`` live in whatever space ``to_frozen`` maps the posterior
    into. When ``to_frozen`` is given as ``(a, b)`` the model's standardized
    output ``m`` becomes ``a * m + b``, which is how the run's per-iteration
    standardization is rewritten into the frozen scoring space.
    """

    def __init__(
        self,
        model,
        target: Tensor,
        radius: float,
        output_indices: list[int] | None = None,
        min_variance: float = 1e-12,
        to_frozen: tuple[Tensor, Tensor] | None = None,
    ):
        super().__init__(model=model)
        self.register_buffer("target", target)
        self.radius_sq = float(radius) ** 2
        self.output_indices = list(range(len(target))) if output_indices is None else list(output_indices)
        self.min_variance = min_variance
        if to_frozen is None:
            self.register_buffer("_gain", torch.ones_like(target))
            self.register_buffer("_offset", torch.zeros_like(target))
        else:
            self.register_buffer("_gain", to_frozen[0])
            self.register_buffer("_offset", to_frozen[1])

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        posterior = self.model.posterior(X)
        idx = torch.as_tensor(self.output_indices, dtype=torch.long, device=posterior.mean.device)
        mean = posterior.mean.index_select(-1, idx)
        var = posterior.variance.index_select(-1, idx)

        gain = self._gain.to(mean)
        mean = gain * mean + self._offset.to(mean)
        var = var * gain.pow(2)

        target = self.target.to(mean)
        # The paper's isotropic approximation: one variance shared by all outputs.
        # Averaging the per-output posterior variances is the natural reading and
        # is what makes the closed form applicable at all.
        s2 = var.mean(dim=-1).clamp_min(self.min_variance)

        offset_sq = ((mean - target) ** 2).sum(dim=-1)
        lam = offset_sq / s2
        x = self.radius_sq / s2

        prob = noncentral_chi2_cdf(x, df=len(self.output_indices), noncentrality=lam)
        return prob.squeeze(-1)


def box_to_ball(
    target_info: dict, y_key: list, mapping: str = "inscribed"
) -> tuple[Tensor, float]:
    """Ball target derived from the box, in the frozen scoring space.

    ``inscribed``     largest ball around the valid centroid that stays in the box
    ``equal_volume``  ball whose volume equals the box's, same center

    The center is the *valid centroid*, not the box center. Windows come from
    quantiles of the native-satisfying tail, so the valid mass presses against
    one face: a box-centered ball covered under 1% of it on L4-1/L5-4. This also
    matches how the source paper picks targets -- k-medoids in output space, at
    data-dense locations.

    Working in the frozen space (rather than the run's standardization) keeps the
    ball TB optimizes identical to the ball `hit_rate_ball` scores.
    """
    from math import gamma, pi

    lows, highs, centers = [], [], []
    for key in y_key:
        info = target_info[key]
        c, sc = float(info["ref_center"]), float(info["ref_scale"])
        lows.append((float(info["lb"]) - c) / sc)
        highs.append((float(info["ub"]) - c) / sc)
        centers.append((float(info["ball_center"]) - c) / sc)

    lo = torch.tensor(lows, dtype=torch.double)
    hi = torch.tensor(highs, dtype=torch.double)
    center = torch.tensor(centers, dtype=torch.double)

    if mapping == "inscribed":
        return center, float(torch.minimum(center - lo, hi - center).min())
    if mapping == "equal_volume":
        k = len(y_key)
        box_volume = float(torch.prod(hi - lo))
        return center, float((box_volume * gamma(k / 2 + 1) / pi ** (k / 2)) ** (1.0 / k))
    if mapping == "box":
        # Single-output only, and there the ball *is* the window: an L2 ball in
        # 1-D is an interval, so centering on the box midpoint with radius equal
        # to the half-width reproduces [L, U] exactly. See `MAPPINGS`.
        if len(y_key) != 1:
            raise ValueError(
                f"the 'box' mapping is exact only for a single output; this task "
                f"has {len(y_key)}"
            )
        return 0.5 * (lo + hi), float(0.5 * (hi - lo))
    raise ValueError(f"unknown ball mapping: {mapping!r}")


def make_tb_acqf_factory(mapping: str = "inscribed"):
    """Build a `get_acqf`-compatible factory for one box-to-ball mapping."""

    def factory(gp_model, train_x, train_y, y_key, weight, target_info, y_mean, y_std):
        keys = list(y_key)
        center, radius = box_to_ball(target_info, keys, mapping)
        kw = {"dtype": train_y.dtype, "device": train_y.device}
        # model space -> frozen space:  y_raw = m * y_std + y_mean,
        #                               y_frozen = (y_raw - ref_center) / ref_scale
        scales = torch.tensor([float(target_info[k]["ref_scale"]) for k in keys], **kw)
        refs = torch.tensor([float(target_info[k]["ref_center"]) for k in keys], **kw)
        y_std_t = torch.as_tensor(y_std, **kw).reshape(-1)[: len(keys)]
        y_mean_t = torch.as_tensor(y_mean, **kw).reshape(-1)[: len(keys)]
        gain = y_std_t / scales
        offset = (y_mean_t - refs) / scales
        return ToleranceBallAcquisition(
            model=gp_model,
            target=center.to(**kw),
            radius=radius,
            output_indices=list(range(len(keys))),
            to_frozen=(gain, offset),
        )

    return factory


tb_acqf_factory = make_tb_acqf_factory("inscribed")


#: The mappings, and what each one is for.
#:
#: `equal_volume` / `inscribed` are the two defensible ways to turn a box target
#: into a ball when there are several outputs; neither is canonical, so both run.
#:
#: `box` exists for a different reason and only makes sense at m = 1. There an L2
#: ball is an interval, so the ball can be made *exactly* the window -- and then
#: the TB acquisition is analytically the range probability itself:
#:
#:     ncx2(df=1, lambda=(mu-c)^2/s^2).cdf(r^2/s^2)
#:         == Phi((c+r-mu)/s) - Phi((c-r-mu)/s)
#:         == P(L <= f(x) <= U)
#:
#: That makes it two things at once. It is an exact cross-check of two independent
#: implementations of the same quantity -- our `_range_probability_on_X` and TB's
#: differentiable Poisson-mixture chi-square series. And it isolates the one thing
#: that separates TB from C1 on a single-output task: not the target, which is now
#: identical, but the *improvement wrapper*. The shipped single-output C1 path is
#: `qEI(CDFRangeObjective)`, and because that objective is a deterministic function
#: of x the expectation collapses, leaving exactly
#:
#:     A_C1(x) = max(0, P(L <= f(x) <= U) - best_f)
#:
#: i.e. the same quantity behind a hinge, tied at zero everywhere below the
#: incumbent. `box` TB is that quantity without the hinge. Comparing M3 against a
#: box-mapped TB therefore measures the plateau, at matched target, surrogate,
#: restart count and restart sampler.
MAPPINGS = ("equal_volume", "inscribed", "box")

#: Which mapping is the headline TB result.
#:
#: `equal_volume`, measured on the frozen calibration samples, covers 83-93% of
#: the box's valid mass on every task; the inscribed ball covers only 6-35%
#: because a ball inscribed in an elongated box is necessarily small. Leading with
#: the inscribed variant would understate TB for a reason that is about the
#: mapping, not the method. The inscribed variant is kept as the containment-
#: guaranteed sensitivity check (nothing it accepts violates the box).
PRIMARY_MAPPING = "equal_volume"


class RangeAwareTBMethod(ProposedMethod):
    """TB with C2 and C3 off -- the acquisition is the only difference."""

    objective_mode = "range"

    def __init__(self, mapping: str = PRIMARY_MAPPING, num_restarts: int = 128,
                 name: str | None = None):
        self.mapping = mapping
        if name is None:
            name = {
                PRIMARY_MAPPING: "M6_range_aware_tb",
                "inscribed": "M6b_range_aware_tb_inscribed",
                "box": "M6c_range_aware_tb_box",
            }[mapping]
        super().__init__(
            ProposedSpec(
                name=name,
                objective="range_prob",  # unused; the factory overrides the acqf
                mean_range_constraints=False,
                mean_filtered_starts=False,
                num_restarts=num_restarts,
            )
        )

    def build_optimize_config(self, variables, q: int) -> dict:
        cfg = super().build_optimize_config(variables, q)
        cfg["_acqf_factory"] = make_tb_acqf_factory(self.mapping)
        return cfg


def make(name: str = "M6_range_aware_tb", num_restarts: int = 128) -> RangeAwareTBMethod:
    if name.endswith("inscribed"):
        mapping = "inscribed"
    elif name.endswith("box"):
        mapping = "box"
    else:
        mapping = PRIMARY_MAPPING
    # Name is the registry key so result directories and result tables agree.
    return RangeAwareTBMethod(mapping=mapping, num_restarts=num_restarts, name=name)
