"""The §4.2 acquisition as written: the *product* of range probabilities.

The paper specifies

    A_range(x) = prod_k P(L_k <= f_k(x) <= U_k)

The shipped multi-output path does something else. `CDFRangeMultiOutputObjective`
returns a *vector* of per-output probabilities and `get_acqf` hands that vector to
qEHVI, so what gets maximized is hypervolume over the probability vector rather
than the product. Those are not variations on a theme. Our task is a *joint*
condition -- all three properties inside their windows at once -- and hypervolume
rewards spread along the Pareto front: a candidate that lifts P1 to 0.9 while P2
sits at 0.01 improves hypervolume and is worthless, since jointly it passes 0.9%
of the time. A product cannot be fooled that way; one small factor sinks it.

This module implements the specified form so the two can be measured against each
other. It lives in `eval/` rather than `src/` deliberately: the production default
does not change until the comparison says it should, and the `_acqf_factory` hook
is already the supported way to swap an acquisition without touching the pipeline.

Four variants, because the shipped path differs from the spec in more than one
way at once and a single replacement could not tell the causes apart:

    M3p_range_prob_product      product, direct, C2/C3 off
    M3q_range_prob_product_ei   product, in qEI, C2/C3 off
    M5p_full_product            product, direct, C2/C3 on -- §4.2 as written
    M5q_full_product_ei         product, in qEI,  C2/C3 on

The C1-only pair is where the aggregation question is decidable. With C2 and C3
on, a smoke test on L3-1 found three *different* acquisitions proposing the same
point to within 1e-4 for three straight iterations: the two-sided mean
constraints left so little admissible volume (2 range-feasible starts out of 2000
draws) that the constraint set chose the candidate and the acquisition only broke
ties. Comparing aggregations there would mostly measure the constraints.

Note on the qEI variants: the product is a deterministic function of x, so
E[(A - best)+] collapses to (A - best)+ -- every candidate below the incumbent
ties at exactly zero and the optimizer sees a plateau. That is not hypothetical;
`scripts/acquisition_auc.py` scores this variant at AUC 0.500 on L3-1/medium,
i.e. no better than shuffling the pool, because the whole pool ties. The shipped
single-output path (`CDFRangeObjective` + qEI) has the same structure. Measuring
it is the point; it is not an oversight being reproduced by accident.

The direct variants avoid a *second* plateau that has nothing to do with qEI: a
float64 product of three tail probabilities underflows to exactly zero, which
would flatten the acquisition wherever the surrogate is pessimistic -- precisely
where the search begins. `_log_normal_interval` keeps those candidates ordered
and differentiable.
"""

from __future__ import annotations

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.acquisition.objective import MCAcquisitionObjective
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor

from eval.methods.proposed import ProposedMethod, ProposedSpec


def _standardized_windows(y_key, target_info, y_mean, y_std):
    """Window bounds in the standardized space the surrogate models.

    Mirrors `_standardized_range_spec` in the production pipeline; outputs whose
    target is not a range are skipped, which on this benchmark means none.
    """
    idx, lo, hi = [], [], []
    for i, key in enumerate(y_key):
        info = target_info[key]
        if info.get("obj") != "range":
            continue
        idx.append(i)
        lo.append((float(info["lb"]) - float(y_mean[i])) / float(y_std[i]))
        hi.append((float(info["ub"]) - float(y_mean[i])) / float(y_std[i]))
    if not idx:
        raise ValueError("no range target found; this factory needs objective_mode='range'")
    return idx, lo, hi


def _log_normal_interval(a: Tensor, b: Tensor) -> Tensor:
    """`log(Phi(b) - Phi(a))` for a standard normal, stable into the far tail.

    A plain `Phi(b) - Phi(a)` underflows to exactly 0 once the window sits ~38
    standard deviations from the posterior mean, and clamping the result at a
    floor makes every such candidate tie. That is not a cosmetic problem: those
    ties are a zero-gradient plateau, and L-BFGS started inside one cannot climb
    out, so the acquisition would look far worse than the form it implements
    actually is. A test pins the ordering that this preserves.

    Whichever side of the window the mean lies on, the computation is moved into
    the tail that is *small*, where `log_ndtr` is accurate, and the subtraction
    then happens between well-separated logs instead of between two numbers both
    rounding to 1.
    """
    flip = a > 0
    lo = torch.where(flip, -b, a)
    hi = torch.where(flip, -a, b)

    log_hi = torch.special.log_ndtr(hi)
    log_lo = torch.special.log_ndtr(lo)
    # clamp keeps exp() strictly below 1 so log1p stays finite if the two logs
    # collapse onto the same float.
    tail = log_hi + torch.log1p(-torch.exp((log_lo - log_hi).clamp(max=-1e-12)))

    # The window straddles the mean: the probability is O(1) and a direct
    # difference is both accurate and cheap.
    straddle = (torch.special.ndtr(hi) - torch.special.ndtr(lo)).clamp_min(1e-300).log()

    return torch.where(hi <= 0, tail, straddle)


def _log_range_probability(posterior, index: Tensor, lower: Tensor, upper: Tensor) -> Tensor:
    """`sum_k log P(L_k <= f_k <= U_k)` from a posterior, shape (...,).

    Summed in log space rather than multiplied: with three windows a few percent
    wide the product reaches 1e-8 routinely and can go far lower, and a float64
    product bottoms out at zero while the sum of logs keeps ranking candidates
    the optimizer still has to choose between.
    """
    mean = posterior.mean.index_select(dim=-1, index=index)
    std = posterior.variance.index_select(dim=-1, index=index).clamp_min(1e-12).sqrt()
    return _log_normal_interval((lower - mean) / std, (upper - mean) / std).sum(dim=-1)


class JointRangeProbability(AcquisitionFunction):
    """`prod_k P(L_k <= f_k(x) <= U_k)`, returned in log space.

    Log rather than raw probability for the reason above, and monotone in the
    product either way, so the argmax the optimizer reports is unchanged.

    Subclasses `AcquisitionFunction` rather than `AnalyticAcquisitionFunction`:
    the latter refuses a multi-output model without a `posterior_transform`, and
    there is no scalarization to supply -- the reduction to a scalar happens
    here, across the per-output windows, and that is the whole content of §4.2.
    """

    def __init__(self, model, index, lower, upper):
        super().__init__(model=model)
        self.register_buffer("index", torch.as_tensor(index, dtype=torch.long))
        self.register_buffer("lower", torch.as_tensor(lower, dtype=torch.double))
        self.register_buffer("upper", torch.as_tensor(upper, dtype=torch.double))

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        posterior = self.model.posterior(X)
        lower = self.lower.to(posterior.mean)
        upper = self.upper.to(posterior.mean)
        log_p = _log_range_probability(posterior, self.index.to(X.device), lower, upper)
        return log_p.squeeze(-1)


class JointRangeProbabilityObjective(MCAcquisitionObjective):
    """The same product, shaped as an MC objective so qEI can wrap it."""

    def __init__(self, model, index, lower, upper):
        super().__init__()
        self.model = model
        self.register_buffer("index", torch.as_tensor(index, dtype=torch.long))
        self.register_buffer("lower", torch.as_tensor(lower, dtype=torch.double))
        self.register_buffer("upper", torch.as_tensor(upper, dtype=torch.double))

    def forward(self, samples: Tensor, X: Tensor | None = None) -> Tensor:
        if X is None:
            raise RuntimeError("X is required for a CDF-based range objective.")
        posterior = self.model.posterior(X)
        log_p = _log_range_probability(
            posterior,
            self.index.to(X.device),
            self.lower.to(posterior.mean),
            self.upper.to(posterior.mean),
        )
        # Exponentiate here: qEI subtracts an incumbent, and differences of logs
        # are ratios, which is not the quantity the paper's argmax is over.
        return log_p.exp().expand(samples.shape[:-1])


def product_acqf_factory(
    gp_model, train_x, train_y, y_key, weight, target_info, y_mean, y_std
):
    """§4.2 as written: maximize the product directly."""
    idx, lo, hi = _standardized_windows(y_key, target_info, y_mean, y_std)
    return JointRangeProbability(model=gp_model, index=idx, lower=lo, upper=hi)


def product_ei_acqf_factory(
    gp_model, train_x, train_y, y_key, weight, target_info, y_mean, y_std
):
    """The product, wrapped in qEI against the best product seen so far."""
    idx, lo, hi = _standardized_windows(y_key, target_info, y_mean, y_std)
    objective = JointRangeProbabilityObjective(
        model=gp_model, index=idx, lower=lo, upper=hi
    )
    with torch.no_grad():
        posterior = gp_model.posterior(train_x)
        index = torch.as_tensor(idx, dtype=torch.long, device=train_x.device)
        best_f = _log_range_probability(
            posterior,
            index,
            torch.as_tensor(lo, dtype=train_x.dtype, device=train_x.device),
            torch.as_tensor(hi, dtype=train_x.dtype, device=train_x.device),
        ).exp().max()

    return qExpectedImprovement(
        model=gp_model,
        best_f=best_f,
        objective=objective,
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([1024])),
        eta=1e-3,
    )


#: (acquisition factory, C2, C3) per variant.
#:
#: The C1-only rows are not optional extras -- they are where the question is
#: actually decidable. A smoke test on L3-1/medium showed all three C2+C3
#: variants proposing the *same* point to within 1e-4 for the first three
#: iterations despite three different acquisitions, because the two-sided mean
#: constraints leave so little admissible volume (2 range-feasible starts out of
#: 2000 draws) that the constraint set picks the candidate and the acquisition
#: only breaks ties. With C2 and C3 off, the acquisition is the only thing
#: choosing, so a difference there is attributable to the aggregation.
#:
#: Both settings are still run: the C1-only pair answers "is the product a better
#: ranking rule", the full pair answers "does it change what the shipped
#: configuration would do", and those are different questions.
VARIANTS: dict[str, tuple] = {
    "M3p_range_prob_product": (product_acqf_factory, False, False),
    "M3q_range_prob_product_ei": (product_ei_acqf_factory, False, False),
    "M5p_full_product": (product_acqf_factory, True, True),
    "M5q_full_product_ei": (product_ei_acqf_factory, True, True),
}


class ProductMethod(ProposedMethod):
    def __init__(self, name: str, num_restarts: int = 128):
        factory, use_c2, use_c3 = VARIANTS[name]
        super().__init__(
            ProposedSpec(
                name=name,
                objective="range_prob",
                mean_range_constraints=use_c2,
                mean_filtered_starts=use_c3,
                num_restarts=num_restarts,
            )
        )
        self._acqf_factory = factory

    def build_optimize_config(self, variables, q: int) -> dict:
        cfg = super().build_optimize_config(variables, q)
        cfg["_acqf_factory"] = self._acqf_factory
        return cfg


def make(name: str, num_restarts: int = 128) -> ProductMethod:
    return ProductMethod(name, num_restarts=num_restarts)
