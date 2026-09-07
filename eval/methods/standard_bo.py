"""Extremum and point-target baselines (paper §6.2, §6.3).

Both run through the production `run_mobo`, so they share the surrogate, the
restart machinery and the acquisition optimizer with the proposed method. Only
the acquisition differs, which is what makes the comparison an acquisition
comparison.

M1 needs no custom acquisition at all: telling the pipeline the outcomes are
plain max/min objectives (`objective_mode="native_extremum"`) makes `get_acqf`
build an ordinary qEHVI. M2 supplies its own builder through the
`_acqf_factory` hook.

The two answer different questions and must not be collapsed into one row:

    M1  extremum-seeking BO, window ignored entirely
    M2  point-target BO, window approximated by its midpoint

Reporting only one of them would let a reader assume the other behaves the same.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.acquisition.objective import MCAcquisitionObjective
from botorch.sampling.normal import SobolQMCNormalSampler
from torch import Tensor

from eval.methods.proposed import ProposedMethod, ProposedSpec


class NegativeTargetDistanceObjective(MCAcquisitionObjective):
    """``-sum_k ((Y_k - t_k) / s_k)^2`` on posterior samples.

    `t_k` is the window midpoint and `s_k` its half-width, both in the
    standardized output space the model works in. Dividing by the half-width is
    what keeps three properties on wildly different scales (y1 ~ 60, y2 ~ 300,
    y3 ~ 6) contributing comparably -- without it y2 would dominate the distance
    and the other two would be effectively ignored.
    """

    def __init__(self, target: Tensor, scale: Tensor):
        super().__init__()
        self.register_buffer("target", target)
        self.register_buffer("scale", scale)

    def forward(self, samples: Tensor, X: Tensor | None = None) -> Tensor:
        z = (samples - self.target.to(samples)) / self.scale.to(samples)
        return -(z ** 2).sum(dim=-1)


def target_distance_acqf_factory(
    gp_model, train_x, train_y, y_key, weight, target_info, y_mean, y_std
):
    """Build qEI on the negative distance to the window midpoint.

    Signature mirrors `get_acqf` so it can be dropped in through
    `optimize_config["_acqf_factory"]`.
    """
    mid, half = [], []
    for i, key in enumerate(y_key):
        info = target_info[key]
        lb, ub = float(info["lb"]), float(info["ub"])
        # The model sees standardized outputs, so the window must be standardized
        # the same way before it can be compared against posterior samples.
        lb_s = (lb - float(y_mean[i])) / float(y_std[i])
        ub_s = (ub - float(y_mean[i])) / float(y_std[i])
        mid.append(0.5 * (lb_s + ub_s))
        half.append(max(0.5 * (ub_s - lb_s), 1e-9))

    target = torch.as_tensor(mid, dtype=train_y.dtype, device=train_y.device)
    scale = torch.as_tensor(half, dtype=train_y.dtype, device=train_y.device)
    objective = NegativeTargetDistanceObjective(target=target, scale=scale)

    with torch.no_grad():
        train_scores = objective(train_y.unsqueeze(0)).squeeze(0)
    best_f = train_scores.nan_to_num(nan=-torch.inf).max()

    return qExpectedImprovement(
        model=gp_model,
        best_f=best_f,
        objective=objective,
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([1024])),
        eta=1e-3,
    )


@dataclass(frozen=True)
class _Baseline:
    name: str
    objective_mode: str
    acqf_factory: object | None


BASELINES = {
    "M1_qehvi_extremum": _Baseline("M1_qehvi_extremum", "native_extremum", None),
    "M2_target_distance": _Baseline("M2_target_distance", "range", target_distance_acqf_factory),
}


class StandardBOMethod(ProposedMethod):
    """Reuses the proposed method's plumbing with C1/C2/C3 all switched off."""

    def __init__(self, baseline: _Baseline, num_restarts: int = 128):
        super().__init__(
            ProposedSpec(
                name=baseline.name,
                objective="qehvi_extremum" if baseline.acqf_factory is None else "target_distance",
                mean_range_constraints=False,
                mean_filtered_starts=False,
                num_restarts=num_restarts,
            )
        )
        self.objective_mode = baseline.objective_mode
        self._acqf_factory = baseline.acqf_factory

    def build_optimize_config(self, variables, q: int) -> dict:
        cfg = super().build_optimize_config(variables, q)
        if self._acqf_factory is not None:
            cfg["_acqf_factory"] = self._acqf_factory
        return cfg


def make(name: str, num_restarts: int = 128) -> StandardBOMethod:
    return StandardBOMethod(BASELINES[name], num_restarts=num_restarts)
