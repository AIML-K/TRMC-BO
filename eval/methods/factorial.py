"""The C1-off cells of the C1/C2/C3 factorial ablation.

The paper's ablation is a cumulative ladder -- C1, C1+C2, C1+C2+C3 -- which
cannot separate "C2 and C3 amplify the range objective" from "C2 and C3 do the
work on their own". This module supplies the four missing cells so the ablation
becomes a full 2x2x2 factorial:

    M13  C1 + C3         (lives in `proposed.VARIANTS`; C1 is on, so no trick)
    M14  C2 only
    M15  C3 only
    M16  C2 + C3

Two traps stand between "flip a flag" and a correct C1-off arm, and both fail
*silently* -- the run completes and the numbers look plausible.

Trap 1: C2/C3 need a live `range_spec`.
    `_standardized_range_spec` (src/bayesian_optimization.py) returns None unless
    some output carries `obj == "range"`, and `optimize_acqf_on_filtered_points`
    gates both the nonlinear constraints and the restart filter on it. Setting
    `objective_mode="native_extremum"` to switch C1 off therefore switches C2 and
    C3 off with it, and the arm silently becomes M1. (`ProposedSpec.objective`
    does not help either: the `_range_objective_mode` key it writes is read by
    nobody.) So these arms keep `objective_mode="range"` -- which is what builds
    the range_spec -- and replace only the acquisition, through the
    `_acqf_factory` hook that `M2_target_distance` already uses.

Trap 2: the sign convention differs between the two paths.
    `run_mobo` calls `negate_y` *before* the acquisition is built, so under
    `objective_mode="native_extremum"` every 'min' output is negated and the GP
    is fit on the negated values. Under `objective_mode="range"` nothing is
    negated. Handing rewritten max/min `target_info` to `get_acqf` from here
    would therefore maximize what should be minimized -- and every task in all
    three suites has at least one 'less' output, so this is not a corner case.
    `extremum_acqf_factory` applies the direction in the *objective weights*
    instead of in the data.

    The two are equivalent. Standardization happens before negation, a scaled RBF
    kernel is invariant under a global sign flip, and the constant mean simply
    fits the flipped value, so the GP fitted on -y has the posterior of the GP
    fitted on y with the mean negated. Maximizing `+1 * samples` of the former is
    the same problem as maximizing `-1 * samples` of the latter.

`eval/scripts/verify_variants.py` is the check that both traps stayed shut: M14
and M16 must report a non-zero nonlinear-constraint count, and all four must
report an acquisition that is not the CDF-range objective.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.objective import (
    WeightedMCMultiOutputObjective,
)
from botorch.acquisition.objective import LinearMCObjective
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions import (
    FastNondominatedPartitioning,
)

from eval.methods.proposed import ProposedMethod, ProposedSpec

#: `get_acqf` uses this sample count for every acquisition; matching it keeps the
#: comparison an acquisition comparison rather than a sampler comparison.
_SAMPLE_SHAPE = torch.Size([1024])


def make_extremum_acqf_factory(signs: list[float]):
    """Build a `get_acqf`-signature factory for the native max/min acquisition.

    `signs[i]` is +1 when output `i` is to be maximized and -1 when minimized,
    read from the task's native targets. The returned factory reproduces
    `get_acqf`'s Case 1 with no range targets -- exactly what
    `M1_qehvi_extremum` optimizes -- but carries the direction in the objective
    weights, for the reason in this module's docstring.
    """

    def extremum_acqf_factory(
        gp_model, train_x, train_y, y_key, weight, target_info, y_mean, y_std
    ):
        if len(signs) != len(y_key):
            raise ValueError(
                f"sign vector has {len(signs)} entries for {len(y_key)} outputs"
            )

        sampler = SobolQMCNormalSampler(sample_shape=_SAMPLE_SHAPE)
        w = weight.to(device=train_y.device, dtype=train_y.dtype)
        signed = w * torch.as_tensor(
            signs, device=train_y.device, dtype=train_y.dtype
        )

        if len(y_key) >= 2:
            with torch.no_grad():
                pred = gp_model.posterior(train_x).mean
            train_obj_Y = pred * signed

            # `get_acqf` builds the reference point from the posterior mean at the
            # training points, not from train_y, and offsets it below the worst
            # one. Mirrored here so the two paths agree cell for cell.
            ref_point = train_obj_Y.nan_to_num(nan=torch.inf).min(dim=0).values - 1e-3
            partitioning = FastNondominatedPartitioning(
                ref_point=ref_point, Y=train_obj_Y
            )
            objective = WeightedMCMultiOutputObjective(
                weights=signed, outcomes=list(range(len(y_key)))
            )
            return qExpectedHypervolumeImprovement(
                model=gp_model,
                ref_point=ref_point,
                objective=objective,
                partitioning=partitioning,
                sampler=sampler,
                constraints=None,
                eta=1e-3,
            )

        # Single output: qEI on the signed column. No range target is passed to
        # the acquisition, so there is no feasible-only incumbent to compute --
        # the window reaches the optimizer through C2/C3 instead, which is the
        # whole point of these arms.
        objective = LinearMCObjective(weights=signed)
        best_f = (train_y[..., 0] * signed[0]).nan_to_num(nan=-torch.inf).max()
        return qExpectedImprovement(
            model=gp_model,
            best_f=best_f,
            objective=objective,
            sampler=sampler,
            constraints=None,
            eta=1e-3,
        )

    return extremum_acqf_factory


@dataclass(frozen=True)
class _Arm:
    name: str
    mean_range_constraints: bool
    mean_filtered_starts: bool


ARMS: dict[str, _Arm] = {
    "M14_constraints_only": _Arm("M14_constraints_only", True, False),
    "M15_starts_only": _Arm("M15_starts_only", False, True),
    "M16_constraints_starts": _Arm("M16_constraints_starts", True, True),
}


class FactorialMethod(ProposedMethod):
    """C2 and/or C3 on the extremum acquisition, with C1 switched off.

    `objective_mode` stays "range" so `range_spec` exists; the acquisition is
    replaced wholesale. See the module docstring for why both halves are needed.
    """

    objective_mode = "range"

    def __init__(self, arm: _Arm, num_restarts: int = 128):
        super().__init__(
            ProposedSpec(
                name=arm.name,
                # Reported as "--" in `ProposedSpec.components`, which is what
                # the ablation tables key on. The field does not switch C1; the
                # acquisition override below is what does.
                objective="qehvi_extremum",
                mean_range_constraints=arm.mean_range_constraints,
                mean_filtered_starts=arm.mean_filtered_starts,
                num_restarts=num_restarts,
            )
        )

    def build_optimize_config(self, variables, q: int) -> dict:
        cfg = super().build_optimize_config(variables, q)
        cfg["_acqf_factory"] = make_extremum_acqf_factory(_native_signs(variables))
        return cfg


def _native_signs(variables) -> list[float]:
    """+1 per output to maximize, -1 per output to minimize, in `y_key` order.

    `BenchmarkVariables._build_target_info` reads the same `native_targets` for
    its "native_extremum" mode, so these are the directions M1 optimizes.
    """
    kinds = {t.name: t.kind for t in variables.task.native_targets}
    signs = []
    for name in variables.y_key:
        kind = kinds.get(name)
        if kind is None:
            raise ValueError(f"no native target for output {name!r}")
        signs.append(1.0 if kind == "greater" else -1.0)
    return signs


def make(name: str, num_restarts: int | None = None) -> FactorialMethod:
    return FactorialMethod(
        ARMS[name], num_restarts=num_restarts if num_restarts is not None else 128
    )
