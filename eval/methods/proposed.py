"""The proposed method, driven through the production `run_mobo`.

Nothing about the acquisition is reimplemented here. This module builds the
`optimize_config` that selects a variant and hands it to
`src/bayesian_optimization.run_mobo`, so the range-probability objective, the
two-sided posterior-mean constraints and the mean-filtered restarts are exactly
the code that ships.

Component names follow the paper:

    C1  range-probability objective          P(L <= f(x) <= U)
    C2  two-sided nonlinear constraints      g_L = mu - L >= 0, g_U = U - mu >= 0
    C3  posterior-mean-filtered restarts     keep starts with L <= mu(x) <= U
    C4a feasibility-weighted ranking         score = P(feasible|x) * acq(x)
    C4b feasibility-filtered restarts        keep the most-feasible restarts

C1-C3 are all statements about the outcome space -- is `f(x)` inside its window.
C4 is a statement about the input space -- can `x` be evaluated at all. Nothing in
C1-C3 can learn the latter, because `initialize_model` drops all-NaN rows from
every GP, so an infeasible observation leaves no trace. C4 is the one consumer of
those rows; it multiplies onto the ladder rather than replacing a rung of it. See
`src/utils/feasibility.py`.

C4 is off by default, so `M3`/`M4`/`M5` produce exactly the candidates they did
before it existed.

`run_mobo` mutates the dict it is given, so every call gets a fresh copy.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from bayesian_optimization import run_mobo  # noqa: E402  (src/ is on sys.path)
from eval.harness.adapter import BenchmarkVariables

Objective = Literal["range_prob", "qehvi_extremum", "target_distance"]


@dataclass(frozen=True)
class ProposedSpec:
    """One cell of the ablation matrix (paper §7.1)."""

    name: str
    objective: Objective = "range_prob"
    mean_range_constraints: bool = True
    mean_filtered_starts: bool = True
    #: C4a -- rank candidates by `P(feasible|x) * acq(x)` instead of `acq(x)`.
    feasibility_weight: bool = False
    #: C4b -- additionally keep only the most-feasible restarts, by rank.
    feasibility_filter: bool = False
    #: Fraction of each drawn restart batch C4b keeps. Rank-based, not an
    #: absolute probability threshold: a threshold was measured to be unusable
    #: because the classifier's probabilities are narrow, problem-dependent in
    #: location, and often constant. See
    #: `src/utils/feasibility.feasibility_start_predicate`.
    feasibility_keep_fraction: float = 0.5
    num_restarts: int = 128

    @property
    def components(self) -> str:
        c1 = "C1" if self.objective == "range_prob" else "--"
        parts = [
            c1,
            "C2" if self.mean_range_constraints else "--",
            "C3" if self.mean_filtered_starts else "--",
        ]
        if self.feasibility_weight:
            parts.append("C4a")
        if self.feasibility_filter:
            parts.append("C4b")
        return "/".join(parts)


#: The variants named in the plan. M0 (random) and M6 (TB) live in their own
#: modules; M1/M2 need the objective switch that lands with milestone M5.
VARIANTS: dict[str, ProposedSpec] = {
    "M3_range_prob": ProposedSpec(
        "M3_range_prob", "range_prob", mean_range_constraints=False, mean_filtered_starts=False
    ),
    "M4_range_prob_constraints": ProposedSpec(
        "M4_range_prob_constraints", "range_prob", mean_range_constraints=True, mean_filtered_starts=False
    ),
    "M5_full": ProposedSpec(
        "M5_full", "range_prob", mean_range_constraints=True, mean_filtered_starts=True
    ),
    # The cell the cumulative ladder skips. C3 filters restarts by the posterior
    # mean but leaves the optimizer free to walk back out of the window, which is
    # the only way to ask whether C3 contributes anything C2 does not already
    # supply. The C1-off half of the factorial lives in `eval/methods/factorial.py`,
    # where switching C1 off without also switching C2/C3 off takes real care.
    "M13_range_prob_starts": ProposedSpec(
        "M13_range_prob_starts", "range_prob",
        mean_range_constraints=False, mean_filtered_starts=True,
    ),
    # C4 ablation: the full method plus the feasibility signal, one component at
    # a time. Separating M10 from M11 is what tells apart "knowing which regions
    # are unmakeable changes what we rank first" from "it changes where we even
    # start looking" -- they can easily disagree, since C4b competes with C3 for
    # the same restart pool while C4a costs nothing.
    "M10_full_feasw": ProposedSpec(
        "M10_full_feasw", "range_prob", mean_range_constraints=True, mean_filtered_starts=True,
        feasibility_weight=True,
    ),
    "M11_full_feasf": ProposedSpec(
        "M11_full_feasf", "range_prob", mean_range_constraints=True, mean_filtered_starts=True,
        feasibility_filter=True,
    ),
    "M12_full_feas": ProposedSpec(
        "M12_full_feas", "range_prob", mean_range_constraints=True, mean_filtered_starts=True,
        feasibility_weight=True, feasibility_filter=True,
    ),
}


class ProposedMethod:
    #: How the outcomes are described to the pipeline; see
    #: `BenchmarkVariables._build_target_info`. Range targets are what select the
    #: range-probability objective, so the proposed variants always use "range".
    objective_mode = "range"

    def __init__(self, spec: ProposedSpec):
        self.spec = spec
        self.name = spec.name

    def build_optimize_config(self, variables: BenchmarkVariables, q: int) -> dict:
        cfg = {
            "q": q,
            "num_restarts": self.spec.num_restarts,
            # BoTorch requires batch_limit == 1 once nonlinear constraints are
            # attached; the production pipeline sets it unconditionally.
            "options": {"batch_limit": 1},
            "return_best_only": False,
            "merge_results": False,
            "acqf_optimizer": "discrete",  # dispatch is by variable type; ours is 'conti'
            "fixed_features_list": [],
        }
        cfg.update(variables.linear_constraints())
        cfg.update({
            "_use_mean_range_constraints": self.spec.mean_range_constraints,
            "_use_mean_filtered_starts": self.spec.mean_filtered_starts,
            "_range_objective_mode": self.spec.objective,
            # Underscore-prefixed, so `run_mobo` strips them before the dict is
            # splatted into BoTorch (which rejects unknown kwargs).
            "_use_feasibility_weight": self.spec.feasibility_weight,
            "_use_feasibility_filter": self.spec.feasibility_filter,
            "_feasibility_keep_fraction": self.spec.feasibility_keep_fraction,
        })
        return cfg

    def _candidates(self, variables, q: int, seed: int, recorder=None) -> np.ndarray:
        """All candidates `run_mobo` produced, raw units, best acquisition first."""
        cfg = self.build_optimize_config(variables, q)
        if recorder is not None:
            cfg["_recorder"] = recorder

        candidates_df, gp_model, fit_result = run_mobo(
            variables.train_x, variables.train_y, variables, cfg
        )
        if recorder is not None:
            recorder.note_fit(fit_result)

        return np.stack([np.asarray(c, dtype=float) for c in candidates_df["candidate"]])

    def propose(self, variables, q: int, seed: int, recorder=None) -> np.ndarray:
        top = self._candidates(variables, q, seed, recorder)[:q]
        return top.reshape(q, variables.task.dim)

    def propose_pool(self, variables, n: int, seed: int, recorder=None) -> np.ndarray:
        """Up to `n` distinct candidates, for batch-recommend protocols.

        One restart yields one candidate, so the pool is capped by
        `num_restarts`; MatFormBench's native protocol asks for 100 suggestions
        against the production default of 128 restarts.
        """
        pool = self._candidates(variables, q=1, seed=seed, recorder=recorder)
        _, keep = np.unique(np.round(pool, 10), axis=0, return_index=True)
        return pool[np.sort(keep)][:n]


def make(name: str, num_restarts: int | None = None) -> ProposedMethod:
    spec = VARIANTS[name]
    if num_restarts is not None and num_restarts != spec.num_restarts:
        spec = replace(spec, num_restarts=num_restarts)
    return ProposedMethod(spec)
