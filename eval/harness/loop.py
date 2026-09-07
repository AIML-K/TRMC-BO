"""Sequential BO campaign.

`run_mobo` proposes once from a fixed dataset; a benchmark run is that call in a
loop against an oracle. This module owns the loop, the initial design, and the
per-iteration trace, and it enforces two protocol rules that keep the comparison
honest:

* The method only ever sees `oracle.observe` (noisy, with MatFormBench's random
  batch effect). Hit rate and violations are scored on `oracle.truth`.
* Every method at a given (task, seed) starts from the *same* initial design and
  gets the *same* number of sequential evaluations. The initial design is drawn
  before any method acts, so its cost is a task-level constant reported
  alongside the run rather than deducted from the optimization budget --
  deducting it would shorten the loop on exactly the tasks where the comparison
  matters most, without making anything fairer.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from botorch.exceptions.errors import CandidateGenerationError

from eval.benchmarks.base import BenchmarkTask, Observation, Oracle, RangeSpec
from eval.harness.adapter import BenchmarkVariables
from eval.harness.recorder import RunRecorder


@dataclass
class CampaignConfig:
    n_init: int = 30
    budget: int = 50
    q: int = 1
    #: Absolute floor of labeled rows needed before a GP can be fit at all.
    #:
    #: Deliberately 2, not "enough to model well". MatFormBench ships exactly 30
    #: initial rows and lets infeasible ones carry NaN -- on L5-4 that leaves as
    #: few as 2 labeled points, and coping with it *is* the sparse-feasibility
    #: task. Extending to a comfortable sample size would quietly delete the
    #: difficulty the level exists to test.
    min_labeled: int = 2
    #: Cap on extra draws used only to clear `min_labeled`.
    max_init_extension: int = 120
    duplicate_tol: float = 1e-8
    max_duplicate_retries: int = 3


@dataclass
class RunResult:
    task_id: str
    width_label: str
    method: str
    seed: int
    X: np.ndarray
    Y_observed: np.ndarray
    Y_true: np.ndarray
    feasible: np.ndarray
    failure_type: list
    is_init: np.ndarray
    hit: np.ndarray
    iteration: np.ndarray
    wall_clock: np.ndarray
    counters: list[dict] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


def _concat(a: Observation, b: Observation) -> Observation:
    return Observation(
        X=np.vstack([a.X, b.X]),
        Y=np.vstack([a.Y, b.Y]),
        feasible=np.concatenate([a.feasible, b.feasible]),
        failure_type=a.failure_type + b.failure_type,
    )


def _is_duplicate(X: np.ndarray, x: np.ndarray, tol: float) -> bool:
    return len(X) > 0 and bool((np.abs(X - x).max(axis=1) <= tol).any())


def build_initial_design(
    oracle: Oracle, cfg: CampaignConfig, seed: int
) -> tuple[Observation, int]:
    """Draw the initial design, extending only far enough to make a GP fittable.

    Returns the observations and the oracle calls spent. The count is recorded,
    not billed to the optimization budget -- see the module docstring.
    """
    X = oracle.initial_design(cfg.n_init, seed=seed)
    obs = oracle.observe(X, seed=seed)
    spent = len(X)

    extra = 0
    while (
        int(np.isfinite(obs.Y).all(axis=1).sum()) < cfg.min_labeled
        and extra < cfg.max_init_extension
    ):
        step = min(cfg.n_init, cfg.max_init_extension - extra)
        X_more = oracle.initial_design(step, seed=seed + 10_000 + extra)
        obs = _concat(obs, oracle.observe(X_more, seed=seed + 10_000 + extra))
        extra += step
        spent += step

    return obs, spent


def run_campaign(
    task: BenchmarkTask,
    range_spec: RangeSpec,
    method,
    oracle: Oracle,
    seed: int,
    cfg: CampaignConfig | None = None,
) -> RunResult:
    cfg = cfg or CampaignConfig()
    rng = np.random.default_rng(seed)
    recorder = RunRecorder()

    # Acquisition optimization draws from torch's global generator (MC samplers,
    # GP fit initialization), so without this a rerun of the same cell gives
    # slightly different proposals. Seeding here makes a cell reproducible, which
    # is what lets `scripts/audit_leakage.py` assert that a method's proposals are
    # bit-identical when the truth oracle is scrambled. Methods start from the same
    # torch state and diverge only through their own draws.
    torch.manual_seed(seed)

    obs, init_spent = build_initial_design(oracle, cfg, seed)
    n_init = len(obs)
    remaining = cfg.budget

    # Two optional oracle capabilities, both absent on the continuous suites and
    # both no-ops when absent. A benchmark whose design space is a finite
    # catalogue supplies them: `enumerate_designs` so restarts start at real
    # designs, `canonicalize` so the duplicate check compares the design that
    # will actually be evaluated rather than the pre-rounding proposal.
    enumerate_designs = getattr(oracle, "enumerate_designs", None)
    canonicalize = getattr(oracle, "canonicalize", None)

    variables = BenchmarkVariables(
        task, range_spec, obs.X, obs.Y, seed=seed,
        objective_mode=getattr(method, "objective_mode", "range"),
        restart_pool=enumerate_designs() if enumerate_designs is not None else None,
    )

    iteration = [0] * n_init
    wall = [0.0] * n_init
    counters: list[dict] = []

    it = 0
    while it < remaining:
        it += 1
        t0 = time.perf_counter()
        variables.set_data(obs.X, obs.Y)

        x_next = None
        for attempt in range(cfg.max_duplicate_retries + 1):
            try:
                proposal = method.propose(variables, cfg.q, seed=seed, recorder=recorder)
            except CandidateGenerationError as err:
                # Acquisition optimization produced nothing at all. Counting it and
                # spending the evaluation on a random design keeps the budget
                # comparable across methods; silently retrying would give a
                # failing method extra attempts the others never get.
                recorder.note("acqf_total_failure")
                recorder.note_text("acqf_total_failure_reason", str(err))
                break
            cand = np.atleast_2d(proposal)[0]
            # On a catalogue benchmark two different proposals can round to the
            # same material. Comparing the raw proposals would let a method
            # re-evaluate an exhausted design every iteration and never be
            # charged a duplicate.
            probe = canonicalize(cand[None, :])[0] if canonicalize is not None else cand
            if not _is_duplicate(obs.X, probe, cfg.duplicate_tol):
                x_next = cand
                break
            recorder.note("duplicate_proposals")
        if x_next is None:
            # Either every retry collapsed onto an existing design, or the
            # acquisition optimizer failed outright. Spend the evaluation on a
            # random feasible design rather than a repeat.
            recorder.note("random_fallback")
            x_next = oracle.initial_design(1, seed=int(rng.integers(1 << 30)))[0]

        new = oracle.observe(x_next[None, :], seed=seed * 1000 + it)
        obs = _concat(obs, new)

        counters.append(recorder.flush(it).counters)
        iteration.append(it)
        wall.append(time.perf_counter() - t0)

    truth = oracle.truth(obs.X)
    hit = range_spec.hits(truth.Y, task.y_names)

    return RunResult(
        task_id=task.task_id,
        width_label=range_spec.width_label,
        method=getattr(method, "name", type(method).__name__),
        seed=seed,
        X=obs.X,
        Y_observed=obs.Y,
        Y_true=truth.Y,
        feasible=obs.feasible,
        failure_type=obs.failure_type,
        is_init=np.arange(len(obs.X)) < n_init,
        hit=hit,
        iteration=np.asarray(iteration),
        wall_clock=np.asarray(wall),
        counters=counters,
        meta={
            "n_init_drawn": n_init,
            "init_oracle_calls": init_spent,
            "budget_for_loop": remaining,
            "labeled_after_init": int(np.isfinite(obs.Y[:n_init]).all(axis=1).sum()),
        },
    )
