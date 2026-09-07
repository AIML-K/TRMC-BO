"""Protocol invariants that make the method comparison mean anything.

`scripts/audit_leakage.py` checks these across every method and every task, but
it takes minutes and nobody runs it on a whim. These are the same assertions on
one small task, fast enough to run on every commit, so a regression is caught
where it is introduced rather than at the next audit.

The invariants:

* no method can reach `oracle.truth` -- proposals are bit-identical when the
  truth oracle is replaced with garbage
* every method at a given seed starts from the same initial design
* every method spends the same number of oracle evaluations
* the point-target baseline reads no window information the proposed method
  does not also read (the specific suspicion that prompted the audit)
* a cell is reproducible, which the truth-scramble check depends on

Kept to three methods and a two-iteration loop on purpose: the property being
tested is structural, and covering all eight here would trade minutes of CI for
no extra coverage.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from eval import methods
from eval.benchmarks.matformbench import ranges
from eval.benchmarks.matformbench.oracle import make_oracle
from eval.benchmarks.matformbench.tasks import load_task
from eval.harness.loop import CampaignConfig, build_initial_design, run_campaign
from eval.scripts.audit_leakage import _FieldSpy

TASK = "L1-1"
WIDTH = "medium"
SEED = 0
CFG = CampaignConfig(n_init=12, budget=2, q=1)
RESTARTS = 4

#: Every method may read the window and its weight. Anything else is an
#: asymmetry that has to be justified in the write-up, not discovered later.
COMMON_FIELDS = {"obj", "lb", "ub", "weight"}

#: The tolerance-ball baseline cannot define a ball without a center and a scale,
#: and those come from the same frozen calibration sample that defines the window.
#: Disclosed rather than removed -- removing it would mean not implementing the
#: published method.
TB_EXTRA_FIELDS = {"ball_center", "ref_center", "ref_scale"}


class _ScrambledTruth:
    def __init__(self, inner):
        self._inner = inner

    def truth(self, X):
        obs = self._inner.truth(X)
        rng = np.random.default_rng(0xBADF00D)
        return replace(obs, Y=rng.normal(1e4, 1e4, size=obs.Y.shape))

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _CountingObserve:
    def __init__(self, inner):
        self._inner = inner
        self.calls = 0

    def observe(self, X, seed=None):
        self.calls += len(np.atleast_2d(X))
        return self._inner.observe(X, seed=seed)

    def __getattr__(self, name):
        return getattr(self._inner, name)


@pytest.fixture(scope="module")
def task_and_spec():
    return load_task(TASK), ranges.load(TASK, WIDTH)


def _run(task, spec, name, oracle, spy=None):
    method = methods.make(name, num_restarts=RESTARTS)
    if spy is None:
        return run_campaign(task, spec, method, oracle, seed=SEED, cfg=CFG)

    from eval.harness import adapter

    original = adapter.BenchmarkVariables._build_target_info
    adapter.BenchmarkVariables._build_target_info = staticmethod(
        lambda t, rs, mode: {k: _FieldSpy(v, spy) for k, v in original(t, rs, mode).items()}
    )
    try:
        return run_campaign(task, spec, method, oracle, seed=SEED, cfg=CFG)
    finally:
        adapter.BenchmarkVariables._build_target_info = staticmethod(original)


@pytest.mark.parametrize("name", ["M2_target_distance", "M3_range_prob", "M5_full"])
def test_proposals_do_not_move_when_the_truth_oracle_is_scrambled(task_and_spec, name):
    task, spec = task_and_spec
    clean = _run(task, spec, name, make_oracle(task))
    dirty = _run(task, spec, name, _ScrambledTruth(make_oracle(task)))

    n0, n1 = int(clean.is_init.sum()), int(dirty.is_init.sum())
    assert n0 == n1
    np.testing.assert_array_equal(clean.X[n0:], dirty.X[n1:])


@pytest.mark.parametrize("name", ["M2_target_distance", "M3_range_prob", "M5_full"])
def test_initial_design_is_the_same_for_every_method(task_and_spec, name):
    task, spec = task_and_spec
    reference, spent = build_initial_design(make_oracle(task), CFG, SEED)

    run = _run(task, spec, name, make_oracle(task))
    n0 = int(run.is_init.sum())
    assert n0 == len(reference)
    np.testing.assert_array_equal(run.X[:n0], reference.X)
    np.testing.assert_array_equal(
        np.nan_to_num(run.Y_observed[:n0], nan=-1e30),
        np.nan_to_num(reference.Y, nan=-1e30),
    )
    assert spent == CFG.n_init  # no method-dependent extension on this task


@pytest.mark.parametrize("name", ["M2_target_distance", "M3_range_prob", "M5_full"])
def test_every_method_spends_the_same_budget(task_and_spec, name):
    task, spec = task_and_spec
    oracle = _CountingObserve(make_oracle(task))
    _run(task, spec, name, oracle)
    assert oracle.calls == CFG.n_init + CFG.budget


def test_point_target_reads_no_more_than_the_proposed_method(task_and_spec):
    """The suspicion that started the audit, pinned as a test.

    Point-target BO beat the proposed method on hit rate, which looks like a leak.
    It is not: it reads strictly the same fields. The finding was that the metric
    was wrong, not the protocol.
    """
    task, spec = task_and_spec
    seen = {}
    for name in ("M2_target_distance", "M5_full", "M6b_range_aware_tb_inscribed"):
        fields: set = set()
        _run(task, spec, name, make_oracle(task), spy=fields)
        seen[name] = fields - {"<iterated>"}

    assert seen["M2_target_distance"] <= COMMON_FIELDS
    assert seen["M5_full"] <= COMMON_FIELDS
    assert seen["M2_target_distance"] <= seen["M5_full"]
    # The one disclosed asymmetry, asserted so it stays the only one.
    assert seen["M6b_range_aware_tb_inscribed"] <= COMMON_FIELDS | TB_EXTRA_FIELDS
    assert seen["M6b_range_aware_tb_inscribed"] & TB_EXTRA_FIELDS


def test_a_cell_is_reproducible(task_and_spec):
    """The truth-scramble test compares runs, so it needs this to hold first."""
    task, spec = task_and_spec
    a = _run(task, spec, "M5_full", make_oracle(task))
    b = _run(task, spec, "M5_full", make_oracle(task))
    np.testing.assert_array_equal(a.X, b.X)
