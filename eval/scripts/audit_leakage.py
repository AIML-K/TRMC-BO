"""Rule out information leakage in the method comparison, empirically.

A baseline that beats the proposed method by more than expected is the classic
symptom of a leak, so the point-target result had to be checked rather than
explained. This script runs the four checks that can actually settle it, on short
campaigns, and prints a pass/fail table. It is meant to be re-runnable: if a
future change starts feeding a method something it should not see, one of these
turns red.

    A1  truth-scramble    replace `oracle.truth` with garbage; a method whose
                          proposals shift was reading the answer key
    A2  initial design    all methods start from byte-identical (X, Y)
    A3  budget parity     all methods spend the same number of oracle calls
    A4  field exposure    which `target_info` fields each method actually reads

A1 relies on campaigns being reproducible, which is why `run_campaign` seeds
torch. If A1 reports a diff on *every* method including random search, suspect
that seeding broke rather than that everything leaks.

A4 is not pass/fail -- it is an inventory. Some asymmetry is by design and has to
be disclosed rather than removed: the tolerance-ball baseline needs a center and
a scale to define its ball at all, and those come from the same frozen
calibration sample that defines the window. What matters is that no method reads
a field derived from the run in progress.

    PYTHONPATH=core:. .venv-eval/bin/python -m eval.scripts.audit_leakage
    PYTHONPATH=core:. .venv-eval/bin/python -m eval.scripts.audit_leakage \
        --task L3-1 --budget 8 --methods M2_target_distance M5_full
"""

from __future__ import annotations

import argparse
import hashlib
import warnings
from dataclasses import replace

import numpy as np

from eval import methods
from eval.benchmarks.registry import resolve
from eval.harness.loop import CampaignConfig, build_initial_design, run_campaign

#: COMBOO is excluded by default: it is deferred pending a verified box-to-ball
#: mapping, and auditing a method whose numbers we do not report wastes minutes.
DEFAULT_METHODS = [
    "M0_random",
    "M1_qehvi_extremum",
    "M2_target_distance",
    "M3_range_prob",
    "M4_range_prob_constraints",
    "M5_full",
    "M6_range_aware_tb",
    "M6b_range_aware_tb_inscribed",
]


def _hash(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for a in arrays:
        # nan_to_num because NaN != NaN would make a failed evaluation hash
        # differently on every call; the sentinel is outside every task's range.
        h.update(np.ascontiguousarray(np.nan_to_num(a, nan=-1e30), dtype=float).tobytes())
    return h.hexdigest()[:16]


class _CountingOracle:
    """Pass-through oracle that counts evaluations and can scramble the truth."""

    def __init__(self, inner, scramble: bool = False):
        self._inner = inner
        self._scramble = scramble
        self.observe_calls = 0
        self.truth_calls = 0

    def initial_design(self, n, seed=None):
        return self._inner.initial_design(n, seed=seed)

    def observe(self, X, seed=None):
        self.observe_calls += len(np.atleast_2d(X))
        return self._inner.observe(X, seed=seed)

    def truth(self, X):
        self.truth_calls += 1
        obs = self._inner.truth(X)
        if not self._scramble:
            return obs
        # Garbage, but garbage of the right shape and dtype: a method that reads
        # truth through any channel would see these values and move.
        rng = np.random.default_rng(0xBADF00D)
        return replace(obs, Y=rng.normal(1e4, 1e4, size=obs.Y.shape))

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _FieldSpy(dict):
    """Dict that records which keys were read.

    `.items()`/`.values()` hand out every value at once, so they are recorded as
    a bulk read rather than pretended to be per-key -- claiming a method touched
    only `lb` when the pipeline iterated the whole dict would be worse than
    admitting the granularity is coarse.
    """

    def __init__(self, base: dict, seen: set):
        super().__init__(base)
        self._seen = seen

    def __getitem__(self, key):
        self._seen.add(str(key))
        return super().__getitem__(key)

    def get(self, key, default=None):
        self._seen.add(str(key))
        return super().get(key, default)

    def items(self):
        self._seen.add("<iterated>")
        return super().items()

    def values(self):
        self._seen.add("<iterated>")
        return super().values()


def _run(task, spec, name, oracle, cfg, seed, spy: set | None = None):
    method = methods.make(name, num_restarts=8)
    if spy is None:
        return run_campaign(task, spec, method, oracle, seed=seed, cfg=cfg)

    # Wrap after construction: the spy has to sit on the dict the adapter builds,
    # so patch `_build_target_info` for the duration of this one campaign.
    from eval.harness import adapter

    original = adapter.BenchmarkVariables._build_target_info

    def wrapped(t, rs, mode):
        info = original(t, rs, mode)
        return {k: _FieldSpy(v, spy) for k, v in info.items()}

    adapter.BenchmarkVariables._build_target_info = staticmethod(wrapped)
    try:
        return run_campaign(task, spec, method, oracle, seed=seed, cfg=cfg)
    finally:
        adapter.BenchmarkVariables._build_target_info = staticmethod(original)


def main() -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="matformbench")
    ap.add_argument("--task", default=None, help="default: the suite's first task")
    ap.add_argument("--width", default=None,
                    help="default: the suite's first condition")
    ap.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    ap.add_argument("--n-init", type=int, default=15)
    ap.add_argument("--budget", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    suite = resolve(args.suite)
    task = suite.load_task(args.task or suite.default_tasks[0])
    width = args.width or suite.default_widths[0]
    spec = suite.load_range_spec(task.task_id, width)
    make_oracle = suite.make_oracle
    cfg = CampaignConfig(n_init=args.n_init, budget=args.budget, q=1)

    print(f"audit: {task.task_id}/{width}  n_init={args.n_init} "
          f"budget={args.budget} seed={args.seed}  restarts=8 (audit setting)\n")

    # -- A2: the initial design is drawn before any method exists, so it can only
    # differ if `build_initial_design` consults the method. Hash it once as the
    # reference every campaign is then checked against.
    ref_obs, ref_spent = build_initial_design(
        _CountingOracle(make_oracle(task)), cfg, args.seed
    )
    ref_init = _hash(ref_obs.X, ref_obs.Y)
    print(f"A2 reference initial design: {len(ref_obs)} rows, hash {ref_init}, "
          f"{ref_spent} oracle calls\n")

    rows = []
    for name in args.methods:
        seen: set = set()
        clean_oracle = _CountingOracle(make_oracle(task))
        clean = _run(task, spec, name, clean_oracle, cfg, args.seed, spy=seen)

        # Fresh oracle for the scrambled run: reusing one would let cached state
        # carry the clean answers across and hide a real leak.
        dirty = _run(task, spec, name, _CountingOracle(make_oracle(task), scramble=True),
                     cfg, args.seed)

        n_init = int(clean.is_init.sum())
        proposed_clean = clean.X[n_init:]
        proposed_dirty = dirty.X[int(dirty.is_init.sum()):]
        same_shape = proposed_clean.shape == proposed_dirty.shape
        a1 = same_shape and np.array_equal(proposed_clean, proposed_dirty)

        a2 = _hash(clean.X[:n_init], clean.Y_observed[:n_init]) == ref_init
        a3_calls = clean_oracle.observe_calls
        a3 = a3_calls == ref_spent + args.budget

        rows.append({
            "method": name,
            "A1": a1,
            "A2": a2,
            "A3": a3,
            "calls": a3_calls,
            "fields": sorted(seen),
            "max_dev": (
                float(np.abs(proposed_clean - proposed_dirty).max())
                if same_shape else float("nan")
            ),
        })
        print(f"  {name:<32} A1={'pass' if a1 else 'FAIL':<4} "
              f"A2={'pass' if a2 else 'FAIL':<4} A3={'pass' if a3 else 'FAIL':<4} "
              f"calls={a3_calls}")

    print("\n" + "=" * 78)
    print(f"{'method':<32} {'A1 truth':<10} {'A2 init':<9} {'A3 budget':<10} deviation")
    print("-" * 78)
    for r in rows:
        dev = "0" if r["A1"] else f"{r['max_dev']:.3g}"
        print(f"{r['method']:<32} {'pass' if r['A1'] else 'FAIL':<10} "
              f"{'pass' if r['A2'] else 'FAIL':<9} "
              f"{'pass' if r['A3'] else 'FAIL':<10} {dev}")

    print("\nA4 -- target_info fields each method read")
    print("-" * 78)
    for r in rows:
        print(f"{r['method']:<32} {', '.join(r['fields']) or '(none)'}")

    failed = [r["method"] for r in rows if not (r["A1"] and r["A2"] and r["A3"])]
    print("\n" + ("ALL PASS -- no method reads the truth oracle, all share one "
                  "initial design and one budget."
                  if not failed else f"FAILURES: {', '.join(failed)}"))
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
