"""Prove the C1/C2/C3 toggles actually reach the optimizer.

If a toggle is silently ignored, every ablation number in the paper is
meaningless while still looking perfectly plausible. So rather than trusting the
wiring, this instruments one acquisition step per variant and reports what
BoTorch was really handed:

    acqf            the acquisition class and its MC objective (C1)
    nonlinear       count of nonlinear inequality constraints attached (C2)
    filtered        whether restart points were filtered by posterior mean (C3)

Expected pattern:

    M1  qEHVI / WeightedMCMultiOutputObjective   nonlinear=0   filtered=no
    M2  qEI   / NegativeTargetDistanceObjective  nonlinear=0   filtered=no
    M3  qEHVI / CDFRangeMultiOutputObjective     nonlinear=0   filtered=no
    M4  qEHVI / CDFRangeMultiOutputObjective     nonlinear=6   filtered=no
    M5  qEHVI / CDFRangeMultiOutputObjective     nonlinear=6   filtered=yes
    M6  ToleranceBall / (none)                   nonlinear=0   filtered=no

The factorial's four extra cells matter most here, because the two ways they can
break are both silent (see `eval/methods/factorial.py`):

    M13 qEHVI / CDFRangeMultiOutputObjective     nonlinear=0   filtered=yes
    M14 qEHVI / WeightedMCMultiOutputObjective   nonlinear=6   filtered=no
    M15 qEHVI / WeightedMCMultiOutputObjective   nonlinear=0   filtered=yes
    M16 qEHVI / WeightedMCMultiOutputObjective   nonlinear=6   filtered=yes

A CDFRange objective on M14-M16 means C1 never switched off; nonlinear=0 on M14
or M16 means the range_spec died with it and C2 switched off unasked. Either one
turns the arm into a duplicate of another row while still producing numbers.

The constraint count is two per property, one per side, so it is a fact about the
task and not a constant: six on MatFormBench (three outputs), two on Olympus
(one), four on HOIP (two).

    PYTHONPATH=core:. .venv-eval/bin/python -m eval.scripts.verify_variants
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np

import bayesian_optimization as bo
from eval.benchmarks.registry import resolve
from eval.harness.adapter import BenchmarkVariables
from eval.methods import make as make_method

#: Methods to probe, in ablation order. `M6c` is single-output only, so it is
#: skipped automatically when the task has several outputs.
ORDER = ["M1_qehvi_extremum", "M2_target_distance", "M3_range_prob",
         "M4_range_prob_constraints", "M13_range_prob_starts", "M5_full",
         "M14_constraints_only", "M15_starts_only", "M16_constraints_starts",
         "M6_range_aware_tb",
         "M6b_range_aware_tb_inscribed", "M6c_range_aware_tb_box"]


class _Probe:
    """Wraps the production functions to record what they received."""

    def __init__(self):
        self.acqf = None
        self.objective = None
        self.n_nonlinear = 0
        self.filtered = False

    def install(self):
        self._orig_optimize = bo.optimize_acqf_on_filtered_points
        self._orig_finalize = bo._finalize_single_restart_optimize_config
        probe = self

        # **kwargs so this keeps working if the production signature grows again.
        def optimize(acq_function, optimize_config, dataloader, cat_dims, dis_dims,
                     range_spec=None, **kwargs):
            probe.acqf = type(acq_function).__name__
            obj = getattr(acq_function, "objective", None)
            probe.objective = "-" if obj is None else type(obj).__name__
            probe.filtered = bool(
                range_spec is not None and kwargs.get("use_mean_filtered_starts", True)
            )
            return probe._orig_optimize(
                acq_function, optimize_config, dataloader, cat_dims, dis_dims,
                range_spec=range_spec, **kwargs,
            )

        def finalize(optimize_config, constraints):
            probe.n_nonlinear = 0 if constraints is None else len(constraints)
            return probe._orig_finalize(optimize_config, constraints)

        bo.optimize_acqf_on_filtered_points = optimize
        bo._finalize_single_restart_optimize_config = finalize

    def restore(self):
        bo.optimize_acqf_on_filtered_points = self._orig_optimize
        bo._finalize_single_restart_optimize_config = self._orig_finalize


def main() -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="matformbench")
    ap.add_argument("--task", default=None, help="default: the suite's first task")
    ap.add_argument("--width", default=None,
                    help="default: the suite's own condition (HOIP has only 'native')")
    # The probe only inspects what was *constructed*, so it does not need the
    # sweep's 128 restarts; a small number keeps this a seconds-long check.
    ap.add_argument("--restarts", type=int, default=8)
    args = ap.parse_args()

    suite = resolve(args.suite)
    task_id = args.task or suite.default_tasks[0]
    width = args.width or suite.default_widths[0]
    task = suite.load_task(task_id)
    oracle = suite.make_oracle(task)
    spec = suite.load_range_spec(task_id, width)

    X = oracle.initial_design(30, seed=0)
    obs = oracle.observe(X, seed=0)

    # Mirror the campaign: a catalogue suite restarts from its catalogue, and C3
    # is a filter *on those restarts*, so probing with the continuous sampler
    # would report on a configuration no cell ever runs.
    enumerate_designs = getattr(oracle, "enumerate_designs", None)
    restart_pool = enumerate_designs() if enumerate_designs is not None else None

    header = f"{'method':26s} {'acquisition':34s} {'objective (C1)':34s} {'C2':>4s} {'C3':>5s}"
    print(f"task={task_id} width={width}")
    print(header)
    print("-" * len(header))

    for name in ORDER:
        try:
            method = make_method(name, args.restarts)
        except (ValueError, KeyError) as err:
            # e.g. the box mapping on a multi-output task. A method that declines
            # is a documented outcome; one that silently degenerates is the bug
            # this script exists to catch.
            print(f"{name:26s} SKIPPED: {err}")
            continue
        variables = BenchmarkVariables(
            task, spec, obs.X, obs.Y, seed=0,
            objective_mode=getattr(method, "objective_mode", "range"),
            restart_pool=restart_pool,
        )
        probe = _Probe()
        probe.install()
        try:
            method.propose(variables, q=1, seed=0)
        except ValueError as err:
            # A method that declines this task is a documented outcome (the box
            # mapping is exact only at m=1); one that silently degenerates is the
            # bug this script exists to catch.
            print(f"{name:26s} DECLINED: {err}")
            continue
        finally:
            probe.restore()
        print(f"{name:26s} {probe.acqf:34s} {probe.objective:34s} "
              f"{probe.n_nonlinear:4d} {'yes' if probe.filtered else 'no':>5s}")


if __name__ == "__main__":
    main()
