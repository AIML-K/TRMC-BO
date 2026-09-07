"""The generic calibration module must not have moved any committed window.

`benchmarks/calibration.py` was extracted from `benchmarks/matformbench/ranges.py`
so a second suite could share it. The 1344 completed MatFormBench cells were
scored against the 15 windows committed under `eval/specs/matformbench/`, so a
change of even one bound would silently invalidate every one of them -- the runs
would still complete and still look plausible.

These tests pin the boundary rather than the arithmetic: the committed JSON is the
contract, and the wrapper must reproduce it byte-for-byte. `calibrate` itself is
covered end-to-end by `calibrate_ranges --check`, which recalibrates from the
oracle and compares; that needs the compiled oracle and 200k evaluations per
task, so it is a script rather than a unit test.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from eval.benchmarks import calibration
from eval.benchmarks.matformbench import ranges
from eval.benchmarks.matformbench.tasks import SELECTED_TASKS

WIDTHS = ("wide", "medium", "narrow")
CELLS = [(t, w) for t in SELECTED_TASKS for w in WIDTHS]


@pytest.mark.parametrize(("task_id", "width"), CELLS)
def test_committed_spec_round_trips(task_id, width):
    """The wrapper reads exactly what is on disk."""
    spec = ranges.load(task_id, width)
    raw = json.loads(ranges.spec_path(task_id, width).read_text(encoding="utf-8"))

    assert spec.lower == raw["lower"]
    assert spec.upper == raw["upper"]
    assert spec.width_label == raw["width_label"] == width
    assert spec.calibration == raw["calibration"]


@pytest.mark.parametrize(("task_id", "width"), CELLS)
def test_scoring_geometry_is_present(task_id, width):
    """`metrics.run_metrics` computes both ball criteria unconditionally.

    Without these two keys it raises a KeyError deep inside scoring, after the
    campaign has already been paid for.
    """
    cal = ranges.load(task_id, width).calibration
    for key in ("y_reference_center", "y_reference_scale"):
        assert key in cal, f"{task_id}/{width} is missing {key}"
    assert all(v > 0 for v in cal["y_reference_scale"].values())


def test_wrapper_delegates_to_the_shared_module():
    """The suite binding is a binding, not a second implementation."""
    assert ranges.SPEC_DIR == calibration.spec_dir("matformbench")
    assert ranges.spec_path("L1-1", "wide") == calibration.spec_path(
        "matformbench", "L1-1", "wide"
    )
    assert ranges.DIFFICULTY is calibration.DIFFICULTY
    assert ranges.calibrate is calibration.calibrate
    assert ranges.sample_design_space is calibration.sample_design_space


def test_spec_dirs_are_per_suite():
    """Two suites must not be able to overwrite each other's frozen windows."""
    from eval.benchmarks.olympus import ranges as olympus_ranges

    assert olympus_ranges.SPEC_DIR != ranges.SPEC_DIR
    assert olympus_ranges.SPEC_DIR.name == "olympus"


def test_single_output_calibration_is_analytic():
    """With one property the bisection solves `target = P(native) * alpha`.

    This is the identity the Olympus specs rely on, and the reason a synthesized
    native threshold has to leave headroom: placing it at the target quantile
    forces alpha = 1 and collapses the window onto the one-sided native target.
    Checked against a synthetic oracle so it needs no benchmark.
    """
    from eval.benchmarks.base import BenchmarkTask, NativeTarget, Observation

    rng = np.random.default_rng(0)
    draws = rng.normal(size=200_000)
    tau = float(np.quantile(draws, 0.25))

    task = BenchmarkTask(
        suite="test", task_id="analytic", dim=1,
        x_names=("x",), y_names=("y",),
        bounds=np.array([[0.0], [1.0]]),
        native_targets=(NativeTarget("y", "less", tau),),
    )

    class _Oracle:
        task = None

        def truth(self, X):
            n = len(X)
            return Observation(
                X=X, Y=draws[:n].reshape(-1, 1),
                feasible=np.ones(n, dtype=bool), failure_type=[None] * n,
            )

    reference = (draws.reshape(-1, 1), 1.0)
    for width, target in calibration.DIFFICULTY.items():
        spec = calibration.calibrate(
            task, _Oracle(), width, reference=reference, n_samples=len(draws)
        )
        alpha = spec.calibration["alpha"]
        # P(native) is 0.25 by construction, so alpha should be target / 0.25.
        assert alpha == pytest.approx(target / 0.25, abs=0.01)
        assert spec.calibration["achieved_fraction_of_feasible"] == pytest.approx(
            target, abs=5e-4
        )
        # Strictly two-sided: the lower bound is a real quantile, not the minimum.
        assert spec.lower["y"] > draws.min()
        assert spec.upper["y"] == pytest.approx(tau)
