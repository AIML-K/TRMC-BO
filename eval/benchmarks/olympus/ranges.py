"""Olympus's frozen target windows.

Same rule as MatFormBench -- see `eval/benchmarks/calibration.py` -- bound to this
suite's spec directory. What differs is where the native threshold comes from:
MatFormBench ships one-sided targets, Olympus ships only "minimize", so the
threshold is *synthesized* and that construction is recorded in the spec so no
reader mistakes it for a benchmark-supplied specification. See
`eval/benchmarks/olympus/tasks.py` for the construction and `NATIVE_QUANTILE` for
why it sits where it does.
"""

from __future__ import annotations

from pathlib import Path

from eval.benchmarks import calibration
from eval.benchmarks.base import RangeSpec
from eval.benchmarks.calibration import (  # noqa: F401  -- re-exported public API
    DIFFICULTY,
    calibrate,
    reference_sample,
    sample_design_space,
)

SUITE = "olympus"

#: The conditions this suite has, for `--widths` defaults and `registry.Suite`.
#: Derived from the shared difficulty table rather than restated, so a suite can
#: never drift out of step with the rule that produces its windows.
WIDTHS: tuple[str, ...] = tuple(calibration.DIFFICULTY)

SPEC_DIR: Path = calibration.spec_dir(SUITE)


def spec_path(task_id: str, width_label: str) -> Path:
    return calibration.spec_path(SUITE, task_id, width_label)


def save(task_id: str, spec: RangeSpec) -> Path:
    return calibration.save(SUITE, task_id, spec)


def load(task_id: str, width_label: str) -> RangeSpec:
    return calibration.load(SUITE, task_id, width_label)
