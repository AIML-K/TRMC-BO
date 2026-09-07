"""MatFormBench's frozen target windows.

The calibration rule and its rationale now live in `eval/benchmarks/calibration.py`,
shared with the Olympus suite. This module is the MatFormBench binding of it and
keeps the public surface callers already use -- `ranges.load(task_id, width)`,
`ranges.calibrate(...)`, `ranges.save(task_id, spec)` -- so nothing downstream
changed when the generic parts moved out.

The 15 committed specs under `eval/specs/matformbench/` were produced before the
move and must keep loading byte-identically; `eval/tests/test_calibration.py`
asserts that against the committed JSON.
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

SUITE = "matformbench"

#: The conditions this suite has, for `--widths` defaults and `registry.Suite`.
#: Derived from the shared difficulty table rather than restated, so a suite can
#: never drift out of step with the rule that produces its windows.
WIDTHS: tuple[str, ...] = tuple(calibration.DIFFICULTY)

#: Kept as a module constant because scripts and tests refer to it by name.
SPEC_DIR: Path = calibration.spec_dir(SUITE)


def spec_path(task_id: str, width_label: str) -> Path:
    return calibration.spec_path(SUITE, task_id, width_label)


def save(task_id: str, spec: RangeSpec) -> Path:
    return calibration.save(SUITE, task_id, spec)


def load(task_id: str, width_label: str) -> RangeSpec:
    return calibration.load(SUITE, task_id, width_label)
