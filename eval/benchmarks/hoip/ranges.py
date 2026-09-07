"""Frozen HOIP windows: a thin binding of the shared spec store to this suite.

Unlike the other two suites there is nothing to calibrate -- HOIP supplies its
own two-sided specification, so `scripts/freeze_hoip_specs.py` writes the
`RangeSpec` directly, by exact enumeration over the catalogue rather than by
sampling. `calibration.save`/`load` are reused unchanged, so the on-disk format
and the loading path stay identical across suites.

The only width label is `native`.
"""

from __future__ import annotations

from eval.benchmarks import calibration
from eval.benchmarks.base import RangeSpec

SUITE = "hoip"

#: HOIP's width axis is replaced by the feasibility axis; see `tasks.py`.
WIDTHS: tuple[str, ...] = ("native",)


def load(task_id: str, width_label: str = "native") -> RangeSpec:
    if width_label not in WIDTHS:
        raise ValueError(
            f"HOIP has one condition, {WIDTHS[0]!r}, not {width_label!r} -- its "
            "specification is benchmark-supplied and the difficulty axis is "
            "input feasibility, not window width. See benchmarks/hoip/tasks.py."
        )
    return calibration.load(SUITE, task_id, width_label)


def save(task_id: str, spec: RangeSpec):
    return calibration.save(SUITE, task_id, spec)
