"""Suite dispatch: the one place that knows which suites exist.

`harness/` is otherwise suite-neutral -- `adapter`, `loop`, `metrics`, `recorder`
and every method reference only `benchmarks/base`. Before this module, the single
exception was three imports inside `runner.run_cell`; now that is a `resolve`
call, and the same call serves the four scripts that used to import MatFormBench
directly (`calibrate_ranges`, `audit_leakage`, `window_geometry`,
`plot_distinct_designs`).

**Import lazily, inside the function that needs it.** Suites are mutually
hostile at import time and the isolation is structural, not incidental:

* MatFormBench's oracle is a compiled CPython 3.10 extension.
* Olympus reads numpy arrays extracted from a TensorFlow checkpoint; if TF were
  ever imported in a worker it would fight 22 single-thread processes for
  threads, and `wall_clock_per_iter` is a reported metric.
* HOIP is a plain lookup over two CSVs and needs neither, which is precisely why
  it must not end up importing either by accident.

`resolve` therefore imports only the suite asked for. A test asserts that running
an Olympus cell leaves no MatFormBench extension in `sys.modules`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from eval.benchmarks.base import BenchmarkTask, Oracle, RangeSpec

#: Suites that have an implementation.
SUITES = ("matformbench", "olympus", "hoip")


@dataclass(frozen=True)
class Suite:
    """The three entry points every suite must provide."""

    name: str
    load_task: Callable[[str], BenchmarkTask]
    make_oracle: Callable[[BenchmarkTask], Oracle]
    load_range_spec: Callable[[str, str], RangeSpec]
    #: Task ids the suite's own configs select, for `--tasks` defaults.
    default_tasks: tuple[str, ...]
    #: Width labels the suite actually has, for `--widths` defaults. Not a
    #: constant: HOIP supplies its own window and varies input feasibility
    #: instead, so it has one condition where the others have three. Every
    #: analysis script defaulted to "medium" and crashed on it -- the right
    #: failure, but the knowledge belongs here with the rest of what is
    #: suite-specific.
    default_widths: tuple[str, ...]


def resolve(suite: str) -> Suite:
    """Look up a suite, importing only that suite's modules."""
    if suite == "matformbench":
        from eval.benchmarks.matformbench import ranges
        from eval.benchmarks.matformbench.oracle import make_oracle
        from eval.benchmarks.matformbench.tasks import SELECTED_TASKS, load_task

        return Suite(
            name=suite,
            load_task=load_task,
            make_oracle=make_oracle,
            load_range_spec=ranges.load,
            default_tasks=tuple(SELECTED_TASKS),
            default_widths=ranges.WIDTHS,
        )

    if suite == "olympus":
        from eval.benchmarks.olympus import ranges
        from eval.benchmarks.olympus.oracle import make_oracle
        from eval.benchmarks.olympus.tasks import SELECTED_TASKS, load_task

        return Suite(
            name=suite,
            load_task=load_task,
            make_oracle=make_oracle,
            load_range_spec=ranges.load,
            default_tasks=tuple(SELECTED_TASKS),
            default_widths=ranges.WIDTHS,
        )

    if suite == "hoip":
        from eval.benchmarks.hoip import ranges
        from eval.benchmarks.hoip.oracle import make_oracle
        from eval.benchmarks.hoip.tasks import SELECTED_TASKS, load_task

        return Suite(
            name=suite,
            load_task=load_task,
            make_oracle=make_oracle,
            load_range_spec=ranges.load,
            default_tasks=tuple(SELECTED_TASKS),
            default_widths=ranges.WIDTHS,
        )

    raise KeyError(f"unknown suite {suite!r}; known: {list(SUITES)}")
