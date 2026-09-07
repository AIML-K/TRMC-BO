"""The table generator's guard against reporting an unfinished sweep.

A missing arm is obvious. A half-finished one is not: it prints a plausible mean
over whichever cells landed first, and the runner walks its matrix in order, so
those cells are biased toward the first task and the widest window rather than
being a random subset.

    PYTHONPATH=core:. .venv-eval/bin/python -m pytest eval/tests/test_emit_tables.py -q
"""

from __future__ import annotations

import pandas as pd

from eval.scripts.emit_tables import (
    MISSING,
    PARTIAL,
    _expected_cells,
    _mean_if_complete,
)


def _counts(spec):
    """spec: {method: [distinct, ...]} -- one entry per completed cell."""
    rows = []
    for method, values in spec.items():
        for i, v in enumerate(values):
            rows.append({"task_id": f"T{i // 3}", "width": "medium",
                         "seed": i % 3, "method": method, "distinct": v})
    return pd.DataFrame(rows)


def test_expected_cells_is_the_longest_arm():
    c = _counts({"complete": [1] * 9, "short": [1] * 2})
    assert _expected_cells(c) == 9


def test_a_complete_arm_reports_its_mean():
    c = _counts({"a": [2.0, 4.0, 6.0]})
    assert _mean_if_complete(c, "a", _expected_cells(c)) == "4.00"


def test_a_half_finished_arm_is_marked_not_averaged():
    """The live case: on a protocol where complete arms have 150 cells, an arm
    with 13 must not print the mean of those 13."""
    c = _counts({"done": [1.0] * 9, "running": [9.0] * 2})
    assert _mean_if_complete(c, "running", _expected_cells(c)) == PARTIAL
    assert _mean_if_complete(c, "done", _expected_cells(c)) == "1.00"


def test_an_absent_arm_is_distinguished_from_a_partial_one():
    """`--` means the method does not run here at all (SCBO on a multi-output
    suite); `(partial)` means it does and has not finished. Collapsing the two
    would read as 'not applicable' for an arm that is merely late."""
    c = _counts({"done": [1.0] * 9})
    assert _mean_if_complete(c, "never_ran", _expected_cells(c)) == MISSING


def test_nothing_claims_completeness_when_no_arm_is_complete():
    """Early in a sweep every arm is short. Scaling to the longest would declare
    the least-unfinished one complete."""
    c = _counts({"a": [1.0], "b": [2.0]})
    assert _mean_if_complete(c, "a", _expected_cells(c)) == "1.00"
    assert _expected_cells(c) == 1


def test_empty_counts_do_not_raise():
    empty = pd.DataFrame(columns=["task_id", "width", "seed", "method", "distinct"])
    assert _expected_cells(empty) == 0
    assert _mean_if_complete(empty, "a", 0) == MISSING


# -- declared task subsets -------------------------------------------------

def _counts_tasks(spec):
    """spec: {method: {task: [distinct, ...]}}"""
    rows = []
    for method, tasks in spec.items():
        for task, values in tasks.items():
            for i, v in enumerate(values):
                rows.append({"task_id": task, "width": "medium", "seed": i,
                             "method": method, "distinct": v})
    return pd.DataFrame(rows)


def test_a_declared_subset_is_reported_and_marked_not_called_partial():
    """COMBOO runs three of MatFormBench's five tasks because the other two cost
    days, not because its sweep is unfinished. Calling that `(partial)` would
    withhold a baseline the paper promises to report."""
    from eval.scripts.emit_tables import SUBSET_MARK

    c = _counts_tasks({
        "full": {t: [1.0, 1.0] for t in ["L1-1", "L2-1", "L3-1", "L4-1", "L5-4"]},
        "M8_comboo": {t: [3.0, 5.0] for t in ["L1-1", "L2-1", "L3-1"]},
    })
    out = _mean_if_complete(c, "M8_comboo", _expected_cells(c), suite="matformbench")
    assert out == "4.00" + SUBSET_MARK


def test_a_subset_method_that_is_also_unfinished_is_still_partial():
    """The marking must not become a blanket exemption: within its declared
    tasks, an incomplete sweep is still incomplete."""
    c = _counts_tasks({
        "full": {t: [1.0, 1.0] for t in ["L1-1", "L2-1", "L3-1", "L4-1", "L5-4"]},
        "M8_comboo": {"L1-1": [3.0]},  # 1 of the 6 cells its subset implies
    })
    assert _mean_if_complete(c, "M8_comboo", _expected_cells(c),
                             suite="matformbench") == PARTIAL


def test_the_subset_only_applies_to_the_suite_it_was_declared_for():
    """COMBOO's restriction is a MatFormBench cost fact; on another suite it
    must be judged normally."""
    c = _counts_tasks({
        "full": {t: [1.0, 1.0] for t in ["a", "b"]},
        "M8_comboo": {"a": [3.0, 3.0]},
    })
    assert _mean_if_complete(c, "M8_comboo", _expected_cells(c),
                             suite="hoip") == PARTIAL


def test_no_suite_given_means_no_subset_handling():
    c = _counts_tasks({
        "full": {t: [1.0, 1.0] for t in ["L1-1", "L2-1", "L3-1", "L4-1", "L5-4"]},
        "M8_comboo": {t: [3.0, 3.0] for t in ["L1-1", "L2-1", "L3-1"]},
    })
    assert _mean_if_complete(c, "M8_comboo", _expected_cells(c)) == PARTIAL
