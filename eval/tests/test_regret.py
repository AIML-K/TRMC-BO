"""The regret curves, and the one correction that is easy to get wrong.

`RangeSpec.violations` wraps its clip in `nan_to_num`, so a design the oracle
never returned a value for scores a violation of exactly 0.0 -- indistinguishable
from a perfect hit. A running minimum over the raw column therefore awards a
regret of zero to a method whose evaluation failed, which would rank the methods
that fail most as the ones that get into spec fastest.

    PYTHONPATH=core:. .venv-eval/bin/python -m pytest eval/tests/test_regret.py -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eval.scripts.plot_regret import violation_regret


class _Task:
    task_id = "T"
    x_names = ["a"]
    y_names = ["y1"]


def _write(root, method, seed, violation, feasible=None, y_true=None, n_init=2):
    d = root / "suite" / "proto" / "T" / "medium" / method / f"seed{seed:03d}"
    d.mkdir(parents=True, exist_ok=True)
    n = len(violation)
    frame = {
        "iteration": [0] * n_init + list(range(1, n + 1)),
        "is_init": [True] * n_init + [False] * n,
        "hit": [False] * (n_init + n),
        "violation_total": [np.nan] * n_init + list(violation),
        "x_a": [0.0] * (n_init + n),
        "y_true_y1": [0.0] * n_init + list(
            y_true if y_true is not None else [1.0] * n
        ),
    }
    if feasible is not None:
        frame["feasible"] = [True] * n_init + list(feasible)
    pd.DataFrame(frame).to_parquet(d / "trace.parquet")


def _run(root, methods, budget):
    return violation_regret(_Task(), methods, budget, root=root,
                            suite="suite", protocol="proto")


def test_regret_is_the_running_minimum(tmp_path):
    _write(tmp_path, "M0_random", 0, [5.0, 3.0, 4.0, 1.0])
    out = _run(tmp_path, ["M0_random"], 4)["M0_random"]
    assert list(out) == [5.0, 3.0, 3.0, 1.0]


def test_a_failed_evaluation_does_not_count_as_a_perfect_hit(tmp_path):
    """The bug. Iteration 2 failed, so `violations` reported 0.0 for it.

    Unmasked, the running min would read [5, 0, 0, 0] and this method would look
    like it reached the window at iteration 2. It never reached it at all.
    """
    _write(tmp_path, "M0_random", 0,
           violation=[5.0, 0.0, 4.0, 2.0],
           feasible=[True, False, True, True])
    out = _run(tmp_path, ["M0_random"], 4)["M0_random"]
    assert list(out) == [5.0, 5.0, 4.0, 2.0]
    assert out[1] != 0.0, "the masking was skipped"


def test_a_missing_output_is_masked_too(tmp_path):
    """A design can be input-feasible and still return no value for a property --
    the HOIP case, where a composition is preparable but its effective mass is a
    censoring sentinel held as missing."""
    _write(tmp_path, "M0_random", 0,
           violation=[5.0, 0.0, 3.0],
           feasible=[True, True, True],
           y_true=[1.0, np.nan, 1.0])
    out = _run(tmp_path, ["M0_random"], 3)["M0_random"]
    assert list(out) == [5.0, 5.0, 3.0]


def test_a_genuine_zero_is_kept(tmp_path):
    """Masking must not swallow a real hit: violation 0 on a feasible, fully
    observed design is the campaign succeeding."""
    _write(tmp_path, "M0_random", 0,
           violation=[5.0, 0.0, 3.0],
           feasible=[True, True, True])
    out = _run(tmp_path, ["M0_random"], 3)["M0_random"]
    assert list(out) == [5.0, 0.0, 0.0]


def test_init_rows_are_excluded(tmp_path):
    """Init designs are not the acquisition's doing and must not enter the curve."""
    _write(tmp_path, "M0_random", 0, [7.0, 6.0], n_init=3)
    out = _run(tmp_path, ["M0_random"], 2)["M0_random"]
    assert list(out) == [7.0, 6.0]


def test_seeds_are_averaged(tmp_path):
    _write(tmp_path, "M0_random", 0, [4.0, 4.0])
    _write(tmp_path, "M0_random", 1, [2.0, 2.0])
    out = _run(tmp_path, ["M0_random"], 2)["M0_random"]
    assert out == pytest.approx([3.0, 3.0])


def test_unrequested_methods_are_ignored(tmp_path):
    _write(tmp_path, "M0_random", 0, [1.0])
    _write(tmp_path, "M5_full", 0, [9.0])
    out = _run(tmp_path, ["M5_full"], 1)
    assert set(out) == {"M5_full"}
