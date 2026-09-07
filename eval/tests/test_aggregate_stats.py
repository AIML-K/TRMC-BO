"""The statistics added for the paper's Table 1: pairing and order sensitivity.

Both answer a reviewer's question rather than a scientific one, so both have to
be right for a reason that is checkable here rather than argued in the text.

    PYTHONPATH=core:. .venv-eval/bin/python -m pytest eval/tests/test_aggregate_stats.py -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eval.harness import aggregate as agg


def _counts(rows):
    """rows: (task_id, width, seed, method, distinct)"""
    return pd.DataFrame(
        [{"task_id": t, "width": w, "seed": s, "method": m, "valid": d, "distinct": d}
         for t, w, s, m, d in rows]
    )


# -- pairing ---------------------------------------------------------------

def test_pairing_counts_wins_per_cell_not_per_mean():
    """The case the difference of means gets wrong.

    `B` wins four cells narrowly and loses one enormously, so it has the better
    record and the worse mean. A reader choosing a method for one campaign wants
    the record; a table printing only the mean hands them the opposite answer.
    """
    rows = []
    for seed, (a, b) in enumerate([(1, 2), (1, 2), (1, 2), (1, 2), (40, 1)]):
        rows += [("T", "medium", seed, "A", a), ("T", "medium", seed, "B", b)]
    out = agg.paired_comparison(_counts(rows), reference="B").iloc[0]

    assert (out["win"], out["tie"], out["loss"]) == (4, 0, 1)
    assert out["mean_diff"] < 0  # B's mean is worse ...
    assert out["win"] > out["loss"]  # ... while its record is better


def test_ties_are_excluded_from_the_test_but_kept_in_n_pairs():
    rows = []
    for seed in range(10):
        d = 3 if seed < 8 else (5, 3)[0]
        rows += [("T", "medium", seed, "A", 3), ("T", "medium", seed, "B", 3)]
    rows += [("T", "wide", 0, "A", 1), ("T", "wide", 0, "B", 4)]
    out = agg.paired_comparison(_counts(rows), reference="B").iloc[0]
    assert out["tie"] == 10
    assert out["n_pairs"] == 11
    assert (out["win"], out["loss"]) == (1, 0)


def test_only_cells_both_methods_ran_are_paired():
    """A method missing a task must not be compared on the tasks it skipped.

    COMBOO is the live case: it does not run on two of the five MatFormBench
    tasks, so pairing it over all five would silently score it on cells it never
    saw.
    """
    rows = [("T1", "medium", 0, "A", 5), ("T1", "medium", 0, "B", 1),
            ("T2", "medium", 0, "B", 9)]
    out = agg.paired_comparison(_counts(rows), reference="B").iloc[0]
    assert out["n_pairs"] == 1
    assert out["loss"] == 1  # B lost the one cell they share


def test_unknown_reference_raises_rather_than_returning_empty():
    with pytest.raises(KeyError):
        agg.paired_comparison(_counts([("T", "w", 0, "A", 1)]), reference="nope")


# -- order sensitivity -----------------------------------------------------

def test_a_dense_cluster_counts_once_under_every_strategy():
    Z = np.array([[0.5, 0.5], [0.501, 0.5], [0.5, 0.502], [0.499, 0.501]])
    for strategy in ("greedy", "shuffled", "maximal"):
        assert agg._distinct_count(Z, 0.1, strategy) == 1.0


def test_greedy_order_dependence_is_real_and_the_bound_sees_past_it():
    """A configuration where arrival order changes the greedy count.

    Three points on a line, spaced 0.6 apart with delta=1.0: taking the middle
    one first blocks both ends for a count of 1, while taking an end first
    yields 2. If this fixture ever stopped discriminating, `order_sensitivity`
    would be reporting three copies of the same number.
    """
    Z = np.array([[0.6], [0.0], [1.2]])  # middle point arrives first
    assert agg._distinct_count(Z, 1.0, "greedy") == 1.0
    assert agg._distinct_count(Z, 1.0, "maximal") == 2.0
    shuffled = agg._distinct_count(Z, 1.0, "shuffled")
    assert 1.0 < shuffled < 2.0


def test_maximal_is_never_below_shuffled_which_is_never_above_it():
    rng = np.random.default_rng(0)
    for _ in range(5):
        Z = rng.random((25, 3))
        g = agg._distinct_count(Z, 0.35, "greedy")
        sh = agg._distinct_count(Z, 0.35, "shuffled")
        mx = agg._distinct_count(Z, 0.35, "maximal")
        assert sh <= mx
        assert g <= mx


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        agg._distinct_count(np.zeros((2, 2)), 0.1, "best")


# -- constraint activity ---------------------------------------------------

class _Task:
    task_id = "T"
    x_names = ["a"]


def _write_trace(root, width, method, seed, n_loop, dropped=None, starved=None):
    d = root / "suite" / "proto" / "T" / width / method / f"seed{seed:03d}"
    d.mkdir(parents=True, exist_ok=True)
    frame = {
        "iteration": list(range(n_loop)),
        "is_init": [False] * n_loop,
        "hit": [False] * n_loop,
        "x_a": [0.0] * n_loop,
    }
    if dropped is not None:
        frame["cnt_hard_constraint_dropped"] = dropped
    if starved is not None:
        frame["cnt_restart_filter_exhausted"] = starved
    pd.DataFrame(frame).to_parquet(d / "trace.parquet")


def test_the_two_inactive_paths_are_both_counted(tmp_path):
    """The bug this function exists to catch.

    `M5_full` records zero `hard_constraint_dropped` on every suite, which reads
    as a perfectly applied constraint. It is not: when C3 finds no in-band start,
    C2 is never *attached*, so it can never be *dropped*. Counting only the first
    path reports 100% active for a method that ran unconstrained a third of the
    time.
    """
    _write_trace(tmp_path, "medium", "M5_full", 0, 10,
                 dropped=[0] * 10, starved=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0])
    out = agg.constraint_activity(_Task(), root=tmp_path, suite="suite",
                                  protocol="proto")
    row = out.iloc[0]
    assert row["dropped_rate"] == 0.0        # the misleading signal on its own
    assert row["starved_rate"] == pytest.approx(0.3)
    assert row["active"] == pytest.approx(0.7)


def test_a_method_with_no_constraint_is_not_reported_as_fully_active():
    """`M3_range_prob` has no C2, so 'the constraint always held' is not a fact
    about it. Reporting 1.0 would put it top of a table it does not belong in."""
    assert "M3_range_prob" not in agg.C2_METHODS
    assert "M5_full" in agg.C2_METHODS


def test_methods_without_c2_get_nan_not_one(tmp_path):
    _write_trace(tmp_path, "medium", "M3_range_prob", 0, 10)
    out = agg.constraint_activity(_Task(), root=tmp_path, suite="suite",
                                  protocol="proto")
    assert np.isnan(out.iloc[0]["active"])


def test_absent_counter_columns_mean_the_counter_never_fired(tmp_path):
    """The recorder only emits keys it was asked to note, so a clean run has no
    column at all -- which must read as zero, not as missing data."""
    _write_trace(tmp_path, "medium", "M16_constraints_starts", 0, 10)
    out = agg.constraint_activity(_Task(), root=tmp_path, suite="suite",
                                  protocol="proto")
    assert out.iloc[0]["active"] == pytest.approx(1.0)


def test_init_rows_are_excluded(tmp_path):
    """Init designs are drawn before any acquisition, so they cannot have had a
    constraint applied and must not dilute the denominator."""
    d = tmp_path / "suite" / "proto" / "T" / "medium" / "M5_full" / "seed000"
    d.mkdir(parents=True)
    pd.DataFrame({
        "iteration": list(range(8)),
        "is_init": [True] * 4 + [False] * 4,
        "hit": [False] * 8,
        "x_a": [0.0] * 8,
        "cnt_restart_filter_exhausted": [np.nan] * 4 + [1, 1, 0, 0],
    }).to_parquet(d / "trace.parquet")
    out = agg.constraint_activity(_Task(), root=tmp_path, suite="suite",
                                  protocol="proto")
    assert out.iloc[0]["active"] == pytest.approx(0.5)  # 2 of 4 loop iterations


def test_paired_p_values_are_holm_corrected():
    """One reference against many methods is a family of tests, and the smallest
    raw p in a dozen comparisons is not evidence on its own."""
    rows = []
    # A: reference wins every cell (tiny p). B: a near-tie (large p).
    for seed in range(12):
        rows += [("T", "medium", seed, "ref", 5),
                 ("T", "medium", seed, "A", 1),
                 ("T", "medium", seed, "B", 5 if seed % 2 else 4)]
    out = agg.paired_comparison(_counts(rows), reference="ref").set_index("method")
    assert "p_holm" in out.columns
    assert (out["p_holm"] >= out["p"] - 1e-12).all(), "correction must not shrink p"
    assert (out["p_holm"] <= 1.0).all()


def test_holm_is_monotone_in_the_raw_p_order():
    """Holm's step-down must never rank a larger raw p below a smaller one."""
    rows = []
    for seed in range(10):
        rows += [("T", "medium", seed, "ref", 9)]
        for j, gap in enumerate([0, 1, 3, 8]):
            rows.append(("T", "medium", seed, f"m{j}", 9 - gap))
    out = agg.paired_comparison(_counts(rows), reference="ref")
    s = out.sort_values("p")
    assert list(s["p_holm"]) == sorted(s["p_holm"]), "p_holm not monotone"


# -- distinct_average_rank --------------------------------------------------

def _dcounts(rows):
    """rows: (task_id, width, method, seed, distinct)"""
    return pd.DataFrame(
        [{"task_id": t, "width": w, "method": m, "seed": s, "distinct": d}
         for t, w, m, s, d in rows]
    )


def test_distinct_average_rank_orders_by_the_diversity_metric():
    """A method that always wins on distinct designs must rank first, even
    though it would rank last by hit rate (the point of the reversal claim)."""
    rows = []
    for seed in range(3):
        rows += [("T", "medium", "A", seed, 10), ("T", "medium", "B", seed, 1)]
    out = agg.distinct_average_rank(_dcounts(rows)).set_index("method")
    assert out.loc["A"]["avg_rank"] == 1.0
    assert out.loc["B"]["avg_rank"] == 2.0


def test_distinct_average_rank_averages_over_task_and_width_cells():
    """A method winning half its cells and losing the other half should land
    near the middle, matching average_rank's own cell-then-mean logic."""
    rows = [
        ("T1", "wide", "A", 0, 5), ("T1", "wide", "B", 0, 1),
        ("T2", "wide", "A", 0, 1), ("T2", "wide", "B", 0, 5),
    ]
    out = agg.distinct_average_rank(_dcounts(rows)).set_index("method")
    assert out.loc["A"]["avg_rank"] == 1.5
    assert out.loc["B"]["avg_rank"] == 1.5


def test_distinct_average_rank_does_not_require_hit_rate_columns():
    """The whole point: this must run on a bare distinct_counts()-shaped frame
    with no found/first_hit/hit_rate columns, which average_rank() requires."""
    rows = [("T", "medium", "A", 0, 3)]
    df = _dcounts(rows)
    assert not {"found", "first_hit", "hit_rate"} & set(df.columns)
    out = agg.distinct_average_rank(df)
    assert len(out) == 1
