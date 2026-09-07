"""Coordinate conventions in the adapter.

The two systems (`item_bound` raw and lower-first, `search_space` normalized and
upper-first) are inherited from the Excel loader and are easy to get backwards.
A silent flip would not crash -- it would just optimize a mirrored design space.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from eval.benchmarks.base import BenchmarkTask, NativeTarget, RangeSpec, SimplexGroup
from eval.harness.adapter import BenchmarkVariables


def _box_task(dim=3):
    return BenchmarkTask(
        suite="t", task_id="box", dim=dim,
        x_names=tuple(f"x{i+1}" for i in range(dim)),
        y_names=("y1",),
        bounds=np.stack([np.full(dim, -1.0), np.full(dim, 1.0)]),
        native_targets=(NativeTarget("y1", "greater", 0.0),),
    )


def _simplex_task():
    return BenchmarkTask(
        suite="t", task_id="simplex", dim=5,
        x_names=tuple(f"x{i+1}" for i in range(5)),
        y_names=("y1",),
        bounds=np.stack([np.full(5, -1.0), np.full(5, 1.0)]),
        native_targets=(NativeTarget("y1", "greater", 0.0),),
        simplex_groups=(SimplexGroup(indices=(0, 1, 2), total=1.0, min_component=0.01),),
    )


def _spec():
    return RangeSpec(lower={"y1": 0.0}, upper={"y1": 1.0}, width_label="t")


def _vars(task):
    n = 4
    X = np.zeros((n, task.dim))
    Y = np.zeros((n, 1))
    return BenchmarkVariables(task, _spec(), X, Y)


def test_item_bound_is_raw_and_lower_first():
    v = _vars(_box_task())
    assert torch.equal(v.item_bound[0], torch.full((3,), -1.0, dtype=torch.double))
    assert torch.equal(v.item_bound[1], torch.full((3,), 1.0, dtype=torch.double))


def test_search_space_is_normalized_and_upper_first():
    v = _vars(_box_task())
    assert torch.equal(v.search_space[0], torch.ones(3, dtype=torch.double))
    assert torch.equal(v.search_space[1], torch.zeros(3, dtype=torch.double))
    # What the production code actually consumes.
    lo, hi = v.search_space.flip([0])
    assert torch.equal(lo, torch.zeros(3, dtype=torch.double))
    assert torch.equal(hi, torch.ones(3, dtype=torch.double))


@pytest.mark.parametrize("raw", [-1.0, -0.25, 0.0, 0.5, 1.0])
def test_normalization_roundtrip(raw):
    v = _vars(_box_task())
    X = np.full((1, 3), raw)
    assert np.allclose(v.to_raw(v.to_normalized(X)), X)


def test_simplex_dims_get_tightened_bounds():
    """A composition component can never be negative, whatever the box declares."""
    v = _vars(_simplex_task())
    lo, hi = v.item_bound.cpu().numpy()
    assert list(lo[:3]) == [0.0, 0.0, 0.0]
    assert list(hi[:3]) == [1.0, 1.0, 1.0]
    assert lo[3] == -1.0 and hi[3] == 1.0  # process dims untouched


def test_simplex_normalization_is_identity():
    """Because simplex dims map onto [0, total], sum == 1 survives normalization."""
    v = _vars(_simplex_task())
    X = np.zeros((1, 5))
    X[0, :3] = [0.2, 0.3, 0.5]
    assert np.allclose(v.to_normalized(X)[0, :3], [0.2, 0.3, 0.5])


def test_linear_constraints_encode_simplex_and_floor():
    v = _vars(_simplex_task())
    con = v.linear_constraints()
    (idx, coef, rhs), = con["equality_constraints"]
    assert idx.tolist() == [0, 1, 2] and coef.tolist() == [1.0] * 3 and rhs == 1.0
    assert len(con["inequality_constraints"]) == 3
    assert all(r == pytest.approx(0.01) for _, _, r in con["inequality_constraints"])


def test_box_task_has_no_linear_constraints():
    con = _vars(_box_task()).linear_constraints()
    assert con["equality_constraints"] is None
    assert con["inequality_constraints"] is None


def test_restart_samples_are_normalized_and_respect_simplex():
    task = _simplex_task()
    v = _vars(task)
    Z = v.sample_raw_candidates(64, v, {})
    assert Z.shape == (64, 1, task.dim)
    assert (Z >= 0).all() and (Z <= 1).all()
    comp = Z[:, 0, :3]
    assert torch.allclose(comp.sum(-1), torch.ones(64, dtype=torch.double))
    assert (comp >= 0.01 - 1e-12).all()


def test_nan_responses_are_excluded_from_the_fit_mask():
    task = _box_task()
    v = _vars(task)
    Y = np.array([[1.0], [np.nan], [3.0], [np.nan]])
    v.set_data(np.zeros((4, 3)), Y)
    assert v.continuous_value_mask.squeeze(-1).tolist() == [True, False, True, False]


def test_target_info_carries_the_range_window():
    info = _vars(_box_task()).target_info["y1"]
    assert info["obj"] == "range"
    assert (info["lb"], info["ub"], info["weight"]) == (0.0, 1.0, 1.0)


def test_target_info_falls_back_when_calibration_is_absent():
    """A spec with no frozen stats must still yield a usable ball geometry."""
    info = _vars(_box_task()).target_info["y1"]
    assert info["ball_center"] == 0.5
    assert info["ref_center"] == 0.5
    assert info["ref_scale"] == 1.0


def test_no_categorical_or_discrete_dims_declared():
    """Empty `candidates_info` is what routes run_mobo to its 'conti' branch."""
    assert _vars(_box_task()).candidates_info == {}


def test_display_names_cover_every_registered_method():
    """A method missing from the table labels would print as a bare key."""
    from eval.harness.aggregate import DISPLAY_NAMES, DISPLAY_ORDER
    from eval.methods import REGISTRY

    assert set(DISPLAY_NAMES) == set(REGISTRY)
    assert set(DISPLAY_ORDER) == set(REGISTRY)


def test_proposed_variants_are_labelled_as_one_family():
    from eval.harness.aggregate import METHOD_NAME, label

    ours = ["M3_range_prob", "M4_range_prob_constraints", "M5_full"]
    assert all(label(m).startswith(METHOD_NAME) for m in ours)
    others = ["M0_random", "M1_qehvi_extremum", "M2_target_distance",
              "M6_range_aware_tb", "M6b_range_aware_tb_inscribed"]
    assert not any(label(m).startswith(METHOD_NAME) for m in others)
