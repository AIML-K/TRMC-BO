"""Anubis baseline: the combination rule, the bootstrap fallback, and the crux
of the whole method -- infeasible rows must be excluded from the base
acquisition's GP fit while still contributing to the feasibility classifier's
training labels.

Modeled on `test_comboo.py`: cheap, hermetic checks against the formulas rather
than a full sweep cell.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from eval.methods.anubis import (
    MIN_PER_CLASS,
    AnubisMethod,
    combine_score,
    feasibility_labels,
    fit_feasibility_classifier,
)


# -- the naive combination rule ----------------------------------------------

def test_zero_feasibility_forces_zero_combined_score():
    base = np.array([0.0, 1.0, 5.0, 100.0])
    p_feasible = np.zeros_like(base)
    assert np.allclose(combine_score(p_feasible, base), 0.0)


def test_certain_feasibility_reduces_to_the_base_score():
    base = np.array([0.0, 1.0, 5.0, 100.0])
    p_feasible = np.ones_like(base)
    assert np.allclose(combine_score(p_feasible, base), base)


def test_combination_is_a_plain_elementwise_product():
    p_feasible = np.array([0.2, 0.5, 0.9])
    base = np.array([10.0, 4.0, 2.0])
    assert np.allclose(combine_score(p_feasible, base), p_feasible * base)


# -- bootstrap fallback --------------------------------------------------

def test_fallback_when_only_one_class_is_represented():
    """All observations feasible so far -- no infeasible example to learn a
    boundary from. Must not crash, and must return P(feasible|x) = 1.0."""
    labels = np.ones(6, dtype=int)
    X = np.random.default_rng(0).random((6, 14))
    classifier = fit_feasibility_classifier(X, labels)

    probe = np.random.default_rng(1).random((5, 14))
    p = classifier(probe)
    assert p.shape == (5,)
    assert np.allclose(p, 1.0)


def test_fallback_when_fewer_than_two_total_observations():
    labels = np.array([1])
    X = np.zeros((1, 14))
    classifier = fit_feasibility_classifier(X, labels)
    p = classifier(np.zeros((3, 14)))
    assert np.allclose(p, 1.0)


def test_fallback_boundary_is_min_per_class():
    """Exactly MIN_PER_CLASS - 1 of the minority class still falls back;
    MIN_PER_CLASS of each does not (and does not crash trying to fit)."""
    assert MIN_PER_CLASS == 2  # pinned so the boundary below stays meaningful

    rng = np.random.default_rng(2)
    X = rng.random((3, 4))
    labels = np.array([1, 1, 0])  # only one infeasible example
    classifier = fit_feasibility_classifier(X, labels)
    assert np.allclose(classifier(rng.random((4, 4))), 1.0)

    X2 = rng.random((4, 4))
    labels2 = np.array([1, 1, 0, 0])  # two of each -> real fit, no crash
    classifier2 = fit_feasibility_classifier(X2, labels2)
    p = classifier2(rng.random((4, 4)))
    assert p.shape == (4,)
    assert np.all((p >= 0.0) & (p <= 1.0))


# -- feasibility labels ---------------------------------------------------

def test_feasibility_label_is_true_if_any_output_was_observed():
    """A censored-but-feasible row (one NaN output) must still label as
    feasible; only an all-NaN row is infeasible -- see the module docstring's
    'What counts as a feasibility label' section."""
    mask = torch.tensor(
        [
            [True, True],    # fully observed -> feasible
            [True, False],   # censored second output -> still feasible
            [False, False],  # nothing observed -> infeasible
        ]
    )
    labels = feasibility_labels(mask)
    assert labels.tolist() == [1, 1, 0]


# -- the crux: infeasible rows excluded from the GP fit, kept for labels -----

class _Vars:
    """Minimal stand-in for `BenchmarkVariables`, carrying only what
    `AnubisMethod._fit_base_acquisition` and `propose` read."""

    def __init__(self, train_x, train_y, mask, item_bound, restart_pool=None):
        self.train_x = train_x
        self.train_y = train_y
        self.continuous_value_mask = mask
        self.item_bound = item_bound
        self.y_key = ["bandgap", "m_star"]
        self.target_info = {
            "bandgap": {"obj": "range", "lb": 0.75, "ub": 1.75},
            "m_star": {"obj": "range", "lb": 0.0, "ub": 4.0},
        }
        self.weight = torch.ones(2, dtype=torch.double)
        self._restart_pool = restart_pool

        class _Task:
            dim = train_x.shape[-1]

        self.task = _Task()

    def to_raw(self, Z):
        lo, hi = self.item_bound.cpu().numpy()
        return np.asarray(Z, dtype=float) * (hi - lo) + lo


def _toy_variables(n_infeasible=2, n_feasible=5, d=3, seed=0):
    rng = np.random.default_rng(seed)
    n = n_infeasible + n_feasible
    X = torch.as_tensor(rng.random((n, d)), dtype=torch.double)
    Y = torch.full((n, 2), float("nan"), dtype=torch.double)
    # Feasible rows get real-looking values inside/near the window.
    Y[n_infeasible:, 0] = torch.as_tensor(rng.uniform(0.8, 1.7, size=n_feasible))
    Y[n_infeasible:, 1] = torch.as_tensor(rng.uniform(0.1, 3.9, size=n_feasible))
    mask = ~torch.isnan(Y)
    item_bound = torch.stack(
        [torch.zeros(d, dtype=torch.double), torch.ones(d, dtype=torch.double)]
    )
    return _Vars(X, Y, mask, item_bound), n_infeasible, n_feasible


def test_infeasible_rows_are_excluded_from_the_base_acquisition_gp_fit():
    variables, n_infeasible, n_feasible = _toy_variables()

    acq_function = AnubisMethod._fit_base_acquisition(variables)
    model = acq_function.model  # ModelListGP, one SingleTaskGP per output
    for sub_model in model.models:
        n_trained = sub_model.train_inputs[0].shape[0]
        assert n_trained == n_feasible, (
            f"expected only the {n_feasible} feasible rows in the GP fit, "
            f"got {n_trained} (infeasible rows leaked in)"
        )


def test_infeasible_rows_still_contribute_feasibility_labels():
    variables, n_infeasible, n_feasible = _toy_variables()
    labels = feasibility_labels(variables.continuous_value_mask)
    assert labels.sum() == n_feasible
    assert (labels == 0).sum() == n_infeasible
    assert len(labels) == n_infeasible + n_feasible


def test_propose_runs_end_to_end_on_a_small_catalogue():
    """Full `propose()` path: fit classifier + base acquisition, score the
    catalogue, return q raw designs of the right shape."""
    d = 3
    variables, _, _ = _toy_variables(n_infeasible=3, n_feasible=6, d=d, seed=1)
    rng = np.random.default_rng(3)
    pool = rng.random((20, d))  # normalized catalogue
    variables._restart_pool = pool

    method = AnubisMethod(num_restarts=8)
    out = method.propose(variables, q=2, seed=0)
    assert out.shape == (2, d)
    assert np.isfinite(out).all()


def test_propose_raises_without_a_restart_pool():
    variables, _, _ = _toy_variables()
    variables._restart_pool = None
    with pytest.raises(ValueError, match="restart_pool"):
        AnubisMethod(num_restarts=4).propose(variables, q=1, seed=0)


def test_propose_falls_back_to_feasibility_only_with_no_feasible_observations():
    """Every row NaN -> nothing to fit a base acquisition on; the score must
    still be well-defined (feasibility alone) rather than crashing."""
    d = 3
    variables, _, _ = _toy_variables(n_infeasible=4, n_feasible=0, d=d, seed=2)
    variables._restart_pool = np.random.default_rng(4).random((10, d))

    out = AnubisMethod(num_restarts=4).propose(variables, q=1, seed=0)
    assert out.shape == (1, d)
    assert np.isfinite(out).all()


def test_unexplored_mask_flags_only_unmatched_rows():
    from eval.methods.anubis import _unexplored_mask

    observed = np.array([[0.1, 0.2], [0.5, 0.5]])
    pool = np.array([[0.1, 0.2], [0.9, 0.9], [0.5, 0.5 + 1e-12]])
    mask = _unexplored_mask(pool, observed)
    assert mask.tolist() == [False, True, False]


def test_unexplored_mask_is_all_true_with_no_observations_yet():
    from eval.methods.anubis import _unexplored_mask

    pool = np.array([[0.1, 0.2], [0.9, 0.9]])
    mask = _unexplored_mask(pool, np.empty((0, 2)))
    assert mask.tolist() == [True, True]


def test_propose_does_not_get_stuck_when_scores_are_degenerate():
    """Regression: with zero feasible observations, every catalogue row scores
    identically (the bootstrap classifier returns a constant P(feasible)=1.0),
    so a plain argsort would return the exact same top-q indices every call.
    Since HOIP's oracle is a deterministic lookup, re-proposing the same
    already-infeasible material teaches the method nothing -- it must instead
    move on to unexplored catalogue rows, or the campaign is stuck at zero
    feasible observations for its entire budget."""
    d = 3
    variables, _, _ = _toy_variables(n_infeasible=4, n_feasible=0, d=d, seed=2)
    variables._restart_pool = np.random.default_rng(4).random((10, d))

    method = AnubisMethod(num_restarts=4)
    proposed = []
    for _ in range(5):
        out = method.propose(variables, q=1, seed=0)
        proposed.append(tuple(out[0].tolist()))
        # Simulate the loop recording this (still-infeasible) observation.
        new_x = torch.as_tensor(out, dtype=variables.train_x.dtype)
        new_y = torch.full((1, 2), float("nan"), dtype=variables.train_y.dtype)
        variables.train_x = torch.cat([variables.train_x, new_x], dim=0)
        variables.train_y = torch.cat([variables.train_y, new_y], dim=0)
        variables.continuous_value_mask = ~torch.isnan(variables.train_y)

    assert len(set(proposed)) == len(proposed), (
        f"proposed the same design more than once with an unexhausted "
        f"catalogue: {proposed}"
    )


def test_propose_falls_back_to_repeats_only_once_the_catalogue_is_exhausted():
    """The opposite edge: once every catalogue row has been observed, there is
    nothing left to explore, and `propose()` must still return a design
    (a repeat) rather than crash on an empty candidate set."""
    d = 2
    variables, _, _ = _toy_variables(n_infeasible=3, n_feasible=0, d=d, seed=5)
    pool = np.array([[0.2, 0.3], [0.7, 0.8]])
    variables._restart_pool = pool
    # Mark both catalogue rows as already observed (infeasible).
    variables.train_x = torch.as_tensor(pool, dtype=variables.train_x.dtype)
    variables.train_y = torch.full((2, 2), float("nan"), dtype=variables.train_y.dtype)
    variables.continuous_value_mask = ~torch.isnan(variables.train_y)

    out = AnubisMethod(num_restarts=4).propose(variables, q=1, seed=0)
    assert out.shape == (1, d)
    assert np.isfinite(out).all()


def test_make_returns_the_registered_name():
    from eval.methods.anubis import make

    method = make()
    assert method.name == "M9_anubis"
    method2 = make("M9_anubis", num_restarts=16)
    assert method2.num_restarts == 16
