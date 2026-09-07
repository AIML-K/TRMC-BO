"""C4 -- the feasibility signal grafted onto the proposed method.

Two things are being pinned here.

**The mechanism.** `src/utils/feasibility.py` holds the classifier, the label
rule, the naive product and the two guards that stop the product from inverting
or flattening the ranking. Cheap hermetic checks against the formulas, in the
style of `test_comboo.py` / `test_anubis.py`, rather than a sweep cell.

**The absence of the mechanism.** C4 is opt-in, and `M3`/`M4`/`M5` results
already on disk are only valid if it is genuinely inert when switched off. The
`run_mobo` identity test at the bottom is the one that matters: same inputs, C4
off, byte-identical candidates.

    PYTHONPATH=core:. .venv-eval/bin/python -m pytest eval/tests/test_feasibility.py -q
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from bayesian_optimization import merge_similar_candidates, run_mobo
from utils.feasibility import (
    DEFAULT_KEEP_FRACTION,
    MIN_PER_CLASS,
    combine_score,
    compose_predicates,
    feasibility_labels,
    feasibility_start_predicate,
    feasibility_weighted_scores,
    fit_feasibility_classifier,
    has_both_classes,
    resolve_feasibility_labels,
)


# -- one classifier, shared with the M9 baseline -----------------------------

def test_the_anubis_baseline_uses_the_same_classifier_object():
    """M10-M12 vs M9 is meant to isolate *where* the feasibility signal is
    applied. That only holds if both sides call the same code, so the baseline
    must re-export these rather than keep its own copy."""
    from eval.methods import anubis

    assert anubis.combine_score is combine_score
    assert anubis.feasibility_labels is feasibility_labels
    assert anubis.fit_feasibility_classifier is fit_feasibility_classifier
    assert anubis.MIN_PER_CLASS is MIN_PER_CLASS


# -- labels ------------------------------------------------------------------

def test_labels_default_to_the_nan_proxy():
    class Loader:
        continuous_value_mask = torch.tensor(
            [[True, True], [True, False], [False, False]]
        )

    assert resolve_feasibility_labels(Loader()).tolist() == [1, 1, 0]


def test_an_explicit_feasibility_column_overrides_the_nan_proxy():
    """A production dataloader where "no output observed" means "not measured"
    rather than "could not be made" can supply the real flag; the NaN proxy is
    only the fallback for loaders that expose nothing else."""

    class Loader:
        continuous_value_mask = torch.tensor([[False, False], [True, True]])
        feasibility_labels = [1, 0]  # deliberately the opposite of the proxy

    assert resolve_feasibility_labels(Loader()).tolist() == [1, 0]


def test_has_both_classes_matches_the_classifier_fallback_boundary():
    assert not has_both_classes(np.array([1, 1, 1, 0]))
    assert has_both_classes(np.array([1, 1, 0, 0]))


# -- the ranking key: sign guard --------------------------------------------

def test_weighted_score_is_the_plain_product_for_nonnegative_acquisitions():
    acq = np.array([10.0, 4.0, 2.0])
    p = np.array([0.2, 0.5, 0.9])
    assert np.allclose(feasibility_weighted_scores(acq, p), combine_score(p, acq))


def test_negative_acquisitions_do_not_invert_the_ranking():
    """A plain product would rank the *least* feasible point first once the
    acquisition goes negative: -10 * 0.1 = -1 beats -10 * 0.9 = -9. Clamping is
    what stops a caller-supplied `_acqf_factory` from silently reversing C4."""
    acq = np.array([-10.0, -10.0])
    p = np.array([0.1, 0.9])

    naive = combine_score(p, acq)
    assert naive[0] > naive[1]  # the inversion this guard exists to prevent

    # Clamping flattens these two to 0, so the flat guard takes over and ranks
    # by feasibility. Either way the inversion is gone: the more feasible point
    # is no longer ranked last.
    guarded = feasibility_weighted_scores(acq, p)
    assert guarded[1] > guarded[0]


def test_negative_acquisitions_are_recorded_not_swallowed():
    notes = []

    class _Recorder:
        def note(self, key, value=1.0):
            notes.append(key)

    feasibility_weighted_scores(np.array([-1.0, 2.0]), np.array([0.5, 0.5]), _Recorder())
    assert "feas_weight_negative_acqf" in notes


def test_a_partially_negative_acquisition_keeps_the_positive_ordering():
    acq = np.array([-5.0, 1.0, 4.0])
    p = np.array([0.9, 0.9, 0.1])
    scores = feasibility_weighted_scores(acq, p)
    assert scores.tolist() == [0.0, pytest.approx(0.9), pytest.approx(0.4)]


# -- the ranking key: flat guard ---------------------------------------------

def test_a_flat_acquisition_falls_back_to_feasibility_alone():
    """At m == 1 the range-probability objective is deterministic in x, so qEI
    is exactly 0 everywhere once any in-window point has been observed. The
    product is then 0 everywhere and carries no ordering; feasibility is the
    only signal left."""
    acq = np.zeros(4)
    p = np.array([0.1, 0.9, 0.4, 0.6])
    assert np.allclose(feasibility_weighted_scores(acq, p), p)


def test_a_flat_nonzero_acquisition_also_falls_back():
    acq = np.full(3, 2.5)
    p = np.array([0.2, 0.8, 0.5])
    assert np.allclose(feasibility_weighted_scores(acq, p), p)


def test_the_flat_case_is_recorded():
    notes = []

    class _Recorder:
        def note(self, key, value=1.0):
            notes.append(key)

    feasibility_weighted_scores(np.zeros(3), np.array([0.1, 0.5, 0.9]), _Recorder())
    assert "feas_weight_acqf_degenerate" in notes


def test_an_almost_flat_acquisition_is_still_a_product():
    """The guard triggers on zero spread, not on small spread -- a genuinely
    tiny difference is still information."""
    acq = np.array([1.0, 1.0 + 1e-12])
    p = np.array([0.9, 0.1])
    scores = feasibility_weighted_scores(acq, p)
    assert scores[0] > scores[1]


# -- C4b: the restart predicate ----------------------------------------------

def _constant_classifier(value):
    return lambda X: np.full(len(np.atleast_2d(X)), value, dtype=float)


def test_the_start_predicate_keeps_the_top_ranked_fraction():
    p_values = np.array([0.05, 0.1, 0.5, 0.99])

    def classifier(X):
        return p_values[: len(np.atleast_2d(X))]

    predicate = feasibility_start_predicate(classifier, keep_fraction=0.5)
    batch = torch.zeros((4, 1, 3), dtype=torch.double)
    mask = predicate(batch)

    assert mask.dtype == torch.bool
    assert mask.shape == (4,)
    assert mask.tolist() == [False, False, True, True]


def test_selection_is_by_rank_not_by_an_absolute_probability():
    """The whole reason C4b ranks instead of thresholding: the classifier's
    probabilities sit in a narrow, problem-dependent band. These four values
    would all clear any threshold below 0.9 and all fail any threshold above it;
    ranking still separates them."""
    p_values = np.array([0.90, 0.91, 0.92, 0.93])

    predicate = feasibility_start_predicate(lambda X: p_values, keep_fraction=0.5)
    assert predicate(torch.zeros((4, 1, 2), dtype=torch.double)).tolist() == [
        False, False, True, True,
    ]


def test_a_constant_classifier_keeps_every_restart():
    """Measured to be the common case, not an edge case: with too few examples
    of either class the classifier is the constant 1.0, and on MatFormBench/L5-4
    a real fit still came out identically 0.139 across 2,000 draws. Cutting an
    arbitrary half on a tie would discard restarts for no reason, so ties all
    survive."""
    predicate = feasibility_start_predicate(_constant_classifier(0.139))
    assert predicate(torch.zeros((6, 1, 2), dtype=torch.double)).all()


def test_the_default_keeps_half():
    """C4b ANDs onto C3's filter, which already starves for in-band restarts
    (0.0008% admitted on MatFormBench/L5-4, 1.4% on HOIP). C4b biases the pool
    C3 then cuts; it does not compete with it for scarcity."""
    assert DEFAULT_KEEP_FRACTION == 0.5


def test_an_invalid_keep_fraction_is_rejected_at_construction():
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            feasibility_start_predicate(_constant_classifier(0.5), keep_fraction=bad)


def test_keeping_everything_is_expressible():
    predicate = feasibility_start_predicate(
        lambda X: np.linspace(0.1, 0.9, len(X)), keep_fraction=1.0
    )
    assert predicate(torch.zeros((5, 1, 2), dtype=torch.double)).all()


def test_the_start_predicate_flattens_the_q_dimension_like_the_mean_filter():
    """Shape contract must match `_make_mean_range_constraints`'s `is_feasible`:
    the leading dimension indexes candidates, everything else is flattened."""
    seen = {}

    def classifier(X):
        seen["shape"] = X.shape
        return np.ones(len(X))

    feasibility_start_predicate(classifier)(torch.zeros((5, 1, 4), dtype=torch.double))
    assert seen["shape"] == (5, 4)


# -- composing C3 and C4b -----------------------------------------------------

def _mask_predicate(values):
    return lambda batch: torch.as_tensor(values, dtype=torch.bool)


def test_composing_nothing_is_none_not_an_always_true_filter():
    """"No predicate" and "a predicate admitting everything" are different
    states to `_sample_initial_conditions`: the first draws once and skips the
    retry loop entirely."""
    assert compose_predicates(None, None) is None


def test_composing_one_predicate_returns_that_exact_object():
    """C4b off must leave C3's filter untouched -- not merely equivalent, the
    same object, so the code path is provably unchanged."""
    predicate = _mask_predicate([True, False])
    assert compose_predicates(predicate, None) is predicate
    assert compose_predicates(None, predicate) is predicate


def test_composing_two_predicates_is_a_conjunction():
    c3 = _mask_predicate([True, True, False, False])
    c4b = _mask_predicate([True, False, True, False])
    composed = compose_predicates(c3, c4b)
    assert composed(torch.zeros((4, 2))).tolist() == [True, False, False, False]


# -- default-off: the guarantee that protects existing results ---------------

def test_merge_ranks_by_acqf_value_by_default():
    assert (
        merge_similar_candidates.__defaults__[-1] == "acqf_value"
    ), "changing this default would silently re-rank every existing config"


def test_proposed_variants_have_c4_off_by_default():
    from eval.methods.proposed import VARIANTS

    for name in ("M3_range_prob", "M4_range_prob_constraints", "M5_full"):
        spec = VARIANTS[name]
        assert not spec.feasibility_weight
        assert not spec.feasibility_filter
        assert "C4" not in spec.components


def test_the_c4_variants_declare_what_they_switch_on():
    from eval.methods.proposed import VARIANTS

    assert VARIANTS["M10_full_feasw"].components == "C1/C2/C3/C4a"
    assert VARIANTS["M11_full_feasf"].components == "C1/C2/C3/C4b"
    assert VARIANTS["M12_full_feas"].components == "C1/C2/C3/C4a/C4b"


def test_the_c4_variants_are_registered_and_named_consistently():
    from eval.methods import REGISTRY, make

    for name in ("M10_full_feasw", "M11_full_feasf", "M12_full_feas"):
        assert name in REGISTRY
        assert make(name, num_restarts=4).name == name


def test_config_carries_the_c4_flags_underscore_prefixed():
    """`run_mobo` strips underscore keys before splatting the dict into BoTorch,
    which rejects unknown kwargs. A C4 key without the prefix would crash every
    call."""
    from eval.methods.proposed import VARIANTS, ProposedMethod

    variables, *_ = _toy_run_mobo_inputs()
    cfg = ProposedMethod(VARIANTS["M12_full_feas"]).build_optimize_config(variables, q=1)

    for key in ("_use_feasibility_weight", "_use_feasibility_filter",
                "_feasibility_keep_fraction"):
        assert key in cfg
    assert cfg["_use_feasibility_weight"] is True
    assert cfg["_feasibility_keep_fraction"] == pytest.approx(0.5)


# -- the identity test --------------------------------------------------------

class _Loader:
    """Minimal stand-in for the dataloader `run_mobo` reads.

    Deliberately deterministic: `sample_raw_candidates` returns a fixed grid
    rather than a random draw, so two `run_mobo` calls are comparable exactly
    instead of approximately.
    """

    semicontinuous_inputs = False
    candidates_info: dict = {}
    item_key: list = []

    def linear_constraints(self) -> dict:
        return {}

    def __init__(self, train_x, train_y, d):
        self.train_x = train_x
        self.train_y = train_y
        self.continuous_value_mask = ~torch.isnan(train_y)
        self.item_bound = torch.stack(
            [torch.zeros(d, dtype=torch.double), torch.ones(d, dtype=torch.double)]
        )
        # Stored [upper; lower]; every production site applies `.flip([0])`.
        self.search_space = torch.stack(
            [torch.ones(d, dtype=torch.double), torch.zeros(d, dtype=torch.double)]
        )
        self.item_tensor = train_x
        self.y_key = ["p1", "p2"]
        self.target_info = {
            "p1": {"obj": "range", "lb": 0.3, "ub": 0.7, "weight": 1.0},
            "p2": {"obj": "range", "lb": 0.3, "ub": 0.7, "weight": 1.0},
        }
        self._d = d

    def sample_raw_candidates(self, n, dataloader, optimize_config):
        grid = torch.linspace(0.05, 0.95, n, dtype=torch.double)
        return grid.reshape(n, 1, 1).expand(n, 1, self._d).contiguous()


def _toy_run_mobo_inputs(n_infeasible=3, n_feasible=7, d=2, seed=0):
    rng = np.random.default_rng(seed)
    n = n_infeasible + n_feasible
    X = torch.as_tensor(rng.random((n, d)), dtype=torch.double)
    Y = torch.full((n, 2), float("nan"), dtype=torch.double)
    Y[n_infeasible:, 0] = torch.as_tensor(rng.uniform(0.2, 0.8, size=n_feasible))
    Y[n_infeasible:, 1] = torch.as_tensor(rng.uniform(0.2, 0.8, size=n_feasible))
    return _Loader(X, Y, d), X, Y


def _base_config(**extras):
    cfg = {
        "q": 1,
        "num_restarts": 4,
        "options": {"batch_limit": 1},
        "return_best_only": False,
        "merge_results": False,
        "acqf_optimizer": "discrete",  # dispatch is by variable type; this is 'conti'
        "fixed_features_list": [],
        "_use_mean_range_constraints": True,
        "_use_mean_filtered_starts": True,
        "_range_objective_mode": "range_prob",
    }
    cfg.update(extras)
    return cfg


def _run(**extras):
    loader, X, Y = _toy_run_mobo_inputs()
    torch.manual_seed(0)
    df, _model, _fit = run_mobo(X, Y, loader, _base_config(**extras))
    return df


@pytest.mark.filterwarnings("ignore")
def test_c4_off_reproduces_the_pre_c4_candidate_ordering():
    """The guarantee the existing result cells rest on. With C4 switched off the
    ranking key must be the acquisition value itself, so `run_mobo` returns the
    same candidates in the same order it did before C4 existed."""
    absent = _run()
    explicitly_off = _run(
        _use_feasibility_weight=False, _use_feasibility_filter=False,
    )

    assert absent["score"].tolist() == absent["acqf_value"].tolist()
    assert np.allclose(
        np.stack(absent["candidate"].to_numpy()),
        np.stack(explicitly_off["candidate"].to_numpy()),
    )
    # Sorted descending by the acquisition value, exactly as before.
    values = np.asarray(absent["acqf_value"].tolist(), dtype=float)
    assert np.all(np.diff(values) <= 0)


@pytest.mark.filterwarnings("ignore")
def test_c4_off_does_not_fit_a_classifier():
    """Not just "the ranking is the same" but "the classifier never ran": with
    C4 off, `p_feasible` is the constant 1.0 placeholder."""
    df = _run()
    assert df["p_feasible"].tolist() == [1.0] * len(df)


@pytest.mark.filterwarnings("ignore")
def test_c4a_reweights_the_ranking_by_the_learned_feasibility():
    df = _run(_use_feasibility_weight=True)

    p = np.asarray(df["p_feasible"].tolist(), dtype=float)
    acq = np.asarray(df["acqf_value"].tolist(), dtype=float)
    score = np.asarray(df["score"].tolist(), dtype=float)

    # The classifier actually ran (3 infeasible + 7 feasible clears MIN_PER_CLASS).
    assert np.any(p < 1.0)
    assert np.all((p >= 0.0) & (p <= 1.0))
    assert np.allclose(score, feasibility_weighted_scores(acq, p))
    assert np.all(np.diff(score) <= 0)  # sorted by the weighted score now


@pytest.mark.filterwarnings("ignore")
def test_c4b_runs_and_keeps_the_acqf_ranking():
    """C4b changes *where the optimizer starts*, not how candidates are ranked;
    with C4a off the ranking key must still be the raw acquisition value."""
    df = _run(_use_feasibility_filter=True)
    assert df["score"].tolist() == df["acqf_value"].tolist()
    assert len(df) > 0


@pytest.mark.filterwarnings("ignore")
def test_the_bootstrap_fallback_is_recorded_when_no_failure_was_seen_yet():
    """Every observation feasible so far: C4 is inert this iteration. "C4 made
    no difference" and "C4 never actually ran" are different findings, so the
    second one is recorded rather than silent."""
    notes = []

    class _Recorder:
        def note(self, key, value=1.0):
            notes.append(key)

        def note_fit(self, *_a, **_k):
            pass

    loader, X, Y = _toy_run_mobo_inputs(n_infeasible=0, n_feasible=8)
    torch.manual_seed(0)
    run_mobo(X, Y, loader, _base_config(
        _use_feasibility_weight=True, _recorder=_Recorder(),
    ))
    assert "feas_bootstrap_fallback" in notes


@pytest.mark.filterwarnings("ignore")
def test_the_c4_switches_are_stripped_before_botorch_sees_the_config():
    """`run_mobo` mutates the config it is given, popping every underscore key
    before splatting the rest into BoTorch (which rejects unknown kwargs). A C4
    switch that survived that strip would crash `optimize_acqf` on every call."""
    loader, X, Y = _toy_run_mobo_inputs()
    cfg = _base_config(
        _use_feasibility_weight=True, _use_feasibility_filter=True, _feasibility_keep_fraction=0.5,
    )
    torch.manual_seed(0)
    run_mobo(X, Y, loader, cfg)

    assert not [k for k in cfg if k.startswith("_")]
