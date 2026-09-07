"""The C1-off factorial arms must be the extremum method plus C2/C3, exactly.

Two failure modes are silent -- the run completes and the numbers look
plausible -- so they are asserted here rather than argued:

1. C1 does not actually switch off, or switching it off takes C2/C3 with it.
   `verify_variants` reports this per-suite from a live campaign step; the
   registry half of it is cheap enough to assert here too.
2. The direction of every 'min' output is inverted. `run_mobo` negates 'min'
   columns *before* the acquisition is built, but only under
   `objective_mode="native_extremum"`; these arms run under "range", where
   nothing is negated, and carry the direction in the objective weights instead.
   Every task in all three suites has at least one 'less' output, so an
   inverted sign would not be a corner case -- it would be every run.

    PYTHONPATH=core:. .venv-eval/bin/python -m pytest eval/tests/test_factorial.py -q
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from bayesian_optimization import get_acqf, initialize_model, negate_y, standardize
from botorch import fit_gpytorch_mll

from eval.methods import FACTORIAL, REGISTRY
from eval.methods import make as make_method
from eval.methods.factorial import make_extremum_acqf_factory

DTYPE = torch.double

#: Both acquisitions are 1024-sample quasi-MC estimates drawn from independently
#: seeded samplers, so their values differ by a few percent even when they are
#: the same function. Anything looser than this stops discriminating a sign flip;
#: `test_a_flipped_sign_would_be_caught` is what holds it honest.
_MC_TOL = 0.10


def _rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


def _fit(train_x, train_y):
    mask = ~torch.isnan(train_y)
    mll, model = initialize_model(train_x, train_y, [], mask)
    fit_gpytorch_mll(mll)
    return model


def _paths(n_out: int, seed: int = 0):
    """Build the same problem twice: M1's negate-the-data path, and ours.

    Returns `(acqf_native, acqf_ours, X_probe)`. The two acquisitions are
    mathematically the same function if the sign handling is right.
    """
    torch.manual_seed(seed)
    y_key = [f"y{i}" for i in range(n_out)]
    train_x = torch.rand(24, 3, dtype=DTYPE)

    # Deliberately asymmetric scales and offsets: a sign error that happened to
    # cancel on centred, unit-variance outputs would survive an easier fixture.
    raw = torch.stack(
        [(train_x.sum(-1) * (i + 2) + 5.0 * i + torch.randn(24, dtype=DTYPE) * 0.1)
         for i in range(n_out)],
        dim=-1,
    )
    mask = torch.ones_like(raw, dtype=torch.bool)
    y_std_t, y_mean, y_std = standardize(raw, mask)

    # Output 0 is maximized, every other output minimized -- the shape the real
    # tasks have (MatFormBench is greater/less/less, HOIP is less/less).
    signs = [1.0] + [-1.0] * (n_out - 1)
    native_info = {
        k: {"obj": "max" if signs[i] > 0 else "min", "weight": 1.0}
        for i, k in enumerate(y_key)
    }
    weight = torch.ones(n_out, dtype=DTYPE)

    # Path A -- what M1 runs: negate the 'min' columns, fit on that, plain get_acqf.
    y_native = negate_y(y_std_t.clone(), y_key, native_info)
    acqf_native = get_acqf(
        _fit(train_x, y_native), train_x, y_native, y_key, weight,
        native_info, y_mean, y_std,
    )

    # Path B -- what the factorial arms run: no negation (objective_mode="range"),
    # direction carried in the objective weights.
    range_info = {
        k: {"obj": "range", "lb": -1e6, "ub": 1e6, "weight": 1.0} for k in y_key
    }
    factory = make_extremum_acqf_factory(signs)
    acqf_ours = factory(
        _fit(train_x, y_std_t), train_x, y_std_t, y_key, weight,
        range_info, y_mean, y_std,
    )

    torch.manual_seed(seed + 1)
    return acqf_native, acqf_ours, torch.rand(40, 1, 3, dtype=DTYPE)


@pytest.mark.parametrize("n_out", [1, 2, 3])
def test_extremum_factory_matches_the_negated_data_path(n_out):
    """Our acquisition ranks candidates the way M1's does, on every output arity.

    Compared by rank rather than by value: both are quasi-MC estimates drawn
    from independently seeded samplers, so the values carry sampling noise while
    the ordering -- which is all the optimizer consumes -- does not.
    """
    acqf_native, acqf_ours, X = _paths(n_out)
    with torch.no_grad():
        a = acqf_native(X).numpy()
        b = acqf_ours(X).numpy()

    corr = _rank_corr(a, b)
    assert corr > 0.95, f"rank correlation {corr:.3f} for {n_out} outputs"

    # And the candidate ours picks -- the only one a q=1 campaign keeps -- is one
    # the native path also rates at the top. Compared by value within tolerance
    # rather than by index: the leaders are often a near-tie that MC noise
    # reorders (0.3143 against 0.3100 at three outputs), which says nothing about
    # the sign handling this test exists to check.
    assert a[int(np.argmax(b))] >= a.max() * (1.0 - _MC_TOL), (
        f"our pick scores {a[int(np.argmax(b))]:.4f} against a best of {a.max():.4f}"
    )


@pytest.mark.parametrize("n_out", [1, 2, 3])
def test_a_flipped_sign_would_be_caught(n_out):
    """The fixture is sharp enough to fail if the direction were inverted.

    Without this, `test_extremum_factory_matches...` could pass on a problem
    where direction happens not to matter, and would then be guarding nothing.
    """
    acqf_native, _, X = _paths(n_out)
    flipped = make_extremum_acqf_factory([-1.0] + [1.0] * (n_out - 1))
    # Rebuild path B's inputs with the wrong signs.
    torch.manual_seed(0)
    y_key = [f"y{i}" for i in range(n_out)]
    train_x = torch.rand(24, 3, dtype=DTYPE)
    raw = torch.stack(
        [(train_x.sum(-1) * (i + 2) + 5.0 * i + torch.randn(24, dtype=DTYPE) * 0.1)
         for i in range(n_out)],
        dim=-1,
    )
    y_std_t, y_mean, y_std = standardize(raw, torch.ones_like(raw, dtype=torch.bool))
    range_info = {
        k: {"obj": "range", "lb": -1e6, "ub": 1e6, "weight": 1.0} for k in y_key
    }
    acqf_bad = flipped(
        _fit(train_x, y_std_t), train_x, y_std_t, y_key,
        torch.ones(n_out, dtype=DTYPE), range_info, y_mean, y_std,
    )
    with torch.no_grad():
        a = acqf_native(X).numpy()
        bad = acqf_bad(X).numpy()
    # Both criteria the passing test uses must reject the inverted arm, or the
    # tolerance in that test has been loosened past the point of guarding anything.
    corr = _rank_corr(a, bad)
    picked = a[int(np.argmax(bad))]
    assert corr < 0.95 or picked < a.max() * (1.0 - _MC_TOL), (
        f"an inverted sign scored corr={corr:.3f}, pick={picked:.4f} of "
        f"{a.max():.4f} -- the fixture is blind"
    )


def test_factorial_covers_all_eight_cells():
    assert len(FACTORIAL) == 8
    assert set(FACTORIAL) == {
        (c1, c2, c3) for c1 in (True, False) for c2 in (True, False)
        for c3 in (True, False)
    }
    for name in FACTORIAL.values():
        assert name in REGISTRY, name


def test_component_labels_match_the_factorial_key():
    """`components` is what the ablation tables key on, so it must not drift."""
    for (c1, c2, c3), name in FACTORIAL.items():
        if name == "M1_qehvi_extremum":
            continue  # its own module's spec; covered by test_fairness
        method = make_method(name)
        parts = method.spec.components.split("/")
        assert (parts[0] == "C1") is c1, name
        assert (parts[1] == "C2") is c2, name
        assert (parts[2] == "C3") is c3, name
