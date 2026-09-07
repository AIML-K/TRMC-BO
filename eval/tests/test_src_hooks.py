"""The hooks added to `src/bayesian_optimization.py` must be invisible by default.

Every one of them is `getattr`-guarded so the production pipeline keeps its exact
behaviour. There is no test suite in `src/`, so these guard the contract from
this side: if a default ever flips, the formulation pipeline silently changes
behaviour and nothing else would catch it.

    PYTHONPATH=core:. .venv-eval/bin/python -m pytest eval/tests -q
"""

from __future__ import annotations

import torch

from bayesian_optimization import (
    _free_dim_mask,
    _raw_candidate_sampler,
)
from sampler import make_samples_drs


class _PlainLoader:
    """Stands in for the real dataloader, which declares neither attribute."""


def test_free_dim_mask_defaults_to_semicontinuous():
    """A loader that says nothing keeps the 'ratio 0 means absent' behaviour."""
    sample = torch.tensor([[0.0, 0.5, 0.0, 0.25]])
    mask = _free_dim_mask(_PlainLoader(), sample)
    assert torch.equal(mask, sample > 0)


def test_free_dim_mask_frees_all_dims_when_opted_out():
    class Loader:
        semicontinuous_inputs = False

    sample = torch.tensor([[0.0, 0.5, 0.0, 0.25]])
    mask = _free_dim_mask(Loader(), sample)
    assert mask.all()
    assert mask.shape == sample.shape


def test_free_dim_mask_collapses_bounds_only_when_enabled():
    """The mask is applied to bounds, so check the effect the caller relies on."""
    search_space = torch.tensor([[1.0, 1.0], [0.0, 0.0]])  # [upper; lower]
    sample = torch.tensor([[0.0, 0.5]])

    locked = search_space.flip([0]) * _free_dim_mask(_PlainLoader(), sample)
    assert torch.equal(locked[:, 0], torch.zeros(2))  # dim 0 pinned to [0, 0]

    class Loader:
        semicontinuous_inputs = False

    free = search_space.flip([0]) * _free_dim_mask(Loader(), sample)
    assert torch.equal(free, search_space.flip([0]))


def test_raw_candidate_sampler_defaults_to_drs():
    assert _raw_candidate_sampler(_PlainLoader()) is make_samples_drs


def test_raw_candidate_sampler_prefers_loader_override():
    sentinel = object()

    class Loader:
        sample_raw_candidates = sentinel

    assert _raw_candidate_sampler(Loader()) is sentinel


# -- C4: the feasibility component must be invisible unless asked for --------

def test_optimize_acqf_on_filtered_points_takes_no_feasibility_predicate_by_default():
    """C4b is opt-in from the call site. If this default ever became a live
    predicate, every existing config would silently start filtering restarts on
    a signal it never asked for."""
    import inspect

    from bayesian_optimization import optimize_acqf_on_filtered_points

    sig = inspect.signature(optimize_acqf_on_filtered_points)
    assert sig.parameters["feas_predicate"].default is None
