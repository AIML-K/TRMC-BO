"""Anubis baseline — BO under *unknown* feasibility constraints.

Hickman, Tom, Zou, Aldeghi & Aspuru-Guzik, *Bayesian optimization with unknown
feasibility constraints for scientific experimentation*, Digital Discovery 2025.
HOIP (`eval/benchmarks/hoip/`) is the perovskite application from that paper: a
composition absent from the near-hull DFT table returns `(nan, nan)`, and
*whether a design can be evaluated at all* is not known ahead of time -- exactly
the problem Anubis exists for. See `eval/README.md`'s "HOIP" section.

## The idea

Some designs cannot be evaluated; that is discovered only after proposing them.
Anubis maintains an online feasibility classifier `P(feasible | x)`, trained on
every design attempted so far (label = whether the oracle returned a real value
or `NaN`), and combines it with an acquisition function computed over the
objective(s) using only the feasible-labeled training data. The source paper
studies several combination rules (naive product, FIA, FCA); **this module
implements only the naive product**,

    score(x) = P(feasible | x) * base_acquisition(x)

and says so here rather than silently narrowing scope. `OVERVIEW.md` S5(e)/S8
independently names exactly this gap -- "binary/feasibility information never
feeds back into the acquisition" -- as a limitation of the production pipeline,
so this baseline also demonstrates what closing that gap looks like.

## Why it is ported rather than installed

The authors ship their method as `atlas`, a library with its own pins. Installing
it would be the same class of risk this project avoids elsewhere: COMBOO's
notebook pins (`eval/methods/comboo.py`) and Olympus's TensorFlow isolation
(`eval/README.md`'s "Olympus" section). So the two moving parts are built
directly on this project's own stack instead:

* the feasibility classifier is `sklearn.gaussian_process.GaussianProcessClassifier`
  with an RBF kernel over the normalized inputs -- the most direct analogue,
  in an already-installed library, of Anubis's own variational-GP classifier;
* the base acquisition is *not* reimplemented. `initialize_model` and `get_acqf`
  (both `src/bayesian_optimization.py`) build exactly the same range-probability
  multi-output objective (`CDFRangeMultiOutputObjective` / `CDFRangeObjective`)
  the proposed method uses -- see `eval/methods/proposed.py`. Only the training
  rows and the final combination are Anubis-specific.

## Two departures from the source paper, both deliberate

**Enumerate and score, not gradient-optimize.** HOIP's design space is a finite
catalogue (`variables._restart_pool`, normalized), exactly the case
`optimize_acqf_on_filtered_points`'s discrete branch in `src/bayesian_optimization.py`
already handles by enumerating candidates and taking the arg-top-q, rather than
running `optimize_acqf`'s gradient-based restarts. The classifier `P(feasible|x)`
also has no obvious gradient-friendly closed form once wrapped around a fitted
`GaussianProcessClassifier`, so scoring every catalogue row sidesteps needing one.
`num_restarts` is accepted by `make()` only for interface consistency with every
other method's factory -- it is unused, because there is nothing to restart:
every candidate is scored exactly once.

**Bootstrap fallback.** `GaussianProcessClassifier.fit` needs both classes
represented to fit anything meaningful (fewer than two examples of either class
makes the boundary undefined, and scikit-learn either errors or returns a
degenerate constant classifier depending on version). Until that data exists,
the deliberate rule implemented here is

    P(feasible | x) = 1.0   for every x

i.e. "no feasibility evidence yet, so do not penalize anything" -- not a silent
degradation, exactly the way `comboo.py`'s "No early stop" section documents its
own edge case. `fit_feasibility_classifier` returns this constant function
whenever fewer than `MIN_PER_CLASS` examples of either class have been observed.

## What counts as a feasibility label

`BenchmarkVariables` does not expose the oracle's `Observation.feasible` flag
directly (see `eval/harness/adapter.py`); methods only ever see `train_y` (raw,
NaN for anything unobserved) and `continuous_value_mask` (`~isnan(train_y)`,
per output). HOIP's own labelling conflates "not in the catalogue" (both outputs
NaN) with "in the catalogue but one output censored" (`eval/README.md`: two
materials carry a censored effective mass, held as NaN because a sentinel is not
a measurement) -- a censored row is still *feasible*. So the label used here is

    feasible_i = continuous_value_mask[i, :].any()

any output observed at all means the design was evaluable, which is exactly the
distinction Anubis's own feasibility flag draws and the only one recoverable
from what `BenchmarkVariables` provides.
"""

from __future__ import annotations

import numpy as np
import torch
from botorch import fit_gpytorch_mll
from botorch.utils.transforms import normalize

from bayesian_optimization import get_acqf, initialize_model, negate_y, standardize

# The classifier, the label rule and the naive-product combination live in
# `src/utils/feasibility.py` so that the production pipeline's C4 component
# (`_use_feasibility_weight` / `_use_feasibility_filter` in `run_mobo`) and this
# baseline share one implementation: a comparison between them then isolates
# *where* the feasibility signal is plugged in rather than *which* classifier it
# is. They are re-exported here because this module is where they were first
# written and where the tests address them. `eval/` imports from `src/`, never
# the reverse, which is why the shared code sits there.
from utils.feasibility import (  # noqa: F401  (re-exported)
    MIN_PER_CLASS,
    combine_score,
    feasibility_labels,
    fit_feasibility_classifier,
)

NAME = "M9_anubis"


def _unexplored_mask(pool: np.ndarray, observed: np.ndarray, atol: float = 1e-9) -> np.ndarray:
    """True for each `pool` row that does not (near-)match any row of
    `observed`, both already in normalized space. A catalogue lookup is
    deterministic, so a previously-observed row carries no further
    information -- see `propose()`'s "Prefer ... never observed" comment for
    why this matters beyond being merely wasteful.
    """
    if len(observed) == 0:
        return np.ones(len(pool), dtype=bool)
    # (n_pool, n_observed) pairwise max-abs-diff; a small catalogue (<=~1300
    # rows here) times <=80 observed rows is cheap enough to broadcast directly.
    close = np.all(np.abs(pool[:, None, :] - observed[None, :, :]) <= atol, axis=-1)
    return ~close.any(axis=-1)


class AnubisMethod:
    """Enumerate the catalogue, score ``P(feasible|x) * base_acquisition(x)``."""

    name = NAME
    #: Range targets are what select `get_acqf`'s CDF-probability path (both the
    #: single- and multi-output branches), which is the base acquisition this
    #: baseline reuses rather than reimplements.
    objective_mode = "range"

    def __init__(self, num_restarts: int = 128):
        # Unused: see the module docstring's "Enumerate and score" section.
        # Kept as a constructor argument so `make()` matches every other
        # method's factory signature.
        self.num_restarts = num_restarts

    # -- base acquisition ----------------------------------------------------

    @staticmethod
    def _fit_base_acquisition(variables):
        """Fit the production multi-output GP and build its acquisition function.

        Not reimplemented: `initialize_model` already fits objective ``j`` only
        on rows where ``continuous_value_mask[:, j]`` is True, which is how an
        infeasible (all-NaN) row is excluded from every output's GP -- and how a
        feasible-but-censored row (`m_star` NaN, `bandgap` observed) still
        contributes to the objective it *does* have. `get_acqf` then builds the
        same range-probability objective `eval/methods/proposed.py` uses.
        """
        train_x_n = normalize(variables.train_x, variables.item_bound)
        train_y_std, y_mean, y_std = standardize(
            variables.train_y, variables.continuous_value_mask
        )
        train_y_std = negate_y(train_y_std, variables.y_key, variables.target_info)

        mll, model = initialize_model(
            train_x_n, train_y_std, [], variables.continuous_value_mask
        )
        fit_gpytorch_mll(mll)

        acq_function = get_acqf(
            model,
            train_x_n,
            train_y_std,
            variables.y_key,
            variables.weight,
            variables.target_info,
            y_mean,
            y_std,
        )
        return acq_function

    # -- Method protocol -------------------------------------------------

    def propose(self, variables, q: int, seed: int, recorder=None) -> np.ndarray:
        pool = variables._restart_pool
        if pool is None:
            raise ValueError(
                "AnubisMethod needs a catalogue benchmark (restart_pool set); "
                "HOIP is the only registered suite that supplies one."
            )
        pool = np.asarray(pool, dtype=float)
        d = variables.task.dim

        train_x_norm = normalize(variables.train_x, variables.item_bound).cpu().numpy()
        labels = feasibility_labels(variables.continuous_value_mask)
        if recorder is not None and (
            int(labels.sum()) < MIN_PER_CLASS or int(len(labels) - labels.sum()) < MIN_PER_CLASS
        ):
            recorder.note("anubis_bootstrap_fallback")

        classifier = fit_feasibility_classifier(train_x_norm, labels)
        p_feasible = classifier(pool)

        if int(labels.sum()) == 0:
            # No feasible observation at all yet: there is nothing to fit a base
            # acquisition on (every output column is entirely NaN), so the score
            # is feasibility alone -- the only signal that exists.
            if recorder is not None:
                recorder.note("anubis_no_feasible_observations")
            scores = p_feasible
        else:
            acq_function = self._fit_base_acquisition(variables)
            pool_t = torch.as_tensor(pool, dtype=torch.double).unsqueeze(1)  # (n, 1, d)
            with torch.no_grad():
                base_acq = acq_function(pool_t).detach().cpu().numpy()
            scores = combine_score(p_feasible, base_acq)

        unexplored = _unexplored_mask(pool, train_x_norm)
        # Prefer catalogue rows never observed before. Without this, a
        # constant score (e.g. every P(feasible|x)=1.0 during the bootstrap
        # fallback, which is exactly what `labels.sum() == 0` guarantees) makes
        # `argsort` return the *same* indices every call -- on a deterministic
        # catalogue lookup, re-proposing an already-infeasible material learns
        # nothing, so the method gets stuck proposing one fixed set of
        # materials for the entire remaining budget instead of exploring. This
        # also improves the ordinary (non-degenerate) case: querying a
        # catalogue row twice is never useful, since a fixed lookup always
        # returns the same answer.
        candidate_idx = np.where(unexplored)[0] if unexplored.any() else np.arange(len(scores))
        ranked = candidate_idx[np.argsort(-scores[candidate_idx])]

        n_pick = min(q, len(ranked))
        top_idx = ranked[:n_pick]
        if n_pick < q:
            # More designs requested than there are (unexplored) catalogue rows
            # -- pad by repeating the best ones rather than crash; the loop's
            # duplicate accounting handles the repeats the same way it does for
            # a re-proposed material.
            top_idx = np.resize(top_idx, q)

        raw = variables.to_raw(pool[top_idx])
        return raw.reshape(q, d)


def make(name: str = NAME, num_restarts: int = 128) -> AnubisMethod:
    return AnubisMethod(num_restarts=num_restarts)
