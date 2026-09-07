"""Random search (paper §6.1).

Establishes how hard each target window is without model guidance. Because the
frozen windows are calibrated to a known valid fraction, this method's hit rate
should land near that fraction -- which makes it a useful sanity check on the
whole harness as well as a baseline.

It draws from the same domain geometry every other method's restarts use, so the
comparison is not confounded by one method sampling the simplex differently.
"""

from __future__ import annotations

import numpy as np


class RandomSearch:
    name = "M0_random"
    objective_mode = "range"

    def __init__(self, num_restarts: int = 0):
        # Accepted for interface symmetry; random search has no restarts.
        self._counter = 0

    def propose(self, variables, q: int, seed: int, recorder=None) -> np.ndarray:
        self._counter += 1
        rng = np.random.default_rng((seed + 1) * 1_000_003 + self._counter)
        # Reuse the adapter's domain sampler so simplex groups and component
        # floors are honoured exactly as they are for the model-based methods.
        variables._rng = rng
        Z = variables.sample_raw_candidates(q, variables, {})
        return variables.to_raw(Z.squeeze(1).cpu().numpy()).reshape(q, variables.task.dim)


def make(_name: str = "M0_random", num_restarts: int = 0) -> RandomSearch:
    return RandomSearch()
