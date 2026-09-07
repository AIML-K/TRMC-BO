"""Common interface every compared method implements.

Keeping this narrow is what lets the sweep treat the proposed method, the
standard-BO ablations, the Range-Aware TB baseline and random search
interchangeably -- and what makes adding COMBOO/SCBO/alpha-GaBO later a matter
of one new file each.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from eval.harness.adapter import BenchmarkVariables


class Method(Protocol):
    name: str

    def propose(
        self,
        variables: BenchmarkVariables,
        q: int,
        seed: int,
        recorder=None,
    ) -> np.ndarray:
        """Return ``q`` designs in raw benchmark units, shape ``(q, d)``.

        ``variables`` already carries the observations so far. Implementations
        must not call the oracle: proposing is free, evaluating is what the
        budget pays for.
        """
        ...
