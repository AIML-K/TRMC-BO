"""Olympus oracles behind the neutral `Oracle` protocol.

Three implementations, one per task kind, plus a pool oracle for the deferred
robustness protocol:

``EmulatorOracle``   pce10 / oer_plate_3496. Olympus's own trained BayesNeuralNet,
                     reimplemented over extracted weights (`data.forward`).
``SurfaceOracle``    branin. The analytical formula ported from Olympus.
``PoolOracle``       lookup against the measured table -- zero model assumptions.
``make_oracle``      dispatch on `task.meta["oracle"]`.

## observe vs. truth

The protocol keeps them apart so no method can score itself on the numbers it
optimizes. What fills the gap differs by task, and both choices are stated rather
than inherited:

**Emulator tasks.** The BNN is Bayesian, so its own posterior supplies the noise
and none has to be invented. `truth` is the noiseless posterior-mean forward pass;
`observe` is one local-reparameterization draw. Note this is *our* definition of
truth, not Olympus's: `Emulator.run(num_samples=1)` returns a single stochastic
draw, so Olympus itself has no noiseless mode.

**Branin is noiseless: `observe` is `truth`.** The Range-Aware BO paper uses it
that way, and a deliberately noiseless benchmark is informative next to
MatFormBench, whose every task is noisy -- it isolates the acquisition comparison
from noise handling. The observe/truth split is then vacuous on this task, which
is a property worth stating, not hiding.

## Input feasibility

Emulator tasks report designs outside the convex hull of the measured
compositions as input-infeasible rather than extrapolating a network that has no
support there. Measured on 200k uniform draws this bites unevenly and the numbers
belong in the write-up:

    pce10             8 facets, 99.89% of the simplex inside -- 0.11% infeasible
    oer_plate_3496    6 facets: the hull *is* the simplex, so nothing is
                      infeasible and this task exercises no input feasibility
    branin            box only, everything feasible

So Olympus contributes almost no input-feasibility difficulty. That is a real
limitation of this suite and the reason MatFormBench keeps the sparse-feasibility
claim (65% of L5-4 proposals are unmanufacturable).
"""

from __future__ import annotations

import numpy as np

from eval.benchmarks import calibration
from eval.benchmarks.base import BenchmarkTask, Observation
from eval.benchmarks.olympus import data as odata


class _BaseOracle:
    """Shared design sampling and bookkeeping."""

    def __init__(self, task: BenchmarkTask):
        self.task = task
        self._calls = 0

    @property
    def n_calls(self) -> int:
        return self._calls

    def initial_design(self, n: int, seed: int) -> np.ndarray:
        """`n` input-feasible designs.

        Olympus ships no design generator, so this is the same uniform reference
        measure the calibration uses -- flat Dirichlet on each simplex group,
        uniform on the box. Using one measure for both means the initial design's
        feasibility rate and the spec's `input_feasible_fraction` are comparable,
        which they are not on MatFormBench (LHS ~21% vs uniform ~33% on L5-4).

        Must work for `n == 1`: `harness/loop.py` uses it as the random fallback
        when the acquisition optimizer fails or proposes a duplicate.
        """
        return calibration.sample_design_space(self.task, n, seed=seed)

    def _observation(
        self, X: np.ndarray, Y: np.ndarray, feasible: np.ndarray
    ) -> Observation:
        """Assemble an `Observation`, NaN-ing the response of infeasible designs.

        `Y` stays 2-D `(n, 1)` even with a single output: `harness/loop.py`
        vstacks it and calls `.all(axis=1)`, both of which silently misbehave on a
        1-D array.
        """
        Y = np.asarray(Y, dtype=float).reshape(len(X), len(self.task.y_names))
        Y = np.where(feasible[:, None], Y, np.nan)
        return Observation(
            X=np.asarray(X, dtype=float),
            Y=Y,
            feasible=feasible,
            failure_type=[None if ok else "OUT_OF_HULL" for ok in feasible],
        )


class EmulatorOracle(_BaseOracle):
    """Olympus's trained BayesNeuralNet over the extracted weights."""

    def __init__(self, task: BenchmarkTask):
        super().__init__(task)
        self.dataset = task.meta["dataset"]
        self.emulator = odata.load_emulator(self.dataset)
        if self.emulator.dim != task.dim:
            raise ValueError(
                f"{task.task_id}: emulator expects {self.emulator.dim} inputs, "
                f"task has {task.dim}"
            )

    def _feasible(self, X: np.ndarray) -> np.ndarray:
        return odata.hull_membership(self.dataset, X)

    def observe(self, X: np.ndarray, seed: int | None = None) -> Observation:
        """One posterior draw per design -- the emulator's own uncertainty."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        rng = np.random.default_rng(seed)
        Y = odata.forward(self.emulator, X, rng)
        self._calls += len(X)
        return self._observation(X, Y, self._feasible(X))

    def truth(self, X: np.ndarray) -> Observation:
        """Noiseless posterior mean. Scoring only, never returned to a method."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        Y = odata.forward(self.emulator, X, mean_pass=True)
        return self._observation(X, Y, self._feasible(X))


class SurfaceOracle(_BaseOracle):
    """An analytical Olympus surface. Noiseless, so `observe` equals `truth`."""

    def __init__(self, task: BenchmarkTask):
        super().__init__(task)
        self.surface = task.meta["surface"]
        if self.surface != "branin":
            raise ValueError(f"unsupported surface {self.surface!r}")

    @staticmethod
    def branin(X: np.ndarray) -> np.ndarray:
        """Olympus's Branin, ported from `surfaces/surface_branin/wrapper_branin.py`.

        Defined on the unit square and rescaled internally -- `x0 = 15*p0 - 5`
        onto [-5, 10], `x1 = 15*p1` onto [0, 15] -- so the [0,1]^2 design box is
        the benchmark's own parameterization, not a normalization we imposed.
        """
        P = np.atleast_2d(np.asarray(X, dtype=float))
        x0 = 15.0 * P[:, 0] - 5.0
        x1 = 15.0 * P[:, 1]
        a, b, c, r, s, t = 1.0, 5.1 / (4 * np.pi**2), 5 / np.pi, 6.0, 10.0, 1 / (8 * np.pi)
        return (a * (x1 - b * x0**2 + c * x0 - r) ** 2 + s * (1 - t) * np.cos(x0) + s)[:, None]

    #: The three global minima, in Olympus's unit-square coordinates. Their own
    #: `minima` property asserts the three values agree.
    MINIMA = np.array([
        [(-np.pi + 5) / 15, 12.275 / 15],
        [(np.pi + 5) / 15, 2.275 / 15],
        [(9.42478 + 5) / 15, 2.475 / 15],
    ])

    def truth(self, X: np.ndarray) -> Observation:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        feasible = np.ones(len(X), dtype=bool)
        return self._observation(X, self.branin(X), feasible)

    def observe(self, X: np.ndarray, seed: int | None = None) -> Observation:
        """Identical to `truth`: this task is noiseless by design."""
        obs = self.truth(X)
        self._calls += len(obs.X)
        return obs


class PoolOracle(_BaseOracle):
    """Nearest measured design, for the model-assumption-free protocol.

    Answers with the measured response of the closest composition in the shipped
    table, so it makes no modelling assumption at all -- the point of running it
    is that the emulator conclusions should not depend on the emulator. Deferred:
    it cannot change the main result, so it runs after the core table exists.
    """

    def __init__(self, task: BenchmarkTask):
        super().__init__(task)
        self.dataset = task.meta["dataset"]
        self._X, self._y = odata.load_measurements(self.dataset)

    def _lookup(self, X: np.ndarray) -> np.ndarray:
        d = np.linalg.norm(X[:, None, :] - self._X[None, :, :], axis=-1)
        return self._y[np.argmin(d, axis=1)][:, None]

    def truth(self, X: np.ndarray) -> Observation:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return self._observation(X, self._lookup(X), odata.hull_membership(self.dataset, X))

    def observe(self, X: np.ndarray, seed: int | None = None) -> Observation:
        obs = self.truth(X)
        self._calls += len(obs.X)
        return obs


def make_oracle(task: BenchmarkTask):
    kind = task.meta.get("oracle")
    if kind == "emulator":
        return EmulatorOracle(task)
    if kind == "surface":
        return SurfaceOracle(task)
    if kind == "pool":
        return PoolOracle(task)
    raise ValueError(f"{task.task_id}: unknown oracle kind {kind!r}")
