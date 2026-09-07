"""Wrap MatFormBench's compiled oracle behind the neutral `Oracle` protocol.

The compiled module (`synthetic_data/oracle_0.cpython-310-*.so`) exposes:

    make_polymer_oracle(dim, level) -> (OracleSuite, index_map)
    suite.evaluate(x_dict, context=None, seed=None) -> dict   # noisy
    suite.evaluate_clean(x_dict)                     -> dict  # noiseless
    suite.generate(n, split, design, seed)           -> list[dict]
    suite.train_box / suite.proposal_box             -> (lo, hi) arrays

Each evaluation returns::

    {"feasible": bool,
     "failure_type": str | None,
     "objectives": {"y1": float | None, ...},
     "aux": {"batch_id": int | None, "in_train_box": bool, "notes": [...]}}

Infeasible designs come back with ``objectives`` all ``None``; we map those to
NaN so the downstream NaN-aware standardization in ``src/bayesian_optimization``
handles them without a special case.

Note that ``suite.validate(df, tag=...)`` is *not* a plain oracle call -- it
requires the caller to supply ``pred_<y>`` columns and exists for the native
recommend protocol. Range-adapted runs use ``evaluate``/``evaluate_clean``.
"""

from __future__ import annotations

import contextlib
import io
import sys

import numpy as np

from eval.benchmarks.base import BenchmarkTask, Observation
from eval.benchmarks.matformbench.tasks import MATFORMBENCH_ROOT, parse_task_id


def _ensure_importable() -> None:
    root = str(MATFORMBENCH_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


@contextlib.contextmanager
def _quiet():
    """Silence the compiled oracle.

    ``evaluate`` writes a stray debug array to stdout on some paths. Left alone
    it drowns the run log at one line per evaluation.
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield


class MatFormBenchOracle:
    """`Oracle` implementation for one MatFormBench task."""

    def __init__(self, task: BenchmarkTask):
        _ensure_importable()
        from synthetic_data.oracle_0 import make_polymer_oracle  # noqa: PLC0415

        level, _ = parse_task_id(task.task_id)
        self.task = task
        self._suite, self._index_map = make_polymer_oracle(dim=task.dim, level=level)
        self._calls = 0

        lo, hi = self._suite.train_box
        if not (np.allclose(lo, task.bounds[0]) and np.allclose(hi, task.bounds[1])):
            raise ValueError(
                f"{task.task_id}: rules train_bounds {task.bounds.tolist()} disagree with "
                f"oracle train_box {(lo.tolist(), hi.tolist())}"
            )

    # -- internals ---------------------------------------------------------

    def _as_dict(self, x: np.ndarray) -> dict[str, float]:
        return {name: float(v) for name, v in zip(self.task.x_names, x)}

    def _collect(self, X: np.ndarray, records: list[dict]) -> Observation:
        m = len(self.task.y_names)
        Y = np.full((len(records), m), np.nan)
        feasible = np.zeros(len(records), dtype=bool)
        failure: list[str | None] = []
        for i, rec in enumerate(records):
            feasible[i] = bool(rec.get("feasible", False))
            failure.append(rec.get("failure_type"))
            objectives = rec.get("objectives") or {}
            for j, name in enumerate(self.task.y_names):
                v = objectives.get(name)
                if v is not None:
                    Y[i, j] = float(v)
        return Observation(X=np.asarray(X, dtype=float), Y=Y, feasible=feasible, failure_type=failure)

    # -- Oracle protocol ---------------------------------------------------

    def observe(self, X: np.ndarray, seed: int | None = None) -> Observation:
        """Noisy evaluation, including MatFormBench's random batch effect.

        A per-row seed is derived from ``seed`` so a campaign replays exactly;
        passing ``seed=None`` leaves the oracle's own RNG in charge.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        records = []
        with _quiet():
            for i, row in enumerate(X):
                row_seed = None if seed is None else int(seed) * 100_003 + i
                records.append(self._suite.evaluate(self._as_dict(row), seed=row_seed))
        self._calls += len(X)
        return self._collect(X, records)

    def truth(self, X: np.ndarray) -> Observation:
        """Noiseless evaluation. Scoring only -- never returned to a method."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        with _quiet():
            records = [self._suite.evaluate_clean(self._as_dict(row)) for row in X]
        return self._collect(X, records)

    def initial_design(self, n: int, seed: int) -> np.ndarray:
        """LHS inside the training box, respecting any simplex group."""
        with _quiet():
            points = self._suite.generate(n, split="train", design="lhs", seed=seed)
        return np.array([[float(p[name]) for name in self.task.x_names] for p in points])

    @property
    def n_calls(self) -> int:
        return self._calls


def make_oracle(task: BenchmarkTask) -> MatFormBenchOracle:
    return MatFormBenchOracle(task)
