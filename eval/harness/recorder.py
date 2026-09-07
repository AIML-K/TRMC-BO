"""Run-scoped telemetry.

Counters the metrics module needs live inside the acquisition optimizer, not in
the campaign loop: how many restarts were kept by the mean filter, how often the
hard constraint had to be dropped, how often a restart raised. Collecting them
through a run-scoped object rather than module globals is what keeps a 24-way
parallel sweep from interleaving counts between runs.

The production code reaches this through `optimize_config["_recorder"]` guarded
by `if rec is not None`, so an unset recorder costs nothing.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class IterationRecord:
    iteration: int
    counters: dict[str, float] = field(default_factory=dict)
    fit: dict = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)


class RunRecorder:
    def __init__(self):
        self._current: dict[str, float] = defaultdict(float)
        self._notes: dict[str, str] = {}
        self._fit: dict = {}
        self.iterations: list[IterationRecord] = []

    # -- called from src/bayesian_optimization -----------------------------

    def note(self, key: str, value: float = 1.0) -> None:
        self._current[key] += value

    def note_max(self, key: str, value: float) -> None:
        self._current[key] = max(self._current.get(key, float("-inf")), value)

    def note_text(self, key: str, value: str) -> None:
        """Record a diagnostic string (kept out of the numeric counters)."""
        self._notes[key] = value

    def note_fit(self, fit_result: dict) -> None:
        self._fit = fit_result or {}

    # -- called from the campaign loop -------------------------------------

    def flush(self, iteration: int) -> IterationRecord:
        rec = IterationRecord(
            iteration=iteration,
            counters=dict(self._current),
            fit=dict(self._fit),
            notes=dict(self._notes),
        )
        self.iterations.append(rec)
        self._current = defaultdict(float)
        self._notes = {}
        self._fit = {}
        return rec
