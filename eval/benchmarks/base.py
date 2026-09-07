"""Benchmark-neutral data model and oracle boundary.

Everything in ``eval/harness`` and ``eval/methods`` is written against the types
defined here, so adding a benchmark suite (Olympus, HOIP) means implementing
``Oracle`` and a task loader -- nothing downstream changes.

Two oracle entry points are deliberately kept separate:

``observe``  what the optimizer is allowed to see (noisy, batch effects, failures)
``truth``    the noiseless response, used only to score hit rate / violations

Mixing the two would let a method score itself against the same numbers it
optimizes, which is exactly what the reproducibility checklist forbids
("실제 oracle 또는 hidden benchmark output으로 hit rate를 계산한다").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np

TargetKind = Literal["greater", "less"]


@dataclass(frozen=True)
class NativeTarget:
    """A benchmark's original one-sided specification, e.g. ``y1 > 61``."""

    name: str
    kind: TargetKind
    value: float

    def satisfied(self, y: np.ndarray) -> np.ndarray:
        return y >= self.value if self.kind == "greater" else y <= self.value


@dataclass(frozen=True)
class SimplexGroup:
    """A set of design coordinates constrained to ``sum == total``.

    MatFormBench's 10D/15D tasks put x1..x6 on a simplex; the 5D tasks have no
    such group. Olympus mixture datasets will reuse this verbatim.
    """

    indices: tuple[int, ...]
    total: float = 1.0
    min_component: float = 0.0


@dataclass(frozen=True)
class BenchmarkTask:
    suite: str
    task_id: str
    dim: int
    x_names: tuple[str, ...]
    y_names: tuple[str, ...]
    bounds: np.ndarray  # (2, dim), row 0 = lower, row 1 = upper  [BoTorch order]
    native_targets: tuple[NativeTarget, ...]
    simplex_groups: tuple[SimplexGroup, ...] = ()
    meta: dict = field(default_factory=dict)

    @property
    def has_simplex(self) -> bool:
        return len(self.simplex_groups) > 0

    def target(self, name: str) -> NativeTarget:
        for t in self.native_targets:
            if t.name == name:
                return t
        raise KeyError(f"{name!r} is not a target of {self.task_id}")


@dataclass(frozen=True)
class RangeSpec:
    """Two-sided target window ``L_k <= y_k <= U_k``, frozen before any run.

    ``calibration`` carries the provenance required by the paper's §5.1
    pre-evaluation table (achieved valid fraction, the alpha that produced it,
    per-property marginals, sample size).
    """

    lower: dict[str, float]
    upper: dict[str, float]
    width_label: str
    calibration: dict = field(default_factory=dict)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.lower)

    def width(self, name: str) -> float:
        return self.upper[name] - self.lower[name]

    def hits(self, Y: np.ndarray, y_names: tuple[str, ...]) -> np.ndarray:
        """Row-wise ``all_k(L_k <= y_k <= U_k)``.

        A failed evaluation carries NaN, and NaN comparisons are False in numpy,
        so failed designs fall out as misses without special-casing.
        """
        ok = np.ones(len(Y), dtype=bool)
        for name in self.lower:
            col = Y[:, y_names.index(name)]
            ok &= (col >= self.lower[name]) & (col <= self.upper[name])
        return ok

    def reference_scaling(self, y_names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
        """Frozen, robust output center/scale for any score needing standardized units.

        Two reasons this is not the mean and standard deviation:

        * It must not move during a run. The surrogate re-standardizes every
          iteration as data arrives; a criterion built on that would drift and
          stop being comparable across methods.
        * It must not be set by outliers. On the simplex tasks y2 has an extreme
          right tail -- L5-4 has p99 = 357 against a maximum near 50,000 -- so its
          standard deviation (438) is an artifact of a handful of blow-ups and
          differs between samples, while IQR/1.349 (12.6) is stable. Using the
          standard deviation collapsed the inscribed ball to 1-2% of the box's
          valid mass on L4/L5, purely because one axis looked 30x narrower than
          it is.
        """
        cal = self.calibration
        center = np.array([cal["y_reference_center"][n] for n in y_names], dtype=float)
        scale = np.array([cal["y_reference_scale"][n] for n in y_names], dtype=float)
        return center, scale

    def ball_center(self, y_names: tuple[str, ...]) -> np.ndarray:
        """Where a ball target should sit, in frozen-standardized units.

        The centroid of the valid region, not the box center. Windows are built
        from quantiles of the native-satisfying tail, so the valid mass presses
        against one face: on L4-1/L5-4 the median valid design sits at 0.13-0.18
        of the y2 window while the box center is nearly empty. A box-centered
        ball covered under 1% of the valid mass there.

        This also matches how the Range-Aware BO paper actually picks targets --
        by k-medoids in output space, i.e. at data-dense locations -- so it is the
        more faithful mapping as well as the more favourable one.
        """
        center, scale = self.reference_scaling(y_names)
        cal = self.calibration
        if "y_valid_centroid" in cal:
            raw = np.array([cal["y_valid_centroid"][n] for n in y_names], dtype=float)
        else:
            raw = np.array([0.5 * (self.lower[n] + self.upper[n]) for n in y_names])
        return (raw - center) / scale

    def _standardized_box(self, y_names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
        center, scale = self.reference_scaling(y_names)
        lo = (np.array([self.lower[n] for n in y_names]) - center) / scale
        hi = (np.array([self.upper[n] for n in y_names]) - center) / scale
        return lo, hi

    def inscribed_ball(self, y_names: tuple[str, ...]) -> tuple[np.ndarray, float]:
        """Largest ball around the valid centroid that stays inside the box.

        Radius is the shortest distance from the centroid to any face, so
        containment still holds exactly: nothing in this ball violates the box.
        """
        lo, hi = self._standardized_box(y_names)
        c = self.ball_center(y_names)
        return c, float(np.min(np.minimum(c - lo, hi - c)))

    def equal_volume_ball(self, y_names: tuple[str, ...]) -> tuple[np.ndarray, float]:
        """Ball with the same volume as the box, in frozen-standardized units.

        The inscribed ball is *contained* in the box but far smaller than it --
        its radius is set by the tightest axis, so an elongated box leaves most
        of its volume outside. Measured on MatFormBench L1-1 the inscribed ball
        covers only 18-36% of the box's valid mass, which would hand the
        Tolerance Ball baseline a 3-5x smaller target than it is scored on.

        Matching volume instead trades containment for size: this ball pokes
        outside the box along the wide axes and misses its corners, so the
        handicap is two-sided rather than uniformly punitive. Neither mapping is
        canonical -- the TB paper defines tolerances directly, never from a box --
        so both are run and reported.
        """
        from math import gamma, pi

        lo, hi = self._standardized_box(y_names)
        k = len(y_names)
        box_volume = float(np.prod(hi - lo))
        radius = (box_volume * gamma(k / 2 + 1) / pi ** (k / 2)) ** (1.0 / k)
        return self.ball_center(y_names), float(radius)

    def ball(self, y_names: tuple[str, ...], mapping: str = "inscribed"):
        if mapping == "inscribed":
            return self.inscribed_ball(y_names)
        if mapping == "equal_volume":
            return self.equal_volume_ball(y_names)
        raise ValueError(f"unknown ball mapping: {mapping!r}")

    def hits_ball(
        self, Y: np.ndarray, y_names: tuple[str, ...], mapping: str = "inscribed"
    ) -> np.ndarray:
        """Ball-criterion membership, for reporting TB on its own terms."""
        ref_center, scale = self.reference_scaling(y_names)
        center, radius = self.ball(y_names, mapping)
        Z = (Y - ref_center) / scale
        d2 = np.nansum((Z - center) ** 2, axis=-1)
        return (d2 <= radius ** 2) & np.isfinite(Y).all(axis=-1)

    def violations(self, Y: np.ndarray, y_names: tuple[str, ...]) -> dict[str, np.ndarray]:
        """Per-property lower/upper excursions, normalized by window width.

        Normalizing by ``U_k - L_k`` is what makes violations comparable across
        the wide/medium/narrow conditions (§8.3).
        """
        low = np.zeros(len(Y))
        high = np.zeros(len(Y))
        for name in self.lower:
            col = Y[:, y_names.index(name)]
            w = self.width(name)
            low += np.nan_to_num(np.clip(self.lower[name] - col, 0.0, None)) / w
            high += np.nan_to_num(np.clip(col - self.upper[name], 0.0, None)) / w
        n = len(self.lower)
        return {"lower": low / n, "upper": high / n, "total": (low + high) / n}


@dataclass
class Observation:
    """Result of evaluating a batch of designs."""

    X: np.ndarray  # (n, dim)
    Y: np.ndarray  # (n, m) -- NaN where the design failed
    feasible: np.ndarray  # (n,) bool, input/experimental feasibility
    failure_type: list[str | None]

    def __len__(self) -> int:
        return len(self.X)


class Oracle(Protocol):
    """The single boundary every benchmark suite is reached through."""

    task: BenchmarkTask

    def observe(self, X: np.ndarray, seed: int | None = None) -> Observation:
        """Noisy evaluation -- the only thing an optimizer may consume."""
        ...

    def truth(self, X: np.ndarray) -> Observation:
        """Noiseless evaluation -- scoring only, never fed back into a method."""
        ...

    def initial_design(self, n: int, seed: int) -> np.ndarray:
        """``n`` input-feasible designs (respects simplex groups if present)."""
        ...
