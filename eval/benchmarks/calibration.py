"""Turn a benchmark's one-sided targets into two-sided windows of a chosen difficulty.

Suite-neutral: `benchmarks/matformbench/ranges.py` and
`benchmarks/olympus/ranges.py` are thin wrappers that bind `suite`. The rule and
the reasoning behind it live here because they are shared, and because a second
suite is exactly when a rule quietly reinterpreted per suite would stop being
comparable.

The paper's range-adapted protocol needs windows whose difficulty is *controlled*
rather than incidental, and which stay comparable to the native protocol. The
rule (plan §1.2):

* keep the native threshold as one bound, so the adapted task is a tightening of
  the original rather than a different task;
* place the opposite bound at a quantile of the clean oracle's output
  distribution, restricted to designs that already satisfy the native side;
* drive every property's quantile from a *single* scalar `alpha`, bisected until
  the joint valid fraction hits the target.

One shared `alpha` matters: tightening each property independently to its own
marginal rate would let the properties contribute unequally to the joint
difficulty, and the three MatFormBench outputs are strongly coupled.

Valid fraction is measured **among input-feasible designs**. On MatFormBench
L4/L5 most of the box fails outright (PRECIP/GEL), and mixing that into the
denominator would conflate input feasibility with outcome-range difficulty -- the
two things §8.4 insists on reporting separately.

The reference measure is uniform over the domain (uniform on the box, uniform on
each simplex via a flat Dirichlet), not the oracle's own generator. Uniform is
the natural reading of "X% of the design space is valid" and does not depend on a
generator's projection details. The two measures disagree -- on L5-4 the oracle's
LHS yields ~21% input-feasible designs against ~33% under uniform -- so
`input_feasible_fraction` in the calibration record is the uniform figure and is
not interchangeable with the initial design's feasibility rate.

## A single output makes the bisection analytic

With one property the rule has a closed form, because the "joint" valid fraction
is the marginal one:

    valid fraction among feasible = P(y satisfies native side) * alpha

so `alpha = target / P(native)`. Two consequences worth stating rather than
rediscovering:

* The bisection still runs and still converges -- it is simply solving a linear
  equation -- so single-output suites reuse this code unchanged. `calibrate`
  records `alpha`, and the identity above is asserted in the Olympus tests.
* The *placement of the synthesized native threshold* decides whether the widest
  condition is two-sided at all. If the threshold is set at the target quantile
  itself (e.g. tau = q10 with a 10% target) then alpha = 1, the opposite bound
  collapses onto the sample extremum, and the window degenerates into the
  one-sided native target. Suites that synthesize a threshold must leave
  headroom; Olympus uses tau = q25 against targets of 10/3/1%, giving
  alpha = 0.40/0.12/0.04.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from eval.benchmarks.base import BenchmarkTask, Oracle, RangeSpec

SPEC_ROOT = Path(__file__).resolve().parents[1] / "specs"

#: Target share of input-feasible designs that should land inside the window.
DIFFICULTY: dict[str, float] = {"wide": 0.10, "medium": 0.03, "narrow": 0.01}


def sample_design_space(task: BenchmarkTask, n: int, seed: int) -> np.ndarray:
    """Draw `n` designs matching the benchmark's own geometry.

    Uniform on the box, Dirichlet on each simplex group with `min_component`
    respected. Deliberately independent of the oracle's own `generate` so that
    calibration costs no benchmark evaluations beyond the scoring pass.
    """
    rng = np.random.default_rng(seed)
    lo, hi = task.bounds
    X = rng.uniform(lo, hi, size=(n, task.dim))
    for group in task.simplex_groups:
        idx = list(group.indices)
        k = len(idx)
        w = rng.dirichlet(np.ones(k), size=n)
        floor = group.min_component / group.total
        X[:, idx] = (floor + w * (1.0 - k * floor)) * group.total
    return X


def _robust_scale(col: np.ndarray) -> float:
    """IQR rescaled to match a Gaussian standard deviation."""
    q75, q25 = np.percentile(col, [75, 25])
    return float((q75 - q25) / 1.349)


def _bounds_at(task: BenchmarkTask, Y: np.ndarray, alpha: float) -> tuple[dict, dict]:
    """Window bounds for a given `alpha`, over already-feasible rows.

    `alpha` is the share of the native-satisfying tail the window keeps:
    `alpha = 1` reproduces the native one-sided target, `alpha -> 0` collapses
    the window onto the threshold.
    """
    lower, upper = {}, {}
    for j, name in enumerate(task.y_names):
        tgt = task.target(name)
        col = Y[:, j]
        col = col[np.isfinite(col)]
        side = col[col >= tgt.value] if tgt.kind == "greater" else col[col <= tgt.value]
        if side.size == 0:
            # No sampled design satisfies the native side; fall back to the full
            # observed support so the caller still gets a usable (if trivial)
            # window and the calibration record shows what happened.
            side = col
        if tgt.kind == "greater":
            lower[name] = float(tgt.value)
            upper[name] = float(np.quantile(side, alpha))
        else:
            upper[name] = float(tgt.value)
            lower[name] = float(np.quantile(side, 1.0 - alpha))
    return lower, upper


def _fraction(task: BenchmarkTask, Y: np.ndarray, lower: dict, upper: dict) -> float:
    ok = np.ones(len(Y), dtype=bool)
    for j, name in enumerate(task.y_names):
        ok &= (Y[:, j] >= lower[name]) & (Y[:, j] <= upper[name])
    return float(ok.mean()) if len(Y) else 0.0


def reference_sample(
    task: BenchmarkTask, oracle: Oracle, n_samples: int, seed: int
) -> tuple[np.ndarray, float]:
    """Clean responses of the input-feasible part of a uniform design sample.

    Returned separately from `calibrate` so the three difficulty levels of a task
    share one scoring pass instead of paying for it three times.
    """
    X = sample_design_space(task, n_samples, seed)
    obs = oracle.truth(X)
    feas = obs.feasible & np.isfinite(obs.Y).all(axis=1)
    if not feas.any():
        raise RuntimeError(f"{task.task_id}: no input-feasible design in {n_samples} draws")
    return obs.Y[feas], float(feas.mean())


def calibrate(
    task: BenchmarkTask,
    oracle: Oracle,
    width_label: str,
    *,
    n_samples: int = 100_000,
    seed: int = 20260820,
    tol: float = 5e-4,
    max_iter: int = 40,
    reference: tuple[np.ndarray, float] | None = None,
) -> RangeSpec:
    target = DIFFICULTY[width_label]

    Y, feasible_fraction = reference if reference is not None else reference_sample(
        task, oracle, n_samples, seed
    )

    native_rate = _fraction(
        task, Y,
        {n: (task.target(n).value if task.target(n).kind == "greater" else -np.inf) for n in task.y_names},
        {n: (task.target(n).value if task.target(n).kind == "less" else np.inf) for n in task.y_names},
    )

    # Valid fraction is monotone non-decreasing in alpha, so plain bisection works.
    lo_a, hi_a = 0.0, 1.0
    lower, upper = _bounds_at(task, Y, hi_a)
    achieved = _fraction(task, Y, lower, upper)
    if achieved < target:
        # Even the native window is tighter than the requested difficulty.
        # Return it as-is rather than silently widening past the native task.
        alpha = hi_a
    else:
        for _ in range(max_iter):
            alpha = 0.5 * (lo_a + hi_a)
            lower, upper = _bounds_at(task, Y, alpha)
            achieved = _fraction(task, Y, lower, upper)
            if abs(achieved - target) <= tol:
                break
            if achieved > target:
                hi_a = alpha
            else:
                lo_a = alpha
        lower, upper = _bounds_at(task, Y, alpha)
        achieved = _fraction(task, Y, lower, upper)

    marginals = {}
    for j, name in enumerate(task.y_names):
        marginals[name] = float(
            ((Y[:, j] >= lower[name]) & (Y[:, j] <= upper[name])).mean()
        )

    # Where the valid mass actually sits. The window is built from quantiles of
    # the native-satisfying tail, so the density piles up against one face rather
    # than around the middle: on L4-1/L5-4 the median valid design sits at 0.13-0.18
    # of the y2 window, and the box center is nearly empty. Any ball centered on
    # the box center therefore misses almost all of it.
    valid = np.ones(len(Y), dtype=bool)
    for j, name in enumerate(task.y_names):
        valid &= (Y[:, j] >= lower[name]) & (Y[:, j] <= upper[name])
    centroid = {
        n: (float(np.median(Y[valid, j])) if valid.any()
            else 0.5 * (lower[n] + upper[n]))
        for j, n in enumerate(task.y_names)
    }

    return RangeSpec(
        lower=lower,
        upper=upper,
        width_label=width_label,
        calibration={
            "rule": "native-threshold-preserved, single-alpha quantile on the satisfying tail",
            "target_fraction_of_feasible": target,
            "achieved_fraction_of_feasible": achieved,
            "achieved_fraction_of_all": achieved * feasible_fraction,
            "alpha": float(alpha),
            "native_joint_rate_of_feasible": native_rate,
            "input_feasible_fraction": feasible_fraction,
            "n_samples": int(n_samples),
            "n_input_feasible": int(len(Y)),
            "marginal_valid_fraction": marginals,
            "y_valid_centroid": centroid,
            # Frozen so any standardized-space criterion (e.g. the tolerance-ball
            # comparison) is defined once and never drifts with the run. Robust
            # statistics, because y2 is heavy-tailed on the simplex tasks and its
            # standard deviation is set by a few blow-ups rather than by the bulk.
            "y_reference_center": {
                n: float(np.median(Y[:, j])) for j, n in enumerate(task.y_names)
            },
            "y_reference_scale": {
                n: float(max(_robust_scale(Y[:, j]), 1e-12))
                for j, n in enumerate(task.y_names)
            },
            # Kept for reference / diagnostics only; not used by any criterion.
            "y_reference_mean": {n: float(Y[:, j].mean()) for j, n in enumerate(task.y_names)},
            "y_reference_std": {n: float(max(Y[:, j].std(), 1e-12)) for j, n in enumerate(task.y_names)},
            "seed": seed,
        },
    )

# -- persistence -----------------------------------------------------------
#
# Frozen windows are committed to git. Every method and seed must optimize
# against byte-identical targets, and a range regenerated on a different machine
# would silently invalidate cross-run comparisons.

def spec_dir(suite: str) -> Path:
    return SPEC_ROOT / suite


def spec_path(suite: str, task_id: str, width_label: str) -> Path:
    return spec_dir(suite) / f"range_spec_{task_id}_{width_label}.json"


def save(suite: str, task_id: str, spec: RangeSpec) -> Path:
    directory = spec_dir(suite)
    directory.mkdir(parents=True, exist_ok=True)
    path = spec_path(suite, task_id, spec.width_label)
    payload = {"task_id": task_id, **asdict(spec)}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load(suite: str, task_id: str, width_label: str) -> RangeSpec:
    payload = json.loads(spec_path(suite, task_id, width_label).read_text(encoding="utf-8"))
    return RangeSpec(
        lower=payload["lower"],
        upper=payload["upper"],
        width_label=payload["width_label"],
        calibration=payload["calibration"],
    )
