"""Metrics for one run (paper §8).

Everything is derived from the per-iteration trace, so a finished run can be
rescored without re-running it. Two conventions matter enough to state:

**Censoring (§8.2).** A run that never hits is recorded as `first_hit = None`
with `found = False`, not as `budget + 1`. Averaging a sentinel would reward the
methods that fail most: pushing failures to `budget + 1` compresses them toward
the successes. Report `found_rate` next to the median over successes, or use
`first_hit_censored` only where a single number is unavoidable.

**Violation normalization (§8.3).** Excursions are divided by the window width
`U_k - L_k` before averaging over properties. Without it y2 (scale ~300) would
swamp y3 (scale ~6), and the wide/medium/narrow conditions could not be compared
at all.

Input feasibility and outcome-range validity are always reported separately, and
their conjunction as a third number -- never collapsed into one "success" rate.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from eval.benchmarks.base import BenchmarkTask, RangeSpec
from eval.harness.loop import RunResult


def _normalized_X(X: np.ndarray, task: BenchmarkTask) -> np.ndarray:
    lo, hi = task.bounds
    span = np.where(hi - lo == 0, 1.0, hi - lo)
    return (X - lo) / span


def delta_uniqueness_curve(
    X: np.ndarray, task: BenchmarkTask, n_grid: int = 50
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy delta-unique subset size against delta.

    A subset is delta-unique when every pair is more than delta apart, so the
    curve says how many *genuinely distinct* valid designs were found rather than
    how many points happened to satisfy the window. This is the diversity notion
    the Range-Aware BO paper scores on, reproduced here so our numbers are
    comparable to theirs.
    """
    if len(X) == 0:
        return np.zeros(n_grid), np.zeros(n_grid)
    Z = _normalized_X(X, task)
    delta_max = float(np.sqrt(task.dim))  # diameter of the unit cube
    deltas = np.linspace(0.0, delta_max, n_grid)
    counts = np.zeros(n_grid)
    for i, delta in enumerate(deltas):
        kept: list[np.ndarray] = []
        for z in Z:
            if all(np.linalg.norm(z - k) > delta for k in kept):
                kept.append(z)
        counts[i] = len(kept)
    return deltas, counts


def _diversity(X: np.ndarray, task: BenchmarkTask, n_eval: int) -> dict:
    if len(X) == 0:
        return {"diversity_norm": 0.0, "mean_pairwise_l2": 0.0, "n_unique": 0}
    deltas, counts = delta_uniqueness_curve(X, task)
    delta_max = deltas[-1] if deltas[-1] > 0 else 1.0
    area = float(np.trapz(counts, deltas))
    Z = _normalized_X(X, task)
    if len(Z) > 1:
        d = np.linalg.norm(Z[:, None, :] - Z[None, :, :], axis=-1)
        pairwise = float(d[np.triu_indices(len(Z), k=1)].mean())
    else:
        pairwise = 0.0
    return {
        "diversity_norm": area / (max(n_eval, 1) * delta_max),
        "mean_pairwise_l2": pairwise,
        "n_unique": int(len(np.unique(np.round(Z, 8), axis=0))),
    }


def iteration_frame(res: RunResult, task: BenchmarkTask, spec: RangeSpec) -> pd.DataFrame:
    """One row per evaluated design; the on-disk trace."""
    viol = spec.violations(res.Y_true, task.y_names)
    frame = pd.DataFrame({
        "iteration": res.iteration,
        "is_init": res.is_init,
        "feasible": res.feasible,
        "failure_type": res.failure_type,
        "hit": res.hit,
        "violation_lower": viol["lower"],
        "violation_upper": viol["upper"],
        "violation_total": viol["total"],
        "wall_clock": res.wall_clock,
    })
    for j, name in enumerate(task.x_names):
        frame[f"x_{name}"] = res.X[:, j]
    for j, name in enumerate(task.y_names):
        frame[f"y_obs_{name}"] = res.Y_observed[:, j]
        frame[f"y_true_{name}"] = res.Y_true[:, j]
    # Counters are per-proposal, so they only exist for the loop rows.
    keys = sorted({k for c in res.counters for k in c})
    for key in keys:
        col = [np.nan] * int(res.is_init.sum()) + [c.get(key, 0.0) for c in res.counters]
        frame[f"cnt_{key}"] = col
    return frame


def run_metrics(res: RunResult, task: BenchmarkTask, spec: RangeSpec) -> dict:
    """Scalar summary of one run. Scored on the noiseless response throughout."""
    loop = ~res.is_init
    n_loop = int(loop.sum())
    hit_loop = res.hit[loop]
    feas_loop = res.feasible[loop]
    viol = spec.violations(res.Y_true, task.y_names)

    # Both ball criteria are reported for every method. Emitting a single
    # "hit_rate_ball" under one default mapping scored the equal-volume TB
    # variant against the inscribed ball -- exactly the optimize-one/score-another
    # mismatch this column exists to avoid.
    hit_ball = {
        mapping: spec.hits_ball(res.Y_true, task.y_names, mapping)[loop]
        for mapping in ("inscribed", "equal_volume")
    }

    hit_idx = np.flatnonzero(hit_loop)
    first_hit = int(res.iteration[loop][hit_idx[0]]) if hit_idx.size else None

    counters: Counter = Counter()
    for c in res.counters:
        counters.update(c)

    out = {
        "task_id": res.task_id,
        "width": res.width_label,
        "method": res.method,
        "seed": res.seed,
        "n_loop": n_loop,

        # -- outcome-range validity ---------------------------------------
        "hit_rate": float(hit_loop.mean()) if n_loop else 0.0,
        "n_valid": int(hit_loop.sum()),
        "found": bool(hit_idx.size),
        "first_hit": first_hit,
        # Only for tables that must show a single number; see module docstring.
        "first_hit_censored": first_hit if first_hit is not None else n_loop + 1,
        "hit_rate_init": float(res.hit[res.is_init].mean()) if res.is_init.any() else 0.0,
        "hit_rate_ball_inscribed": float(hit_ball["inscribed"].mean()) if n_loop else 0.0,
        "hit_rate_ball_equal_volume": float(hit_ball["equal_volume"].mean()) if n_loop else 0.0,
        "n_valid_ball_inscribed": int(hit_ball["inscribed"].sum()),
        "n_valid_ball_equal_volume": int(hit_ball["equal_volume"].sum()),

        # -- input feasibility (kept separate) ----------------------------
        "invalid_rate": float((~feas_loop).mean()) if n_loop else 0.0,
        "feasible_rate": float(feas_loop.mean()) if n_loop else 0.0,
        "combined_success_rate": float((feas_loop & hit_loop).mean()) if n_loop else 0.0,

        # -- range violation ----------------------------------------------
        "violation_lower": float(np.nanmean(viol["lower"][loop])) if n_loop else np.nan,
        "violation_upper": float(np.nanmean(viol["upper"][loop])) if n_loop else np.nan,
        "violation_total": float(np.nanmean(viol["total"][loop])) if n_loop else np.nan,
        # y2 blows up by orders of magnitude on the simplex tasks, and violations
        # are width-normalized, so a single such design can dominate the mean.
        "violation_total_median": float(np.nanmedian(viol["total"][loop])) if n_loop else np.nan,

        # -- optimizer health ---------------------------------------------
        "acqf_total_failure_rate": counters.get("acqf_total_failure", 0.0) / max(n_loop, 1),
        "random_fallback_rate": counters.get("random_fallback", 0.0) / max(n_loop, 1),
        "duplicate_proposal_rate": counters.get("duplicate_proposals", 0.0) / max(n_loop, 1),

        # -- cost ----------------------------------------------------------
        "wall_clock_total": float(res.wall_clock[loop].sum()) if n_loop else 0.0,
        "wall_clock_per_iter": float(np.median(res.wall_clock[loop])) if n_loop else 0.0,
    }
    out.update({f"cnt_{k}": v for k, v in counters.items()})
    out.update(_diversity(res.X[loop][hit_loop], task, n_loop))
    out.update({f"meta_{k}": v for k, v in res.meta.items()})

    # Failure modes, since L3-L5 differ in *why* a design is infeasible.
    failures = Counter(f for f, m in zip(res.failure_type, loop) if m and f)
    out.update({f"fail_{k}": v for k, v in failures.items()})
    return out


def aggregate(rows: list[dict]) -> pd.DataFrame:
    """Seed-level rows -> one row per (task, width, method) with spread.

    `first_hit` is summarized as the median over runs that actually hit, paired
    with `found_rate`. A mean over the censored column is reported too, but only
    because some tables need one number -- it is the misleading one.
    """
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    keys = ["task_id", "width", "method"]
    numeric = df.select_dtypes(include=[np.number]).columns.difference(["seed"])

    grouped = df.groupby(keys, sort=False)
    out = grouped[list(numeric)].agg(["mean", "std"])
    out.columns = [f"{c}_{s}" for c, s in out.columns]

    out["found_rate"] = grouped["found"].mean()
    out["n_seeds"] = grouped.size()
    out["first_hit_median_when_found"] = grouped.apply(
        lambda g: g.loc[g["found"], "first_hit"].median() if g["found"].any() else np.nan,
        include_groups=False,
    )
    # The stability number §8.5 asks for.
    out["hit_rate_seed_std"] = grouped["hit_rate"].std()
    return out.reset_index()
