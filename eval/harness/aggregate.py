"""Turn run directories into the paper's tables and curves.

Reads the `metrics.json` each cell wrote (scalars) and the `trace.parquet`
(per-iteration), so nothing is recomputed from scratch and a sweep can be
re-tabulated without re-running.

Two reporting rules are enforced here rather than left to the caller:

* Method labels are human-readable in every output. Registry keys like "M2" are
  directory names, not something a reader should have to decode.
* Native-protocol and range-adapted results are never mixed into one table. The
  backbone requires them reported separately, and they are not comparable --
  different targets, different evaluation protocol.
* `first_hit` is summarized as a median over the runs that actually hit, always
  printed next to `found_rate`. A mean over a `budget + 1` sentinel makes the
  methods that fail most look closest to the ones that succeed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from eval.harness.runner import RESULT_ROOT

#: Working name for the proposed method until the paper settles on one.
METHOD_NAME = "TRMC-BO"

#: Registry key -> label for tables and figures.
#:
#: Keys stay as they are: they name result directories, and renaming them would
#: orphan every run already on disk. Translation happens only at presentation
#: time, because "M2" tells a reader nothing about what was compared.
DISPLAY_NAMES: dict[str, str] = {
    "M0_random": "Random search",
    "M1_qehvi_extremum": "Standard BO (extremum-seeking)",
    "M2_target_distance": "Point-target BO (target-distance qEI)",
    "M3_range_prob": f"{METHOD_NAME} (range probability only)",
    "M4_range_prob_constraints": f"{METHOD_NAME} (+ two-sided constraints)",
    "M5_full": f"{METHOD_NAME} (full)",
    # The cells that complete the C1/C2/C3 factorial. Labels name the components
    # rather than a rung, because these are not on the ladder -- three of them
    # switch C1 off, which is the comparison they exist to make.
    "M13_range_prob_starts": f"{METHOD_NAME} (C1+C3, no constraints)",
    "M14_constraints_only": f"{METHOD_NAME} (C2 only, no range objective)",
    "M15_starts_only": f"{METHOD_NAME} (C3 only, no range objective)",
    "M16_constraints_starts": f"{METHOD_NAME} (C2+C3, no range objective)",
    # The C4 feasibility variants. Labels name *where* the signal is applied,
    # because that is the only thing that differs between the three.
    "M10_full_feasw": f"{METHOD_NAME} (full + feasibility-weighted ranking)",
    "M11_full_feasf": f"{METHOD_NAME} (full + feasibility-filtered restarts)",
    "M12_full_feas": f"{METHOD_NAME} (full + feasibility, both)",
    # The §4.2 product variants. Labels say which aggregation and which
    # acquisition form, because "product" alone would not distinguish the two
    # ways the shipped path departs from the spec.
    "M3p_range_prob_product": f"{METHOD_NAME} (range-probability product)",
    "M3q_range_prob_product_ei": f"{METHOD_NAME} (product in qEI)",
    "M5p_full_product": f"{METHOD_NAME} (full, product)",
    "M5q_full_product_ei": f"{METHOD_NAME} (full, product in qEI)",
    "M6_range_aware_tb": "Range-Aware TB (equal-volume ball)",
    "M6b_range_aware_tb_inscribed": "Range-Aware TB (inscribed ball)",
    "M6c_range_aware_tb_box": "Range-Aware TB (window as ball, no hinge)",
    "M8_comboo": "COMBOO (optimistic constraints)",
    "M9_anubis": "Anubis (feasibility-weighted acquisition)",
    "M7_scbo": "SCBO (trust-region constrained BO)",
}

#: Presentation order: baselines, then the ablation ladder, then the closest
#: published competitor. Reads as an argument rather than a leaderboard --
#: `average_rank` is where performance ordering belongs.
DISPLAY_ORDER = [
    "M0_random",
    "M1_qehvi_extremum",
    "M2_target_distance",
    "M3_range_prob",
    "M4_range_prob_constraints",
    "M13_range_prob_starts",
    "M5_full",
    "M14_constraints_only",
    "M15_starts_only",
    "M16_constraints_starts",
    "M10_full_feasw",
    "M11_full_feasf",
    "M12_full_feas",
    "M3p_range_prob_product",
    "M3q_range_prob_product_ei",
    "M5p_full_product",
    "M5q_full_product_ei",
    "M6_range_aware_tb",
    "M6b_range_aware_tb_inscribed",
    "M6c_range_aware_tb_box",
    "M8_comboo",
    "M9_anubis",
    "M7_scbo",
]


#: What the proposed method is called in rendered output. The internal name
#: identifies the affiliation, which would undo a double-blind submission the
#: moment a figure generated here is dropped into the paper, so anything
#: producing paper output sets `PAPER_METHOD_NAME` and everything else keeps the
#: name the issue threads use.
INTERNAL_METHOD_NAME = "TRMC-BO"


def label(method: str) -> str:
    name = DISPLAY_NAMES.get(method, method)
    override = os.environ.get("PAPER_METHOD_NAME")
    if override:
        name = name.replace(INTERNAL_METHOD_NAME, override)
    return name


def _with_labels(df: pd.DataFrame, order: bool = True) -> pd.DataFrame:
    """Replace the `method` key with its label, optionally in canonical order."""
    out = df.copy()
    if order:
        rank = {m: i for i, m in enumerate(DISPLAY_ORDER)}
        out = out.assign(_o=out["method"].map(lambda m: rank.get(m, len(rank))))
        out = out.sort_values(["_o", "method"]).drop(columns="_o")
    out.insert(0, "method_label", out["method"].map(label))
    return out.drop(columns="method").rename(columns={"method_label": "method"}).reset_index(drop=True)


def load_runs(root: Path = RESULT_ROOT, suite: str = "matformbench",
              protocol: str = "range_adapted") -> pd.DataFrame:
    """One row per completed cell."""
    base = root / suite / protocol
    rows = []
    for path in sorted(base.glob("*/*/*/*/metrics.json")):
        if not (path.parent / "run.json").exists():
            continue  # incomplete cell
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    if not rows:
        raise FileNotFoundError(f"no completed runs under {base}")
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """Mean +/- std per group, plus the censoring-aware first-hit summary."""
    by = by or ["task_id", "width", "method"]
    grouped = df.groupby(by, sort=False)

    numeric = [c for c in df.select_dtypes(include=[np.number]).columns if c != "seed"]
    out = grouped[numeric].agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]

    out["n_seeds"] = grouped.size()
    out["found_rate"] = grouped["found"].mean()
    out["first_hit_median"] = grouped.apply(
        lambda g: g.loc[g["found"], "first_hit"].median() if g["found"].any() else np.nan,
        include_groups=False,
    )
    out["hit_rate_seed_std"] = grouped["hit_rate"].std()
    return out.reset_index()


def core_table(df: pd.DataFrame, width: str | None = None,
               sort_by: str | None = None) -> pd.DataFrame:
    """Backbone §1.7: one row per method, averaged over tasks.

    Rows come out in `DISPLAY_ORDER` unless `sort_by` names a column, so the
    ablation ladder stays adjacent and readable.
    """
    sub = df if width is None else df[df["width"] == width]
    per_cell = summarize(sub, by=["task_id", "width", "method"])
    rows = []
    for method, group in per_cell.groupby("method", sort=False):
        rows.append({
            "method": method,
            "hit_rate": group["hit_rate_mean"].mean(),
            "found_rate": group["found_rate"].mean(),
            "first_hit_median": group["first_hit_median"].median(),
            "violation_lower": group["violation_lower_mean"].mean(),
            "violation_upper": group["violation_upper_mean"].mean(),
            "violation_total_median": group["violation_total_median_mean"].mean(),
            "invalid_rate": group["invalid_rate_mean"].mean(),
            "hit_ball_eqvol": group["hit_rate_ball_equal_volume_mean"].mean(),
            "hit_ball_insc": group["hit_rate_ball_inscribed_mean"].mean(),
            "diversity": group["diversity_norm_mean"].mean(),
            "seed_std": group["hit_rate_seed_std"].mean(),
            "acqf_failure_rate": group["acqf_total_failure_rate_mean"].mean(),
            "sec_per_iter": group["wall_clock_per_iter_mean"].mean(),
        })
    table = pd.DataFrame(rows)
    if sort_by:
        return _with_labels(table.sort_values(sort_by, ascending=False), order=False)
    return _with_labels(table)


def average_rank(df: pd.DataFrame, metric: str = "hit_rate") -> pd.DataFrame:
    """Average rank across (task, width) cells.

    The Range-Aware BO paper reports average rank rather than a mean over
    heterogeneous tasks, and for good reason: hit rates differ by an order of
    magnitude between wide and narrow, so a raw mean is dominated by the easy
    conditions.
    """
    per_cell = summarize(df, by=["task_id", "width", "method"])
    col = f"{metric}_mean"
    per_cell["rank"] = per_cell.groupby(["task_id", "width"])[col].rank(
        ascending=False, method="average"
    )
    out = per_cell.groupby("method")["rank"].agg(["mean", "std", "count"])
    out.columns = ["avg_rank", "rank_std", "n_cells"]
    # Performance ordering belongs here, so labels are applied without reordering.
    return _with_labels(out.sort_values("avg_rank").reset_index(), order=False)


def distinct_average_rank(counts: pd.DataFrame, value: str = "distinct") -> pd.DataFrame:
    """`average_rank`'s ranking logic, for a `distinct_counts()` frame.

    `average_rank` cannot rank by `distinct`: it is built on `summarize()`, which
    needs `found`/`first_hit`/`hit_rate` columns that only exist on a
    `metrics.json`-derived frame. `distinct_counts()` (from `trace.parquet`) has
    none of those, only `task_id, width, method, seed, valid, distinct`. This is
    that same rank-over-(task, width)-cells logic applied to the frame that
    actually has the diversity metric.
    """
    per_cell = counts.groupby(["task_id", "width", "method"])[value].mean().reset_index()
    per_cell["rank"] = per_cell.groupby(["task_id", "width"])[value].rank(
        ascending=False, method="average"
    )
    out = per_cell.groupby("method")["rank"].agg(["mean", "std", "count"])
    out.columns = ["avg_rank", "rank_std", "n_cells"]
    return _with_labels(out.sort_values("avg_rank").reset_index(), order=False)


def discovery_curves(
    root: Path = RESULT_ROOT, suite: str = "matformbench",
    protocol: str = "range_adapted", task_id: str | None = None,
    width: str | None = None,
) -> pd.DataFrame:
    """Cumulative valid discoveries against iteration, averaged over seeds."""
    base = root / suite / protocol
    frames = []
    for path in sorted(base.glob("*/*/*/*/trace.parquet")):
        method, width_dir, task_dir = path.parts[-3], path.parts[-4], path.parts[-5]
        if task_id and task_dir != task_id:
            continue
        if width and width_dir != width:
            continue
        trace = pd.read_parquet(path, columns=["iteration", "is_init", "hit"])
        loop = trace[~trace["is_init"]].sort_values("iteration")
        frames.append(pd.DataFrame({
            "task_id": task_dir, "width": width_dir, "method": method,
            "seed": path.parts[-2],
            "iteration": loop["iteration"].to_numpy(),
            "cumulative_valid": np.cumsum(loop["hit"].to_numpy()),
        }))
    if not frames:
        raise FileNotFoundError(f"no traces under {base}")
    allf = pd.concat(frames, ignore_index=True)
    curves = (
        allf.groupby(["task_id", "width", "method", "iteration"])["cumulative_valid"]
        .agg(["mean", "std", "count"]).reset_index()
    )
    curves["method_label"] = curves["method"].map(label)
    return curves


def render(table: pd.DataFrame, float_fmt: str = "{:.4g}") -> str:
    return table.to_string(index=False, float_format=lambda v: float_fmt.format(v))


# ---------------------------------------------------------------------------
# Distinct valid designs
#
# `hit_rate` counts evaluations, not designs. A method that converges onto one
# in-window point and re-proposes nearby scores well on it while returning a
# single usable recipe: the duplicate guard in the campaign loop rejects only
# exact repeats (tolerance 1e-8), so near-duplicates all count as hits.
#
# The metric below counts designs that are pairwise further apart than `delta`
# in the normalized design space -- the delta-uniqueness notion the Range-Aware
# BO paper scores on. Reported next to the raw count, never instead of it: the
# two answer different questions ("how often does a proposal meet spec" vs "how
# many materially different qualifying formulations came out"), and which one is
# primary is a claim the write-up has to make explicitly.
#
# `delta` is not derived from any physical tolerance the benchmark provides, so
# the ranking must be shown to be delta-invariant -- see `delta_sweep`.

DEFAULT_DELTA = 0.1


def _normalize_designs(X: np.ndarray, task) -> np.ndarray:
    lo, hi = task.bounds
    span = np.where(hi - lo == 0, 1.0, hi - lo)
    return (X - lo) / span


def _greedy_distinct(Z: np.ndarray, delta: float) -> np.ndarray:
    """Indices of a greedily chosen subset that is pairwise > delta apart."""
    kept: list[int] = []
    for i, z in enumerate(Z):
        if all(np.linalg.norm(z - Z[j]) > delta for j in kept):
            kept.append(i)
    return np.asarray(kept, dtype=int)


def valid_designs(
    task, root: Path = RESULT_ROOT, suite: str = "matformbench",
    protocol: str = "range_adapted", width: str | None = None,
    max_seed: int | None = 10, exclude_fallback: bool = False,
) -> pd.DataFrame:
    """Every in-window proposal, one row per design, with a `distinct` flag.

    Reads traces rather than `metrics.json` because the design coordinates are
    needed, and so this works on results produced before the metric existed.

    `exclude_fallback` drops iterations the loop spent on a random design because
    the acquisition returned nothing usable (`loop.py` notes `random_fallback`
    when every duplicate retry collapsed or the optimizer failed outright). It
    exists because on HOIP that safety net, not the acquisition, produced most of
    what the methods "found": M5_full's 1.00 distinct designs on `dense` are 0.00
    once those iterations are removed. Any suite with a non-zero fallback rate is
    under the same suspicion, so both counts are reported rather than one.

    A cell whose trace has no `cnt_random_fallback` column had no fallback: the
    recorder only emits keys that were noted at least once, and `metrics.py`
    reads the rate as `counters.get("random_fallback", 0.0)`. So an absent column
    means the two counts coincide, which is the case for every MatFormBench and
    Olympus cell measured so far.
    """
    base = root / suite / protocol / task.task_id
    xcols = [f"x_{n}" for n in task.x_names]
    frames: list = []
    cells: list = []
    for path in sorted(base.glob("*/*/*/trace.parquet")):
        width_dir, method, seed_dir = path.parts[-4], path.parts[-3], path.parts[-2]
        if width and width_dir != width:
            continue
        seed = int(seed_dir.replace("seed", ""))
        if max_seed is not None and seed >= max_seed:
            continue
        trace = pd.read_parquet(path)
        loop = trace[~trace["is_init"]].sort_values("iteration")
        if exclude_fallback and "cnt_random_fallback" in loop.columns:
            loop = loop[loop["cnt_random_fallback"].fillna(0.0) <= 0]
        hits = loop[loop["hit"]][["is_init", "hit", "iteration"] + xcols]
        # A cell with no hits still has to appear downstream as a zero. Dropping
        # it here would remove it from the averages and quietly favour whichever
        # method fails most often.
        frames.append(hits.assign(task_id=task.task_id, width=width_dir,
                                  method=method, seed=seed))
        cells.append((task.task_id, width_dir, method, seed))
    if not cells:
        return pd.DataFrame(columns=xcols + ["task_id", "width", "method", "seed",
                                             "iteration", "distinct"])
    out = pd.concat(frames, ignore_index=True)
    out.attrs["cells"] = cells
    flags = np.zeros(len(out), dtype=bool)
    for _, idx in out.groupby(["width", "method", "seed"]).groups.items():
        pos = out.index.get_indexer(idx)
        Z = _normalize_designs(out.loc[idx, xcols].to_numpy(float), task)
        flags[pos[_greedy_distinct(Z, DEFAULT_DELTA)]] = True
    return out.assign(distinct=flags)


#: How many random orders `_distinct_count` averages over for the non-greedy
#: strategies. Twenty is enough to separate the strategies on every suite here
#: while keeping the whole delta audit a seconds-long post-hoc pass over
#: `trace.parquet` -- nothing is re-run.
_ORDER_RESAMPLES = 20


def _distinct_count(Z: np.ndarray, delta: float, strategy: str = "greedy",
                    seed: int = 0) -> float:
    """Size of a delta-separated subset of `Z`, under one of three orderings.

    The greedy rule keeps the first point and then every later point further than
    `delta` from all kept ones, so it depends on the order the designs arrive in.
    That is a property worth stating rather than hiding, and it cuts both ways:

    ``greedy``
        Evaluation order -- the order the campaign actually produced. This is the
        default and the one the tables report, because it is the causally honest
        count: it is what a chemist reading the campaign log top to bottom would
        have in hand, and it never credits a method for an ordering it did not
        run.
    ``shuffled``
        Mean over `_ORDER_RESAMPLES` random orders. Answers the reviewer's
        question directly -- does the ranking survive if the arrival order is
        scrambled?
    ``maximal``
        Best over the same random orders: a lower bound on the true
        delta-packing number of the recovered set. Labelled a bound rather than
        a maximum because computing the exact one is a maximum-independent-set
        problem, which is NP-hard and not worth solving for a robustness check.

    A dense cluster contributes exactly 1 under all three, which is the intent:
    the metric counts materially different formulations, not proposals.
    """
    if strategy == "greedy":
        return float(len(_greedy_distinct(Z, delta)))
    if strategy not in ("shuffled", "maximal"):
        raise ValueError(f"unknown strategy {strategy!r}")
    if len(Z) == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    sizes = []
    for _ in range(_ORDER_RESAMPLES):
        order = rng.permutation(len(Z))
        sizes.append(len(_greedy_distinct(Z[order], delta)))
    return float(np.mean(sizes) if strategy == "shuffled" else np.max(sizes))


def order_sensitivity(tasks: dict, delta: float = DEFAULT_DELTA, **kw) -> pd.DataFrame:
    """The headline mean under all three orderings, per method.

    If the three columns rank the methods the same way, the greedy rule's order
    dependence is not carrying the result -- which is the only thing this table
    is asked to establish.
    """
    frames = [distinct_counts(tasks, delta=delta, strategy=st, **kw).assign(strategy=st)
              for st in ("greedy", "shuffled", "maximal")]
    out = pd.concat(frames, ignore_index=True)
    wide = out.groupby(["method", "strategy"])["distinct"].mean().unstack("strategy")
    wide = wide[["greedy", "shuffled", "maximal"]]
    for col in wide.columns:
        wide[f"rank_{col}"] = wide[col].rank(ascending=False, method="min").astype(int)
    return _with_labels(wide.reset_index())


def distinct_counts(tasks: dict, delta: float = DEFAULT_DELTA,
                    strategy: str = "greedy", **kw) -> pd.DataFrame:
    """Per-(task, width, method, seed) counts of valid and distinct designs.

    Every completed cell appears, including cells where the method found nothing:
    those are genuine zeros and must not be silently dropped from the averages.
    """
    rows = []
    for task in tasks.values():
        df = valid_designs(task, **kw)
        if df.empty:
            continue
        xcols = [f"x_{n}" for n in task.x_names]
        found = {}
        for (w, m, s), g in df.groupby(["width", "method", "seed"]):
            Z = _normalize_designs(g[xcols].to_numpy(float), task)
            found[(w, m, s)] = (len(g), _distinct_count(Z, delta, strategy))
        for tid, w, m, s in df.attrs.get("cells", []):
            valid, distinct = found.get((w, m, s), (0, 0))
            rows.append({"task_id": tid, "width": w, "method": m, "seed": s,
                         "valid": valid, "distinct": distinct})
    return pd.DataFrame(rows)


def seed_bootstrap_ci(counts: pd.DataFrame, value: str = "distinct",
                      n_boot: int = 10_000, alpha: float = 0.05,
                      seed: int = 0) -> pd.DataFrame:
    """Percentile bootstrap interval for each method's headline mean.

    The resampling unit is the **seed**, not the cell. Within a seed every method
    starts from a byte-identical initial design, so cells sharing a seed are not
    independent draws and resampling them individually would report an interval
    several times too narrow. Resampling seeds keeps that dependence intact.

    The statistic is the same one the tables print: mean over (task, width) of
    the per-cell value, here computed by averaging a seed's cells and then
    averaging the resampled seeds. That is only identical to the printed grand
    mean when the seed is present in every cell, which the sweep guarantees, so a
    partially-filled matrix is dropped rather than silently reported on.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for method, g in counts.groupby("method"):
        per_seed = g.groupby("seed")[value].mean()
        n_cells = g.groupby("seed").size()
        if per_seed.empty or n_cells.nunique() != 1:
            rows.append({"method": method, "mean": per_seed.mean() if len(per_seed) else np.nan,
                         "ci_lo": np.nan, "ci_hi": np.nan, "n_seeds": len(per_seed),
                         "ragged": n_cells.nunique() != 1})
            continue
        v = per_seed.to_numpy(float)
        draws = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
        lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
        rows.append({"method": method, "mean": float(v.mean()),
                     "ci_lo": float(lo), "ci_hi": float(hi),
                     "n_seeds": len(v), "ragged": False})
    return _with_labels(pd.DataFrame(rows))


def paired_comparison(counts: pd.DataFrame, reference: str = "M5_full",
                      value: str = "distinct") -> pd.DataFrame:
    """Win/tie/loss against `reference`, paired within (task, width, seed).

    A difference of means says how far apart two methods are on average; it does
    not say whether one beats the other *on the same problem*, which is the
    question a reader with one campaign to spend is actually asking. Pairing
    answers that, and it is legitimate here for a specific reason: within a seed
    every method starts from a byte-identical initial design and optimizes a
    byte-identical frozen window (`eval/scripts/audit_leakage.py` asserts both).
    So a (task, width, seed) cell is the *same* problem handed to two methods,
    and the pair is a matched observation rather than two independent draws.

    The p-value is a two-sided exact sign test on the non-tied pairs. A sign test
    rather than a t-test because the per-cell counts are small integers with a
    floor at zero and a task-dependent ceiling -- distinctly not normal -- and
    rather than Wilcoxon signed-rank because the magnitudes are not on a
    meaningful common scale across tasks whose valid regions differ in capacity
    by three orders of magnitude (7 on HOIP against 4,000 on the formulation
    tasks). Ties are excluded from the test, which is standard and is also why
    `n_pairs` is reported beside it: an 8-1 record over 45 cells is a weaker
    claim than 8-1 over 9.
    """
    from scipy.stats import binomtest

    key = ["task_id", "width", "seed"]
    if reference not in set(counts["method"]):
        raise KeyError(f"reference method {reference!r} is not in these runs")
    ref = counts[counts["method"] == reference].set_index(key)[value]

    rows = []
    for method, g in counts.groupby("method"):
        if method == reference:
            continue
        other = g.set_index(key)[value]
        common = ref.index.intersection(other.index)
        if len(common) == 0:
            continue
        diff = (ref.loc[common] - other.loc[common]).to_numpy(float)
        win, loss = int((diff > 0).sum()), int((diff < 0).sum())
        tie = int(len(diff) - win - loss)
        n = win + loss
        p = float(binomtest(win, n, 0.5).pvalue) if n else float("nan")
        rows.append({
            "method": method, "win": win, "tie": tie, "loss": loss,
            "n_pairs": len(diff), "mean_diff": float(diff.mean()), "p": p,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # Holm-Bonferroni across the comparisons in this call. One reference is
    # tested against every other method at once, so the family is exactly this
    # table and correcting over it is the honest unit -- reporting a dozen
    # uncorrected p-values invites picking the smallest. Holm rather than plain
    # Bonferroni because it is uniformly more powerful at the same familywise
    # error rate, and rather than Benjamini-Hochberg because the claim being
    # defended is "this method beats that one", where a false positive is a
    # wrong claim about a specific baseline, not a tolerable share of a
    # discovery list.
    out = out.sort_values("p").reset_index(drop=True)
    m = len(out)
    adjusted, running = [], 0.0
    for i, raw in enumerate(out["p"]):
        val = min(1.0, (m - i) * float(raw))
        running = max(running, val)  # enforce monotonicity
        adjusted.append(running)
    out["p_holm"] = adjusted
    return _with_labels(out)


#: The two independent ways the two-sided posterior-mean constraint (C2) can be
#: absent from an iteration on which it was nominally switched on. Both are
#: needed: reading only the first makes the full method look perfectly clean.
#:
#:   hard_constraint_dropped
#:       C2 was attached, every restart was rejected by botorch's own recheck of
#:       the initial conditions, and `_optimize_with_soft_fallback` re-ran the
#:       iteration with `nonlinear_inequality_constraints = None`. This is the
#:       C3-off failure mode: unfiltered starts rarely satisfy the window.
#:   restart_filter_exhausted
#:       C3 was on but found *zero* in-band starts, so `_sample_initial_conditions`
#:       returned the unfiltered pool with `hard_ok=False` -- and C2 was then never
#:       attached in the first place. Because it was never attached it can never be
#:       "dropped", so this path leaves `hard_constraint_dropped` at zero.
_C2_INACTIVE_COUNTERS = ("cnt_hard_constraint_dropped", "cnt_restart_filter_exhausted")

#: Methods that attach the two-sided posterior-mean constraint at all. Listed
#: rather than inferred, because there is no positive telemetry for "C2 was
#: attached this iteration" -- only for the two ways it can go missing. Without
#: the list a method that never had a constraint is indistinguishable from one
#: whose constraint held every time: both record zero of both counters, and both
#: would be reported as 100% active. Keep in step with the `VARIANTS` and `ARMS`
#: tables in `eval/methods/`.
C2_METHODS = frozenset({
    "M4_range_prob_constraints",
    "M5_full",
    "M10_full_feasw",
    "M11_full_feasf",
    "M12_full_feas",
    "M5p_full_product",
    "M5q_full_product_ei",
    "M14_constraints_only",
    "M16_constraints_starts",
})


def constraint_activity(
    task, root: Path = RESULT_ROOT, suite: str = "matformbench",
    protocol: str = "range_adapted", width: str | None = None,
    max_seed: int | None = 10,
) -> pd.DataFrame:
    """Share of loop iterations on which the window constraint was really in force.

    botorch requires initial conditions that already satisfy a nonlinear
    constraint, so C2 cannot be applied to a start that is out of band. C3 is what
    supplies in-band starts. When it is off, or when it is on but starves, the
    optimizer silently proceeds without the constraint -- the run completes, the
    numbers look plausible, and nothing in the headline tables shows it.

    This counts that, the same way the campaign already counts random-fallback
    iterations, and for the same reason: an ablation row is only the configuration
    it is named after if the component was actually switched on.

    Returns one row per method with `active` in [0, 1] and the two inactive paths
    broken out, because they mean different things -- see `_C2_INACTIVE_COUNTERS`.
    Methods that never attach the constraint get NaN rather than 1.0: they are not
    perfectly active, the question does not apply to them.
    """
    base = root / suite / protocol / task.task_id
    rows = []
    for path in sorted(base.glob("*/*/*/trace.parquet")):
        width_dir, method, seed_dir = path.parts[-4], path.parts[-3], path.parts[-2]
        if width and width_dir != width:
            continue
        seed = int(seed_dir.replace("seed", ""))
        if max_seed is not None and seed >= max_seed:
            continue
        trace = pd.read_parquet(path)
        loop = trace[~trace["is_init"]]
        if loop.empty:
            continue
        counts = {}
        for col in _C2_INACTIVE_COUNTERS:
            # An absent column means the counter never fired: the recorder only
            # emits keys it was asked to note at least once.
            counts[col] = (
                int((loop[col].fillna(0.0) > 0).sum()) if col in loop.columns else 0
            )
        inactive = counts[_C2_INACTIVE_COUNTERS[0]] + counts[_C2_INACTIVE_COUNTERS[1]]
        rows.append({
            "task_id": task.task_id, "width": width_dir, "method": method,
            "seed": seed, "n_loop": len(loop),
            "dropped": counts["cnt_hard_constraint_dropped"],
            "starved": counts["cnt_restart_filter_exhausted"],
            "inactive": inactive,
        })
    if not rows:
        return pd.DataFrame(columns=["method", "active", "dropped_rate",
                                     "starved_rate", "n_cells"])
    df = pd.DataFrame(rows)
    agg = df.groupby("method").agg(
        n_loop=("n_loop", "sum"), dropped=("dropped", "sum"),
        starved=("starved", "sum"), inactive=("inactive", "sum"),
        n_cells=("seed", "count"),
    ).reset_index()
    agg["dropped_rate"] = agg["dropped"] / agg["n_loop"]
    agg["starved_rate"] = agg["starved"] / agg["n_loop"]
    agg["active"] = 1.0 - agg["inactive"] / agg["n_loop"]
    off = ~agg["method"].isin(C2_METHODS)
    agg.loc[off, ["active", "dropped_rate", "starved_rate"]] = np.nan
    return _with_labels(
        agg[["method", "active", "dropped_rate", "starved_rate", "n_cells"]]
    )


def constraint_activity_all(tasks: dict, **kw) -> pd.DataFrame:
    """`constraint_activity` pooled over every task of a protocol.

    Only methods that attach the constraint are returned; for the rest the
    question is not applicable and a row of blanks would invite the reader to
    compare them against the ones it does apply to.
    """
    frames = [constraint_activity(t, **kw) for t in tasks.values()]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=["method", "active", "dropped_rate",
                                     "starved_rate", "n_cells"])
    out = pd.concat(frames, ignore_index=True)
    # Weight by cells so a task with more widths does not count the same as one
    # with fewer.
    out["_w"] = out["n_cells"]
    grouped = out.groupby("method").apply(
        lambda g: pd.Series({
            "active": np.average(g["active"], weights=g["_w"]),
            "dropped_rate": np.average(g["dropped_rate"], weights=g["_w"]),
            "starved_rate": np.average(g["starved_rate"], weights=g["_w"]),
            "n_cells": int(g["n_cells"].sum()),
        }),
        include_groups=False,
    ).reset_index()
    return grouped.dropna(subset=["active"]).reset_index(drop=True)


def main_table(counts: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    """Distinct designs alongside raw hits, with the concentration ratio.

    `concentration` = distinct / valid is what makes the mechanism legible: a
    method near 1.0 returns a different recipe every time it succeeds, while a
    low ratio means it kept re-sampling one basin.
    """
    per_cell = counts.groupby(["task_id", "width", "method"]).agg(
        distinct=("distinct", "mean"), distinct_sd=("distinct", "std"),
        valid=("valid", "mean"),
    ).reset_index()
    met = metrics.groupby("method").agg(
        first_hit=("first_hit_censored", "mean"),
        violation=("violation_total_median", "mean"),
        sec_per_iter=("wall_clock_per_iter", "mean"),
    )
    rows = []
    for method, g in per_cell.groupby("method"):
        d, v = g["distinct"].mean(), g["valid"].mean()
        r = met.loc[method] if method in met.index else None
        rows.append({
            "method": method,
            "distinct": d,
            "valid": v,
            "concentration": d / v if v > 0 else np.nan,
            "distinct_cv": g["distinct_sd"].mean() / d if d > 0 else np.nan,
            "first_hit": None if r is None else r["first_hit"],
            "violation": None if r is None else r["violation"],
            "sec_per_iter": None if r is None else r["sec_per_iter"],
        })
    return _with_labels(pd.DataFrame(rows))


def fallback_comparison(tasks: dict, metrics: pd.DataFrame | None = None,
                        delta: float = DEFAULT_DELTA, **kw) -> pd.DataFrame:
    """Distinct designs with and without the loop's random-fallback iterations.

    The two columns coincide exactly when a method never fell back, which is the
    whole point of printing them side by side: it turns "these numbers are not a
    fallback-policy comparison" from an assurance into a column the reader can
    check. Where they diverge, `distinct_nonfb` is the one that describes the
    acquisition.
    """
    allc = distinct_counts(tasks, delta=delta, **kw)
    nonfb = distinct_counts(tasks, delta=delta, exclude_fallback=True, **kw)
    out = (allc.groupby("method")["distinct"].mean().rename("distinct")
           .to_frame()
           .join(nonfb.groupby("method")["distinct"].mean().rename("distinct_nonfb")))
    out["lost_to_fallback"] = out["distinct"] - out["distinct_nonfb"]
    if metrics is not None and "random_fallback_rate" in metrics.columns:
        out = out.join(
            metrics.groupby("method")["random_fallback_rate"].mean().rename("fallback_rate")
        )
    return _with_labels(out.reset_index())


def delta_sweep(tasks: dict, deltas=(0.05, 0.1, 0.2, 0.4), **kw) -> pd.DataFrame:
    """Distinct counts across delta, to show the ranking does not depend on it."""
    out = None
    for d in deltas:
        c = distinct_counts(tasks, delta=d, **kw)
        col = c.groupby("method")["distinct"].mean().rename(f"d>{d}")
        out = col.to_frame() if out is None else out.join(col)
    return _with_labels(out.reset_index())


# ---------------------------------------------------------------------------
# MatFormBench's own batch protocol
#
# A different question from the sequential sweep, and the two must not be mixed
# in one table. Here the benchmark scores a *batch* of 100 suggestions produced
# from the initial design in one shot -- no feedback loop -- against its own
# composite: recommend, top-k over 5 rounds, design-set-size sensitivity, and
# stability across 10 internal seeds. We supply proposals; every metric is
# computed by MatFormBench's compiled scorers, so these numbers are directly
# comparable to the baselines it ships and not to anything we defined.
# ---------------------------------------------------------------------------

#: Sub-scores of MatFormBench's `Total_Score`, in the order its own report lists
#: them. Kept explicit rather than globbed so a schema change surfaces as a
#: KeyError instead of a silently shorter table.
NATIVE_SCORES = [
    "Total_Score", "Score_Success", "Score_Efficiency",
    "Score_Explore", "Score_Robust", "Score_Stability", "HV",
]


def load_native(root: Path = RESULT_ROOT, suite: str = "matformbench") -> pd.DataFrame:
    """One row per native-protocol result file."""
    rows = []
    for path in sorted((root / suite / "native").glob("*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        res = d["results"]
        row = {
            # "L1" + "Dataset-1" -> "L1-1", the id used everywhere else.
            "task_id": f"{d['task']['level']}-{d['task']['dataset'].split('-')[-1]}",
            "method": d["algorithm"]["optimizer"]["params"]["method"],
            "seed": d["task"]["seed"],
            "hit_rate_overall": res["hit_rate"]["hit_rate_overall"],
            "hit_at_least_1": res["hit_rate"]["hit_rate_at_least_1"],
            "diversity": res["diversity_score"],
        }
        row.update({k: res["score_pack"][k] for k in NATIVE_SCORES})
        # Stability is reported per internal seed; its spread is the point of the
        # sub-score, so keep the spread rather than only the mean.
        by_seed = list(res["stability"]["Stability_HitRateBySeed"].values())
        row["stability_hit_mean"] = float(np.mean(by_seed))
        row["stability_hit_std"] = float(np.std(by_seed))
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"no native results under {root / suite / 'native'}")
    return pd.DataFrame(rows)


def native_table(df: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """Mean over seeds, one row per method (or per `by` group)."""
    keys = by or ["method"]
    cols = NATIVE_SCORES + [
        "hit_rate_overall", "hit_at_least_1", "diversity",
        "stability_hit_mean", "stability_hit_std",
    ]
    out = df.groupby(keys)[cols].mean().reset_index()
    out.insert(len(keys), "n_seeds", df.groupby(keys)["seed"].nunique().values)
    return _with_labels(out) if keys == ["method"] else out
