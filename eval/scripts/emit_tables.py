"""Generate the paper's result tables as .tex, from the runs on disk.

Until now every number in `paper/sections/results.tex` was transcribed by hand
from `eval.scripts.report`. That is why Appendix D was still a TODO: keeping a
hand-copied table in step with a sweep that is still landing cells is work
nobody does twice. It also means a stale table is indistinguishable from a
current one. This script removes the transcription step.

    PYTHONPATH=core:. PAPER_METHOD_NAME=TRMC-BO \\
        .venv-eval/bin/python -m eval.scripts.emit_tables --out paper/sections/tables

Emits, one file each:

    tab_main.tex          headline: distinct designs with a 95% seed-bootstrap
                          interval, beside raw hits, for all three protocols
    tab_paired.tex        win/tie/loss against the full method, paired within
                          (task, width, seed), with a two-sided sign test
    tab_factorial.tex     the ablation: range objective x optimizer treatment
    tab_activity.tex      how often the window constraint was really applied --
                          the table that keeps tab_factorial honest
    tab_order.tex         the headline under three delta-uniqueness orderings
    tab_comboo.tex        every method restricted to the tasks COMBOO completed
    tab_cost.tex          median seconds per iteration, per method and suite

An arm whose sweep has not finished prints `(partial)` rather than a mean over
whichever cells landed first; see `_mean_if_complete`.

`PAPER_METHOD_NAME` must be set for anything going into the submission: the
internal name identifies the affiliation and the submission is double-blind.
The script refuses to write a double-blind-unsafe table rather than leaving it
to be noticed in proof.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from eval.benchmarks.registry import resolve
from eval.harness import aggregate as agg

#: (label, suite, protocol) for the three protocols the headline table spans.
#: HOIP is deliberately absent: it is claim (iv)'s negative result and has its
#: own table, and averaging it into a headline would bury exactly the finding the
#: paper reports.
PROTOCOLS = [
    ("MatFormBench", "matformbench", "range_adapted"),
    ("Olympus (branin)", "olympus", "branin"),
    ("Olympus (mixtures)", "olympus", "emulator"),
]

#: Row order for the headline table: baselines, then the ablation. Anything on
#: disk that is not listed goes to the factorial table instead of being dropped.
MAIN_ROWS = [
    "M0_random",
    "M1_qehvi_extremum",
    "M2_target_distance",
    "M6_range_aware_tb",
    "M6b_range_aware_tb_inscribed",
    None,  # \midrule -- isolates COMBOO's declared-subset row from the
           # full-suite baselines above and SCBO below (see SUBSET_MARK).
    "M8_comboo",
    None,  # \midrule
    "M7_scbo",
    None,  # \midrule
    # The compact headline carries only the selected configuration -- published
    # baselines above the rule, the one method below it. The ablation ladder
    # (M3/M13/M16 vs M5) is reported in the text as a short arrow-chain and in
    # full in `tab_factorial`, not as extra rows here; see FACTORIAL_ROWS.
    "M5_full",
]

#: The ablation grid, as (range objective, optimizer treatment, method).
#:
#: NOT a 2x2x2 over C1/C2/C3. C2 and C3 are not independent factors: the
#: acquisition optimizer requires initial conditions that already satisfy a
#: nonlinear constraint, and C3 is what produces them. So the cell "C2 without
#: C3" is not a design point -- it is C2 attached to starts that mostly fail the
#: recheck, after which the optimizer proceeds unconstrained on 9-83% of
#: iterations depending on the suite. `M4_range_prob_constraints` and
#: `M14_constraints_only` are those cells; they belong in the constraint-activity
#: table, not here, where quoting them as an ablation rung would misdescribe them.
FACTORIAL_ROWS = [
    ("--", "none", "M1_qehvi_extremum"),
    ("\\Cone", "none", "M3_range_prob"),
    ("--", "\\Cthree", "M15_starts_only"),
    ("\\Cone", "\\Cthree", "M13_range_prob_starts"),
    ("--", "\\Ctwo+\\Cthree", "M16_constraints_starts"),
    ("\\Cone", "\\Ctwo+\\Cthree", "M5_full"),
]

MISSING = "--"

#: Methods that ran but are out of scope for this submission, filtered from every
#: generated table so the decision is enforced in one place rather than
#: remembered per table.
#:
#: The C4 feasibility variants are complete on HOIP (30 cells each) and
#: MatFormBench (150 each). This version uses C4 only to test whether a
#: feasibility classifier rescues claim (iv) -- that result is hand-written in
#: the HOIP appendix -- and does not report the formulation-suite result. See
#: `eval/predictions_c4.md`, whose C4p is refuted there.
EXCLUDED_METHODS = frozenset({
    "M10_full_feasw",
    "M11_full_feasf",
    "M12_full_feas",
})


def _drop_excluded(df, column: str = "method"):
    """Remove out-of-scope methods, whether the column holds keys or labels."""
    if df is None or len(df) == 0 or column not in df.columns:
        return df
    labels = {agg.label(m) for m in EXCLUDED_METHODS}
    return df[~df[column].isin(EXCLUDED_METHODS | labels)].reset_index(drop=True)


def _esc(text: str) -> str:
    return text.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")


#: Seeds 0-9 are the protocol. `aggregate.valid_designs` caps there, so the
#: metrics frame has to cap identically or the two disagree on the denominator:
#: L1-1/wide/M0_random has ten extra seeds on disk, which made `tab_diversity`
#: print an exact-unique count (1.9) larger than the same rows' hit count (1.6)
#: in `tab_main` -- impossible, since exact-unique is a subset of the hits.
MAX_SEED = 10


def _load(suite: str, protocol: str, delta: float):
    """(counts, metrics, tasks) for one protocol, or None if it has no runs."""
    S = resolve(suite)
    metrics = agg.load_runs(suite=suite, protocol=protocol)
    metrics = metrics[metrics["seed"] < MAX_SEED].reset_index(drop=True)
    if metrics.empty:
        return None
    tasks = {t: S.load_task(t) for t in sorted(metrics["task_id"].unique())}
    counts = agg.distinct_counts(tasks, delta=delta, suite=suite, protocol=protocol)
    return _drop_excluded(counts), _drop_excluded(metrics), tasks


def _fmt(x: float, nd: int = 2) -> str:
    return MISSING if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


#: Marker for a cell whose sweep has not finished. A partially-filled arm is far
#: more dangerous than a missing one: it prints a plausible number computed over
#: whichever (task, width, seed) cells happened to land first, which are not a
#: random subset -- the runner walks the matrix in order, so an incomplete arm is
#: biased toward the first task and the widest window.
PARTIAL = "(partial)"


#: Methods that deliberately run on a subset of a protocol's tasks, with the
#: tasks they cover. Needed because "ran 90 of 150 cells" has two very different
#: causes and the cell count cannot tell them apart: an unfinished sweep, which
#: must not be reported, and a declared restriction, which must be reported and
#: marked. COMBOO's is cost -- one L4-1 cell ran over 9.5 hours without
#: finishing -- and it is documented in the appendix rather than hidden in an
#: omission.
TASK_SUBSETS: dict[str, dict[str, list[str]]] = {
    "M8_comboo": {"matformbench": ["L1-1", "L2-1", "L3-1"]},
}

#: Appended to a mean computed over a declared task subset, so it is never
#: silently compared against a full-suite average in the same column.
SUBSET_MARK = "$^{\\dagger}$"


def _subset_tasks(key: str, suite: str) -> list[str] | None:
    return TASK_SUBSETS.get(key, {}).get(suite)


def _mean_if_complete(counts, key: str, expected: int, nd: int = 2,
                      suite: str | None = None) -> str:
    """Mean distinct designs for `key`, or a marker if its sweep is unfinished.

    A method with a declared task restriction is judged complete against that
    restriction and marked, rather than being called partial for tasks it was
    never going to run.
    """
    g = counts[counts["method"] == key]
    if g.empty:
        return MISSING
    subset = _subset_tasks(key, suite) if suite else None
    if subset is not None:
        g = g[g["task_id"].isin(subset)]
        if g.empty:
            return MISSING
        # Expected cells scale with the share of tasks this method covers.
        n_tasks = max(counts["task_id"].nunique(), 1)
        expected = int(round(expected * len(subset) / n_tasks))
        if len(g) < expected:
            return PARTIAL
        return _fmt(g["distinct"].mean(), nd) + SUBSET_MARK
    if expected and len(g) < expected:
        return PARTIAL
    return _fmt(g["distinct"].mean(), nd)


def _expected_cells(counts) -> int:
    """How many cells a complete arm has: the most any method on this protocol has.

    Read off the data rather than the config so it stays right when a suite gains
    a task, and so a protocol where every arm is short is not silently rescaled --
    if nothing is complete, nothing claims to be.
    """
    if counts.empty:
        return 0
    return int(counts.groupby("method").size().max())


def _num(v) -> float | None:
    """Comparable float, or None for anything that must not rank (NaN, missing)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(f) else f


def _emphasize(rows: list[dict], ncols: int, higher_is_better: bool = True) -> None:
    """Bold the best cell of each column and underline the runner-up, in place.

    Emphasis is generated rather than applied by hand after the fact. It was
    hand-applied once, and when the ablation sweep completed and every number
    moved, the hand-edited tables were the ones that did not get regenerated --
    they carried an incomplete-sweep mean for a whole revision. Anything a
    reader sees has to come out of the same run of this script as the numbers.

    `rows` are dicts with `cells` (rendered strings) and `vals` (float or None,
    None meaning the cell is missing, partial, or otherwise not comparable --
    a declared task subset does not compete for "best", since it averages
    different tasks than the column it sits in).
    """
    for j in range(ncols):
        live = [(r["vals"][j], i) for i, r in enumerate(rows)
                if r["vals"][j] is not None]
        # Rank on the value as *printed*, not the underlying float. Three arms
        # sharing a displayed 0.076 differ in the sixth decimal, and ranking on
        # the float underlined exactly one of them -- three identical numbers in
        # a column with one marked reads as a typo, not as a tie.
        groups: dict[str, list[int]] = {}
        for v, i in live:
            groups.setdefault(rows[i]["cells"][j].partition(" \\tiny[")[0],
                              []).append(i)
        order = sorted(groups, key=lambda k: float(k.rstrip(SUBSET_MARK) or "nan"),
                       reverse=higher_is_better)
        for rank, key in enumerate(order[:2]):
            cmd = "textbf" if rank == 0 else "underline"
            for i in groups[key]:
                cell = rows[i]["cells"][j]
                # Emphasize the estimate, not the interval trailing it: a bolded
                # \tiny bootstrap interval reads as a second headline number.
                head, sep, tail = cell.partition(" \\tiny[")
                rows[i]["cells"][j] = f"\\{cmd}{{{head}}}{sep}{tail}"


def _wrap(body: str, header: str, caption: str, label: str,
          colspec: str, size: str = "\\footnotesize") -> str:
    return (
        "% GENERATED by eval/scripts/emit_tables.py -- edit that, not this file.\n"
        "\\begin{table}[t]\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\centering\n"
        f"{size}\n"
        f"\\begin{{tabular}}{{{colspec}}}\n"
        "\\toprule\n"
        f"{header}"
        "\\midrule\n"
        f"{body}"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def emit_main(data: dict, name: str) -> str:
    """Headline table: distinct designs with a bootstrap interval, beside hits.

    The `uniq.` column the hand-written table carried is dropped, not lost: it
    was exactly distinct/hits, so a reader can still form it, and the space buys
    the interval -- which a reader cannot reconstruct from anything printed.
    """
    per = {}
    counts_by_title: dict = {}
    for title, suite, protocol in PROTOCOLS:
        d = data.get((suite, protocol))
        if d is None:
            continue
        counts, metrics, _ = d
        # `seed_bootstrap_ci` returns rows keyed by display label (it calls
        # `_with_labels`), while `counts` is still keyed by registry name. Index
        # the interval by label and the hit count by name rather than assuming
        # they agree.
        ci = agg.seed_bootstrap_ci(counts, value="distinct").set_index("method")
        hits = counts.groupby("method")["valid"].mean()
        per[title] = (ci, hits)
        counts_by_title[title] = counts

    lines = []
    rows: list[dict] = []
    for key in MAIN_ROWS:
        if key is None:
            rows.append({"rule": True})
            continue
        cells, vals, seen = [], [], False
        for title, suite, _ in PROTOCOLS:
            if title not in per:
                cells += [MISSING, MISSING]
                vals += [None, None]
                continue
            ci, hits = per[title]
            lbl = agg.label(key)
            if lbl not in ci.index:
                cells += [MISSING, MISSING]
                vals += [None, None]
                continue
            seen = True
            # An unfinished arm must not print an interval either: a bootstrap
            # over the cells that landed first is a confident statement about a
            # biased subset.
            c = counts_by_title[title]
            status = _mean_if_complete(c, key, _expected_cells(c), suite=suite)
            if status == PARTIAL:
                cells += [PARTIAL, PARTIAL]
                vals += [None, None]
                continue
            # A declared task subset keeps its interval -- it is a complete
            # measurement of fewer tasks -- but carries the mark so the column is
            # not read as a like-for-like average.
            mark = SUBSET_MARK if status.endswith(SUBSET_MARK) else ""
            r = ci.loc[lbl]
            cells.append(
                f"{_fmt(r['mean'])}{mark} "
                f"\\tiny[{_fmt(r['ci_lo'], 1)},{_fmt(r['ci_hi'], 1)}]"
            )
            # Two decimals, matching the distinct column. At one decimal a cell
            # where every hit is already distinct (random on MatFormBench,
            # standard BO on branin: distinct == hits exactly) prints as
            # "1.65 / 1.6", contradicting the caption's hits >= distinct.
            cells.append(_fmt(hits.get(key, float("nan"))))
            # A subset average does not compete for "best" against full-suite
            # rows; it is a mean over different tasks.
            vals.append(None if mark else _num(r["mean"]))
            vals.append(None if mark else _num(hits.get(key, float("nan"))))
        if not seen:
            continue
        label = agg.label(key)
        if key == "M8_comboo":
            # State the subset inline rather than relying on the caption's
            # dagger alone -- a reader scanning the table should see it here.
            label = f"{label} (3-task subset)"
        rows.append({"label": label, "cells": cells, "vals": vals})

    body_rows = [r for r in rows if not r.get("rule")]
    _emphasize(body_rows, 2 * len(PROTOCOLS))
    for r in rows:
        if r.get("rule"):
            lines.append("\\midrule\n")
        else:
            lines.append(f"{_esc(r['label'])} & " + " & ".join(r["cells"]) + " \\\\\n")

    header = (
        "& \\multicolumn{2}{c}{MatFormBench$^{\\ast}$} & \\multicolumn{2}{c}{Olympus (branin)}"
        " & \\multicolumn{2}{c}{Olympus (mixtures)} \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
        "Method & distinct & hits & distinct & hits & distinct & hits \\\\\n"
        "& designs & & designs & & designs & \\\\\n"
    )
    caption = (
        "\\textbf{Headline result.} Counts per run in $50$ evaluations, averaged "
        "over a suite's tasks, three window widths and $10$ seeds; higher is "
        "better, \\textbf{bold} best in a column, \\underline{underline} the "
        "runner-up. \\emph{Distinct designs} are valid designs mutually more than "
        "$\\delta=0.1$ apart, with a $95\\%$ percentile bootstrap interval "
        "resampled over seeds; \\emph{hits} are all in-window evaluations, so "
        "hits $\\ge$ distinct. SCBO is single-output and COMBOO multi-output, so "
        "each runs where the other cannot. $\\dagger$: average over a declared "
        "task subset (Appendix~\\ref{app:comboo}). $\\ast$: the exploratory "
        "suite (Section~\\ref{sec:protocol})."
    )
    return _wrap("".join(lines), header, caption, "tab:main", "lrrrrrr")


def emit_paired(data: dict, reference: str = "M5_full") -> str:
    lines = []
    for title, suite, protocol in PROTOCOLS:
        d = data.get((suite, protocol))
        if d is None:
            continue
        counts = d[0]
        if reference not in set(counts["method"]):
            continue
        pc = agg.paired_comparison(counts, reference=reference)
        lines.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{{title}}}}} \\\\\n")
        for _, r in pc.iterrows():
            p = "$<10^{-4}$" if r["p_holm"] < 1e-4 else f"${r['p_holm']:.3f}$"
            lines.append(
                f"\\quad {_esc(r['method'])} & {int(r['win'])} & {int(r['tie'])} & "
                f"{int(r['loss'])} & {int(r['n_pairs'])} & {p} \\\\\n"
            )
    header = ("Method & win & tie & loss & cells & $p_{\\mathrm{Holm}}$ \\\\\n")
    caption = (
        "\\textbf{Paired comparison.} Each cell is one (task, width, seed); within "
        "a seed every method starts from a byte-identical initial design on a "
        "byte-identical window, so a cell is the same problem for any two "
        "methods. \\emph{win} counts cells where \\methodname\\ (full) recovered "
        "more distinct designs, and win/tie/loss sum to \\emph{cells}. $p$ is a "
        "two-sided sign test on the non-tied pairs, Holm-corrected within each "
        "suite block."
    )
    return _wrap("".join(lines), header, caption, "tab:paired", "lrrrrr")


#: Rows for the compact rank-reversal table (Table 2). Published baselines only,
#: plus the full method -- `M4_range_prob_constraints` is deliberately absent:
#: it names a configuration that ran unconstrained on 9.5-58.7% of its
#: iterations (see FACTORIAL_ROWS's note), so a rank built on it would be a rank
#: built on a method that wasn't run. No replacement row: the point of Table 2 is
#: the reversal on the methods a reader already knows, not the ablation.
REVERSAL_ROWS = [
    "M6b_range_aware_tb_inscribed",
    "M6_range_aware_tb",
    "M2_target_distance",
    "M5_full",
]

#: Confirmatory suites only. The reversal claim is scoped to Olympus by design
#: (non-negotiable claim 2): MatFormBench is the exploratory suite where the
#: metric was found, not where it is confirmed.
REVERSAL_PROTOCOLS = [
    ("Olympus (branin)", "olympus", "branin"),
    ("Olympus (mixtures)", "olympus", "emulator"),
]


def emit_reversal(data: dict) -> str:
    """Table 2: hit-rate rank vs. distinct-design rank, confirmatory suites only.

    Generated rather than hand-written for the same reason `tab_main` is: a
    hand-written version of this table carried an invalid ablation row
    (`\\Cone+\\Ctwo`) for one whole revision before anyone noticed, because
    nothing regenerated it when the ablation was corrected elsewhere.
    """
    lines = []
    for title, suite, protocol in REVERSAL_PROTOCOLS:
        d = data.get((suite, protocol))
        if d is None:
            continue
        # Rank within REVERSAL_ROWS only, not against every ablation/product
        # variant the sweep ran. Ranking against the full field mixes the
        # displayed baselines' standing with methods this table never shows, and
        # turns a crisp 1-4 reversal into an uninterpretable 7-of-14 -- the
        # comparison this table makes is among the rows it prints, not the whole
        # registry.
        counts = d[0][d[0]["method"].isin(REVERSAL_ROWS)]
        metrics = d[1][d[1]["method"].isin(REVERSAL_ROWS)]
        hits = agg.average_rank(metrics, metric="hit_rate").set_index("method")
        div = agg.distinct_average_rank(counts).set_index("method")
        lines.append((title, hits, div))

    if not lines:
        return _wrap("", "Method & rank by hit rate & rank by distinct designs \\\\\n",
                     "Rank reversal: no confirmatory runs found.",
                     "tab:reversal", "lrr")

    ncols = 2 * len(lines)
    header_cols = " & ".join(f"\\multicolumn{{2}}{{c}}{{{t}}}" for t, _, _ in lines)
    cmids = "".join(f"\\cmidrule(lr){{{2*i+2}-{2*i+3}}}" for i in range(len(lines)))
    # Two subheader lines, each spanning the full width: "1.3" is a rank, and a
    # single-word column head ("by diversity") left a reader to guess whether it
    # was a rank or a count.
    sub_top = " & ".join("mean rank by & mean rank by" for _ in lines)
    sub_bot = " & ".join("hit rate & distinct designs" for _ in lines)
    header = (f"& {header_cols} \\\\\n{cmids}\n"
              f"Method & {sub_top} \\\\\n& {sub_bot} \\\\\n")

    rows = []
    for key in REVERSAL_ROWS:
        lbl = agg.label(key)
        cells, vals = [], []
        seen = False
        for _, hits, div in lines:
            if lbl not in hits.index or lbl not in div.index:
                cells += [MISSING, MISSING]
                vals += [None, None]
                continue
            seen = True
            cells.append(_fmt(hits.loc[lbl, "avg_rank"], 1))
            cells.append(_fmt(div.loc[lbl, "avg_rank"], 1))
            vals.append(_num(hits.loc[lbl, "avg_rank"]))
            vals.append(_num(div.loc[lbl, "avg_rank"]))
        if not seen:
            continue
        rows.append({"label": lbl, "cells": cells, "vals": vals})

    # Best rank per column, not our own row: the point of the table is that the
    # column winners swap between the two metrics, which is only visible if the
    # emphasis tracks the winner rather than the proposed method. Lower is better.
    _emphasize(rows, 2 * len(lines), higher_is_better=False)
    body = [f"{_esc(r['label'])} & " + " & ".join(r["cells"]) + " \\\\\n" for r in rows]

    caption = (
        "\\textbf{The two metrics rank the evaluated target-range methods "
        "oppositely.} Cells are \\emph{ranks, not scores}: each method's mean rank "
        "among the four shown, over the (task, width) cells of the confirmatory "
        "Olympus tasks, so $1.0$ is best and $4.0$ worst. \\textbf{Bold} marks the "
        "best rank in a column, \\underline{underline} the runner-up."
    )
    return _wrap("".join(body), header, caption, "tab:reversal", "l" + "rr" * len(lines))


def emit_factorial(data: dict, delta: float) -> str:
    lines: list[str] = []
    rows: list[dict] = []
    for obj, treat, key in FACTORIAL_ROWS:
        cells = []
        for title, suite, protocol in PROTOCOLS:
            d = data.get((suite, protocol))
            if d is None:
                cells.append(MISSING)
                continue
            counts = d[0]
            cells.append(_mean_if_complete(counts, key, _expected_cells(counts),
                                           suite=suite))
        rows.append({"label": f"{obj} & {treat}", "cells": cells,
                     "vals": [_num(c) for c in cells]})

    # Emphasis marks the best cell of each column, not our own row. Bolding
    # `M5_full` unconditionally put bold on branin's 12.17 while the no-\Cone
    # rung scored 13.20 in the same column -- the table asserted a win the
    # numbers beside it denied, which is the one thing a reader checks here.
    _emphasize(rows, len(PROTOCOLS))
    for r in rows:
        lines.append(f"{r['label']} & " + " & ".join(r["cells"]) + " \\\\\n")

    header = (
        "\\multicolumn{2}{c}{configuration} & "
        "\\multicolumn{3}{c}{mean distinct designs per run} \\\\\n"
        "\\cmidrule(lr){1-2}\\cmidrule(lr){3-5}\n"
        "objective & optimizer & MatFormBench & branin & mixtures \\\\\n"
    )
    caption = (
        "\\textbf{Ablation.} The range objective crossed with the optimizer's "
        "window handling at three levels: none, restart filtering alone "
        "(\\Cthree), and restart filtering plus the posterior-mean constraint "
        "(\\Ctwo+\\Cthree). Cells are distinct valid designs per run at "
        "$\\delta=0.1$, averaged over a suite's tasks, three widths and $10$ "
        "seeds; higher is better, \\textbf{bold} best in a column, "
        "\\underline{underline} the runner-up. \\Ctwo\\ has no level of its own: "
        "the optimizer needs starts that already satisfy the constraint, and "
        "\\Cthree\\ is what supplies them (Table~\\ref{tab:activity})."
    )
    return _wrap("".join(lines), header, caption, "tab:factorial", "ccrrr")


def emit_activity(data: dict) -> str:
    """How often the window constraint was actually applied.

    The table that makes the ablation above honest: without it nothing tells a
    reader that the configuration named `C1+C2` spent most of its budget
    unconstrained.
    """
    lines = []
    for title, suite, protocol in PROTOCOLS:
        d = data.get((suite, protocol))
        if d is None:
            continue
        act = _drop_excluded(agg.constraint_activity_all(
            d[2], suite=suite, protocol=protocol))
        if act is None or act.empty:
            continue
        lines.append(f"\\multicolumn{{4}}{{l}}{{\\emph{{{title}}}}} \\\\\n")
        for _, r in act.iterrows():
            lines.append(
                f"\\quad {_esc(r['method'])} & {_fmt(100 * r['active'], 1)} & "
                f"{_fmt(100 * r['dropped_rate'], 1)} & "
                f"{_fmt(100 * r['starved_rate'], 1)} \\\\\n"
            )
    header = "Configuration & active \\% & dropped \\% & starved \\% \\\\\n"
    caption = (
        "\\textbf{Was the window constraint in force?} Percentages of a "
        "configuration's loop iterations, summing to $100$. \\emph{dropped}: every "
        "unfiltered start failed the optimizer's own feasibility recheck, so the "
        "constraint was removed and the iteration retried without it. "
        "\\emph{starved}: \\Cthree\\ found no in-band start, so the constraint was "
        "never attached, which is why that path never registers as dropped. Both "
        "are measured properties of our implementation; no cell is emphasized."
    )
    return _wrap("".join(lines), header, caption, "tab:activity", "lrrr", "\\small")


def emit_order(data: dict, delta: float) -> str:
    lines = []
    for title, suite, protocol in PROTOCOLS:
        d = data.get((suite, protocol))
        if d is None:
            continue
        counts, _, tasks = d
        os_df = _drop_excluded(agg.order_sensitivity(
            tasks, delta=delta, suite=suite, protocol=protocol))

        # An unfinished arm gets the same treatment here as everywhere else: its
        # mean is over whichever cells landed first, so it is marked rather than
        # printed, and it does not rank. Without this the table ranked a 35-of-150
        # arm first on MatFormBench, above the full method, on a biased subset.
        expected = _expected_cells(counts)
        n_cells = counts.groupby("method").size()
        complete = {
            agg.label(k): (n >= expected or _subset_tasks(k, suite) is not None)
            for k, n in n_cells.items()
        }
        os_df = os_df.assign(
            _complete=os_df["method"].map(complete).fillna(False).astype(bool))

        # Re-rank over the rows that actually compete. `order_sensitivity` ranks
        # the whole registry, so dropping rows afterwards left gaps -- MatFormBench
        # printed 3, 5, 6, ... with 1, 2 and 4 belonging to rows this submission
        # does not show, which reads as a suppressed result. The rank a reader can
        # check is the rank among the rows they can see and compare.
        for col in ("greedy", "shuffled", "maximal"):
            os_df[f"rank_{col}"] = (
                os_df[col].where(os_df["_complete"])
                .rank(ascending=False, method="min"))

        lines.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{{title}}}}} \\\\\n")
        for _, r in os_df.iterrows():
            if not r["_complete"]:
                cells = [PARTIAL] * 3 + [MISSING] * 3
            else:
                cells = [_fmt(r["greedy"]), _fmt(r["shuffled"]), _fmt(r["maximal"]),
                         f"{int(r['rank_greedy'])}", f"{int(r['rank_shuffled'])}",
                         f"{int(r['rank_maximal'])}"]
            lines.append(f"\\quad {_esc(r['method'])} & " + " & ".join(cells) + " \\\\\n")
    header = (
        "& \\multicolumn{3}{c}{mean distinct designs per run} & "
        "\\multicolumn{3}{c}{rank within suite} \\\\\n"
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
        "Method & greedy & shuffled & bound & greedy & shuffled & bound \\\\\n"
    )
    caption = (
        "\\textbf{The greedy rule's order dependence does not carry the result.} "
        "Left block: mean distinct designs per run, higher better. Right block: "
        "rank within the suite, $1$ best. \\emph{greedy} counts in evaluation "
        "order, as every other table does; \\emph{shuffled} averages $20$ random "
        "orders; \\emph{bound} takes the best of them, a lower bound on the "
        "$\\delta$-packing number of the recovered set. Arms whose sweep is "
        "unfinished are marked \\emph{(partial)} and do not rank."
    )
    return _wrap("".join(lines), header, caption, "tab:order", "lrrrrrr", "\\scriptsize")


#: The MatFormBench tasks COMBOO completed. Its feasibility probe made the two
#: simplex tasks unaffordable (Appendix: COMBOO's Task Coverage), so the headline
#: column averages five tasks for every other method and three for it. Comparing
#: those two numbers directly would be a mistake, hence the like-for-like table.
COMBOO_TASKS = ["L1-1", "L2-1", "L3-1"]


def emit_comboo(data: dict, delta: float) -> str:
    """Every method restricted to the tasks COMBOO actually ran."""
    d = data.get(("matformbench", "range_adapted"))
    if d is None:
        return _wrap("", "Method & distinct & hits \\\\\n",
                     "COMBOO comparison: no MatFormBench runs found.",
                     "tab:comboo", "lrr")
    counts = d[0]
    sub = counts[counts["task_id"].isin(COMBOO_TASKS)]
    ci = agg.seed_bootstrap_ci(sub, value="distinct").set_index("method")
    hits = sub.groupby("method")["valid"].mean()
    pc = agg.paired_comparison(sub, reference="M8_comboo")
    pc = pc.set_index("method") if not pc.empty else pc

    lines = []
    for key in [k for k in MAIN_ROWS if k]:
        lbl = agg.label(key)
        if lbl not in ci.index:
            continue
        r = ci.loc[lbl]
        if key == "M8_comboo":
            rec = "(reference)"
        elif not len(pc) or lbl not in pc.index:
            rec = MISSING
        else:
            q = pc.loc[lbl]
            # `paired_comparison` scores wins for the REFERENCE, so invert to
            # read as "this method against COMBOO".
            rec = f"{int(q['loss'])}/{int(q['tie'])}/{int(q['win'])}"
        lines.append(
            f"{_esc(lbl)} & {_fmt(r['mean'])} \\tiny[{_fmt(r['ci_lo'], 1)},"
            f"{_fmt(r['ci_hi'], 1)}] & {_fmt(hits.get(key, float('nan')))} & "
            f"{rec} \\\\\n"
        )
    header = ("Method & distinct designs & hits & win/tie/loss \\\\\n"
              "& ($\\delta{=}0.1$) & & vs COMBOO \\\\\n")
    caption = (
        "\\textbf{COMBOO, like for like.} Every method restricted to the three "
        "MatFormBench tasks COMBOO completed. The first two columns are counts "
        "per run over those tasks, three widths and $10$ seeds. "
        "\\emph{win/tie/loss} counts the $90$ (task, width, seed) cells the row's "
        "method won, tied and lost against COMBOO on distinct designs, so it "
        "sums to $90$. The two simplex tasks are absent for cost reasons given "
        "in this appendix."
    )
    return _wrap("".join(lines), header, caption, "tab:comboo", "lrrr", "\\small")


def emit_diversity(data: dict) -> str:
    """Distinct-design counts beside the diversity measures that do not threshold.

    The primary metric picks one threshold and counts. A reviewer is entitled to
    ask whether the ordering is an artefact of thresholding at all, so the same
    runs are also scored by measures that never threshold: mean pairwise distance
    between qualifying designs, and the normalized area under the whole
    delta-uniqueness curve (`metrics._diversity`), which integrates over every
    delta rather than choosing one.
    """
    lines = []
    for title, suite, protocol in PROTOCOLS:
        d = data.get((suite, protocol))
        if d is None:
            continue
        counts, metrics, _ = d
        dist = counts.groupby("method")["distinct"].mean()
        cols = ["diversity_norm", "mean_pairwise_l2", "n_unique"]
        agg_m = metrics.groupby("method")[cols].mean()
        lines.append(f"\\multicolumn{{5}}{{l}}{{\\emph{{{title}}}}} \\\\\n")
        order = [m for m in agg.DISPLAY_ORDER if m in agg_m.index]
        # Emphasis is per suite block, not down the whole table: the three
        # suites are different problems and a cross-suite "best" means nothing.
        block: list[dict] = []
        for m in order:
            if m not in dist.index:
                continue
            # Skip arms whose sweep is unfinished, for the same reason the other
            # tables do.
            if _mean_if_complete(counts, m, _expected_cells(counts),
                                 suite=suite) == PARTIAL:
                continue
            r = agg_m.loc[m]
            cells = [_fmt(dist[m]), _fmt(r["diversity_norm"], 3),
                     _fmt(r["mean_pairwise_l2"], 3), _fmt(r["n_unique"])]
            block.append({"label": f"\\quad {_esc(agg.label(m))}",
                          "cells": cells, "vals": [_num(c) for c in cells]})
        # Only the first three columns are emphasized. The fourth is the
        # delta->0 limit, where "more" is not "better": a method that returns
        # fifty near-identical designs leads it, which is the whole failure this
        # paper is about. Bolding it would have the table assert the opposite of
        # what its own caption argues.
        _emphasize(block, 3)
        for r in block:
            lines.append(f"{r['label']} & " + " & ".join(r["cells"]) + " \\\\\n")

    header = ("Method & distinct designs & area under the & mean pairwise & "
              "exactly distinct \\\\\n"
              "& ($\\delta{=}0.1$) & $\\delta$-curve & distance & designs \\\\\n")
    caption = (
        "\\textbf{The ordering does not depend on thresholding.} Four scorings of "
        "the same in-window evaluations, per run and averaged over a suite's "
        "tasks, three widths and $10$ seeds. Higher is better in the first three "
        "columns, where \\textbf{bold} is the best method within a suite block "
        "and \\underline{underline} the runner-up. \\emph{Area under the "
        "$\\delta$-curve} integrates the distinct count over every threshold "
        "(unitless); \\emph{mean pairwise} is the distance between qualifying "
        "designs in normalized design space. \\emph{Exactly distinct} is the "
        "$\\delta\\to0$ limit, where near-identical designs all count, so the "
        "concentrating baselines lead it; it is left unemphasized."
    )
    return _wrap("".join(lines), header, caption, "tab:diversity", "lrrrr",
                 "\\scriptsize")


def emit_cost(data: dict) -> str:
    lines = []
    per = {}
    for title, suite, protocol in PROTOCOLS:
        d = data.get((suite, protocol))
        if d is None:
            continue
        per[title] = d[1].groupby("method")["wall_clock_per_iter"].median()
    methods = sorted({m for s in per.values() for m in s.index},
                     key=lambda m: agg.DISPLAY_ORDER.index(m)
                     if m in agg.DISPLAY_ORDER else 999)
    for m in methods:
        cells = [_fmt(per[t].get(m, float("nan")), 1) if t in per else MISSING
                 for t, _, _ in PROTOCOLS]
        lines.append(f"{_esc(agg.label(m))} & " + " & ".join(cells) + " \\\\\n")
    header = "Method & MatFormBench & branin & mixtures \\\\\n"
    caption = (
        "\\textbf{Computational cost.} Median wall-clock seconds per campaign "
        "iteration at $128$ multi-start points, on one pinned CPU core; lower is "
        "cheaper. Nothing is emphasized: random search costs $0.0$ by "
        "construction."
    )
    return _wrap("".join(lines), header, caption, "tab:cost", "lrrr", "\\small")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/sections/tables")
    ap.add_argument("--delta", type=float, default=agg.DEFAULT_DELTA)
    ap.add_argument("--reference", default="M5_full")
    ap.add_argument("--only", nargs="*", default=None,
                    help="regenerate only these tables (bare names, e.g. `main "
                         "factorial`). The order table resamples 20 permutations "
                         "per cell and dominates the runtime, so iterating on the "
                         "others is worth the flag.")
    ap.add_argument("--allow-internal-name", action="store_true",
                    help="write tables carrying the internal method name anyway")
    args = ap.parse_args()

    if not os.environ.get("PAPER_METHOD_NAME") and not args.allow_internal_name:
        raise SystemExit(
            "PAPER_METHOD_NAME is unset, so every proposed-method row would "
            "carry the internal name into a double-blind submission. Set it "
            "(PAPER_METHOD_NAME=TRMC-BO) or pass --allow-internal-name."
        )

    data = {}
    for title, suite, protocol in PROTOCOLS:
        d = _load(suite, protocol, args.delta)
        if d is None:
            print(f"[warn] no runs for {suite}/{protocol}; its columns will be '{MISSING}'")
        data[(suite, protocol)] = d

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # Thunks rather than values: `--only` must skip the *computation*, not just
    # the write, since the order table is most of the runtime.
    builders = {
        "main": lambda: emit_main(data, args.reference),
        "paired": lambda: emit_paired(data, args.reference),
        "reversal": lambda: emit_reversal(data),
        "factorial": lambda: emit_factorial(data, args.delta),
        "activity": lambda: emit_activity(data),
        "order": lambda: emit_order(data, args.delta),
        "comboo": lambda: emit_comboo(data, args.delta),
        "diversity": lambda: emit_diversity(data),
        "cost": lambda: emit_cost(data),
    }
    wanted = list(builders) if not args.only else list(args.only)
    unknown = [w for w in wanted if w not in builders]
    if unknown:
        raise SystemExit(f"unknown table(s) {unknown}; known: {sorted(builders)}")
    written = {f"tab_{name}.tex": builders[name]() for name in wanted}
    for fname, text in written.items():
        (out / fname).write_text(text, encoding="utf-8")
        print(f"wrote {out / fname}")


if __name__ == "__main__":
    main()
