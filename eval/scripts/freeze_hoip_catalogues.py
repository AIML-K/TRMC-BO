"""Choose and freeze the three HOIP catalogues -- the suite's feasibility axis.

    PYTHONPATH=core:. .venv-eval/bin/python -m eval.scripts.freeze_hoip_catalogues
    #   --check   re-derive and compare against the committed file, writing nothing

Why an axis at all
------------------

The one thing HOIP is here to settle is where TRMC-BO's diversity advantage
stops surviving unknown input feasibility. MatFormBench measured 24%, 41% and
69% infeasible and the reversal appears only at 69% -- one task. HOIP's own
catalogue is 91.3% infeasible, which would add a second isolated point rather
than a boundary.

So the suite runs three *nested restrictions of the same chemistry*. Restricting
the molcat / metal / halogen lists changes how much of the design space is
manufacturable while leaving the oracle, the window, and the outcome landscape
untouched.

What is held fixed, and why that is the whole trick
--------------------------------------------------

An axis that varies two things measures neither. The axis under test is
infeasibility, so outcome difficulty has to be held -- and here it can be held
*exactly*, not approximately: every rectangle considered is required to contain
**all seven** valid materials of the full catalogue. The valid set is therefore
literally identical across the three tasks, which means the delta-separated
capacity -- the ceiling on the primary metric -- is identical too. A distinct
count of 4 means the same thing on every rung.

The selection rule, stated before it was run
--------------------------------------------

`full` is the declared design space. For the other two, among all
molcat x metal x halogen rectangles that

  (a) contain every valid material of the full catalogue, and
  (b) are large enough that one campaign (n_init + budget evaluations) covers no
      more than a stated fraction of them,

take the one with the **highest feasible rate**; ties broken by smallest size,
then lexicographically. Constraint (b) is what stops the easy rung from being a
near-exhaustive enumeration: total feasibility is capped at 111 materials, so a
high feasible rate can only be bought with a small catalogue, and past some point
"search" stops being search.

The rule uses the oracle and no method. That is the same standing the calibrated
windows have on the other two suites, and it is frozen here before any method
runs.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from eval.benchmarks.hoip import data as hd

SPEC_PATH = Path(__file__).resolve().parents[1] / "specs" / "hoip" / "catalogues.json"

#: One campaign's oracle calls: `n_init` + `budget` from the sweep config.
CAMPAIGN_SIZE = 80

#: task_id -> the largest share of its catalogue one campaign may touch.
#: `full` is unrestricted by construction (80 / 1276 = 6.3%).
COVERAGE_CEILING = {"dense": 1 / 3, "restricted": 1 / 5}


def _valid_keys() -> set[tuple[str, str, str]]:
    table = hd.load_table()
    lo, hi = hd.NATIVE_WINDOW["bandgap"]
    cap = hd.NATIVE_WINDOW["m_star"][1]
    ok = (table.bandgap >= lo) & (table.bandgap <= hi) & (table.m_star < cap)
    return set(zip(table.molcat[ok], table.metal[ok], table.halogen[ok]))


def _describe(molcats, metals, halogens, feasible, valid) -> dict:
    n = len(molcats) * len(metals) * len(halogens)
    f = sum(1 for k in feasible if k[0] in set(molcats) and k[1] in set(metals)
            and k[2] in set(halogens))
    v = sum(1 for k in valid if k[0] in set(molcats) and k[1] in set(metals)
            and k[2] in set(halogens))
    return {
        "molcats": list(molcats), "metals": list(metals), "halogens": list(halogens),
        "n_materials": n, "n_feasible": f, "n_valid": v,
        "feasible_rate": f / n, "infeasible_rate": 1 - f / n,
        "valid_share_of_feasible": v / f if f else 0.0,
        "valid_share_of_space": v / n,
        "campaign_coverage": CAMPAIGN_SIZE / n,
    }


def select(task_id: str) -> dict:
    """The rectangle the rule picks for `task_id`. Exhaustive, not heuristic."""
    feasible = set(hd.response_map())
    valid = _valid_keys()

    if task_id == "full":
        return _describe(hd.MOLCATS, hd.METALS, hd.HALOGENS, feasible, valid)

    ceiling = COVERAGE_CEILING[task_id]
    min_size = int(np.ceil(CAMPAIGN_SIZE / ceiling))

    # (a) forces these in; everything else is optional.
    need_mc = {k[0] for k in valid}
    need_me = {k[1] for k in valid}
    need_hal = {k[2] for k in valid}
    opt_mc = [m for m in hd.MOLCATS if m not in need_mc]
    opt_me = [m for m in hd.METALS if m not in need_me]
    opt_hal = [h for h in hd.HALOGENS if h not in need_hal]

    core_mc = tuple(m for m in hd.MOLCATS if m in need_mc)
    core_me = tuple(m for m in hd.METALS if m in need_me)
    core_hal = tuple(h for h in hd.HALOGENS if h in need_hal)

    best = None
    for r in range(len(opt_mc) + 1):
        for extra_mc in itertools.combinations(opt_mc, r):
            molcats = tuple(m for m in hd.MOLCATS if m in need_mc or m in extra_mc)
            for hal_extra in range(len(opt_hal) + 1):
                for extra_hal in itertools.combinations(opt_hal, hal_extra):
                    halogens = tuple(h for h in hd.HALOGENS
                                     if h in need_hal or h in extra_hal)
                    # For a fixed molcat/halogen choice the feasible rate is an
                    # average over metals, so for each metal count the best set is
                    # the top-scoring prefix. Scanning every count is therefore
                    # exact, not greedy.
                    per_metal = {
                        e: sum(1 for mc in molcats for h in halogens
                               if (mc, e, h) in feasible)
                        for e in hd.METALS
                    }
                    ranked = sorted(opt_me, key=lambda e: (-per_metal[e], e))
                    for k in range(len(ranked) + 1):
                        metals = tuple(m for m in hd.METALS
                                       if m in need_me or m in ranked[:k])
                        n = len(molcats) * len(metals) * len(halogens)
                        if n < min_size:
                            continue
                        cand = _describe(molcats, metals, halogens, feasible, valid)
                        key = (-cand["feasible_rate"], cand["n_materials"],
                               cand["molcats"], cand["metals"], cand["halogens"])
                        if best is None or key < best[0]:
                            best = (key, cand)
    if best is None:
        raise RuntimeError(f"no rectangle satisfies the rule for {task_id!r}")
    entry = best[1]
    entry["rule"] = (
        f"highest feasible rate among molcat x metal x halogen rectangles that "
        f"contain all {len(valid)} valid materials and have campaign coverage "
        f"<= {ceiling:.3f} ({CAMPAIGN_SIZE} evaluations, so n >= {min_size})"
    )
    return entry


def build() -> dict:
    valid = _valid_keys()
    tasks = {tid: select(tid) for tid in ("dense", "restricted", "full")}
    # Nesting is not required by the rule, but if it holds the axis is a pure
    # restriction and worth recording.
    order = ["dense", "restricted", "full"]
    nested = all(
        set(tasks[a][k]) <= set(tasks[b][k])
        for a, b in zip(order, order[1:])
        for k in ("molcats", "metals", "halogens")
    )
    return {
        "campaign_size": CAMPAIGN_SIZE,
        "coverage_ceiling": COVERAGE_CEILING,
        "valid_materials": sorted("/".join(k) for k in valid),
        "nested": nested,
        "tasks": tasks,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="compare against the committed file instead of writing it")
    args = ap.parse_args()

    payload = build()
    header = (f"{'task':11s} {'n':>5s} {'feasible':>9s} {'infeasible':>11s} "
              f"{'valid':>6s} {'V/F':>7s} {'coverage':>9s}")
    print(header)
    print("-" * len(header))
    for tid in ("dense", "restricted", "full"):
        t = payload["tasks"][tid]
        print(f"{tid:11s} {t['n_materials']:5d} {t['n_feasible']:9d} "
              f"{t['infeasible_rate']:10.1%} {t['n_valid']:6d} "
              f"{t['valid_share_of_feasible']:7.3f} {t['campaign_coverage']:8.1%}")
    for tid in ("dense", "restricted"):
        t = payload["tasks"][tid]
        print(f"\n{tid}: {len(t['molcats'])} molcats {t['molcats']}")
        print(f"{'':<{len(tid)}}  {len(t['metals'])} metals {t['metals']}")
        print(f"{'':<{len(tid)}}  {len(t['halogens'])} halogens {t['halogens']}")
    print(f"\nnested: {payload['nested']}")

    if args.check:
        committed = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
        same = committed == payload
        print("\ncatalogues match the committed file" if same
              else "\nMISMATCH against the committed file")
        raise SystemExit(0 if same else 1)

    SPEC_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPEC_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nwrote {SPEC_PATH}")


if __name__ == "__main__":
    main()
