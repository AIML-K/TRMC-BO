"""HOIP tasks: one real chemistry at three levels of unknown input feasibility.

| task_id      | materials | infeasible | valid | role |
| ------------ | --------: | ---------: | ----: | ---- |
| `dense`      |       240 |      60.0% |     7 | below the MatFormBench reversal |
| `restricted` |       408 |      73.5% |     7 | just above it |
| `full`       |     1,276 |      91.3% |     7 | the benchmark's own catalogue |

These are nested restrictions of the *same* design space, frozen by
`scripts/freeze_hoip_catalogues.py` before any method ran; see that module for
the selection rule. Every rectangle it considers is required to contain all seven
valid materials, so **the valid set is identical across the three tasks** -- the
delta-separated capacity, and hence the ceiling on the primary metric, does not
move along the axis. Only the share of the space that is manufacturable does.

That is the entire point. Across 2,424 completed cells on the other two suites,
TRMC-BO's advantage in distinct qualifying designs holds at 24% and 41%
infeasible and reverses at 69% -- on a single task, with nothing else in the
world testing it. This ladder brackets that crossing inside one chemistry.

## The specification is the benchmark's own, and that is a first

MatFormBench supplied one-sided thresholds and Olympus supplied only
`minimize`, so on both suites the two-sided window had to be constructed by
`benchmarks/calibration.py`. HOIP supplies one outright. Upstream's success test
is

    measurement['bandgap'] < 0.5 and measurement['m_star'] < 4.

against `bandgap = abs(gap - 1.25)`, i.e.

    0.75 <= Eg <= 1.75  and  m* <= 4

so this suite runs **one condition, `native`**, and has no wide/medium/narrow
axis at all. The width axis is replaced by the feasibility axis above -- windows
are what the other two suites vary, and varying it here would mean discarding the
one benchmark-supplied specification the paper has.

The raw gap is kept as the modelled output rather than upstream's folded
`abs(gap - 1.25)`. Folding is exactly the transformation that destroys the
two-sidedness under test, and it hands the surrogate a non-monotone response with
a crease at the target. Nothing is lost by not folding: `M2_target_distance`
aiming at the window midpoint *is* the folded objective, so the comparison
against upstream's formulation survives as a baseline rather than as the task
definition.

## Native targets, and why both are caps

`native_targets` is what `M1_qehvi_extremum` reads -- the "standard BO applied to
this task" baseline, which ignores the window rather than approximating it. An
extremum-seeking method has no reading of a two-sided specification, so it is
given the specification's **upper bounds** (`Eg <= 1.75`, `m* <= 4`) and the
lower bound is discarded. That is the same convention the shared calibrator uses
on the other two suites, where the native threshold is kept as one bound and the
opposite bound is the two-sided addition; here the addition is supplied rather
than calibrated.

`m*`'s lower bound is 0, which is physical rather than chosen -- an effective
mass cannot be negative. It is a real bound on the GP posterior mean, not a
vacuous one, so C2 attaches four constraints on these tasks (two sides x two
properties).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from eval.benchmarks.base import BenchmarkTask, NativeTarget
from eval.benchmarks.hoip import data as hd

SUITE = "hoip"

#: Frozen catalogue definitions, committed alongside the windows.
CATALOGUES_PATH = Path(__file__).resolve().parents[2] / "specs" / SUITE / "catalogues.json"

SELECTED_TASKS: tuple[str, ...] = ("dense", "restricted", "full")

#: Physical floor on the effective mass. Not a modelling choice.
M_STAR_FLOOR = 0.0


def load_catalogues() -> dict:
    if not CATALOGUES_PATH.exists():
        raise FileNotFoundError(
            f"{CATALOGUES_PATH} is missing -- the HOIP catalogues are not frozen yet:\n"
            "    PYTHONPATH=core:. .venv-eval/bin/python -m "
            "eval.scripts.freeze_hoip_catalogues"
        )
    return json.loads(CATALOGUES_PATH.read_text(encoding="utf-8"))


def catalogue(task_id: str) -> dict:
    frozen = load_catalogues()
    if task_id not in frozen["tasks"]:
        raise ValueError(
            f"unknown HOIP task {task_id!r}; known: {sorted(frozen['tasks'])}"
        )
    return frozen["tasks"][task_id]


def materials(task_id: str) -> list[tuple[str, str, str]]:
    """Every candidate material in the catalogue, in a fixed canonical order.

    Order is declaration order, not the frozen file's, so a reordering of the
    JSON can never silently permute the design space.
    """
    entry = catalogue(task_id)
    molcats = [m for m in hd.MOLCATS if m in set(entry["molcats"])]
    metals = [m for m in hd.METALS if m in set(entry["metals"])]
    halogens = [h for h in hd.HALOGENS if h in set(entry["halogens"])]
    return [(mc, me, ha) for mc in molcats for me in metals for ha in halogens]


def task_geometry(task_id: str) -> dict:
    """Coordinates and bounds -- everything that needs no oracle.

    Bounds are the per-coordinate extent of the descriptors *of the options this
    catalogue actually contains*, so normalized space is [0, 1]^14 with both ends
    attained. `metrics._normalized_X` and the delta-uniqueness curve then work
    unchanged, and delta is comparable across the three tasks only in the sense
    that each is normalized to its own box -- which is stated in the results
    rather than assumed away.
    """
    keys = materials(task_id)
    X = hd.encode(keys)
    lows, highs = X.min(axis=0), X.max(axis=0)
    entry = catalogue(task_id)
    return {
        "dim": X.shape[1],
        "x_names": hd.descriptor_names(),
        "y_names": hd.Y_NAMES,
        "bounds": np.stack([lows, highs]),
        "meta": {
            "catalogue": task_id,
            "n_materials": len(keys),
            "n_feasible": entry["n_feasible"],
            "infeasible_rate": entry["infeasible_rate"],
            "n_valid_native": entry["n_valid"],
            "molcats": entry["molcats"],
            "metals": entry["metals"],
            "halogens": entry["halogens"],
            "campaign_coverage": entry["campaign_coverage"],
            "selection_rule": entry.get("rule", "the full declared design space"),
        },
    }


def load_task(task_id: str) -> BenchmarkTask:
    geom = task_geometry(task_id)
    gap_lo, gap_hi = hd.NATIVE_WINDOW["bandgap"]
    m_cap = hd.NATIVE_WINDOW["m_star"][1]

    meta = dict(geom["meta"])
    meta["specification"] = {
        "constructed": False,
        "source": "upstream success test: abs(bandgap - 1.25) < 0.5 and m_star < 4.",
        "window": {"bandgap": [gap_lo, gap_hi], "m_star": [M_STAR_FLOOR, m_cap]},
        "note": (
            "benchmark-supplied, unlike MatFormBench (one-sided thresholds) and "
            "Olympus (synthesized). m_star's lower bound is the physical floor."
        ),
    }
    meta["oracle"] = "lookup"
    meta["noiseless"] = True

    return BenchmarkTask(
        suite=SUITE,
        task_id=task_id,
        dim=geom["dim"],
        x_names=geom["x_names"],
        y_names=geom["y_names"],
        bounds=geom["bounds"],
        # Upper bounds only: see the module docstring on what M1 is allowed to read.
        native_targets=(
            NativeTarget(name="bandgap", kind="less", value=float(gap_hi)),
            NativeTarget(name="m_star", kind="less", value=float(m_cap)),
        ),
        simplex_groups=(),
        meta=meta,
    )


def load_selected() -> dict[str, BenchmarkTask]:
    return {tid: load_task(tid) for tid in SELECTED_TASKS}
