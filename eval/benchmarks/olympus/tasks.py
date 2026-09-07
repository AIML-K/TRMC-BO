"""Olympus task definitions in the neutral `BenchmarkTask` model.

Three tasks, chosen to answer two different questions:

| task_id          | kind                 | dim | inputs  | role |
| ---------------- | -------------------- | --- | ------- | ---- |
| `branin`         | analytical surface   | 2   | box     | §5.2 -- a plain BO landscape, plottable |
| `pce10`          | real polymer blend   | 4   | simplex | §5.3 -- real mixture formulation |
| `wf3`            | real polymer blend   | 4   | simplex | §5.3 -- same design space as pce10, different property |
| `thin_film`      | real perovskite film | 3   | simplex | §5.3 -- smallest simplex, fully feasible |

**Branin** is the analytical pick because it has three global minima, so a window
placed near the optimum splits the valid region into disconnected components --
exactly the diversity claim under test -- and because 2-D plots without a PCA
projection. The Range-Aware BO paper also visualizes Branin.

**Which Olympus datasets are mixture problems at all.** Its declaration is
unreliable in *both* directions, so `eval/scripts/audit_olympus_datasets.py`
ignores it and searches the measurements for an affine dependency -- a subset of
columns whose row-sum is constant, i.e. an indicator vector in the null space of
the centred design matrix. That finds groups whatever the config says, and finds
them on subsets, the way MatFormBench puts only x1..x6 on a simplex.

Of 43 datasets, 8 have a constant-sum subset and 4 are usable:

| dataset | dim | n | total | interior | hull-feasible | emulator R2 | goal |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| photo_pce10 | 4 | 1040 | 1 | 80.7% | 99.9% | 0.924/0.919 | min |
| photo_wf3 | 4 | 1040 | 1 | 80.7% | 99.9% | 0.971/0.881 | min |
| p3ht | 5 | 178 | 100 | 91.0% | **29.5%** | 0.692/0.834 | **max** |
| thin_film | 3 | 94 | 1 | 56.4% | 100% | 0.904/0.969 | min |

and 4 are not: the `oer_plate` family sums to exactly 1 but never uses more than
4 of its 6 components, so a uniform Dirichlet design queries a stratum with no
training data (`data.interior_support`).

Two datasets contradict their own declaration, in opposite directions, and each
cost a selected task:

* `colors_bob` **declares simplex and is not one** -- not one of its 241 rows sums
  to 1 (0.64 to 4.40, median 2.68), because its five features are independent dye
  volumes. Trusting it confined the search to a 4-D slice of a box the emulator
  was trained across, and made `hull_membership` -- which drops a coordinate,
  valid only on a simplex -- project a box-shaped cloud into a meaningless "22.6%
  infeasible". Its sweep was discarded.
* `p3ht` **declares `"none"` and is one** -- five composition percentages summing
  to 100 +- 0.1. Olympus simply missed it.

**What each task is for.**

`pce10` and `wf3` measure the *same 1,040 blends* with a different polymer:
identical design matrices, responses correlating at r = 0.742 overall and 0.86 on
the 975 rows where they should differ most. They are **not two independent
samples** -- report them as one design space carrying two properties, which makes
the pair a controlled test of transfer across objectives. Give r alongside.

`p3ht` was the intended independent evidence -- different chemistry, 5 components
in percent, the only *maximize* target -- and **is rejected**: its emulator
predicts exactly 0 for 41.6% of its own 178 training designs, though not one
measured conductivity is 0. See `data.activation_floor_rate`. Support for the
maximize direction is kept (`native_kind`, `native_quantile`), since it cost
nothing and a later suite will need it.

`thin_film` holds the opposite end of both axes: the smallest simplex (3
components) and no input-feasibility pressure at all (its hull is the whole
simplex). 94 measurements, so its evidence is thin.

**So three of Olympus's eight constant-sum datasets are usable, each failing for a
different reason that the others' checks cannot see**: `colors_bob` is not a
simplex, the `oer_plate` family has no interior, and `p3ht`'s emulator is flat on
its own data. `tasks._measured_simplex_total` runs all three checks.

## The specification window is constructed, and that has to be said

Olympus supplies no target window -- every dataset carries only
`default_goal: minimize`. A two-sided specification is therefore *synthesized*:

    tau = q25(response) under the uniform reference measure
    native target = "response <= tau"

and the shared calibrator then places the lower bound so the valid fraction among
input-feasible designs hits 10% / 3% / 1%. Write-ups must state that the
specification is constructed rather than benchmark-supplied.

`NATIVE_QUANTILE = 0.25` is not free choice. With a single output the calibration
is analytic: valid fraction = `P(response <= tau) * alpha`. Putting tau at the
target quantile itself -- tau = q10 against a 10% target -- forces `alpha = 1`,
which collapses the lower bound onto the sample minimum and turns the widest
condition back into the one-sided native target, deleting the two-sidedness under
test at a third of the width axis. q25 gives alpha = 0.40 / 0.12 / 0.04, so all
three conditions are strictly two-sided. See `benchmarks/calibration.py`.

## Why tau is frozen on disk rather than computed on demand

tau is a quantile of the *oracle's* response under the uniform measure, so
computing it needs oracle evaluations. Recomputing it per worker would spend that
cost 810 times and, worse, would let the task definition drift with any change to
sampling. It is frozen next to the windows by
`eval/scripts/synthesize_olympus_targets.py` and committed, exactly like the
windows themselves.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from eval.benchmarks.base import BenchmarkTask, NativeTarget, SimplexGroup
from eval.benchmarks.olympus import data as odata

SUITE = "olympus"

#: Share of designs the synthesized one-sided specification admits.
#:
#: For a `minimize` task the cap is `tau = q25(response)` and the native side is
#: `response <= tau`; for a `maximize` task it is `tau = q75` and `response >= tau`.
#: Either way `P(native) = 0.25`, which is what keeps the closed form
#: `valid fraction = P(native) * alpha` -- and hence alpha = 0.40/0.12/0.04 -- the
#: same across tasks of both directions. See the module docstring for why the 0.25
#: cannot be tightened to the target itself.
NATIVE_SHARE = 0.25


def native_quantile(goal: str) -> float:
    """Quantile of the response to place the synthesized threshold at."""
    return NATIVE_SHARE if goal == "minimize" else 1.0 - NATIVE_SHARE


def native_kind(goal: str) -> str:
    """`less` for a minimize task, `greater` for a maximize one."""
    return "less" if goal == "minimize" else "greater"

#: Frozen synthesized thresholds, committed alongside the window specs.
TARGETS_PATH = Path(__file__).resolve().parents[2] / "specs" / SUITE / "native_targets.json"

#: task_id -> how to build it. `dataset` selects the Olympus dataset for emulator
#: tasks; `surface` selects an analytical surface.
SELECTED_TASKS: dict[str, dict] = {
    "branin": {"oracle": "surface", "surface": "branin"},
    "pce10": {"oracle": "emulator", "dataset": "photo_pce10"},
    "wf3": {"oracle": "emulator", "dataset": "photo_wf3"},
    "thin_film": {"oracle": "emulator", "dataset": "thin_film"},
}


#: How much of a dataset must actually lie on the simplex before its declared
#: simplex constraint is believed, and how much of it must use every component.
#:
#: Both thresholds exist because a declaration was wrong and a declaration was
#: incomplete, and each cost a task. `SUM_TOLERANCE` is loose enough for
#: `thin_film`, whose measurements are rounded to two decimals so 4 of 94 rows sum
#: to 0.99 or 1.01.
SIMPLEX_SHARE_FLOOR = 0.95
SUM_TOLERANCE = 0.02


def _measured_simplex_total(dataset: str, dim: int) -> float:
    """Check Olympus's declared simplex constraint against its own measurements.

    Two independent failures, both found only after a task had been selected and
    one of them only after a 270-cell sweep had run:

    **`colors_bob` declares `"parameters": "simplex"` and is not a simplex
    dataset.** Not one of its 241 rows sums to 1 -- they range 0.64 to 4.40,
    median 2.68 -- because its five features are independent dye volumes (Red,
    Orange, Yellow, Blue, Green), not composition fractions. Believing the
    declaration made the adapter impose `sum(x) == 1`, so the search was confined
    to a 4-D slice while the emulator had been trained across the whole box. Its
    "22.6% input-infeasible" was an artifact of `hull_membership` dropping a
    coordinate -- valid only on a simplex -- and so projecting a box-shaped cloud.

    **The `oer_plate` family is a genuine simplex with no interior.** Every row
    sums to exactly 1, but at most 4 of 6 components are ever non-zero, so uniform
    Dirichlet designs query a stratum with no training data at all. See
    `data.interior_support`.

    A convex-hull feasibility test detects neither: for `colors_bob` it is
    computed in the wrong space, and for `oer_plate` the hull of those faces is
    the whole simplex. Hence a check against the raw measurements.
    """
    X, _ = odata.load_measurements(dataset)
    sums = X.sum(axis=1)
    total = float(np.median(sums))
    on_simplex = float(np.isclose(sums, total, atol=SUM_TOLERANCE * max(total, 1.0)).mean())
    if on_simplex < SIMPLEX_SHARE_FLOOR:
        raise ValueError(
            f"{dataset}: only {on_simplex:.1%} of its {len(X)} measurements have a "
            f"constant component sum (median {total:.4g}), so it is not a mixture "
            "problem -- its features are not composition fractions."
        )
    floored = odata.activation_floor_rate(dataset)
    if floored > odata.FLOOR_RATE_CEILING:
        raise ValueError(
            f"{dataset}: the emulator predicts the activation floor for "
            f"{floored:.1%} of its own training designs (ceiling "
            f"{odata.FLOOR_RATE_CEILING:.0%}); its response distribution over any "
            "reference measure collapses and a synthesized window degenerates"
        )
    support = odata.interior_support(dataset)
    if support < odata.SUPPORT_FLOOR:
        raise ValueError(
            f"{dataset}: only {support:.1%} of measurements use every component "
            f"(floor {odata.SUPPORT_FLOOR:.0%}); a uniform Dirichlet design would "
            "query a stratum the emulator never trained on"
        )
    return total


# -- geometry, which needs no target ----------------------------------------

def task_geometry(task_id: str) -> dict:
    """Everything about a task except its synthesized threshold.

    Split out because the threshold is derived *from* the oracle, and building the
    oracle needs the geometry: `synthesize_olympus_targets.py` uses this to break
    the cycle.
    """
    if task_id not in SELECTED_TASKS:
        raise ValueError(f"unknown Olympus task {task_id!r}; known: {sorted(SELECTED_TASKS)}")
    entry = SELECTED_TASKS[task_id]

    if entry["oracle"] == "surface":
        # Olympus defines every surface on the unit square and rescales inside.
        # `wrapper_branin.py`: x0 = 15*p0 - 5, x1 = 15*p1.
        return {
            "dim": 2,
            "x_names": ("p0", "p1"),
            "y_names": ("f",),
            "bounds": np.stack([np.zeros(2), np.ones(2)]),
            "simplex_groups": (),
            # Branin is a minimization surface; stated rather than defaulted.
            "meta": {"oracle": "surface", "surface": entry["surface"],
                     "default_goal": "minimize"},
        }

    dataset = entry["dataset"]
    config = odata.load_config(dataset)
    x_names = tuple(p["name"] for p in config["parameters"])
    y_names = tuple(m["name"] for m in config["measurements"])
    dim = len(x_names)

    lows = np.array([float(p["low"]) for p in config["parameters"]])
    highs = np.array([float(p["high"]) for p in config["parameters"]])
    # The declared constraint is not consulted: `colors_bob` declares a simplex it
    # does not have and `p3ht` has one it does not declare. The measurements decide.
    total = _measured_simplex_total(dataset, dim)
    if len(y_names) != 1:
        raise ValueError(f"{dataset}: expected one measurement, found {y_names}")
    goal = config.get("default_goal")
    if goal not in ("minimize", "maximize"):
        raise ValueError(f"{dataset}: unsupported default_goal {goal!r}")

    return {
        "dim": dim,
        "x_names": x_names,
        "y_names": y_names,
        "bounds": np.stack([lows, highs]),
        "simplex_groups": (SimplexGroup(indices=tuple(range(dim)), total=total),),
        "meta": {
            "oracle": "emulator",
            "dataset": dataset,
            "default_goal": goal,
            "simplex_total": total,
            "n_measurements": len(odata.load_measurements(dataset)[0]),
        },
    }


# -- the frozen synthesized thresholds --------------------------------------

def load_native_targets() -> dict:
    if not TARGETS_PATH.exists():
        raise FileNotFoundError(
            f"{TARGETS_PATH} is missing -- the synthesized specification caps are "
            "not frozen yet. Generate them with\n"
            "    PYTHONPATH=core:. .venv-eval/bin/python -m "
            "eval.scripts.synthesize_olympus_targets"
        )
    return json.loads(TARGETS_PATH.read_text(encoding="utf-8"))


def save_native_targets(payload: dict) -> Path:
    TARGETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    TARGETS_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return TARGETS_PATH


def provisional_task(task_id: str) -> BenchmarkTask:
    """A task with a placeholder threshold, for synthesizing the real one.

    tau is a quantile of the oracle's response, so it cannot be known before the
    oracle has been built and sampled -- but building an oracle needs a task. This
    breaks that cycle. The placeholder target is `response <= +inf`, i.e. vacuous,
    and nothing that consumes a provisional task reads `native_targets`:
    `make_oracle` reads `meta`, and `sample_design_space` reads bounds and simplex
    groups. Never hand one of these to a campaign.
    """
    geom = task_geometry(task_id)
    (y_name,) = geom["y_names"]
    return BenchmarkTask(
        suite=SUITE,
        task_id=task_id,
        dim=geom["dim"],
        x_names=geom["x_names"],
        y_names=geom["y_names"],
        bounds=geom["bounds"],
        native_targets=(
            NativeTarget(
                name=y_name,
                kind=native_kind(geom["meta"].get("default_goal", "minimize")),
                # Vacuous either way, so a provisional task can never look valid.
                value=float("-inf") if geom["meta"].get("default_goal") == "maximize"
                else float("inf"),
            ),
        ),
        simplex_groups=geom["simplex_groups"],
        meta={**geom["meta"], "provisional": True},
    )


def load_task(task_id: str) -> BenchmarkTask:
    geom = task_geometry(task_id)
    frozen = load_native_targets()
    if task_id not in frozen["tasks"]:
        raise KeyError(
            f"{task_id!r} has no frozen specification cap in {TARGETS_PATH}; "
            "re-run eval.scripts.synthesize_olympus_targets"
        )
    entry = frozen["tasks"][task_id]
    (y_name,) = geom["y_names"]

    goal = geom["meta"].get("default_goal", "minimize")
    targets = (
        NativeTarget(name=y_name, kind=native_kind(goal), value=float(entry["tau"])),
    )
    meta = dict(geom["meta"])
    # Provenance travels with the task so no result can be read without it.
    meta["specification"] = {
        "constructed": True,
        "rule": (f"tau = q{int(native_quantile(goal) * 100)}(response) under the uniform "
                 f"reference measure; native side is 'response {native_kind(goal)} tau' "
                 f"for a {goal} task"),
        **{k: v for k, v in entry.items() if k != "tau"},
    }

    return BenchmarkTask(
        suite=SUITE,
        task_id=task_id,
        dim=geom["dim"],
        x_names=geom["x_names"],
        y_names=geom["y_names"],
        bounds=geom["bounds"],
        native_targets=targets,
        simplex_groups=geom["simplex_groups"],
        meta=meta,
    )


def load_selected() -> dict[str, BenchmarkTask]:
    return {tid: load_task(tid) for tid in SELECTED_TASKS}
