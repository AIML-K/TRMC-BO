"""Load MatFormBench task definitions into the neutral `BenchmarkTask` model.

Two files describe a task and neither alone is enough:

``task_registry.json``          features + the one-sided target thresholds
``synthetic_data/<L>/<D>_rules.json``  design box, simplex groups, noise, failure windows

The registry is what MatFormBench's own runner reads; the rules file is emitted
alongside the shipped 30-row CSV and is the only place the design space is
written down. (MatFormBench's own baselines never read it -- they infer bounds
from the training data's min/max, which silently ignores the simplex groups on
the 10D/15D tasks.)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from eval.benchmarks.base import BenchmarkTask, NativeTarget, SimplexGroup

SUITE = "matformbench"

#: Repo-relative location of the third-party clone. Overridable for CI.
MATFORMBENCH_ROOT = Path(
    os.environ.get(
        "MATFORMBENCH_ROOT",
        Path(__file__).resolve().parents[3] / "eval" / "external" / "MatFormBench",
    )
)

#: The five tasks selected in the plan (§1.1).
#:
#: L1-L3 are held at 5D so that only the structural difficulty changes
#: (smooth -> coupled/noisy -> local invalid). L4/L5 then escalate dimension,
#: and both carry a 6-component simplex group, which is where the input-space
#: feasibility machinery actually gets exercised.
SELECTED_TASKS: dict[str, tuple[str, str]] = {
    "L1-1": ("L1", "Dataset-1"),  # smooth response, 5D
    "L2-1": ("L2", "Dataset-1"),  # coupled / noisy response, 5D
    "L3-1": ("L3", "Dataset-1"),  # local invalid regions, 5D
    "L4-1": ("L4", "Dataset-1"),  # multimodal, 10D, simplex on x1..x6
    "L5-4": ("L5", "Dataset-4"),  # sparse feasibility, 15D, simplex on x1..x6
}

_KIND = {"greater": "greater", "less": "less"}


def _registry_path() -> Path:
    return MATFORMBENCH_ROOT / "task_registry.json"


def _rules_path(level: str, dataset: str) -> Path:
    return MATFORMBENCH_ROOT / "synthetic_data" / level / f"{dataset}_rules.json"


def seed_csv_path(level: str, dataset: str) -> Path:
    """The 30-row dataset MatFormBench ships for this task."""
    return MATFORMBENCH_ROOT / "synthetic_data" / level / f"{dataset}.csv"


def parse_task_id(task_id: str) -> tuple[str, str]:
    """``"L4-1"`` -> ``("L4", "Dataset-1")``, for any of the 30 tasks."""
    if task_id in SELECTED_TASKS:
        return SELECTED_TASKS[task_id]
    level, _, n = task_id.partition("-")
    if not level.startswith("L") or not n.isdigit():
        raise ValueError(f"unrecognized MatFormBench task id: {task_id!r}")
    return level, f"Dataset-{n}"


def load_task(task_id: str) -> BenchmarkTask:
    level, dataset = parse_task_id(task_id)

    with open(_registry_path(), encoding="utf-8") as f:
        registry = json.load(f)
    entry = registry[level][dataset]

    with open(_rules_path(level, dataset), encoding="utf-8") as f:
        rules = json.load(f)

    x_names = tuple(entry["features"])
    dim = len(x_names)
    if dim != rules["dim"]:
        raise ValueError(
            f"{task_id}: registry lists {dim} features but rules file says dim={rules['dim']}"
        )

    lo, hi = rules["train_bounds"]
    bounds = np.stack([np.full(dim, float(lo)), np.full(dim, float(hi))])

    targets = tuple(
        NativeTarget(name=t["header"], kind=_KIND[t["type"]], value=float(t["value"]))
        for t in entry["targets"]
    )

    simplex = tuple(
        SimplexGroup(
            indices=tuple(int(i) for i in g["idx"]),
            total=float(g.get("total", 1.0)),
            min_component=float(g.get("min_component", 0.0)),
        )
        for g in (rules.get("simplex_groups") or [])
    )

    return BenchmarkTask(
        suite=SUITE,
        task_id=task_id,
        dim=dim,
        x_names=x_names,
        y_names=tuple(t.name for t in targets),
        bounds=bounds,
        native_targets=targets,
        simplex_groups=simplex,
        meta={
            "level": level,
            "dataset": dataset,
            # The design box the oracle will still answer on -- wider than the
            # training box by `proposal_extension`. We optimize inside
            # train_bounds; this is recorded so extrapolation stays a choice.
            "proposal_extension": rules.get("proposal_extension"),
            "failure_windows": rules.get("failure_windows") or {},
            "conditional_gates": rules.get("conditional_gates") or [],
            "noise": rules.get("noise") or {},
            "n_batches": rules.get("n_batches"),
            "plugins": rules.get("plugins") or [],
            "index_map": rules.get("index_map") or {},
            "censor_y2_low": rules.get("censor_y2_low"),
            "censor_y2_high": rules.get("censor_y2_high"),
        },
    )


def load_selected() -> dict[str, BenchmarkTask]:
    return {tid: load_task(tid) for tid in SELECTED_TASKS}
