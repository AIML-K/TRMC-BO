"""Run the {task x width x method x seed} matrix.

Design constraints that shaped this:

* **Resume must be free.** A full sweep is thousands of runs; a crash three
  quarters through must not cost the first three quarters. Every cell owns a
  directory and writes `run.json` last, so its presence means "this cell is
  complete" and the cell is skipped.
* **One process per cell, one thread per process.** The GP fits are small, so
  24 processes each spawning 24 BLAS threads would thrash. Workers pin
  themselves to a single thread and parallelism comes from the process pool.
* **Each worker builds its own oracle.** The MatFormBench oracle is a compiled
  extension with internal RNG state; sharing one across processes would couple
  runs that are supposed to be independent.
* **Suites are resolved per cell, lazily.** `benchmarks.registry.resolve` is
  called inside `run_cell` so a worker imports only the suite it is running --
  see that module for why the isolation has to be structural.
* **A failing cell must not kill the sweep.** Exceptions are caught, written to
  `error.json`, and reported in the summary.
"""

from __future__ import annotations

import json
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

RESULT_ROOT = Path(__file__).resolve().parents[1] / "results"


@dataclass(frozen=True)
class SweepCell:
    suite: str
    protocol: str
    task_id: str
    width: str
    method: str
    seed: int

    def path(self, root: Path) -> Path:
        return root / self.suite / self.protocol / self.task_id / self.width / self.method / f"seed{self.seed:03d}"


def _pin_to_one_thread() -> None:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "1"
    import torch

    torch.set_num_threads(1)


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return None if not np.isfinite(value) else value
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def run_cell(cell: SweepCell, cfg: dict, root: Path) -> dict:
    """Execute one cell and write its artifacts. Safe to call in a worker."""
    _pin_to_one_thread()

    import warnings

    warnings.filterwarnings("ignore")

    from eval.benchmarks.registry import resolve
    from eval.harness.loop import CampaignConfig, run_campaign
    from eval.harness.metrics import iteration_frame, run_metrics
    from eval.methods import make as make_method

    out = cell.path(root)
    out.mkdir(parents=True, exist_ok=True)

    try:
        suite = resolve(cell.suite)
        task = suite.load_task(cell.task_id)
        spec = suite.load_range_spec(cell.task_id, cell.width)
        oracle = suite.make_oracle(task)
        method = make_method(cell.method, cfg.get("num_restarts", 128))

        campaign = CampaignConfig(
            n_init=cfg.get("n_init", 30),
            budget=cfg.get("budget", 50),
            q=cfg.get("q", 1),
        )
        result = run_campaign(task, spec, method, oracle, seed=cell.seed, cfg=campaign)

        frame = iteration_frame(result, task, spec)
        frame.to_parquet(out / "trace.parquet", index=False)

        metrics = run_metrics(result, task, spec)
        (out / "metrics.json").write_text(
            json.dumps(_json_safe(metrics), indent=2, sort_keys=True), encoding="utf-8"
        )
        # Written last: its presence is the completion marker resume relies on.
        (out / "run.json").write_text(
            json.dumps(
                _json_safe({
                    "cell": asdict(cell),
                    "campaign": asdict(campaign),
                    "range_spec_calibration": spec.calibration,
                    "oracle_calls": getattr(oracle, "n_calls", None),
                    # Suite-specific telemetry, absent on suites that publish none.
                    "oracle_diagnostics": (
                        oracle.diagnostics() if hasattr(oracle, "diagnostics") else None
                    ),
                    "meta": result.meta,
                }),
                indent=2, sort_keys=True,
            ),
            encoding="utf-8",
        )
        (out / "error.json").unlink(missing_ok=True)
        return {"cell": asdict(cell), "status": "ok", "hit_rate": metrics["hit_rate"]}

    except Exception as exc:  # noqa: BLE001 -- one bad cell must not stop the sweep
        (out / "error.json").write_text(
            json.dumps(
                {"cell": asdict(cell), "error": repr(exc), "traceback": traceback.format_exc()},
                indent=2,
            ),
            encoding="utf-8",
        )
        return {"cell": asdict(cell), "status": "error", "error": repr(exc)}


def build_matrix(cfg: dict) -> list[SweepCell]:
    return [
        SweepCell(
            suite=cfg.get("suite", "matformbench"),
            protocol=cfg.get("protocol", "range_adapted"),
            task_id=task_id,
            width=width,
            method=method,
            seed=seed,
        )
        for task_id in cfg["tasks"]
        for width in cfg["widths"]
        for method in cfg["methods"]
        for seed in range(cfg["n_seeds"])
    ]


def pending(cells: list[SweepCell], root: Path) -> list[SweepCell]:
    return [c for c in cells if not (c.path(root) / "run.json").exists()]


def run_sweep(
    cfg: dict,
    root: Path = RESULT_ROOT,
    workers: int | None = None,
    force: bool = False,
    progress=print,
) -> dict:
    cells = build_matrix(cfg)
    todo = cells if force else pending(cells, root)
    workers = workers or max(1, (os.cpu_count() or 4) - 2)

    progress(f"{len(cells)} cells, {len(todo)} to run, {workers} workers")
    if not todo:
        return {"total": len(cells), "ran": 0, "errors": []}

    errors, done = [], 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_cell, c, cfg, root): c for c in todo}
        for fut in as_completed(futures):
            res = fut.result()
            done += 1
            if res["status"] == "error":
                errors.append(res)
                progress(f"[{done}/{len(todo)}] ERROR {res['cell']} :: {res['error']}")
            elif done % 25 == 0 or done == len(todo):
                progress(f"[{done}/{len(todo)}] ok")

    if errors:
        progress(f"{len(errors)} cell(s) failed; see error.json under {root}")
    return {"total": len(cells), "ran": done, "errors": errors}
