"""Run the MatFormBench native protocol with our method injected.

    PYTHONPATH=core:. .venv-eval/bin/python -m eval.scripts.run_native \
        --tasks L1-1 L2-1 L3-1 L4-1 L5-4 --method M5_full

MatFormBench is driven from its own directory (it writes `synthetic_data/<L>/...`
and reads relative paths), so the working directory is switched for the call and
restored afterwards. Its repository is never modified: the algorithm class is
injected into ALGORITHM_REGISTRY in this process only, and the algorithm card is
written to a temporary file.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path

import yaml

from eval.benchmarks.matformbench.tasks import MATFORMBENCH_ROOT, SELECTED_TASKS, parse_task_id

NATIVE_RESULTS = Path(__file__).resolve().parents[1] / "results" / "matformbench" / "native"


@contextlib.contextmanager
def _dataframe_accepting_metrics():
    """Work around an upstream type mismatch in MatFormBench.

    `team_test_v2.main` calls `_metrics1(data=validate_df, ...)` and
    `_metrics3(data=validate_df, ...)`, but the shipped compiled metrics declare
    `data: str` and reject a DataFrame outright. Their own error text ("data 必须是
    CSV、Excel 文件路径，或 pd.DataFrame") shows a DataFrame branch was intended,
    but the Cython signature makes it unreachable -- so the published runner
    cannot execute for *any* algorithm, ours included. Verified by running their
    own `algo_card_GA_GPR.yaml`, which fails on the identical line.

    The metric computation itself is untouched: the frame is spilled to a
    temporary CSV and the path is handed to the original function.
    """
    import tempfile

    import pandas as pd

    from inverse_metrics import metrics as metrics_mod

    originals = {}

    def _coerce_targets(args, kwargs):
        """`_metrics6` calls `_metrics1` with `targets` as a list of dicts, while
        the compiled signature declares `str`. Same class of mismatch as `data`."""
        if "targets" in kwargs and isinstance(kwargs["targets"], (list, dict)):
            kwargs = {**kwargs, "targets": json.dumps(kwargs["targets"], ensure_ascii=False)}
        elif args and isinstance(args[0], (list, dict)):
            args = (json.dumps(args[0], ensure_ascii=False), *args[1:])
        return args, kwargs

    def wrap(fn):
        def wrapper(data=None, *args, **kwargs):
            args, kwargs = _coerce_targets(args, kwargs)
            if isinstance(data, pd.DataFrame):
                with tempfile.TemporaryDirectory() as tmp:
                    path = os.path.join(tmp, "frame.csv")
                    data.to_csv(path, index=False)
                    return fn(path, *args, **kwargs)
            return fn(data, *args, **kwargs)

        return wrapper

    try:
        for name in ("_metrics1", "_metrics3"):
            originals[name] = getattr(metrics_mod, name)
            setattr(metrics_mod, name, wrap(originals[name]))
        # team_test_v2 imported the names directly, so rebind there as well.
        import team_test_v2

        for name in originals:
            if hasattr(team_test_v2, name):
                setattr(team_test_v2, name, getattr(metrics_mod, name))
        yield
    finally:
        for name, fn in originals.items():
            setattr(metrics_mod, name, fn)
            import team_test_v2 as t

            if hasattr(t, name):
                setattr(t, name, fn)


@contextlib.contextmanager
def _inside_matformbench():
    previous = Path.cwd()
    root = str(MATFORMBENCH_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(previous)


def _card(task_id: str, method: str, seed: int, cfg: dict) -> dict:
    level, dataset = parse_task_id(task_id)
    return {
        "submission": {
            "algo_name": f"TRMC_{method}",
            "description": "Two-sided range-constrained BO (proposal only; scoring is MatFormBench's)",
        },
        "task": {"level": level, "dataset": dataset,
                 "split": cfg["split"], "design": cfg["design"], "seed": seed},
        "algorithm": {
            "optimizer": {
                "name": f"trmc_range_bo_{method.lower()}",
                "params": {"method": method, "level": level, "dataset": dataset,
                           "n_generate": cfg["n_generate"], "random_state": seed},
            },
            "surrogate_model": {"name": "none", "params": {}},
        },
        "evaluation": {
            "recommend": {"n_suggestions": cfg["n_suggestions"], "n_choose": cfg["n_choose"]},
            "topk": {"K": cfg["K"], "n_rounds": cfg["n_rounds"], "min_success": cfg["min_success"]},
            "dss": {"count": cfg["dss_count"]},
        },
        "mlflow": {"enabled": False},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=list(SELECTED_TASKS))
    ap.add_argument("--method", default="M5_full")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0])
    ap.add_argument("--n-suggestions", type=int, default=100)
    ap.add_argument("--n-choose", type=int, default=5)
    ap.add_argument("--n-generate", type=int, default=128)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--n-rounds", type=int, default=5)
    ap.add_argument("--min-success", type=int, default=5)
    ap.add_argument("--split", default="train")
    ap.add_argument("--design", default="lhs")
    args = ap.parse_args()

    cfg = {
        "n_suggestions": args.n_suggestions, "n_choose": args.n_choose,
        "n_generate": args.n_generate, "K": args.K, "n_rounds": args.n_rounds,
        "min_success": args.min_success, "split": args.split, "design": args.design,
        "dss_count": [10, 15, 30, 50, 100],
    }

    NATIVE_RESULTS.mkdir(parents=True, exist_ok=True)

    with _inside_matformbench():
        from eval.benchmarks.matformbench.native_plugin import register

        added = register((args.method,))
        print(f"registered: {added}")

        import team_test_v2

        stack = contextlib.ExitStack()
        stack.enter_context(_dataframe_accepting_metrics())

        for task_id in args.tasks:
            for seed in args.seeds:
                card = _card(task_id, args.method, seed, cfg)
                card_path = Path("inverse_algo_card") / f"_trmc_{args.method}_{task_id}_{seed}.yaml"
                card_path.write_text(yaml.safe_dump(card, sort_keys=False), encoding="utf-8")
                print(f"\n=== {task_id} seed{seed} ===")
                # MatFormBench writes results under its own outputs/, which lives
                # inside the gitignored third-party clone. Snapshot what appears
                # so results survive a re-clone and sit with everything else.
                before = {
                    q: q.stat().st_mtime for q in Path("outputs").glob("*.json")
                } if Path("outputs").exists() else {}
                try:
                    team_test_v2.main(
                        config_path=str(card_path), task_registry_path="task_registry.json"
                    )
                except Exception as exc:  # noqa: BLE001
                    import traceback

                    tb = traceback.format_exc()
                    out = NATIVE_RESULTS / f"{task_id}_{args.method}_seed{seed}_error.json"
                    out.write_text(
                        json.dumps({"error": repr(exc), "traceback": tb}, indent=2),
                        encoding="utf-8",
                    )
                    print(f"  ERROR: {exc!r}\n{tb}")
                finally:
                    card_path.unlink(missing_ok=True)
                    for produced in Path("outputs").glob("*.json"):
                        if before.get(produced) == produced.stat().st_mtime:
                            continue
                        target = NATIVE_RESULTS / f"{task_id}_{args.method}_seed{seed}_{produced.name}"
                        target.write_bytes(produced.read_bytes())
                        print(f"  -> {target.relative_to(NATIVE_RESULTS.parents[2])}")
        stack.close()


if __name__ == "__main__":
    main()
