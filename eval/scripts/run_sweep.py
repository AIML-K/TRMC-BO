"""Run a sweep from a config file.

    PYTHONPATH=core:. .venv-eval/bin/python -m eval.scripts.run_sweep \
        --config eval/configs/matformbench_range_adapted.yaml

Re-running skips completed cells, so an interrupted sweep resumes for free.
Use --smoke for a two-cell shakedown before committing hours of compute.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from eval.harness.runner import RESULT_ROOT, build_matrix, pending, run_sweep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--results", default=str(RESULT_ROOT))
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="re-run completed cells")
    ap.add_argument("--dry-run", action="store_true", help="list the matrix and exit")
    ap.add_argument("--smoke", action="store_true",
                    help="one task, one width, 1 seed -- shakedown only")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.smoke and args.results == str(RESULT_ROOT):
        # A shakedown writes complete-looking cells at budget 5 into the very
        # paths the real sweep resumes from, so the real sweep would skip them
        # and the campaign would silently be 5 iterations deep in a dozen cells.
        # Send it somewhere resume never looks unless told otherwise.
        args.results = str(RESULT_ROOT / "_smoke")
    if args.smoke:
        # The middle difficulty where a suite has three; the only one where it has
        # one. `widths[1:2]` alone silently produced an empty matrix on HOIP,
        # which reads exactly like "nothing left to run".
        widths = cfg["widths"][1:2] or cfg["widths"][:1]
        cfg = {**cfg, "tasks": cfg["tasks"][:1], "widths": widths, "n_seeds": 1,
               "budget": min(cfg.get("budget", 50), 5)}

    root = Path(args.results)
    cells = build_matrix(cfg)
    todo = cells if args.force else pending(cells, root)

    if args.dry_run:
        print(f"{len(cells)} cells total, {len(todo)} pending")
        for cell in todo[:20]:
            print("  ", cell.task_id, cell.width, cell.method, f"seed{cell.seed}")
        if len(todo) > 20:
            print(f"   ... and {len(todo) - 20} more")
        return

    summary = run_sweep(cfg, root=root, workers=args.workers, force=args.force)
    print(f"done: {summary['ran']} run, {len(summary['errors'])} error(s)")


if __name__ == "__main__":
    main()
