"""Does an acquisition prefer the candidates that actually pass?

A campaign score answers "which method wins" but not "why". This asks the
narrower question directly: freeze one surrogate, draw a pool of candidates,
score them with each acquisition, and check the ranking against ground truth.
AUC = 1.0 means every in-window candidate outranks every out-of-window one, 0.5
means the acquisition is no better than shuffling the pool.

It is far cheaper than a sweep -- one GP fit per (task, width, seed) instead of
one per iteration -- so it can cover many settings and expose a ranking
difference before committing hours to campaigns. It is also strictly weaker
evidence: a one-step ranking says nothing about how a method explores over 50
iterations, and an acquisition that ranks well can still get stuck. Read it as a
mechanism check on a campaign result, never as a replacement for one.

The surrogate is built by replaying `run_mobo`'s own preamble -- same
normalization, standardization, negation and `initialize_model` -- so the
acquisitions being compared see exactly the model they would see in a real run.

    PYTHONPATH=core:. .venv-eval/bin/python -m eval.scripts.acquisition_auc
    PYTHONPATH=core:. .venv-eval/bin/python -m eval.scripts.acquisition_auc \
        --tasks L3-1 L4-1 --widths medium narrow --seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import torch
from botorch import fit_gpytorch_mll

from bayesian_optimization import (  # noqa: E402  (src/ is on sys.path)
    get_acqf,
    initialize_model,
    negate_y,
    standardize,
)
from botorch.utils.transforms import normalize  # noqa: E402

from eval.benchmarks.matformbench import ranges
from eval.benchmarks.matformbench.oracle import make_oracle
from eval.benchmarks.matformbench.tasks import load_task
from eval.harness.adapter import BenchmarkVariables
from eval.harness.loop import CampaignConfig, build_initial_design
from eval.methods.range_product import product_acqf_factory, product_ei_acqf_factory
from eval.methods.standard_bo import target_distance_acqf_factory

#: label -> factory with `get_acqf`'s signature.
ACQUISITIONS = {
    "qEHVI over P vector (shipped)": get_acqf,
    "product, direct (paper 4.2)": product_acqf_factory,
    "product, in qEI": product_ei_acqf_factory,
    "point-target distance qEI": target_distance_acqf_factory,
}


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUC. NaN when the pool is all-pass or all-fail."""
    pos, neg = labels.sum(), (~labels).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average ranks within ties, or an acquisition that returns a constant would
    # score 1.0 or 0.0 by argsort order alone rather than the 0.5 it deserves.
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    return float((ranks[labels].sum() - pos * (pos + 1) / 2) / (pos * neg))


def build_surrogate(task, spec, seed: int, n_init: int):
    """Replay `run_mobo`'s preamble to get the model the acquisitions would see."""
    oracle = make_oracle(task)
    obs, _ = build_initial_design(oracle, CampaignConfig(n_init=n_init), seed)
    variables = BenchmarkVariables(task, spec, obs.X, obs.Y, seed=seed)

    torch.manual_seed(seed)
    train_x = normalize(variables.train_x, variables.item_bound.to(variables.train_x))
    train_y_std, y_mean, y_std = standardize(
        variables.train_y, variables.continuous_value_mask
    )
    train_y_std = negate_y(train_y_std, variables.y_key, variables.target_info)
    weight = torch.as_tensor(
        [variables.target_info[k]["weight"] for k in variables.y_key],
        dtype=train_y_std.dtype,
    )
    mll, gp_model = initialize_model(
        train_x, train_y_std, [], variables.continuous_value_mask
    )
    fit_gpytorch_mll(mll)
    return {
        "gp_model": gp_model, "train_x": train_x, "train_y": train_y_std,
        "y_key": variables.y_key, "weight": weight,
        "target_info": variables.target_info, "y_mean": y_mean, "y_std": y_std,
    }, variables, oracle


def score_pool(acqf, Z: torch.Tensor, chunk: int = 64) -> np.ndarray:
    """Acquisition values on `(n, 1, d)` candidates, chunked to bound memory."""
    out = []
    with torch.no_grad():
        for i in range(0, len(Z), chunk):
            out.append(acqf(Z[i:i + chunk]).reshape(-1).cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=["L1-1", "L2-1", "L3-1", "L4-1", "L5-4"])
    ap.add_argument("--widths", nargs="*", default=["wide", "medium", "narrow"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--n-init", type=int, default=30)
    ap.add_argument("--pool", type=int, default=600)
    args = ap.parse_args()

    print(f"acquisition ranking AUC -- pool={args.pool} candidates, "
          f"n_init={args.n_init}, seeds={args.seeds}")
    print("AUC 0.5 = no better than shuffling the pool; 1.0 = every passing "
          "candidate ranked above every failing one\n")

    per_cell: list[dict] = []
    for task_id in args.tasks:
        task = load_task(task_id)
        for width in args.widths:
            spec = ranges.load(task_id, width)
            for seed in args.seeds:
                kw, variables, oracle = build_surrogate(task, spec, seed, args.n_init)

                # Same sampler the acquisition optimizer draws its restarts from,
                # so the pool has the geometry the optimizer would actually search.
                Z = variables.sample_raw_candidates(args.pool, variables, {})
                X_raw = variables.to_raw(Z.squeeze(1).cpu().numpy())
                truth = oracle.truth(X_raw)
                ok = truth.feasible & np.isfinite(truth.Y).all(axis=1)
                labels = ok & spec.hits(truth.Y, task.y_names)

                row = {"task": task_id, "width": width, "seed": seed,
                       "base_rate": float(labels.mean())}
                for name, factory in ACQUISITIONS.items():
                    try:
                        row[name] = auc(score_pool(factory(**kw), Z), labels)
                    except Exception as err:  # a factory that cannot build here
                        row[name] = float("nan")
                        row.setdefault("errors", []).append(f"{name}: {err}")
                per_cell.append(row)
                print(f"  {task_id}/{width:<6} seed {seed}  pass rate "
                      f"{row['base_rate']:.3f}  " +
                      "  ".join(f"{n.split(',')[0].split(' ')[0]}={row[n]:.3f}"
                                for n in ACQUISITIONS))

    names = list(ACQUISITIONS)
    print("\n" + "=" * 92)
    print(f"{'task/width':<16}{'pass rate':>10}" + "".join(f"{n[:20]:>21}" for n in names))
    print("-" * 92)
    for task_id in args.tasks:
        for width in args.widths:
            rows = [r for r in per_cell if r["task"] == task_id and r["width"] == width]
            if not rows:
                continue
            base = np.mean([r["base_rate"] for r in rows])
            cells = [np.nanmean([r[n] for r in rows]) for n in names]
            best = int(np.nanargmax(cells)) if not np.all(np.isnan(cells)) else -1
            print(f"{task_id + '/' + width:<16}{base:>10.3f}" + "".join(
                f"{('*' if i == best else '') + f'{v:.3f}':>21}" for i, v in enumerate(cells)
            ))
    print("-" * 92)
    overall = [np.nanmean([r[n] for r in per_cell]) for n in names]
    print(f"{'mean':<16}{'':>10}" + "".join(f"{v:>21.3f}" for v in overall))
    print("\n* = best in that row. Cells with a pass rate of 0.000 contribute no "
          "AUC (nothing to rank) and are averaged out.")

    errs = [(r["task"], r["width"], r["seed"], e)
            for r in per_cell for e in r.get("errors", [])]
    if errs:
        print("\nfactories that failed to build:")
        for t, w, s, e in errs:
            print(f"  {t}/{w} seed {s}: {e}")


if __name__ == "__main__":
    main()
