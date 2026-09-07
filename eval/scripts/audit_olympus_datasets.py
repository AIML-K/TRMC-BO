"""Find which Olympus datasets are actually mixture (simplex) problems.

    PYTHONPATH=core:. .venv-eval/bin/python -m eval.scripts.audit_olympus_datasets

**Olympus's own declaration cannot be trusted, in either direction**, which is why
this reads the measurements rather than `config.json`:

* `colors_bob` declares `"parameters": "simplex"` and is not one -- not a single
  row sums to 1 (0.64 to 4.40), because its features are independent dye volumes.
* the four `oer_plate` datasets declare it correctly, but never use more than 4 of
  6 components, so a uniform Dirichlet design queries a stratum with no training
  data at all.

Both cost a selected task. So detection here is by **affine dependency**: a
mixture constraint means some subset S of the input columns satisfies
`sum_{j in S} x_j = c` for every row, i.e. the centred design matrix has an
indicator vector in its null space. Taking the null space via SVD finds such
groups whatever the declaration says, and finds them on *subsets* -- MatFormBench's
10-D/15-D tasks put only x1..x6 on a simplex and leave the rest as process
variables, and an Olympus dataset could be shaped the same way.

Reported per dataset, because each has separately disqualified a candidate:

``group``       the columns that sum to a constant, and to what
``interior``    share of rows using *every* column of that group (oer_plate: 0%)
``dtypes``      whether any input is categorical -- the harness adapter is
                continuous-only, so a categorical dataset is unusable regardless
``emulator``    whether a trained BayesNeuralNet ships, and its published R2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OLYMPUS = REPO / "eval" / "external" / "olympus" / "src" / "olympus"

#: A singular value below this (relative to the largest) counts as a null
#: direction. Deliberately loose -- thin_film's compositions are rounded to two
#: decimals, which puts its null direction at 3.1e-3 of the leading one, just past
#: a 3e-3 cut. The explicit constant-sum check below is what actually rejects a
#: false positive, so this only has to avoid missing a real group; the gap to the
#: next singular value is ~200x in every dataset that has one.
NULL_RTOL = 2e-2

#: How far a subset's row-sum may stray from its median before it is not a
#: constant sum. thin_film's rounding gives 1e-2 on a total of 1.
SUM_RTOL = 0.02

#: How close a null vector's non-zero entries must be to equal before it is read
#: as an indicator vector, i.e. a plain sum rather than a weighted one.
INDICATOR_RTOL = 0.05


def load(dataset: str) -> tuple[dict, pd.DataFrame | None, str]:
    d = OLYMPUS / "datasets" / f"dataset_{dataset}"
    config = json.loads((d / "config.json").read_text(encoding="utf-8"))
    try:
        frame = pd.read_csv(d / "data.csv", header=None)
    except Exception as exc:  # noqa: BLE001 -- report, do not abort the audit
        return config, None, f"unreadable: {type(exc).__name__}"
    return config, frame, ""


def simplex_groups(X: np.ndarray) -> list[tuple[tuple[int, ...], float]]:
    """Column subsets whose row-sum is constant, found via the null space.

    A mixture group means `X[:, S] @ 1 = c`, so `1_S` lies in the null space of
    the *centred* matrix. Any null vector proportional to a 0/1 pattern is such a
    group; a null vector with unequal weights is some other affine relation and is
    not a mixture constraint.
    """
    if X.shape[0] <= X.shape[1]:
        return []
    centred = X - X.mean(axis=0)
    scale = float(np.abs(centred).max())
    if scale == 0:
        return []
    _, sv, vt = np.linalg.svd(centred / scale, full_matrices=False)

    found = []
    for value, vec in zip(sv, vt):
        if value > NULL_RTOL * sv[0]:
            continue
        mag = np.abs(vec)
        big = mag > INDICATOR_RTOL * mag.max()
        if not big.any():
            continue
        # Indicator-like: every participating entry has the same magnitude and
        # the same sign, and the rest are ~0.
        if mag[big].std() > INDICATOR_RTOL * mag[big].mean():
            continue
        if len(set(np.sign(vec[big]))) != 1:
            continue
        idx = tuple(int(i) for i in np.flatnonzero(big))
        total = float(np.median(X[:, list(idx)].sum(axis=1)))
        spread = float(np.abs(X[:, list(idx)].sum(axis=1) - total).max())
        if spread > SUM_RTOL * max(abs(total), 1.0):
            continue
        found.append((idx, total))
    return found


def emulator_info(dataset: str) -> str:
    matches = sorted((OLYMPUS / "emulators").glob(f"emulator_{dataset}_*"))
    if not matches:
        return "none"
    kind = matches[0].name[len(f"emulator_{dataset}_"):]
    info = matches[0] / "Model" / "training_completed.info"
    if not info.exists():
        return kind
    scores = {}
    for line in info.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            label, _, value = line.strip().partition("=")
            scores[label.strip().replace(" ", "_").lower()] = float(value)
    train = scores.get("train_r2")
    valid = scores.get("validation_r2")
    if train is None:
        return kind
    return f"{kind} R2 {train:.3f}/{valid:.3f}"


def audit(dataset: str) -> dict:
    config, frame, err = load(dataset)
    params = config.get("parameters", [])
    measurements = config.get("measurements", [])
    declared = config.get("constraints", {}).get("parameters")
    types = sorted({p.get("type", "?") for p in params})

    row = {
        "dataset": dataset,
        "dim": len(params),
        "n": 0 if frame is None else len(frame),
        "declared": declared,
        "types": "+".join(types),
        "n_targets": len(measurements),
        "goal": config.get("default_goal"),
        "emulator": emulator_info(dataset),
        "group": "",
        "total": np.nan,
        "interior": np.nan,
        "note": err,
    }
    if frame is None or "categorical" in types or len(params) < 2:
        if "categorical" in types:
            row["note"] = "categorical inputs -- adapter is continuous-only"
        return row

    block = frame.iloc[:, : len(params)]
    numeric = block.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        row["note"] = "non-numeric values in the parameter block"
        return row

    X = numeric.to_numpy(float)
    groups = simplex_groups(X)
    if not groups:
        row["note"] = row["note"] or "no constant-sum subset"
        return row

    # Report the widest group; a dataset with several would need the full list.
    idx, total = max(groups, key=lambda g: len(g[0]))
    row["group"] = ("all" if len(idx) == len(params)
                    else "+".join(str(i) for i in idx))
    row["total"] = total
    sub = X[:, list(idx)]
    row["interior"] = float(((sub > 0).sum(axis=1) >= len(idx)).mean())
    if len(groups) > 1:
        row["note"] = f"{len(groups)} constant-sum subsets"
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None)
    args = ap.parse_args()

    names = args.datasets or sorted(
        p.name[len("dataset_"):] for p in (OLYMPUS / "datasets").glob("dataset_*")
    )
    rows = [audit(name) for name in names]
    df = pd.DataFrame(rows)

    mixture = df[df["group"] != ""].copy()
    other = df[df["group"] == ""].copy()

    pd.set_option("display.width", 250)
    print(f"{len(df)} datasets scanned; {len(mixture)} have a constant-sum column subset\n")

    print("=== mixture candidates (found in the data, not the declaration) ===")
    show = mixture.sort_values(["interior", "n"], ascending=[False, False])
    print(show[["dataset", "dim", "n", "declared", "types", "group", "total",
                "interior", "goal", "emulator", "note"]].to_string(index=False))

    print("\n=== declaration vs. data disagreements ===")
    for _, r in df.iterrows():
        declared_simplex = r["declared"] == "simplex"
        found = r["group"] != ""
        if declared_simplex and not found:
            print(f"  {r['dataset']:16s} declares simplex, none found in the data "
                  f"({r['note']})")
        elif found and not declared_simplex:
            print(f"  {r['dataset']:16s} sums to {r['total']:.4g} on {r['group']} "
                  f"but declares {r['declared']!r}")

    print("\n=== not mixture problems ===")
    print(other[["dataset", "dim", "n", "declared", "types", "note"]].to_string(index=False))


if __name__ == "__main__":
    main()
