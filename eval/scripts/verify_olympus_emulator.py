"""Fidelity gate for the extracted Olympus emulators (milestone O1).

Runs in the throwaway TensorFlow venv, never at sweep time:

    .venv-olympus/bin/python eval/scripts/verify_olympus_emulator.py

It imports `benchmarks/olympus/data.py` -- the module the sweep actually uses --
and compares it against TFP's own `DenseLocalReparameterization` loaded with the
same extracted weights. Checking a private copy of the forward pass would say
nothing about the one that runs.

Three checks, because each can fail independently:

**(A) Structure.** Set every kernel posterior scale to ~0 so TFP's layers become
deterministic, then compare against our `mean_pass`. This catches a transposed
weight, a dropped bias, the wrong activation, a wrong layer count or a wrong
output activation -- everything except the noise model. The residual floor is
float32 accumulation over 48-wide matmuls, ~1e-6, since TFP runs float32 and we
run float64.

**(B) Noise model.** With the real scales, compare the sampled mean against TFP's
at increasing sample counts. A correct noise model leaves only Monte-Carlo error,
so the gap must fall like 1/sqrt(K); a systematic error (missing softplus, wrong
variance formula, noise on the bias) leaves a floor that does not move. Testing
"is the gap small" at one K cannot tell those apart, which is why K is swept.

**(C) End-to-end R2.** Reproduce Olympus's own scoring protocol -- ten averaged
stochastic passes, scored in *scaled* target space -- over the shipped
measurements, and print it beside the R2 the emulator recorded at training time.

On (C), note what cannot be checked: Olympus's `Dataset` splits with
`random_seed=None`, i.e. `np.random.seed(None)`, so the train/validation split
behind the published figures is not reproducible even in principle. An all-rows
score also necessarily mixes the 80% the emulator trained on with the 20% outer
test set it never saw. So (C) is an end-to-end sanity check whose number is
*expected* to sit below the published train R2; the load-bearing evidence is (A),
which pins our forward pass to TFP's at float32 round-off, and (B), which pins the
noise. A wrong transform or activation would show up in (C) as a collapse, not as
a few points.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO)]

from eval.benchmarks.olympus import data as odata  # noqa: E402

#: Untransformed scale that makes softplus underflow, i.e. a deterministic layer.
SILENT = -60.0

#: Structural agreement is bounded by float32 accumulation in TFP, not by us.
STRUCTURE_TOL = 5e-6


def build_reference(emu: odata.Emulator, *, silence: bool = False):
    """TFP's own layer stack, loaded with the extracted weights."""
    import tensorflow as tf
    import tensorflow_probability as tfp

    # TFP's activation, chosen to mirror Olympus's own `act_funcs` table so the
    # reference is theirs and not ours. `linear` is passed as None because
    # tf.keras treats that as the identity.
    tf_act = {
        "linear": None,
        "leaky_relu": lambda y: tf.nn.leaky_relu(y, 0.2),
        "relu": tf.nn.relu,
        "sigmoid": tf.nn.sigmoid,
        "softplus": tf.nn.softplus,
        "softsign": tf.nn.softsign,
    }
    hidden = tf_act[emu.hidden_act]
    out = tf_act[emu.out_act]

    layers = [
        tfp.layers.DenseLocalReparameterization(emu.kernel_loc[i].shape[1], activation=hidden)
        for i in range(emu.n_layers - 1)
    ]
    layers.append(
        tfp.layers.DenseLocalReparameterization(emu.kernel_loc[-1].shape[1], activation=out)
    )
    net = tf.keras.Sequential(layers)
    net.build((None, emu.dim))

    with np.load(odata.ARTIFACT_DIR / f"emulator_{emu.dataset}.npz") as raw:
        for i, layer in enumerate(layers):
            scale = raw[f"w{i}_untransformed_scale"]
            assign = {
                "kernel_posterior_loc": raw[f"w{i}_loc"],
                "kernel_posterior_untransformed_scale": (
                    np.full_like(scale, SILENT) if silence else scale
                ),
                "bias_posterior_loc": raw[f"b{i}_loc"],
            }
            for var in layer.trainable_variables:
                matched = [k for k in assign if k in var.name]
                if len(matched) != 1:
                    raise RuntimeError(f"cannot map TFP variable {var.name!r}")
                var.assign(assign[matched[0]].astype(np.float32))
    return net


def scaled(emu: odata.Emulator, X: np.ndarray) -> np.ndarray:
    """Feature-transform the inputs the way the emulator was trained."""
    forward_x, _ = odata.TRANSFORMS[emu.feature_transform]
    return forward_x(np.asarray(X, dtype=float), emu.x_stats).astype(np.float32)


def unscaled_target(emu: odata.Emulator, y: np.ndarray) -> np.ndarray:
    """Undo the target transform, so TFP's raw output and ours are compared.

    `forward` back-transforms; TFP returns the scaled target. Comparing the
    networks rather than the networks plus a constant means undoing it here.
    """
    forward_y, _ = odata.TRANSFORMS[emu.target_transform]
    return forward_y(np.asarray(y, dtype=float), emu.y_stats)


def probe_designs(dataset: str, n: int, seed: int) -> np.ndarray:
    """Uniform draws on the dataset's own simplex, at the dataset's own scale.

    The scale is read from the measurements, not assumed to be 1: `p3ht` records
    composition in percent and sums to 100. Probing it with draws that sum to 1
    would compare the two implementations far outside the trained range -- which
    checks (A) and (B) would survive, being numerical identities, but it makes the
    reported magnitudes meaningless.
    """
    X, _ = odata.load_measurements(dataset)
    total = float(np.median(X.sum(axis=1)))
    return np.random.default_rng(seed).dirichlet(np.ones(X.shape[1]), size=n) * total


def check_structure(emu: odata.Emulator, X: np.ndarray) -> tuple[float, bool]:
    net = build_reference(emu, silence=True)
    # TFP returns the scaled target; our forward() back-transforms, so undo it
    # to compare the networks rather than the networks plus a constant.
    ref = net(scaled(emu, X)).numpy().ravel()
    mine = unscaled_target(emu, odata.forward(emu, X, mean_pass=True)).ravel()
    gap = float(np.abs(ref - mine).max())
    return gap, gap <= STRUCTURE_TOL


def check_noise(emu: odata.Emulator, X: np.ndarray, counts: tuple[int, ...]) -> list[dict]:
    net = build_reference(emu)
    rows = []
    for k in counts:
        ref = net(np.tile(scaled(emu, X), (k, 1))).numpy().reshape(k, len(X))
        rng = np.random.default_rng(7)
        mine = np.stack([
            unscaled_target(emu, odata.forward(emu, X, rng)).ravel() for _ in range(k)
        ])
        rows.append({
            "k": k,
            "gap": float(np.abs(ref.mean(0) - mine.mean(0)).max()),
            "mc_floor": float(ref.std(0).max() / np.sqrt(k)),
        })
    return rows


def check_r2(emu: odata.Emulator, num_samples: int = 10) -> dict:
    X, y = odata.load_measurements(emu.dataset)
    rng = np.random.default_rng(0)
    draws = np.mean([odata.forward(emu, X, rng).ravel() for _ in range(num_samples)], axis=0)
    mean_pass = odata.forward(emu, X, mean_pass=True).ravel()

    def r2(pred: np.ndarray) -> float:
        # Scored in scaled target space, as Olympus scores it.
        target = unscaled_target(emu, y)
        scaled_pred = unscaled_target(emu, pred)
        return float(1 - ((target - scaled_pred) ** 2).sum() / ((target - target.mean()) ** 2).sum())

    return {
        "n": len(X),
        "r2_sampled": r2(draws),
        "r2_mean_pass": r2(mean_pass),
        "rmse_raw": float(np.sqrt(((y - mean_pass) ** 2).mean())),
        "published": emu.meta["published_scores"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*",
                    default=["photo_pce10", "photo_wf3", "colors_bob", "oer_plate_3496"])
    ap.add_argument("--points", type=int, default=48)
    ap.add_argument("--counts", type=int, nargs="*", default=[1000, 4000, 16000])
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    failures = []
    for dataset in args.datasets:
        emu = odata.load_emulator(dataset)
        X = probe_designs(dataset, args.points, args.seed)
        print(f"\n=== {dataset} ({emu.n_layers} layers, dim {emu.dim}, "
              f"act {emu.hidden_act}/{emu.out_act}, "
              f"{emu.feature_transform}->{emu.target_transform}) ===")

        gap, ok = check_structure(emu, X)
        print(f"  (A) structure   max|TFP_noiseless - mean_pass| = {gap:.3e}  "
              f"tol {STRUCTURE_TOL:.0e}  {'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"{dataset}: structure {gap:.3e}")

        rows = check_noise(emu, X, tuple(args.counts))
        for prev, row in zip([None] + rows[:-1], rows):
            ratio = f"  shrink {prev['gap'] / row['gap']:.2f}x" if prev else ""
            print(f"  (B) noise K={row['k']:<6d} gap {row['gap']:.3e}  "
                  f"MC floor {row['mc_floor']:.3e}{ratio}")
        # Over the full sweep of K the gap must fall roughly like sqrt(K).
        span = (rows[-1]["k"] / rows[0]["k"]) ** 0.5
        observed = rows[0]["gap"] / rows[-1]["gap"]
        verdict = "PASS" if observed >= 0.6 * span else "FAIL"
        print(f"  (B) overall     shrink {observed:.2f}x vs sqrt(K) prediction "
              f"{span:.2f}x  {verdict}")
        if verdict == "FAIL":
            failures.append(f"{dataset}: noise does not scale as 1/sqrt(K)")

        r = check_r2(emu)
        pub = r["published"]
        print(f"  (C) R2 all {r['n']} rows: sampled {r['r2_sampled']:.4f}  "
              f"mean pass {r['r2_mean_pass']:.4f}   raw RMSE {r['rmse_raw']:.5f}")
        print(f"      published (unreproducible split): train {pub['train_r2']:.4f}  "
              f"valid {pub['validation_r2']:.4f}")

    print()
    if failures:
        print("FAILURES:")
        for f in failures:
            print("  -", f)
        raise SystemExit(1)
    print(f"all {len(args.datasets)} emulator(s) passed the fidelity gate")


if __name__ == "__main__":
    main()
