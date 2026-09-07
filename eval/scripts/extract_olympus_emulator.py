"""Extract a shipped Olympus BayesNeuralNet emulator into plain arrays.

Run **once**, in the throwaway `.venv-olympus` (TensorFlow), never at sweep time:

    .venv-olympus/bin/python eval/scripts/extract_olympus_emulator.py \
        --datasets photo_pce10 photo_wf3

Why extraction rather than loading the emulator in the experiment venv:

* `olympus` core pins `SQLAlchemy==1.4.45` and `olympus/__init__.py` eagerly
  imports `.databases`/`.emulators`; mlflow (pulled in by MatFormBench) also uses
  sqlalchemy. Installing the package is not an option.
* TensorFlow present at sweep time would compete for threads with 22 single-thread
  workers, and `wall_clock_per_iter` is a reported metric.
* Suite isolation becomes structural: an Olympus cell loads numpy arrays, so it
  cannot drag TF into a MatFormBench run or vice versa.

Two artifacts per dataset land in `eval/data/olympus/`: an `.npz` of weights and
scaler statistics, and a `.json` recording architecture, transforms and
provenance. `benchmarks/olympus/oracle.py` reimplements the forward pass from
those, and `--verify` checks it against this checkpoint's published scores.

## What is actually stored in the checkpoint

Each `tfp.layers.DenseLocalReparameterization` keeps `kernel_posterior_loc` and
`kernel_posterior_untransformed_scale`, plus a `bias_posterior_loc`. There is no
`bias_posterior_untransformed_scale`: TFP's default bias posterior is singular,
i.e. a point mass at `loc`, so only the kernel is random. `model/Variable` is the
pre-softplus observation scale, shaped `(batch_size, targets_dim)` because the
graph builds it from `y_pred.get_shape()`; every row is the same trained scalar.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OLYMPUS_SRC = REPO / "eval" / "external" / "olympus" / "src" / "olympus"
OUT_DIR = REPO / "eval" / "data" / "olympus"

#: Why the emulator's aleatoric scale is recorded but never used.
#:
#: `wrapper_bayes_neural_net._build_inference` builds it as
#: `softplus(Variable(tf.ones(self.y_pred.get_shape())))`, and `y_pred` is shaped
#: `(batch_size, targets_dim)`. The observation noise is therefore learned *per
#: position within the prediction batch* rather than per output: the checkpoints
#: hold 50 distinct values (photo_pce10 sigma 0.196-0.232, photo_wf3 0.104-0.120),
#: so the same design gets a different aleatoric scale depending on where it lands
#: in a batch. That is an upstream defect, not a modelling choice.
#:
#: It costs us nothing, because it never enters the returned prediction. Olympus's
#: `Emulator.run` back-transforms `y_pred` alone and hands `sigma_al` back as a
#: separate value its callers ignore. Our `observe` noise is the kernel-posterior
#: (local reparameterization) noise carried inside `y_pred`, which is well defined.
OBS_SCALE_NOTE = "recorded for provenance; batch-position-indexed upstream, unused here"

#: float32 machine epsilon, which TFP's `default_loc_scale_fn` adds to softplus
#: before using the result as the posterior scale. Dropping it changes the
#: sampled noise by ~1e-7 -- negligible numerically, kept so the reimplementation
#: is the same function rather than approximately the same function.
SOFTPLUS_EPS = float(np.finfo(np.float32).eps)


# -- reading emulator.pickle without importing olympus ----------------------
#
# The pickle references only pure-python olympus classes plus a pandas DataFrame,
# but unpickling them would run `olympus/__init__.py` (sqlalchemy) and would break
# on the pandas version skew of the embedded frame. Every non-numpy class is
# therefore replaced by a placeholder that accepts state and any attribute
# access; we read the restored `__dict__` and nothing else.
#
# `__getattr__` is load-bearing: `DataTransformer` pickles its transform list as
# *bound methods*, which unpickle as `getattr(obj, "_forward_standardize")`.

class _Stub:
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        self.__dict__.update(state if isinstance(state, dict) else {"_state": state})

    def __call__(self, *args, **kwargs):
        return _Stub()

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Stub()


class _ShimUnpickler(pickle.Unpickler):
    _made: dict[str, type] = {}

    def find_class(self, module, name):
        if module.startswith(("olympus", "pandas")):
            key = f"{module}.{name}"
            self._made.setdefault(key, type(name, (_Stub,), {"_shim_path": key}))
            return self._made[key]
        return super().find_class(module, name)


def read_emulator_pickle(path: Path) -> dict:
    """Model hyperparameters and the two fitted DataTransformers."""
    with open(path, "rb") as handle:
        emu = _ShimUnpickler(handle).load()

    model = emu.__dict__["model"].__dict__
    feat = emu.__dict__["feature_transformer"].__dict__
    targ = emu.__dict__["target_transformer"].__dict__

    for name, tr in (("feature", feat), ("target", targ)):
        if not tr.get("trained"):
            raise RuntimeError(f"{path.parent.name}: {name} transformer is untrained")

    return {
        "hidden_depth": int(model["hidden_depth"]),
        "hidden_nodes": int(model["hidden_nodes"]),
        "hidden_act": str(model["hidden_act"]),
        "out_act": str(model["out_act"]),
        "batch_size": int(model["batch_size"]),
        "task": str(model["task"]),
        "feature_transform": list(feat["transformations"]),
        "target_transform": list(targ["transformations"]),
        "x_mean": np.asarray(feat["_mean"], dtype=np.float64),
        "x_stddev": np.asarray(feat["_stable_stddev"], dtype=np.float64),
        "x_min": np.asarray(feat["_stable_min"], dtype=np.float64),
        "x_max": np.asarray(feat["_stable_max"], dtype=np.float64),
        "y_mean": np.asarray(targ["_mean"], dtype=np.float64),
        "y_stddev": np.asarray(targ["_stable_stddev"], dtype=np.float64),
        "y_min": np.asarray(targ["_stable_min"], dtype=np.float64),
        "y_max": np.asarray(targ["_stable_max"], dtype=np.float64),
    }


# -- reading the checkpoint --------------------------------------------------

def read_checkpoint(model_dir: Path) -> dict[str, np.ndarray]:
    """Layer weights, by position. Requires TensorFlow; no graph is built.

    Optimizer slots (`.../Adam`, `.../Adam_1`, `beta*_power`) are dropped -- they
    are training state, not the model.
    """
    import tensorflow as tf  # noqa: PLC0415  -- only this venv has it

    reader = tf.train.load_checkpoint(str(model_dir / "model.ckpt"))
    names = [n for n in reader.get_variable_to_shape_map() if "/Adam" not in n]

    prefix = "model/sequential/dense_local_reparameterization"
    out: dict[str, np.ndarray] = {}
    index = 0
    while True:
        scope = prefix if index == 0 else f"{prefix}_{index}"
        loc = f"{scope}/kernel_posterior_loc"
        if loc not in names:
            break
        out[f"w{index}_loc"] = reader.get_tensor(loc).astype(np.float64)
        out[f"w{index}_untransformed_scale"] = reader.get_tensor(
            f"{scope}/kernel_posterior_untransformed_scale"
        ).astype(np.float64)
        out[f"b{index}_loc"] = reader.get_tensor(
            f"{scope}/bias_posterior_loc"
        ).astype(np.float64)
        if f"{scope}/bias_posterior_untransformed_scale" in names:
            raise RuntimeError(
                f"{scope}: bias posterior is not singular in this checkpoint; the "
                "reimplementation assumes a deterministic bias"
            )
        index += 1

    if not out:
        raise RuntimeError(f"{model_dir}: no dense_local_reparameterization variables")

    # Trained pre-softplus observation noise, kept for provenance only -- see
    # `OBS_SCALE_NOTE`. Stored in full because the rows are *not* identical.
    out["obs_scale_untransformed"] = reader.get_tensor("model/Variable").astype(np.float64)
    out["_n_layers"] = np.array(index)
    return out


def published_scores(model_dir: Path) -> dict[str, float]:
    """The R2/RMSD Olympus recorded when it trained this emulator.

    Measured with `num_samples=10` in *scaled* target space, so the reproduction
    must average ten stochastic passes and score before back-transforming.
    """
    text = (model_dir / "training_completed.info").read_text(encoding="utf-8")
    scores = {}
    for line in text.splitlines():
        # "Train R2=0.9244..." -- the split label matters: both the train and the
        # validation line key on "R2=", so partitioning on "=" alone loses one.
        line = line.strip()
        if not line or "=" not in line:
            continue
        label, _, value = line.partition("=")
        scores[label.strip().replace(" ", "_").lower()] = float(value)
    return scores


def extract(dataset: str) -> tuple[Path, Path]:
    emu_dir = OLYMPUS_SRC / "emulators" / f"emulator_{dataset}_BayesNeuralNet"
    meta = read_emulator_pickle(emu_dir / "emulator.pickle")
    weights = read_checkpoint(emu_dir / "Model")

    n_layers = int(weights.pop("_n_layers"))
    if n_layers != meta["hidden_depth"] + 1:
        raise RuntimeError(
            f"{dataset}: checkpoint has {n_layers} layers but the pickle says "
            f"hidden_depth={meta['hidden_depth']} (expected {meta['hidden_depth'] + 1})"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    npz_path = OUT_DIR / f"emulator_{dataset}.npz"
    arrays = dict(weights)
    for key in ("x_mean", "x_stddev", "x_min", "x_max",
                "y_mean", "y_stddev", "y_min", "y_max"):
        arrays[key] = meta[key]
    np.savez(npz_path, **arrays)

    json_path = OUT_DIR / f"emulator_{dataset}.json"
    json_path.write_text(
        json.dumps(
            {
                "dataset": dataset,
                "source": "aspuru-guzik-group/olympus, shipped BayesNeuralNet emulator",
                "model": "BayesNeuralNet (tfp DenseLocalReparameterization stack)",
                "n_layers": n_layers,
                "hidden_depth": meta["hidden_depth"],
                "hidden_nodes": meta["hidden_nodes"],
                "hidden_act": meta["hidden_act"],
                "out_act": meta["out_act"],
                "leaky_relu_alpha": 0.2,
                "batch_size": meta["batch_size"],
                "task": meta["task"],
                "feature_transform": meta["feature_transform"],
                "target_transform": meta["target_transform"],
                "softplus_eps": SOFTPLUS_EPS,
                "bias_posterior": "singular (point mass at loc)",
                "obs_scale_untransformed": OBS_SCALE_NOTE,
                "obs_scale_softplus_range": [
                    float(np.log1p(np.exp(-np.abs(o))) + max(o, 0.0))
                    for o in (
                        float(weights["obs_scale_untransformed"].min()),
                        float(weights["obs_scale_untransformed"].max()),
                    )
                ],
                "published_scores": published_scores(emu_dir / "Model"),
                "published_scores_protocol": (
                    "num_samples=10 averaged stochastic passes, scored in scaled "
                    "target space (before back_transform)"
                ),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return npz_path, json_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=["photo_pce10", "photo_wf3"])
    args = ap.parse_args()

    for dataset in args.datasets:
        npz_path, json_path = extract(dataset)
        meta = json.loads(json_path.read_text(encoding="utf-8"))
        with np.load(npz_path) as data:
            shapes = ", ".join(
                f"{k}{tuple(data[k].shape)}" for k in sorted(data) if k.startswith("w")
            )
        print(f"{dataset}: {meta['n_layers']} layers, {shapes}")
        print(
            f"  act {meta['hidden_act']}/{meta['out_act']}, "
            f"transforms {meta['feature_transform']}->{meta['target_transform']}, "
            f"published {meta['published_scores']}"
        )
        print(f"  wrote {npz_path.relative_to(REPO)}, {json_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
