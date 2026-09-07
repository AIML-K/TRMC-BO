"""Register the proposed method as a MatFormBench algorithm (backbone §5.1 native).

The native protocol is not sequential BO. It is a batch-recommend evaluation:
propose `n_suggestions` candidates from a fixed 30-row dataset, `chosen_best`
selects `n_choose`, and the benchmark's own metrics take over (`recommend`,
`topk` over `n_rounds`, `dss` across dataset sizes, `stability` across dataset
seeds). Running inside it is what makes our numbers comparable to the 20-plus
baselines MatFormBench ships.

Fidelity comes from subclassing `BaseStandaloneInverseRecommender` and overriding
only `_run_direct_algorithm`. Everything else -- candidate scoring with their
Ridge+TabPFN ensemble, fitness and violation, `chosen_best`, the result frame
layout -- stays theirs. Our contribution is the proposal mechanism, so that is
the only thing replaced. Their repository is never modified: the class is
injected into `ALGORITHM_REGISTRY` at run time (see `eval/scripts/run_native.py`).

## Turning a one-sided spec into a window

Our method optimizes `P(L <= f(x) <= U)`, which needs both bounds finite, while
native targets are one-sided (`y1 > 61`). The free bound is taken from the
observed support of the training data:

    y_k > tau   ->  [tau, max(observed y_k) + margin * range]
    y_k < tau   ->  [min(observed y_k) - margin * range, tau]

This is the loosest finite window consistent with the native spec, so the native
task is not silently made harder than it is. It is a protocol adapter, not the
range-adapted experiment -- that one uses the frozen calibrated windows.
"""

from __future__ import annotations

import numpy as np

from eval.benchmarks.base import BenchmarkTask, NativeTarget, RangeSpec, SimplexGroup
from eval.harness.adapter import BenchmarkVariables
from eval.methods import make as make_method

#: How far past the observed support the open side of a window is placed.
OPEN_SIDE_MARGIN = 0.5


def _resolve_base_class():
    """Imported lazily so this module is importable without MatFormBench present."""
    from inverse_algorithms_v2.standalone_generative import (  # noqa: PLC0415
        BaseStandaloneInverseRecommender,
    )

    return BaseStandaloneInverseRecommender


def _task_from_context(
    features: list[str], target_specs: list[dict], level: str | None, dataset: str | None,
    bounds: np.ndarray,
) -> BenchmarkTask:
    """Prefer the real design space; fall back to the data's bounding box.

    MatFormBench's own baselines infer bounds from the training data's min/max,
    which ignores the simplex groups on the 10D/15D tasks entirely. When the
    algorithm card names the task we load the rules file instead and get both the
    true box and the simplex structure.
    """
    if level and dataset:
        from eval.benchmarks.matformbench.tasks import load_task  # noqa: PLC0415

        return load_task(f"{level}-{dataset.split('-')[-1]}")

    kind = {"greater": "greater", "less": "less"}
    return BenchmarkTask(
        suite="matformbench",
        task_id="native-unknown",
        dim=len(features),
        x_names=tuple(features),
        y_names=tuple(spec["header"] for spec in target_specs),
        bounds=bounds,
        native_targets=tuple(
            NativeTarget(spec["header"], kind[spec["type"]], float(spec["value"]))
            for spec in target_specs
        ),
        simplex_groups=(),
    )


def _window_from_native(
    task: BenchmarkTask, Y: np.ndarray, margin: float = OPEN_SIDE_MARGIN
) -> RangeSpec:
    lower, upper, center, scale, centroid = {}, {}, {}, {}, {}
    for j, name in enumerate(task.y_names):
        col = Y[:, j]
        col = col[np.isfinite(col)]
        lo_obs = float(col.min()) if col.size else 0.0
        hi_obs = float(col.max()) if col.size else 1.0
        span = max(hi_obs - lo_obs, 1e-9)

        tgt = task.target(name)
        if tgt.kind == "greater":
            lower[name] = float(tgt.value)
            upper[name] = hi_obs + margin * span
        else:
            lower[name] = lo_obs - margin * span
            upper[name] = float(tgt.value)

        center[name] = float(np.median(col)) if col.size else 0.0
        q75, q25 = (np.percentile(col, [75, 25]) if col.size else (1.0, 0.0))
        scale[name] = float(max((q75 - q25) / 1.349, 1e-9))
        centroid[name] = 0.5 * (lower[name] + upper[name])

    return RangeSpec(
        lower=lower, upper=upper, width_label="native",
        calibration={
            "rule": f"native one-sided spec, open side at observed support + {margin} x range",
            "y_reference_center": center,
            "y_reference_scale": scale,
            "y_valid_centroid": centroid,
        },
    )


def build_recommender_class(method_name: str = "M5_full"):
    """Create a MatFormBench-compatible class that proposes with `method_name`."""
    base = _resolve_base_class()

    class TRMCRangeBORecommender(base):
        """Proposes with the range-constrained BO method; scores with MatFormBench's."""

        def __init__(self, features, target, surrogate_params=None,
                     objective_weights=None, random_state=42, **optimizer_params):
            self._method_name = optimizer_params.pop("method", method_name)
            self._level = optimizer_params.pop("level", None)
            self._dataset = optimizer_params.pop("dataset", None)
            self._n_suggestions = int(optimizer_params.pop("n_generate", 128))
            super().__init__(
                features=features,
                target=target,
                optimizer_params=optimizer_params,
                objective_weights=objective_weights,
                random_state=random_state,
            )

        def _run_direct_algorithm(self):
            bounds = np.stack([
                self._feature_bounds[:, 0], self._feature_bounds[:, 1]
            ])
            task = _task_from_context(
                self.feature_cols, self.target_specs, self._level, self._dataset, bounds
            )
            spec = _window_from_native(task, self._Y)

            variables = BenchmarkVariables(
                task, spec, self._X, self._Y, seed=self.random_state,
                objective_mode="range",
            )
            method = make_method(self._method_name)
            pool = method.propose_pool(variables, self._n_suggestions, seed=self.random_state)

            # `_finalize` clips to the data box, scores with their ensemble
            # predictor and computes fitness / total_violation, so ranking and
            # `chosen_best` stay entirely MatFormBench's.
            return self._finalize(pool)

    TRMCRangeBORecommender.__name__ = f"TRMCRangeBO_{method_name}"
    return TRMCRangeBORecommender


def register(method_names: tuple[str, ...] = ("M5_full", "M3_range_prob")) -> list[tuple[str, str]]:
    """Inject one registry entry per method. Returns the keys added."""
    from inverse_algorithms_v2 import registry  # noqa: PLC0415

    added = []
    for name in method_names:
        key = (f"trmc_range_bo_{name.lower()}", "none")
        registry.ALGORITHM_REGISTRY[key] = build_recommender_class(name)
        added.append(key)
    return added
