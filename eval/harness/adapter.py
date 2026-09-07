"""Present a `BenchmarkTask` to the production BO code as if it were a dataloader.

`run_mobo(train_x, train_y, dataloader, optimize_config)` reads a specific set of
attributes off its `dataloader` argument. `BenchmarkVariables` synthesizes
exactly that surface from a benchmark task, so the three components under test
(range-probability objective, two-sided mean constraints, mean-filtered restarts)
run in `src/` unmodified rather than being reimplemented here.

Two coordinate systems are in play and mixing them is the easiest way to get
silently wrong results:

    raw         the benchmark's own units. Process dims live on [-1, 1],
                simplex dims on [0, 1].
    normalized  [0, 1]^d, what the GP and the acquisition optimizer see.
                `run_mobo` produces it via `normalize(train_x, item_bound)`.

`item_bound` is raw and ordered [lower; upper]. `search_space` is normalized and
ordered [upper; lower] -- flipped relative to BoTorch, which is why the
production code writes `search_space.flip([0])` at every use site. Both
conventions are inherited, not chosen here.
"""

from __future__ import annotations

import numpy as np
import torch

from eval.benchmarks.base import BenchmarkTask, RangeSpec

DTYPE = torch.double


class BenchmarkVariables:
    """Dataloader-shaped view of a benchmark task.

    Beyond the attributes `run_mobo` reads, this declares two opt-outs that the
    production code resolves with `getattr` defaults:

    ``semicontinuous_inputs = False``
        The formulation pipeline treats a coordinate of 0 as "ingredient
        omitted" and freezes it during acquisition optimization. Benchmark
        coordinates are DoE-coded process variables where 0 is the center point,
        so every coordinate must stay free.

    ``sample_raw_candidates``
        Replaces the Dirichlet-Rescale sampler, which needs the Excel
        group/ratio schema this task has no counterpart for.

    ``restart_pool``
        Optional (n, dim) array of raw designs the acquisition restarts must be
        drawn from, instead of the continuous domain. `None` -- the default, and
        what both finished suites use -- leaves the sampler exactly as it was.
        A catalogue benchmark supplies one: its design space is a finite set, so
        restarting anywhere else would seed the optimizer at points no material
        occupies. It is the same idea as the Dirichlet mirror used for simplex
        tasks, and because every method's restarts (random search included) come
        through here, it applies to all of them equally.
    """

    def __init__(
        self,
        task: BenchmarkTask,
        range_spec: RangeSpec,
        X: np.ndarray,
        Y: np.ndarray,
        *,
        objective_mode: str = "range",
        device: torch.device | str = "cpu",
        seed: int = 0,
        restart_pool: np.ndarray | None = None,
    ):
        self.task = task
        self.range_spec = range_spec
        self.objective_mode = objective_mode
        self.device = torch.device(device)
        self._rng = np.random.default_rng(seed)

        self.item_key = list(task.x_names)
        self.y_key = list(task.y_names)
        self.x_key = list(task.x_names)

        self.item_bound = self._raw_bounds(task).to(self.device)
        # Normalized bounds are [0, 1] by construction; stored max-first to match
        # the loader this stands in for.
        d = task.dim
        self.search_space = torch.stack(
            [torch.ones(d, dtype=DTYPE), torch.zeros(d, dtype=DTYPE)]
        ).to(self.device)

        # No categorical or discrete inputs => `cat_dims`/`dis_dims` stay empty
        # and `optimize_acqf_on_filtered_points` takes its 'conti' branch.
        self.candidates_info: dict = {}
        self.x_info = {name: {"item_val_type": "conti"} for name in task.x_names}
        self.ratio_info: dict = {}
        self.orgroup_info: dict = {}

        self.target_info = self._build_target_info(task, range_spec, objective_mode)

        self._simplex_dims = [i for g in task.simplex_groups for i in g.indices]
        # Stored normalized, which is what `sample_raw_candidates` returns.
        self._restart_pool = (
            None if restart_pool is None
            else np.clip(self.to_normalized(np.asarray(restart_pool, dtype=float)), 0.0, 1.0)
        )
        self.set_data(X, Y)

    # -- data --------------------------------------------------------------

    def set_data(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Install the current observation set (raw units; NaN for failures)."""
        self.train_x = torch.as_tensor(np.asarray(X), dtype=DTYPE, device=self.device)
        self.train_y = torch.as_tensor(np.asarray(Y), dtype=DTYPE, device=self.device)
        self.item_tensor = self.train_x
        self.x = self.train_x
        # `initialize_model` fits objective j only on rows where y_j was observed,
        # which is how failed evaluations (NaN) stay out of the GP.
        self.continuous_value_mask = ~torch.isnan(self.train_y)
        self.weight = torch.ones(len(self.y_key), dtype=DTYPE, device=self.device)

    @staticmethod
    def _build_target_info(
        task: BenchmarkTask, range_spec: RangeSpec, objective_mode: str
    ) -> dict:
        """Describe the outcomes in the vocabulary the production pipeline uses.

        ``range``
            Each output is a two-sided window. This is what selects the
            range-probability objective (C1) inside ``get_acqf``.
        ``native_extremum``
            Each output becomes a plain max/min objective following the
            benchmark's original one-sided direction, so ``get_acqf`` builds an
            ordinary qEHVI with no range handling at all. This is the honest
            "standard BO applied to this task" baseline: it ignores the window
            rather than approximating it, which is a different question from the
            point-target baseline.
        """
        if objective_mode == "range":
            # `ref_*` and `ball_center` are ignored by the production pipeline but
            # let a ball-based acquisition express its target in the *frozen*
            # scoring space rather than in run_mobo's per-iteration
            # standardization -- otherwise it would optimize one ball and be
            # scored against a slightly different one.
            cal = range_spec.calibration
            centroid = cal.get("y_valid_centroid", {})
            ref_center = cal.get("y_reference_center", {})
            ref_scale = cal.get("y_reference_scale", {})
            info = {}
            for name in task.y_names:
                lb = float(range_spec.lower[name])
                ub = float(range_spec.upper[name])
                info[name] = {
                    "obj": "range",
                    "lb": lb,
                    "ub": ub,
                    "weight": 1.0,
                    "ball_center": float(centroid.get(name, 0.5 * (lb + ub))),
                    "ref_center": float(ref_center.get(name, 0.5 * (lb + ub))),
                    "ref_scale": float(ref_scale.get(name, max(ub - lb, 1e-12))),
                }
            return info
        if objective_mode == "native_extremum":
            return {
                t.name: {
                    "obj": "max" if t.kind == "greater" else "min",
                    "weight": 1.0,
                }
                for t in task.native_targets
            }
        raise ValueError(f"unknown objective_mode: {objective_mode!r}")

    @staticmethod
    def _raw_bounds(task: BenchmarkTask) -> torch.Tensor:
        """Per-dimension raw bounds, tightened on simplex dims.

        The rules file declares one box for every dimension, but a simplex
        component can never be negative. Handing [-1, 1] to those dims would put
        half their normalized range outside the feasible set and turn the
        sum == 1 equality into an affine constraint for no reason.
        """
        bounds = torch.as_tensor(task.bounds.copy(), dtype=DTYPE)
        for group in task.simplex_groups:
            idx = list(group.indices)
            bounds[0, idx] = 0.0
            bounds[1, idx] = float(group.total)
        return bounds

    # -- coordinate transforms --------------------------------------------

    def to_normalized(self, X_raw: np.ndarray) -> np.ndarray:
        lo, hi = self.item_bound.cpu().numpy()
        return (np.asarray(X_raw, dtype=float) - lo) / (hi - lo)

    def to_raw(self, Z: np.ndarray) -> np.ndarray:
        lo, hi = self.item_bound.cpu().numpy()
        return np.asarray(Z, dtype=float) * (hi - lo) + lo

    # -- production-code opt-outs -----------------------------------------

    semicontinuous_inputs = False

    def sample_raw_candidates(self, num_restarts: int, dataloader, optimize_config):
        """Acquisition-restart initial conditions, normalized, shape (n, 1, d).

        Mirrors the benchmark's own design geometry: a draw from the design
        catalogue when the task has one, otherwise a Dirichlet draw rescaled to
        respect `min_component` on each simplex group and uniform elsewhere. Kept
        independent of the oracle so drawing restarts costs no evaluations.
        """
        if self._restart_pool is not None:
            n = len(self._restart_pool)
            idx = self._rng.choice(n, size=num_restarts, replace=num_restarts > n)
            Z = self._restart_pool[idx]
            return torch.as_tensor(Z, dtype=DTYPE, device=self.device).unsqueeze(1)

        d = self.task.dim
        Z = self._rng.random((num_restarts, d))
        for group in self.task.simplex_groups:
            idx = list(group.indices)
            k = len(idx)
            w = self._rng.dirichlet(np.ones(k), size=num_restarts)
            floor = group.min_component / group.total
            w = floor + w * (1.0 - k * floor)
            # Simplex dims are normalized onto [0, total] => normalized == raw/total.
            Z[:, idx] = w
        return torch.as_tensor(Z, dtype=DTYPE, device=self.device).unsqueeze(1)

    # -- constraints for the acquisition optimizer -------------------------

    def linear_constraints(self) -> dict:
        """`equality_constraints` / `inequality_constraints` in normalized space.

        Simplex dims normalize onto [0, total], so `sum(z) == 1` holds unchanged
        and `min_component` becomes a plain lower bound on each component.
        """
        eq, ineq = [], []
        for group in self.task.simplex_groups:
            idx = torch.tensor(group.indices, dtype=torch.long, device=self.device)
            coef = torch.ones(len(group.indices), dtype=DTYPE, device=self.device)
            eq.append((idx, coef, 1.0))
            if group.min_component > 0:
                floor = group.min_component / group.total
                for i in group.indices:
                    ineq.append((
                        torch.tensor([i], dtype=torch.long, device=self.device),
                        torch.ones(1, dtype=DTYPE, device=self.device),
                        floor,
                    ))
        return {
            "equality_constraints": eq or None,
            "inequality_constraints": ineq or None,
        }

    def to(self, device):
        self.device = torch.device(device)
        for name in ("item_bound", "search_space", "train_x", "train_y",
                     "item_tensor", "x", "continuous_value_mask", "weight"):
            setattr(self, name, getattr(self, name).to(self.device))
        return self
