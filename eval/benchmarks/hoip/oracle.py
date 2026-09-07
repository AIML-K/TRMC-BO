"""Lookup oracle over a finite catalogue of real materials.

`observe` **is** `truth`. There is no emulator and no measurement model here: the
responses are the DFT numbers upstream tabulated, returned unchanged. Branin is
the only other noiseless oracle in this harness, and the reason to say so plainly
is that it makes HOIP the one suite whose conclusions cannot be an artifact of a
surrogate -- which is also what the deferred Olympus "pool protocol" was for.

## Snapping, and why the snapped design is what gets returned

The design space is 11 x 29 x 4 categorical choices. It is presented to the
optimizer as the 14-D space of the benchmark's own physicochemical descriptors
(6 per molcat, 4 per metal, 4 per halogen), which is the representation Anubis
itself found effective -- their `desc_` arms. The production acquisition path
then runs exactly as it does on MatFormBench and Olympus, with no categorical
branch and no adapter surgery, so C1/C2/C3 are the shipped components rather than
re-implementations.

A proposal therefore lands anywhere in the descriptor hull, and only 1,276 of
those points are materials. `snap` maps a proposal to one, **per block**: the
nearest molcat by its 6 descriptors, the nearest metal by its 4, the nearest
halogen by its 4, each in the coordinates normalized by this task's own bounds.
Per block rather than jointly because it is the exact inverse of the encoding --
the encoding is a concatenation, so its inverse is a concatenation of inverses --
and because a joint nearest-neighbour would let the block with the widest
descriptor spread decide the other two.

`Observation.X` carries the **snapped** coordinates, not the proposal, and that
one decision is what keeps the rest of the harness correct without changing it:

* the GP trains on real materials rather than on hull interior points that no
  material occupies;
* `loop._is_duplicate` sees a re-proposal of an exhausted material as the
  duplicate it is, so a method that keeps landing on the same material pays for
  it through `duplicate_proposal_rate` instead of quietly re-scoring a hit;
* `metrics` scores the material actually evaluated.

The relax-then-round gap this introduces -- the acquisition was maximized at the
unsnapped point -- is measured rather than assumed away: `snap_distance` and the
acquisition re-scored at the snapped point are recorded per proposal.

## What infeasible means here, exactly

A composition absent from `df_results.csv` returns `(nan, nan)` upstream, and
that absence is the unknown input feasibility this suite exists to measure. It
conflates two things -- not a stable perovskite, and stable but metallic so the
effective mass is undefined (8 such compositions; see
`scripts/audit_hoip_catalogue.py`). Both are inherited unchanged; a metallic
material fails the window regardless of how it is labelled.

Two feasible materials carry a censored effective mass (`>1000` upstream). They
are feasible with `m_star = NaN`: a censoring sentinel is not a measurement.
"""

from __future__ import annotations

import numpy as np

from eval.benchmarks.base import BenchmarkTask, Observation
from eval.benchmarks.hoip import data as hd
from eval.benchmarks.hoip import tasks as htasks

#: `failure_type` labels. Kept distinct because they are different events:
#: the first is an infeasible design, the second a feasible one with an
#: unobserved output.
FAIL_INFEASIBLE = "not_in_catalogue"
FAIL_CENSORED = "m_star_censored"


class HOIPOracle:
    """Catalogue lookup with per-block snapping. Deterministic; `seed` is ignored."""

    def __init__(self, task: BenchmarkTask):
        self.task = task
        self.n_calls = 0

        self._keys = htasks.materials(task.task_id)
        self._X = hd.encode(self._keys)

        lo, hi = task.bounds
        span = np.where(hi - lo == 0, 1.0, hi - lo)
        self._lo, self._span = lo, span

        # Per-block option tables, in the same normalized coordinates the
        # proposals arrive in, so snapping and `to_normalized` never disagree.
        entry = htasks.catalogue(task.task_id)
        declared = {"molcats": hd.MOLCATS, "metals": hd.METALS, "halogens": hd.HALOGENS}
        descriptors = hd.load_descriptors()
        self._blocks = []
        for block, slot in hd.block_slices().items():
            options = [o for o in declared[block] if o in set(entry[block])]
            raw = np.asarray(
                [descriptors[block]["values"][o] for o in options], dtype=float
            )
            # Both forms are kept on purpose. The normalized table is what the
            # nearest-neighbour search runs in, so no descriptor's units decide
            # the match. The raw table is what gets *returned*: rebuilding a
            # coordinate as `z * span + lo` would leave a relative float error,
            # and `molcat_scf_e` spans ~1e4, so that lands within an order of
            # magnitude of `loop.CampaignConfig.duplicate_tol` (1e-8) -- close
            # enough that two evaluations of one material could stop comparing
            # equal. Returning the stored value makes the snap exact.
            self._blocks.append(
                (block, slot, options, (raw - lo[slot]) / span[slot], raw)
            )

        responses = hd.response_map()
        censored = {
            (r.molcat, r.metal, r.halogen)
            for r in hd.load_table().itertuples()
            if r.censored
        }
        self._responses = responses
        self._censored = censored
        #: Relax-then-round step size, one entry per observed proposal. A
        #: discrete space presented continuously has to report this rather than
        #: let it disappear inside the oracle.
        self._snap_distances: list[float] = []

    # -- geometry ----------------------------------------------------------

    def enumerate_designs(self) -> np.ndarray:
        """Every material in the catalogue, raw coordinates, canonical order.

        The design space is finite, so any diagnostic that samples it on the other
        suites can be exact here. `scripts/valid_region_geometry.py` uses this.
        """
        return self._X.copy()

    def snap(self, X: np.ndarray) -> tuple[np.ndarray, list[tuple[str, str, str]], np.ndarray]:
        """Nearest catalogue material per block.

        Returns the snapped raw coordinates, the material keys, and the L2
        distance in normalized coordinates from each proposal to its material --
        the size of the relax-then-round step, which a discrete benchmark
        presented continuously has to report rather than hide.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        Z = (X - self._lo) / self._span
        snapped = np.empty_like(X)
        picks = []
        for _block, slot, options, table, raw in self._blocks:
            d = np.linalg.norm(Z[:, None, slot] - table[None, :, :], axis=-1)
            choice = np.argmin(d, axis=1)
            picks.append([options[i] for i in choice])
            snapped[:, slot] = raw[choice]
        keys = list(zip(*picks))
        distance = np.linalg.norm(Z - (snapped - self._lo) / self._span, axis=1)
        return snapped, keys, distance

    # -- the oracle boundary ----------------------------------------------

    def canonicalize(self, X: np.ndarray) -> np.ndarray:
        """The design that would actually be evaluated for each proposal.

        `harness/loop.py` calls this before its duplicate check, so two proposals
        rounding to the same material count as the repeat they are. Absent on the
        continuous suites, where it would be the identity.
        """
        return self.snap(X)[0]

    def truth(self, X: np.ndarray) -> Observation:
        """Noiseless evaluation. Scoring only -- never fed back into a method."""
        snapped, keys, _distance = self.snap(X)
        Y = np.full((len(keys), len(self.task.y_names)), np.nan)
        feasible = np.zeros(len(keys), dtype=bool)
        failure: list[str | None] = []
        for i, key in enumerate(keys):
            if key not in self._responses:
                failure.append(FAIL_INFEASIBLE)
                continue
            gap, m_star = self._responses[key]
            Y[i] = (gap, m_star)
            feasible[i] = True
            failure.append(FAIL_CENSORED if key in self._censored else None)
        return Observation(X=snapped, Y=Y, feasible=feasible, failure_type=failure)

    def observe(self, X: np.ndarray, seed: int | None = None) -> Observation:
        """What the optimizer may consume. The lookup is noiseless, so this is
        `truth` -- stated rather than defaulted, the way Branin's is."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        self.n_calls += len(X)
        self._snap_distances.extend(float(d) for d in self.snap(X)[2])
        return self.truth(X)

    def initial_design(self, n: int, seed: int) -> np.ndarray:
        """`n` distinct materials, drawn uniformly from the catalogue.

        Uniform over *materials*, which is the natural reading of "X% of the
        design space is manufacturable" for an enumerable catalogue, and the
        measure every count in `specs/hoip/` is computed under.

        Infeasible draws are not replaced. On `full` roughly 9% of the catalogue
        is feasible, so 30 draws label about 3 materials and the loop's
        `min_labeled` extension will often fire -- coping with that is the
        sparse-feasibility task, exactly as it is on MatFormBench L5-4, and the
        extra oracle calls are reported rather than billed to the budget.

        Distinct, and correct at `n == 1`, because the loop draws single points
        from here as its random fallback.
        """
        rng = np.random.default_rng(seed)
        size = min(int(n), len(self._keys))
        idx = rng.choice(len(self._keys), size=size, replace=False)
        return self._X[idx]


    def diagnostics(self) -> dict:
        """Suite-specific numbers the recorder folds into `run.json`.

        `snap_distance` is the L2 step from proposal to material in normalized
        coordinates. Near zero means the acquisition optimizer was already
        landing on the catalogue; large values mean the reported acquisition
        value belongs to a point that was never evaluated, and the comparison has
        to be read with that in mind.
        """
        d = np.asarray(self._snap_distances, dtype=float)
        if not d.size:
            return {"n_snapped": 0}
        return {
            "n_snapped": int(d.size),
            "snap_distance_mean": float(d.mean()),
            "snap_distance_median": float(np.median(d)),
            "snap_distance_max": float(d.max()),
            "snap_distance_zero_rate": float((d <= 1e-12).mean()),
        }


def make_oracle(task: BenchmarkTask) -> HOIPOracle:
    return HOIPOracle(task)
