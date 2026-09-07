"""Contracts for the HOIP suite.

Three groups.

The first pins the `Oracle` / `BenchmarkTask` contracts that fail *silently* --
a 1-D `Y`, a missing calibration key, an `initial_design` that cannot produce one
point. Each of those yields a run that completes and looks plausible while being
wrong.

The second pins what is specific to a **catalogue** benchmark presented in a
continuous descriptor space: snapping has to be idempotent and separable, the
snapped design is what the harness must see, and the composition parse has to be
unambiguous. That last one is this suite's counterpart of Olympus's pinned
`colors_bob` / `oer_plate` counter-examples: upstream recovers (molcat, metal,
halogen) by sequential substring search over lists in which `S` is a metal and
also the tail of `H3S` and `MS`, which is exactly the shape of parse that
relabels materials without ever failing.

The third pins that the two guarded harness hooks HOIP needs -- `restart_pool`
and `canonicalize` -- change nothing when they are absent, because two finished
suites and 2,424 completed cells depend on that.
"""

from __future__ import annotations

import numpy as np
import pytest

from eval.benchmarks import calibration
from eval.benchmarks.hoip import data as hd
from eval.benchmarks.hoip import ranges as hranges
from eval.benchmarks.hoip import tasks as htasks
from eval.benchmarks.hoip.oracle import FAIL_CENSORED, FAIL_INFEASIBLE, make_oracle
from eval.harness.adapter import BenchmarkVariables

ALL_TASKS = tuple(htasks.SELECTED_TASKS)
WIDTH = "native"

pytestmark = pytest.mark.filterwarnings("ignore")


def _artifacts_present() -> bool:
    return (hd.DATA_ROOT / "df_results.csv").exists() and (
        hd.DATA_ROOT / "descriptors.json"
    ).exists()


needs_data = pytest.mark.skipif(
    not _artifacts_present(),
    reason="HOIP data not fetched; run eval.scripts.fetch_hoip_data",
)


# -- contracts the harness relies on -----------------------------------------

@needs_data
@pytest.mark.parametrize("task_id", ALL_TASKS)
def test_task_declares_a_window_for_every_output(task_id):
    task = htasks.load_task(task_id)
    spec = hranges.load(task_id, WIDTH)
    assert set(spec.lower) == set(task.y_names) == set(spec.upper)
    for name in task.y_names:
        assert spec.lower[name] < spec.upper[name]


@needs_data
@pytest.mark.parametrize("task_id", ALL_TASKS)
def test_scoring_geometry_present(task_id):
    """`metrics.run_metrics` computes both ball criteria unconditionally."""
    task = htasks.load_task(task_id)
    spec = hranges.load(task_id, WIDTH)
    center, scale = spec.reference_scaling(task.y_names)
    assert np.isfinite(center).all() and (scale > 0).all()
    for mapping in ("inscribed", "equal_volume"):
        c, r = spec.ball(task.y_names, mapping)
        assert np.isfinite(c).all() and r > 0


@needs_data
@pytest.mark.parametrize("task_id", ALL_TASKS)
def test_observation_Y_is_always_2d(task_id):
    task = htasks.load_task(task_id)
    oracle = make_oracle(task)
    for n in (1, 5):
        obs = oracle.observe(oracle.initial_design(n, seed=3))
        assert obs.Y.ndim == 2 and obs.Y.shape == (n, len(task.y_names))
        assert obs.X.shape == (n, task.dim)
        assert len(obs.failure_type) == n == len(obs.feasible)


@needs_data
@pytest.mark.parametrize("task_id", ALL_TASKS)
def test_initial_design_is_distinct_and_works_for_one_point(task_id):
    """The loop draws single points from here as its random fallback."""
    task = htasks.load_task(task_id)
    oracle = make_oracle(task)
    assert oracle.initial_design(1, seed=0).shape == (1, task.dim)
    X = oracle.initial_design(40, seed=1)
    assert len({tuple(row) for row in X}) == len(X)


@needs_data
@pytest.mark.parametrize("task_id", ALL_TASKS)
def test_all_inputs_are_continuous(task_id):
    """The adapter takes the 'conti' branch; a categorical dim would silently
    change which acquisition optimizer the ablation is measuring."""
    task = htasks.load_task(task_id)
    spec = hranges.load(task_id, WIDTH)
    oracle = make_oracle(task)
    obs = oracle.observe(oracle.initial_design(5, seed=0))
    variables = BenchmarkVariables(task, spec, obs.X, obs.Y)
    assert variables.candidates_info == {}
    assert {v["item_val_type"] for v in variables.x_info.values()} == {"conti"}
    assert not task.simplex_groups


@needs_data
@pytest.mark.parametrize("task_id", ALL_TASKS)
def test_lookup_is_noiseless(task_id):
    task = htasks.load_task(task_id)
    oracle = make_oracle(task)
    X = oracle.initial_design(20, seed=7)
    a, b = oracle.observe(X, seed=1), oracle.observe(X, seed=999)
    truth = oracle.truth(X)
    for other in (b, truth):
        assert np.array_equal(a.Y, other.Y, equal_nan=True)
        assert np.array_equal(a.feasible, other.feasible)


@needs_data
def test_infeasible_designs_carry_nan_and_a_failure_type():
    task = htasks.load_task("full")
    oracle = make_oracle(task)
    obs = oracle.observe(oracle.enumerate_designs())
    infeasible = ~obs.feasible
    assert infeasible.any()
    assert np.isnan(obs.Y[infeasible]).all()
    assert all(obs.failure_type[i] == FAIL_INFEASIBLE for i in np.flatnonzero(infeasible))
    assert int(obs.feasible.sum()) == len(hd.response_map())


@needs_data
def test_a_censored_effective_mass_is_feasible_but_unobserved():
    """`>1000` upstream is a censoring sentinel. Feeding 1000.0 to the GP would
    put a fictitious design three orders of magnitude past the bulk."""
    task = htasks.load_task("full")
    oracle = make_oracle(task)
    obs = oracle.observe(oracle.enumerate_designs())
    censored = np.array([f == FAIL_CENSORED for f in obs.failure_type])
    assert censored.sum() == len(hd.censored_rows()) == 2
    assert obs.feasible[censored].all()
    assert np.isnan(obs.Y[censored, task.y_names.index("m_star")]).all()
    assert np.isfinite(obs.Y[censored, task.y_names.index("bandgap")]).all()
    assert np.nanmax(obs.Y[:, task.y_names.index("m_star")]) < hd.CENSOR_SENTINEL


# -- what is specific to a catalogue in a continuous coat ---------------------

@needs_data
@pytest.mark.parametrize("task_id", ALL_TASKS)
def test_snapping_is_idempotent_on_the_catalogue(task_id):
    task = htasks.load_task(task_id)
    oracle = make_oracle(task)
    X = oracle.enumerate_designs()
    snapped, keys, distance = oracle.snap(X)
    assert np.allclose(distance, 0.0)
    assert np.allclose(snapped, X)
    assert keys == htasks.materials(task_id)


@needs_data
@pytest.mark.parametrize("task_id", ALL_TASKS)
def test_snapping_lands_on_a_catalogue_material_from_anywhere(task_id):
    task = htasks.load_task(task_id)
    oracle = make_oracle(task)
    rng = np.random.default_rng(0)
    lo, hi = task.bounds
    # Deliberately outside the box as well as inside: the acquisition optimizer
    # is bounded to [0, 1] but nothing guarantees a baseline is.
    X = lo + rng.uniform(-0.2, 1.2, size=(200, task.dim)) * (hi - lo)
    _snapped, keys, _d = oracle.snap(X)
    catalogue = set(htasks.materials(task_id))
    assert all(k in catalogue for k in keys)


@needs_data
def test_snapping_is_separable_across_blocks():
    """Per block, not jointly: the inverse of a concatenation is a concatenation
    of inverses, and a joint nearest neighbour would let the widest-spread block
    decide the other two."""
    task = htasks.load_task("full")
    oracle = make_oracle(task)
    keys = htasks.materials("full")
    X = hd.encode(keys[:50])
    # Perturbing only the halogen coordinates must change only the halogen.
    slot = hd.block_slices()["halogens"]
    rng = np.random.default_rng(1)
    P = X.copy()
    P[:, slot] += rng.normal(0, 1e-3, size=(len(P), slot.stop - slot.start))
    _s, moved, _d = oracle.snap(P)
    for original, new in zip(keys[:50], moved):
        assert original[:2] == new[:2]


@needs_data
def test_the_observation_carries_the_snapped_design_not_the_proposal():
    """What the GP trains on, what the duplicate check compares, and what
    `metrics` scores all have to be the design that was actually evaluated."""
    task = htasks.load_task("dense")
    oracle = make_oracle(task)
    lo, hi = task.bounds
    rng = np.random.default_rng(2)
    X = lo + rng.uniform(0, 1, size=(30, task.dim)) * (hi - lo)
    obs = oracle.observe(X)
    assert not np.allclose(obs.X, X)
    assert np.allclose(obs.X, oracle.canonicalize(X))
    catalogue = {tuple(np.round(r, 12)) for r in oracle.enumerate_designs()}
    assert all(tuple(np.round(r, 12)) in catalogue for r in obs.X)


@needs_data
def test_the_composition_parse_is_unambiguous():
    """Upstream's sequential substring search is exposed to `MS` reading as
    `M` + `S`, and to `S` being both a metal and the tail of `H3S`. Ours matches
    by exact concatenation, so any collision is reported rather than resolved by
    list order."""
    assert hd.parse_ambiguities() == {}
    assert hd.roundtrip_failures() == []
    assert hd.parse_composition("H_3SInCl_3-1D-W") == ("H3S", "In", "Cl")
    assert hd.parse_composition("MSPbBr_3-1D-R") == ("MS", "Pb", "Br")
    assert hd.parse_composition("NH_4SnI_3-2D") == ("NH4", "Sn", "I")
    assert hd.parse_composition("GCdBr_3-1D-W\\bullet") == ("G", "Cd", "Br")


@needs_data
def test_the_catalogue_matches_its_declaration():
    assert not any(hd.undeclared_options().values())
    assert hd.duplicate_compositions() == []
    assert hd.si_table_agreement()["in_lookup_not_in_si"] == []
    for block, report in hd.descriptor_completeness().items():
        assert not report["missing_options"], block
        assert not report["nonfinite_options"], block


@needs_data
def test_the_ladder_varies_feasibility_and_holds_the_valid_set_fixed():
    """The axis under test is input feasibility. If the valid set moved too, the
    distinct-design ceiling would move with it and the axis would be unreadable."""
    frozen = htasks.load_catalogues()
    entries = [frozen["tasks"][t] for t in ("dense", "restricted", "full")]
    rates = [e["infeasible_rate"] for e in entries]
    assert rates == sorted(rates), rates
    assert rates[0] < 0.65 < rates[1] < rates[2]

    valid_sets = []
    for task_id in ("dense", "restricted", "full"):
        task = htasks.load_task(task_id)
        spec = hranges.load(task_id, WIDTH)
        oracle = make_oracle(task)
        obs = oracle.truth(oracle.enumerate_designs())
        hit = spec.hits(obs.Y, task.y_names)
        _s, keys, _d = oracle.snap(obs.X)
        valid_sets.append({k for k, h in zip(keys, hit) if h})
    assert valid_sets[0] == valid_sets[1] == valid_sets[2]
    assert len(valid_sets[0]) == 7

    # Nested, so the axis is a pure restriction of one design space.
    assert frozen["nested"] is True


@needs_data
def test_the_window_is_the_benchmarks_own_not_a_calibrated_one():
    spec = hranges.load("full", WIDTH)
    assert spec.lower["bandgap"] == 0.75 and spec.upper["bandgap"] == 1.75
    assert spec.upper["m_star"] == 4.0
    assert spec.calibration["constructed"] is False
    assert spec.calibration["exact_enumeration"] is True
    # It is a specification, so it is not tuned to a difficulty target.
    assert spec.calibration["target_fraction_of_feasible"] is None
    with pytest.raises(ValueError, match="one condition"):
        hranges.load("full", "medium")


# -- the guarded harness hooks ------------------------------------------------

@needs_data
def test_a_restart_pool_confines_restarts_to_the_catalogue():
    task = htasks.load_task("dense")
    spec = hranges.load("dense", WIDTH)
    oracle = make_oracle(task)
    obs = oracle.observe(oracle.initial_design(10, seed=0))
    pool = oracle.enumerate_designs()
    variables = BenchmarkVariables(
        task, spec, obs.X, obs.Y, seed=0, restart_pool=pool
    )
    Z = variables.sample_raw_candidates(64, variables, {}).squeeze(1).cpu().numpy()
    _s, keys, distance = oracle.snap(variables.to_raw(Z))
    assert np.allclose(distance, 0.0)
    assert set(keys) <= set(htasks.materials("dense"))


def test_absent_hooks_leave_the_continuous_suites_untouched():
    """Two finished suites and 2,424 completed cells depend on this."""
    import inspect

    from eval.harness import loop

    signature = inspect.signature(BenchmarkVariables.__init__)
    assert signature.parameters["restart_pool"].default is None

    source = inspect.getsource(loop.run_campaign)
    assert 'getattr(oracle, "enumerate_designs", None)' in source
    assert 'getattr(oracle, "canonicalize", None)' in source
    # Both are read through `getattr` with a None default and both call sites are
    # guarded, so an oracle that publishes neither behaves exactly as before.
    assert "if canonicalize is not None else cand" in source
    assert "if enumerate_designs is not None else None" in source


@needs_data
def test_running_hoip_pulls_in_neither_tensorflow_nor_matformbench():
    """Suite isolation is structural. HOIP needs two CSVs and no framework at
    all, which is exactly why an accidental import must fail a test."""
    import subprocess
    import sys
    import textwrap
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    script = textwrap.dedent(
        """
        import sys
        from eval.benchmarks.registry import resolve
        suite = resolve("hoip")
        task = suite.load_task("dense")
        oracle = suite.make_oracle(task)
        oracle.observe(oracle.initial_design(5, seed=0))
        bad = [m for m in sys.modules
               if m.split(".")[0] in ("tensorflow", "tensorflow_probability", "olympus")
               or "matformbench" in m.lower()]
        assert not bad, bad
        print("clean")
        """
    )
    env = {"PYTHONPATH": f"{repo / 'src'}:{repo}", "PATH": "/usr/bin:/bin"}
    out = subprocess.run(
        [sys.executable, "-c", script], cwd=repo, env=env,
        capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stderr
    assert "clean" in out.stdout


@needs_data
def test_frozen_specs_are_reproducible():
    """A window regenerated on another machine would silently invalidate every
    cross-run comparison, so the derivation is checked, not trusted."""
    from eval.scripts.freeze_hoip_specs import build_spec

    for task_id in ALL_TASKS:
        rebuilt = build_spec(task_id)
        committed = hranges.load(task_id, WIDTH)
        assert rebuilt.lower == committed.lower
        assert rebuilt.upper == committed.upper
        assert rebuilt.calibration["n_valid"] == committed.calibration["n_valid"]


@needs_data
def test_frozen_catalogues_are_reproducible():
    from eval.scripts.freeze_hoip_catalogues import build

    assert build() == htasks.load_catalogues()


@needs_data
def test_spec_files_are_committed_where_the_loader_looks():
    for task_id in ALL_TASKS:
        assert calibration.spec_path("hoip", task_id, WIDTH).exists()
    assert htasks.CATALOGUES_PATH.exists()
