"""Contracts for the Olympus suite.

Two groups. The first pins the `Oracle`/`BenchmarkTask` contracts that fail
*silently* -- a 1-D `Y`, a missing calibration key, an `initial_design` that
cannot produce one point. Each of those produces a run that completes and looks
plausible while being wrong, which is why they are tests rather than assertions
in a review checklist.

The second pins the two acquisition identities that make this suite worth running
at all. With a single output the ball target can be made exactly the window, and
then the Range-Aware TB acquisition *is* the range probability; C1's shipped
single-output path is that same probability behind an improvement hinge. Together
they let the plateau be measured at matched target, surrogate and restarts. If
either identity broke, the §5.2/§5.3 mechanism argument would be measuring
something else.
"""

from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import torch

from eval.benchmarks import calibration
from eval.benchmarks.olympus import data as odata
from eval.benchmarks.olympus import ranges as oranges
from eval.benchmarks.olympus import tasks as otasks
from eval.benchmarks.olympus.oracle import SurfaceOracle, make_oracle

REPO_ROOT = Path(__file__).resolve().parents[2]

WIDTHS = ("wide", "medium", "narrow")
EMULATOR_TASKS = ("pce10", "wf3", "thin_film")
ALL_TASKS = tuple(otasks.SELECTED_TASKS)

pytestmark = pytest.mark.filterwarnings("ignore")


def _artifacts_present() -> bool:
    return all(
        (odata.ARTIFACT_DIR / f"emulator_{otasks.SELECTED_TASKS[t]['dataset']}.npz").exists()
        for t in EMULATOR_TASKS
    )


needs_data = pytest.mark.skipif(
    not (odata.OLYMPUS_ROOT.exists() and _artifacts_present()),
    reason="Olympus clone or extracted emulator artifacts absent",
)


# -- task and oracle contracts ----------------------------------------------

@needs_data
@pytest.mark.parametrize("task_id", ALL_TASKS)
def test_task_loads_and_declares_a_window_for_every_output(task_id):
    task = otasks.load_task(task_id)
    assert task.suite == "olympus"
    assert task.task_id == task_id
    assert len(task.y_names) == 1
    assert task.bounds.shape == (2, task.dim)
    # `adapter.BenchmarkVariables` iterates y_names against the spec.
    for width in WIDTHS:
        spec = oranges.load(task_id, width)
        assert set(spec.lower) == set(task.y_names) == set(spec.upper)
        assert spec.upper[task.y_names[0]] > spec.lower[task.y_names[0]]


@needs_data
@pytest.mark.parametrize("task_id", ALL_TASKS)
@pytest.mark.parametrize("width", WIDTHS)
def test_scoring_geometry_present(task_id, width):
    """`metrics.run_metrics` computes both ball criteria unconditionally."""
    cal = oranges.load(task_id, width).calibration
    assert "y_reference_center" in cal and "y_reference_scale" in cal
    assert all(v > 0 for v in cal["y_reference_scale"].values())


@needs_data
@pytest.mark.parametrize("task_id", ALL_TASKS)
def test_observation_Y_is_always_2d(task_id):
    """m == 1 must still give `(n, 1)`; `loop` vstacks and does `.all(axis=1)`."""
    task = otasks.load_task(task_id)
    oracle = make_oracle(task)
    for n in (1, 5):
        X = oracle.initial_design(n, seed=3)
        for obs in (oracle.observe(X, seed=1), oracle.truth(X)):
            assert obs.Y.ndim == 2
            assert obs.Y.shape == (n, 1)
            assert obs.feasible.shape == (n,)
            assert len(obs.failure_type) == n


@needs_data
@pytest.mark.parametrize("task_id", ALL_TASKS)
def test_initial_design_handles_n_equals_one_and_respects_the_simplex(task_id):
    """`loop.py` calls `initial_design(1, ...)` as its random fallback."""
    task = otasks.load_task(task_id)
    oracle = make_oracle(task)
    for n in (1, 2, 37):
        X = oracle.initial_design(n, seed=11)
        assert X.shape == (n, task.dim)
        lo, hi = task.bounds
        assert (X >= lo - 1e-12).all() and (X <= hi + 1e-12).all()
        for group in task.simplex_groups:
            idx = list(group.indices)
            assert np.allclose(X[:, idx].sum(axis=1), group.total, atol=1e-12)
            assert (X[:, idx] >= 0).all()


@needs_data
@pytest.mark.parametrize("task_id", ALL_TASKS)
def test_all_inputs_are_continuous(task_id):
    """`harness/adapter.py` hardcodes the 'conti' branch."""
    task = otasks.load_task(task_id)
    assert task.meta.get("categorical_dims") in (None, (), [])
    assert task.bounds.dtype.kind == "f"


@needs_data
@pytest.mark.parametrize("task_id", EMULATOR_TASKS)
def test_truth_is_deterministic_and_observe_is_reproducibly_noisy(task_id):
    task = otasks.load_task(task_id)
    oracle = make_oracle(task)
    X = oracle.initial_design(24, seed=5)

    # `equal_nan=True` matters: infeasible designs carry NaN, and NaN != NaN would
    # make a deterministic oracle look non-deterministic. On pce10 this never
    # showed (0.1% infeasible, so 24 draws contain none); on colors_bob 75% of
    # uniform draws fall outside the measurement hull and it shows immediately.
    same = lambda a, b: np.array_equal(a, b, equal_nan=True)  # noqa: E731

    assert same(oracle.truth(X).Y, oracle.truth(X).Y)
    # Same seed replays; different seeds differ; neither equals the mean pass.
    assert same(oracle.observe(X, seed=7).Y, oracle.observe(X, seed=7).Y)
    assert not same(oracle.observe(X, seed=7).Y, oracle.observe(X, seed=8).Y)
    assert not same(oracle.observe(X, seed=7).Y, oracle.truth(X).Y)
    # The NaN pattern itself must be identical -- feasibility is deterministic
    # even though the response is not.
    assert np.array_equal(
        np.isnan(oracle.observe(X, seed=7).Y), np.isnan(oracle.truth(X).Y)
    )


@needs_data
def test_branin_is_noiseless_by_design():
    """Stated in `oracle.py`: observe == truth on the analytical surface."""
    task = otasks.load_task("branin")
    oracle = make_oracle(task)
    X = oracle.initial_design(16, seed=2)
    assert np.array_equal(oracle.observe(X, seed=1).Y, oracle.truth(X).Y, equal_nan=True)
    assert np.array_equal(
        oracle.observe(X, seed=1).Y, oracle.observe(X, seed=999).Y, equal_nan=True
    )


def test_branin_matches_its_three_documented_minima():
    """Olympus's own `minima` property asserts the three values agree."""
    values = SurfaceOracle.branin(SurfaceOracle.MINIMA).ravel()
    assert values == pytest.approx(0.397887, abs=1e-5)
    assert values.max() - values.min() < 1e-9
    # And that value is the global minimum over the design box.
    grid = np.random.default_rng(0).random((200_000, 2))
    assert SurfaceOracle.branin(grid).min() > values[0] - 1e-9


@needs_data
@pytest.mark.parametrize("task_id", EMULATOR_TASKS)
def test_emulator_tasks_have_interior_training_support(task_id):
    """The check a hull test cannot make; see `data.interior_support`.

    `oer_plate_3496` was selected as a task and then rejected on exactly this:
    its measurements never use more than 4 of 6 components, so a uniform
    Dirichlet design -- which always uses all of them -- lands entirely outside
    the emulator's training stratum, while the hull reports 100% feasible.
    """
    dataset = otasks.SELECTED_TASKS[task_id]["dataset"]
    support = odata.interior_support(dataset)
    assert support >= odata.SUPPORT_FLOOR, (
        f"{task_id} ({dataset}): only {support:.1%} of measurements use every "
        f"component, below the {odata.SUPPORT_FLOOR:.0%} floor -- a uniform design "
        "would query a stratum the emulator never saw"
    )


@needs_data
def test_the_rejected_oer_plate_datasets_would_fail_that_check():
    """Pins the counter-example, so the guard is known to have teeth."""
    for dataset in ("oer_plate_3496", "oer_plate_3851"):
        assert odata.interior_support(dataset) == 0.0


@needs_data
@pytest.mark.parametrize("task_id", EMULATOR_TASKS)
def test_measured_designs_are_inside_their_own_hull(task_id):
    """An exact predicate reported 22 of pce10's own points as outside.

    Faces are not an edge case -- a blend omitting an ingredient sits on one, and
    the acquisition optimizer pushes toward exactly those boundaries.
    """
    dataset = otasks.SELECTED_TASKS[task_id]["dataset"]
    X, _ = odata.load_measurements(dataset)
    assert odata.hull_membership(dataset, X).all()


@needs_data
def test_infeasible_designs_carry_nan_and_a_failure_type():
    """Outside the hull the emulator has no support, so no number is reported."""
    task = otasks.load_task("pce10")
    oracle = make_oracle(task)
    # A pure vertex mixture that pce10's hull excludes, if any exists; otherwise
    # this task simply has no infeasible region and the assertion is vacuous.
    rng = np.random.default_rng(0)
    X = rng.dirichlet(np.ones(task.dim), size=20_000)
    obs = oracle.truth(X)
    outside = ~obs.feasible
    if outside.any():
        assert np.isnan(obs.Y[outside]).all()
        assert all(obs.failure_type[i] == "OUT_OF_HULL" for i in np.flatnonzero(outside))
    assert np.isfinite(obs.Y[obs.feasible]).all()


# -- the calibration is analytic at m == 1 ----------------------------------

@needs_data
@pytest.mark.parametrize("task_id", ALL_TASKS)
def test_window_alpha_matches_the_closed_form(task_id):
    """valid fraction = P(y <= tau) * alpha, with P(y <= tau) = 0.25 by design."""
    frozen = otasks.load_native_targets()["tasks"][task_id]
    assert frozen["fraction_satisfying_native"] == pytest.approx(otasks.NATIVE_SHARE, abs=5e-3)
    for width, target in calibration.DIFFICULTY.items():
        cal = oranges.load(task_id, width).calibration
        assert cal["alpha"] == pytest.approx(target / otasks.NATIVE_SHARE, abs=0.02)
        assert cal["achieved_fraction_of_feasible"] == pytest.approx(target, abs=5e-3)


@needs_data
@pytest.mark.parametrize("task_id", ALL_TASKS)
def test_every_width_is_strictly_two_sided(task_id):
    """The whole point of q25: no width may collapse to the one-sided target.

    At tau = q10 the widest condition would force alpha = 1 and put the lower
    bound on the sample minimum, silently deleting two-sidedness.
    """
    frozen = otasks.load_native_targets()["tasks"][task_id]
    y_name = otasks.load_task(task_id).y_names[0]
    for width in WIDTHS:
        spec = oranges.load(task_id, width)
        assert spec.calibration["alpha"] < 1.0
        assert spec.lower[y_name] > frozen["response_min"]
        assert spec.upper[y_name] == pytest.approx(frozen["tau"])


# -- the acquisition identities ---------------------------------------------

def _toy_gp(seed: int = 0, d: int = 3):
    from botorch.models import SingleTaskGP

    torch.manual_seed(seed)
    X = torch.rand(14, d, dtype=torch.double)
    raw = X.sum(-1, keepdim=True) * 2.0 + 1.0
    mean, std = raw.mean(), raw.std()
    return SingleTaskGP(X, (raw - mean) / std), X, (raw - mean) / std, mean, std


def test_box_mapped_tb_is_exactly_the_range_probability():
    """TB's chi-square series and our normal-CDF form must agree.

    Two independent implementations of `P(L <= f(x) <= U)`: a differentiable
    Poisson-mixture noncentral chi-square (df=1) against a difference of normal
    CDFs. Agreement to machine precision checks both.
    """
    from eval.methods.range_aware_tb import make_tb_acqf_factory

    gp, X, Y, mean, std = _toy_gp()
    lower_raw, upper_raw = 3.0, 4.5
    target_info = {"y": {
        "obj": "range", "lb": lower_raw, "ub": upper_raw, "weight": 1.0,
        "ball_center": 0.5 * (lower_raw + upper_raw),
        "ref_center": float(mean), "ref_scale": float(std),
    }}
    acqf = make_tb_acqf_factory("box")(
        gp, X, Y, ["y"], torch.ones(1, dtype=torch.double), target_info,
        torch.tensor([mean]), torch.tensor([std]),
    )

    Xt = torch.rand(9, 1, X.shape[-1], dtype=torch.double)
    with torch.no_grad():
        tb = acqf(Xt)
        post = gp.posterior(Xt)
        m = post.mean[..., 0]
        s = post.variance[..., 0].clamp_min(1e-12).sqrt()
        normal = torch.distributions.Normal(torch.zeros_like(m), torch.ones_like(m))
        prob = (normal.cdf(((upper_raw - mean) / std - m) / s)
                - normal.cdf(((lower_raw - mean) / std - m) / s)).squeeze(-1)

    assert torch.allclose(tb, prob, atol=1e-12)


def test_box_mapping_refuses_multiple_outputs():
    """A ball equals a box only in 1-D; anywhere else this must not degenerate."""
    from eval.methods.range_aware_tb import box_to_ball

    info = {n: {"lb": 0.0, "ub": 1.0, "ball_center": 0.5, "ref_center": 0.0,
                "ref_scale": 1.0} for n in ("y1", "y2")}
    with pytest.raises(ValueError, match="single output"):
        box_to_ball(info, ["y1", "y2"], "box")


def test_c1_single_output_acquisition_is_the_probability_behind_a_hinge():
    """`qEI(CDFRangeObjective)` collapses to `max(0, P - best_f)`.

    The objective is a deterministic function of x, so the expectation over
    posterior samples has nothing to average and qEI degenerates to its own
    integrand. That hinge -- everything below the incumbent tied at exactly zero --
    is the mechanism Olympus is run to measure, so it is pinned here.

    This code path is exercised by no other suite: MatFormBench is m = 3
    throughout and takes the qEHVI branch.
    """
    from botorch.acquisition.monte_carlo import qExpectedImprovement
    from botorch.sampling.normal import SobolQMCNormalSampler

    from utils.probability_objective import CDFRangeObjective

    gp, X, Y, mean, std = _toy_gp(seed=1)
    lower_s, upper_s = (3.0 - float(mean)) / float(std), (4.5 - float(mean)) / float(std)
    best_f = 0.35

    acqf = qExpectedImprovement(
        model=gp, best_f=best_f,
        objective=CDFRangeObjective(model=gp, output_index=0, lower=lower_s, upper=upper_s),
        sampler=SobolQMCNormalSampler(torch.Size([256])),
    )
    Xt = torch.rand(24, 1, X.shape[-1], dtype=torch.double)
    with torch.no_grad():
        value = acqf(Xt)
        post = gp.posterior(Xt)
        m = post.mean[..., 0]
        s = post.variance[..., 0].clamp_min(1e-12).sqrt()
        normal = torch.distributions.Normal(torch.zeros_like(m), torch.ones_like(m))
        prob = (normal.cdf((upper_s - m) / s) - normal.cdf((lower_s - m) / s)).squeeze(-1)

    assert torch.allclose(value, (prob - best_f).clamp_min(0.0), atol=1e-7)
    # The plateau is real, not a rounding artifact: below the incumbent it is
    # identically zero, so the acquisition carries no gradient there.
    assert (value == 0).any()


# -- suite isolation --------------------------------------------------------

@needs_data
def test_running_olympus_pulls_in_neither_tensorflow_nor_matformbench():
    """The isolation is structural; see `benchmarks/registry.py`.

    Olympus reads numpy arrays extracted from a TF checkpoint precisely so that no
    worker imports TensorFlow, which would contend for threads with 22
    single-thread processes while `wall_clock_per_iter` is being reported. The
    converse matters too: an Olympus cell must not load MatFormBench's compiled
    CPython-3.10 extension.

    Run in a subprocess because the claim is about a *worker*, not about pytest.
    In-process this would only measure test ordering -- other tests in this suite
    import MatFormBench, so `oracle_0` is already resident by the time it runs.
    """
    import subprocess

    program = textwrap.dedent(
        """
        import sys
        from eval.benchmarks.registry import resolve

        suite = resolve("olympus")
        task = suite.load_task("pce10")
        oracle = suite.make_oracle(task)
        oracle.observe(oracle.initial_design(4, seed=0), seed=0)
        oracle.truth(oracle.initial_design(4, seed=0))
        suite.load_range_spec("pce10", "medium")

        leaked = sorted(
            name for name in sys.modules
            if name == "tensorflow"
            or name.startswith("tensorflow.")
            or "oracle_0" in name
            or name.startswith("synthetic_data")
        )
        print("LEAKED:" + ",".join(leaked))
        """
    )
    env = {**os.environ, "PYTHONPATH": f"src{os.pathsep}."}
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True, text=True, cwd=REPO_ROOT, env=env, timeout=600,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    leaked = [
        line[len("LEAKED:"):] for line in result.stdout.splitlines()
        if line.startswith("LEAKED:")
    ]
    assert leaked == [""], f"an Olympus cell imported: {leaked}"
