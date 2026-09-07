#!/usr/bin/env bash
#
# Reproduce the MatFormBench experiments end to end.
#
#   ./eval/scripts/reproduce_matformbench.sh status     # what is done, what is left
#   ./eval/scripts/reproduce_matformbench.sh setup      # env + third-party clone (once)
#   ./eval/scripts/reproduce_matformbench.sh verify     # tests + C1/C2/C3 toggle check
#   ./eval/scripts/reproduce_matformbench.sh sweep      # main range-adapted matrix
#   ./eval/scripts/reproduce_matformbench.sh comboo     # COMBOO baseline  (read the warning)
#   ./eval/scripts/reproduce_matformbench.sh native     # MatFormBench's own protocol
#   ./eval/scripts/reproduce_matformbench.sh audit      # fairness / leakage checks
#   ./eval/scripts/reproduce_matformbench.sh auc        # acquisition ranking diagnostic
#   ./eval/scripts/reproduce_matformbench.sh product    # §4.2 product-aggregation sweep
#   ./eval/scripts/reproduce_matformbench.sh report     # tables from whatever is complete
#   ./eval/scripts/reproduce_matformbench.sh all        # audit -> sweep -> product -> native -> report
#
# Every sweep is resumable: re-running skips cells that already have a run.json,
# so an interrupted run costs only the cell that was in flight.
#
# ---------------------------------------------------------------------------
# Two things to know before starting
#
# 1. Run one stage at a time. `wall_clock_per_iter` is a reported metric, so a
#    second heavy job on the same machine corrupts the timing numbers for both.
#    `all` therefore runs stages sequentially, never in parallel.
#
# 2. Do not regenerate the target windows. `eval/specs/matformbench/*.json` is
#    committed on purpose: every method and seed must optimize byte-identical
#    targets, and regenerating them on another machine would silently
#    invalidate cross-run comparisons. `calibrate` exists below only to *verify*
#    the committed files reproduce.
# ---------------------------------------------------------------------------
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# This checkout ships core/ as compiled .so only -- no src/ present, and none
# needed. PY points at the sibling .venv-eval this proof-of-concept reuses (or,
# inside the Docker image, at /opt/venv-eval); a bare host checkout creates its
# own (see eval/README.md's Environment section).
PY="${PY:-../.venv-eval/bin/python}"
export PYTHONPATH="core:."
WORKERS="${WORKERS:-22}"
RESULTS="eval/results/matformbench"
CFG_MAIN="eval/configs/matformbench_range_adapted.yaml"
CFG_COMBOO="eval/configs/matformbench_comboo.yaml"
CFG_PRODUCT="eval/configs/matformbench_product.yaml"

log() { printf '\n[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

require_env() {
  [ -x "$PY" ] || die "$PY not found — run '$0 setup' first."
  [ -d eval/external/MatFormBench ] || die "MatFormBench clone missing — run '$0 setup' first."
}

# Bracketed first character so the pattern cannot match the shell that is asking.
# `pgrep -f` sees full command lines, and any wrapper quoting the plain string --
# a status check, a `bash -c`, an editor -- counts as a hit. That already produced
# one false "a sweep is running" reading.
SWEEP_PAT='run_swee[p] --config'

warn_if_busy() {
  if pgrep -f "$SWEEP_PAT" >/dev/null 2>&1; then
    echo
    echo "WARNING: a sweep is already running. Timing metrics from a shared machine"
    echo "         are not usable. Ctrl-C now, or continue and discard sec_per_iter."
    echo
    sleep 5
  fi
}

# ---------------------------------------------------------------------------

cmd_setup() {
  log "creating the Python 3.10 environment"
  # MatFormBench ships compiled modules for CPython 3.10 while the production
  # project pins 3.11, so experiments get their own interpreter. `.venv/` is
  # left untouched.
  uv venv --python 3.10 .venv-eval

  log "cloning MatFormBench (third-party, never modified)"
  mkdir -p eval/external
  [ -d eval/external/MatFormBench ] || \
    git clone --depth 1 https://github.com/DeepVerse/MatFormBench.git eval/external/MatFormBench

  log "installing dependencies"
  # Order matters: MatFormBench's unpinned requirements pull numpy 2.x, and the
  # second command pins it back. The compiled oracle works under both, but the
  # production GP stack is pinned against 1.26.
  VIRTUAL_ENV=.venv-eval uv pip install -r eval/external/MatFormBench/requirements.txt
  VIRTUAL_ENV=.venv-eval uv pip install \
      torch==2.6.0 botorch==0.15.0 gpytorch==1.14 linear-operator==0.6 \
      drs==2.0.0 scikit-learn==1.6.1 scipy==1.15.2 numpy==1.26.4 pandas==2.2.3 \
      threadpoolctl tqdm pyarrow pytest
  # Undeclared MatFormBench dependency: without it their Ridge+TabPFN scorer
  # silently degrades to Ridge-only behind a warning, changing every native score.
  VIRTUAL_ENV=.venv-eval uv pip install tabpfn

  log "setup complete — next: $0 verify"
}

cmd_verify() {
  require_env
  log "unit tests"
  $PY -m pytest eval/tests -q || die "tests failed"

  log "C1/C2/C3 toggles actually reaching BoTorch"
  # If a toggle were silently ignored every ablation number would be meaningless
  # while still looking plausible, so this is checked rather than trusted.
  $PY -m eval.scripts.verify_variants

  log "committed target windows load"
  $PY - <<'PYEOF'
from eval.benchmarks.matformbench import ranges
from eval.benchmarks.matformbench.tasks import SELECTED_TASKS
n = 0
for t in SELECTED_TASKS:
    for w in ("wide", "medium", "narrow"):
        s = ranges.load(t, w)
        c = s.calibration
        assert "y_reference_center" in c and "y_reference_scale" in c, (t, w)
        n += 1
        print(f"  {t:6s} {w:6s} achieved {c['achieved_fraction_of_feasible']:.2%} "
              f"(target {c['target_fraction_of_feasible']:.0%})")
print(f"{n} frozen specs OK")
PYEOF
}

cmd_calibrate() {
  require_env
  echo
  echo "This OVERWRITES eval/specs/matformbench/*.json, which are committed."
  echo "Only do this if you intend to change the calibration rule — regenerating"
  echo "them invalidates comparison against results produced with the old windows."
  read -r -p "Type 'overwrite' to proceed: " reply
  [ "$reply" = "overwrite" ] || { echo "aborted"; return 0; }
  $PY -m eval.scripts.calibrate_ranges
}

cmd_sweep() {
  require_env; warn_if_busy
  # 5 tasks x 3 widths x 8 methods x 10 seeds = 1200 cells.
  # Measured: ~9.5 h at 22 workers, 0 errors. Per-cell cost varies ~100x across
  # methods -- extremum-seeking qEHVI is 46.7 min while the proposed variants are
  # 3.2-10.7 min, because chasing extrema grows the Pareto front and qEHVI's
  # three-objective box decomposition grows with it.
  log "range-adapted sweep (1200 cells, ~9.5 h at ${WORKERS} workers)"
  $PY -m eval.scripts.run_sweep --config "$CFG_MAIN" --workers "$WORKERS"
}

cmd_comboo() {
  require_env; warn_if_busy
  cat <<'EOF'

  ------------------------------------------------------------------------
  COST WARNING — measured, not estimated

    L1-1, L2-1   0.4 min/cell   (COMBOO declares infeasibility and exits early)
    L3-1         5-7 min/cell
    L4-1         >9.5 h/cell and did not finish   <-- 10D simplex
    L5-4         unmeasured, 15D simplex, likely worse

  COMBOO runs a full 128-restart auxiliary feasibility probe *in addition to*
  the constrained solve, every iteration. On the simplex tasks that combines
  with an equality constraint, 6 component floors and 6 nonlinear constraints.

  The 5D tasks (L1-1, L2-1, L3-1) complete in minutes. The simplex tasks
  (L4-1, L5-4) did not complete in 9.5 h and were stopped. If you only want the
  tractable part, restrict the config's `tasks:` list to the 5D tasks.
  ------------------------------------------------------------------------

EOF
  read -r -p "Proceed with the full 150-cell config? [y/N] " reply
  [ "$reply" = "y" ] || { echo "skipped"; return 0; }
  log "COMBOO sweep"
  $PY -m eval.scripts.run_sweep --config "$CFG_COMBOO" --workers "$WORKERS"
}

cmd_native() {
  require_env; warn_if_busy
  # MatFormBench's own batch protocol (recommend / topk / dss / stability) so our
  # numbers sit alongside the baselines it ships. ~2.8 min per task-seed.
  #
  # The published runner cannot execute as shipped: team_test_v2 passes a
  # DataFrame to compiled metrics that declare `data: str`. Their own
  # algo_card_GA_GPR.yaml fails on the identical line. run_native.py wraps the
  # two affected functions to spill a frame to a temp CSV; the metric
  # computation itself is untouched. See eval/README.md.
  log "native protocol (5 tasks x 3 seeds x 2 methods, ~1.5 h)"
  for m in M5_full M3_range_prob; do
    log "  method $m"
    $PY -m eval.scripts.run_native --method "$m" --seeds 0 1 2
  done
}

cmd_audit() {
  require_env
  # Cheap and decisive, so it runs before anything expensive: if a method can
  # reach the truth oracle or gets a different initial design, every number
  # produced afterwards is void. One task per geometry -- box, simplex, sparse --
  # because the constraint plumbing differs between them and a pass on the 5D box
  # says nothing about the 15D simplex.
  for t in L1-1 L4-1 L5-4; do
    log "leakage / fairness audit on $t"
    $PY -m eval.scripts.audit_leakage --task "$t" || die "audit FAILED on $t"
  done
  log "all audits passed"
}

cmd_auc() {
  require_env
  # One GP fit per (task, width, seed) rather than one per iteration, so this
  # answers "does this acquisition rank passing candidates above failing ones"
  # in minutes instead of hours. It cannot answer how a method explores over 50
  # iterations -- that is what the sweeps are for.
  log "acquisition ranking AUC (15 settings x 5 seeds)"
  $PY -m eval.scripts.acquisition_auc --seeds 0 1 2 3 4
}

cmd_product() {
  require_env; warn_if_busy
  # 5 tasks x 3 widths x 4 variants x 5 seeds = 300 cells.
  #
  # Only the new variants run. M3_range_prob and M5_full already exist at these
  # seeds from the main sweep and are read from there; the cells land under the
  # same range_adapted tree so `report` picks them all up together.
  log "§4.2 product-aggregation sweep (300 cells)"
  $PY -m eval.scripts.run_sweep --config "$CFG_PRODUCT" --workers "$WORKERS"
}

cmd_report() {
  require_env
  log "tables from completed cells"
  $PY - <<'PYEOF'
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
pd.set_option("display.width", 250)
from eval.harness.aggregate import load_runs, core_table, average_rank, discovery_curves, render

df = load_runs()
# M0_random has extra seeds from an earlier 20-seed launch; level the comparison.
df = df[df.seed < 10]
print(f"\n{len(df)} completed cells, {df.method.nunique()} methods\n")

print("=" * 110)
print("CORE TABLE - mean over tasks x widths x seeds")
print("=" * 110)
t = core_table(df)
print(render(t[["method", "hit_rate", "found_rate", "first_hit_median",
                "violation_lower", "violation_upper", "invalid_rate", "sec_per_iter"]]))

print("\n" + "=" * 110)
print("AVERAGE RANK over (task, width) cells — lower is better")
print("=" * 110)
print(render(average_rank(df)))

print("\n" + "=" * 110)
print("HIT RATE BY WINDOW WIDTH")
print("=" * 110)
from eval.harness.aggregate import label, DISPLAY_ORDER
piv = df.pivot_table(index="method", columns="width", values="hit_rate", aggfunc="mean")
piv = piv.reindex([m for m in DISPLAY_ORDER if m in piv.index])
piv = piv[[c for c in ("wide", "medium", "narrow") if c in piv.columns]]
piv.index = [label(m) for m in piv.index]
print(render(piv.reset_index().rename(columns={"index": "method"})))

print("\n" + "=" * 110)
print("CUMULATIVE VALID DESIGNS FOUND (of 50 proposals)")
print("=" * 110)
c = discovery_curves()
agg = c.groupby(["method", "iteration"])["mean"].mean().reset_index()
cols = [1, 10, 20, 30, 50]
print(f"{'method':40s}" + "".join(f"{'it'+str(i):>8s}" for i in cols))
for m in DISPLAY_ORDER:
    s = agg[agg.method == m].set_index("iteration")["mean"]
    if s.empty:
        continue
    print(f"{label(m):40s}" + "".join(f"{s.get(i, float('nan')):8.2f}" for i in cols))

# MatFormBench's own batch protocol, scored by its compiled metrics rather than
# ours. Printed separately and never merged into the tables above: it has no
# feedback loop, so a row here is not comparable to a row from the sequential
# sweep even for the same method.
try:
    from eval.harness.aggregate import load_native, native_table
    nat = load_native()
    print("\n" + "=" * 110)
    print("NATIVE PROTOCOL - scored by MatFormBench, not by us")
    print("=" * 110)
    print(render(native_table(nat)))
    print()
    print(render(native_table(nat, by=["task_id", "method"])))
except FileNotFoundError:
    print("\nno native-protocol results yet (run '$0 native')")
PYEOF
}

cmd_status() {
  [ -x "$PY" ] || { echo "environment not set up — run '$0 setup'"; return 0; }
  echo
  echo "  environment      $([ -x "$PY" ] && "$PY" --version 2>&1 || echo missing)"
  echo "  MatFormBench     $([ -d eval/external/MatFormBench ] && echo present || echo MISSING)"
  echo "  frozen specs     $(ls eval/specs/matformbench/*.json 2>/dev/null | wc -l)/15"
  echo
  echo "  range-adapted    $(find $RESULTS/range_adapted -name run.json 2>/dev/null | wc -l) cells complete"
  echo "  errors           $(find $RESULTS -name error.json 2>/dev/null | wc -l)"
  echo "  native results   $(ls $RESULTS/native/*.json 2>/dev/null | wc -l) files"
  echo "  product variants $(find $RESULTS/range_adapted -name run.json -path '*M[35][pq]*' 2>/dev/null | wc -l)/300 cells"
  echo
  echo "  per method:"
  find $RESULTS/range_adapted -name run.json 2>/dev/null \
    | awk -F/ '{print $(NF-2)}' | sort | uniq -c | sort -rn | sed 's/^/    /'
  echo
  if pgrep -f "$SWEEP_PAT" >/dev/null 2>&1; then
    echo "  RUNNING: $(pgrep -cf "$SWEEP_PAT") sweep processes"
  else
    echo "  no sweep running"
  fi
  echo
}

cmd_all() {
  # Audit first: it is minutes, and a failure invalidates everything downstream.
  # COMBOO is left out of `all` on purpose -- it prompts, and its simplex cells
  # did not finish in 9.5 h. Run it deliberately or not at all.
  cmd_audit
  cmd_sweep
  cmd_product
  cmd_native
  cmd_report
}

case "${1:-status}" in
  setup)     cmd_setup ;;
  verify)    cmd_verify ;;
  calibrate) cmd_calibrate ;;
  sweep)     cmd_sweep ;;
  comboo)    cmd_comboo ;;
  native)    cmd_native ;;
  audit)     cmd_audit ;;
  auc)       cmd_auc ;;
  product)   cmd_product ;;
  report)    cmd_report ;;
  status)    cmd_status ;;
  all)       cmd_all ;;
  *) sed -n '2,33p' "$0" | sed 's/^# \{0,1\}//' ; exit 1 ;;
esac
