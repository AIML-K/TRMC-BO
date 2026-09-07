#!/usr/bin/env bash
#
# Reproduce the Olympus experiments end to end.
#
#   ./eval/scripts/reproduce_olympus.sh status    # what is done, what is left
#   ./eval/scripts/reproduce_olympus.sh setup     # env + third-party clone + emulator extraction (once)
#   ./eval/scripts/reproduce_olympus.sh verify    # emulator fidelity + C1/C2/C3 toggle check
#   ./eval/scripts/reproduce_olympus.sh branin    # analytical protocol sweep (paper §5.2)
#   ./eval/scripts/reproduce_olympus.sh emulator  # real mixture protocol sweep (paper §5.3)
#   ./eval/scripts/reproduce_olympus.sh factorial # C1/C2/C3 factorial ladder (both protocols)
#   ./eval/scripts/reproduce_olympus.sh audit     # dataset simplex/feasibility audit
#   ./eval/scripts/reproduce_olympus.sh report    # tables + figures from whatever is complete
#   ./eval/scripts/reproduce_olympus.sh all       # audit -> branin -> emulator -> report
#
# Every sweep is resumable: re-running skips cells that already have a run.json,
# so an interrupted run costs only the cell that was in flight.
#
# ---------------------------------------------------------------------------
# Two things to know before starting
#
# 1. Run one stage at a time. `wall_clock_per_iter` is a reported metric, so a
#    second heavy job on the same machine corrupts the timing numbers for both.
#
# 2. Do not regenerate the target windows or the extracted emulator weights.
#    `eval/specs/olympus/*.json` and `eval/data/olympus/*` are produced once by
#    `setup`/`verify` and then used as-is; `eval/specs/olympus/*.json` is
#    committed on purpose so every method and seed optimizes byte-identical
#    targets. `verify` only checks these reproduce; it never overwrites them.
# ---------------------------------------------------------------------------
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PY="${PY:-../.venv-eval/bin/python}"
PY_TF="${PY_TF:-../.venv-olympus/bin/python}"
export PYTHONPATH="core:."
WORKERS="${WORKERS:-22}"
RESULTS="eval/results/olympus"
CFG_BRANIN="eval/configs/olympus_branin.yaml"
CFG_EMULATOR="eval/configs/olympus_emulator.yaml"
CFG_FACT_BRANIN="eval/configs/factorial_olympus.yaml"
CFG_FACT_EMULATOR="eval/configs/factorial_olympus_emulator.yaml"

log() { printf '\n[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

require_env() {
  [ -x "$PY" ] || die "$PY not found — run '$0 setup' first."
  [ -f eval/specs/olympus/native_targets.json ] || \
    die "olympus target windows missing — run '$0 setup' first."
}

SWEEP_PAT='run_swee[p] --config eval/configs/(olympus|factorial_olympus)'

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
  log "cloning Olympus (third-party, never modified)"
  mkdir -p eval/external
  [ -d eval/external/olympus ] || \
    git clone --depth 1 https://github.com/aspuru-guzik-group/olympus.git eval/external/olympus

  log "creating the throwaway TensorFlow environment"
  # Olympus's emulator checkpoints need TF/TFP to load. This venv exists only to
  # lift the weights out once; it is never used at sweep time, so TensorFlow
  # never competes for threads with the 22 single-thread sweep workers.
  uv venv --python 3.10 .venv-olympus
  VIRTUAL_ENV=.venv-olympus uv pip install \
      "tensorflow-cpu==2.15.*" "tensorflow-probability==0.23.0" numpy==1.26.4

  log "extracting emulator weights to numpy"
  "$PY_TF" eval/scripts/extract_olympus_emulator.py \
      --datasets photo_pce10 photo_wf3 oer_plate_3496

  log "setup complete — next: $0 verify"
}

cmd_verify() {
  [ -x "$PY_TF" ] && [ -f eval/data/olympus/emulator_photo_pce10.npz ] || \
    die "extracted emulator weights missing — run '$0 setup' first."
  [ -x "$PY" ] || die "$PY not found — run '$0 setup' first."

  log "gating extracted weights against TFP"
  "$PY_TF" eval/scripts/verify_olympus_emulator.py

  log "unit tests"
  $PY -m pytest eval/tests -q || die "tests failed"

  log "committed target windows load"
  require_env
  $PY -m eval.scripts.calibrate_ranges --suite olympus --check

  log "C1/C2/C3 toggles actually reaching BoTorch (expect qEI, 2 constraints)"
  $PY -m eval.scripts.verify_variants --suite olympus --task pce10

  log "SCBO port (the trust-region baseline registered on the single-output tasks)"
  $PY -m eval.scripts.verify_scbo_port
}

cmd_synthesize() {
  require_env
  echo
  echo "This OVERWRITES eval/specs/olympus/*.json, which are committed."
  echo "Only do this if you intend to change the target/window construction --"
  echo "regenerating them invalidates comparison against results produced with"
  echo "the old windows."
  read -r -p "Type 'overwrite' to proceed: " reply
  [ "$reply" = "overwrite" ] || { echo "aborted"; return 0; }
  $PY -m eval.scripts.synthesize_olympus_targets
  $PY -m eval.scripts.calibrate_ranges --suite olympus
}

cmd_audit() {
  require_env
  log "auditing all Olympus datasets for declared-vs-actual simplex structure"
  # Cheap and decisive: of 43 datasets only 3 (branin excluded, analytical) are
  # usable mixture tasks, and this is what rules the rest out. Run before any
  # sweep so a stale conclusion about a dataset is never trusted silently.
  $PY -m eval.scripts.audit_olympus_datasets

  log "leakage / fairness audit"
  $PY -m eval.scripts.audit_leakage --task branin --suite olympus

  log "what the valid region actually is, before any method runs"
  $PY -m eval.scripts.valid_region_geometry --suite olympus
}

cmd_branin() {
  require_env; warn_if_busy
  log "Olympus analytical protocol (branin, paper §5.2)"
  $PY -m eval.scripts.run_sweep --config "$CFG_BRANIN" --workers "$WORKERS"
}

cmd_emulator() {
  require_env; warn_if_busy
  [ -f eval/data/olympus/emulator_photo_pce10.npz ] || \
    die "extracted emulator weights missing — run '$0 setup' first."
  log "Olympus emulator protocol (pce10 / wf3 / thin_film, paper §5.3)"
  $PY -m eval.scripts.run_sweep --config "$CFG_EMULATOR" --workers "$WORKERS"
}

cmd_factorial() {
  require_env; warn_if_busy
  log "C1/C2/C3 factorial ladder — analytical protocol"
  $PY -m eval.scripts.run_sweep --config "$CFG_FACT_BRANIN" --workers "$WORKERS"
  log "C1/C2/C3 factorial ladder — emulator protocol"
  $PY -m eval.scripts.run_sweep --config "$CFG_FACT_EMULATOR" --workers "$WORKERS"
}

cmd_report() {
  require_env
  # Figures below carry the method's display name baked into pixels, same as
  # the .tex tables emit_tables produces — both must render under the
  # double-blind paper name, not the internal one, or a regenerated figure
  # silently reverts to the internal name.
  : "${PAPER_METHOD_NAME:?set PAPER_METHOD_NAME before emitting paper output}"
  log "tables for the branin protocol"
  $PY -m eval.scripts.report --suite olympus --protocol branin
  log "tables for the emulator protocol"
  $PY -m eval.scripts.report --suite olympus --protocol emulator

  log "regret curves"
  $PY -m eval.scripts.plot_regret --suite olympus --protocol branin \
      --out paper/figures/regret_branin.png
  log "discovery curves"
  $PY -m eval.scripts.plot_discovery_curves --suite olympus --protocol branin \
      --out paper/figures/curves_branin.png

  log "MatFormBench distinct-design scatter, all tasks"
  $PY -m eval.scripts.plot_distinct_designs_grid --suite matformbench \
      --protocol range_adapted --tasks L1-1 L2-1 L3-1 L4-1 L5-4 --width medium \
      --out paper/figures/distinct_all_tasks.png

  log "mixture-task capacity (needed for mixture-task coverage regret)"
  $PY -m eval.scripts.valid_region_geometry --suite olympus \
      --tasks pce10 thin_film wf3 --json eval/specs/capacity.json
  log "discovery curves, per mixture task"
  $PY -m eval.scripts.plot_discovery_curves_grid --suite olympus --protocol emulator \
      --tasks pce10 thin_film wf3 --out paper/figures/curves_olympus_emulator.png
  log "regret curves, per mixture task"
  $PY -m eval.scripts.plot_regret_grid --suite olympus --protocol emulator \
      --tasks pce10 thin_film wf3 --out paper/figures/regret_olympus_emulator.png

  log "paper tables (.tex)"
  $PY -m eval.scripts.emit_tables --out paper/sections/tables
}

cmd_status() {
  [ -x "$PY" ] || { echo "environment not set up — run '$0 setup'"; return 0; }
  echo
  echo "  environment       $([ -x "$PY" ] && "$PY" --version 2>&1 || echo missing)"
  echo "  olympus clone     $([ -d eval/external/olympus ] && echo present || echo MISSING)"
  echo "  emulator weights  $(ls eval/data/olympus/*.npz 2>/dev/null | wc -l) files"
  echo "  frozen specs      $(ls eval/specs/olympus/*.json 2>/dev/null | wc -l)"
  echo
  echo "  branin sweep      $(find $RESULTS/branin -name run.json 2>/dev/null | wc -l) cells complete"
  echo "  emulator sweep    $(find $RESULTS/emulator -name run.json 2>/dev/null | wc -l) cells complete"
  echo "  errors            $(find $RESULTS -name error.json 2>/dev/null | wc -l)"
  echo
  echo "  per method (branin):"
  find $RESULTS/branin -name run.json 2>/dev/null \
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
  cmd_audit
  cmd_branin
  cmd_emulator
  cmd_report
}

case "${1:-status}" in
  setup)      cmd_setup ;;
  verify)     cmd_verify ;;
  synthesize) cmd_synthesize ;;
  branin)     cmd_branin ;;
  emulator)   cmd_emulator ;;
  factorial)  cmd_factorial ;;
  audit)      cmd_audit ;;
  report)     cmd_report ;;
  status)     cmd_status ;;
  all)        cmd_all ;;
  *) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//' ; exit 1 ;;
esac
