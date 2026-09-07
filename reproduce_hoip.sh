#!/usr/bin/env bash
#
# Reproduce the HOIP experiments end to end. No third-party environment is
# involved at any step: HOIP is a lookup over two CSVs and adds no dependency.
#
#   ./eval/scripts/reproduce_hoip.sh status   # what is done, what is left
#   ./eval/scripts/reproduce_hoip.sh setup    # env + fetch data (once)
#   ./eval/scripts/reproduce_hoip.sh verify   # audit + C1/C2/C3 toggle check
#   ./eval/scripts/reproduce_hoip.sh sweep    # main catalogue sweep (M0-M9)
#   ./eval/scripts/reproduce_hoip.sh report   # tables from whatever is complete
#   ./eval/scripts/reproduce_hoip.sh all      # verify -> sweep -> report
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
# 2. Do not regenerate the catalogues or target windows.
#    `eval/specs/hoip/*.json` is committed on purpose: every method and seed
#    must optimize byte-identical targets over a byte-identical feasibility
#    ladder. `freeze` exists below only to *verify* the committed files
#    reproduce (`--check`), not to overwrite them casually.
# ---------------------------------------------------------------------------
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# This checkout ships core/ as compiled .so only -- no src/ present, and none
# needed. PY points at the sibling .venv-eval this proof-of-concept reuses;
# a real public checkout creates its own (see eval/README.md's Environment section).
PY="${PY:-../.venv-eval/bin/python}"
export PYTHONPATH="core:."
WORKERS="${WORKERS:-22}"
RESULTS="eval/results/hoip"
CFG_MAIN="eval/configs/hoip_catalogue.yaml"

log() { printf '\n[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

require_env() {
  [ -x "$PY" ] || die "$PY not found — run '$0 setup' first."
  [ -f eval/data/hoip/df_results.csv ] || die "HOIP data missing — run '$0 setup' first."
}

SWEEP_PAT='run_swee[p] --config eval/configs/hoip_catalogue'

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
  [ -x "$PY" ] || die "$PY not found — HOIP shares .venv-eval with MatFormBench; \
run './eval/scripts/reproduce_matformbench.sh setup' first."

  log "fetching the two CSVs and descriptors"
  # The shipped pickles are opened once here and re-emitted as JSON, never
  # unpickled at sweep time.
  $PY -m eval.scripts.fetch_hoip_data

  log "setup complete — next: $0 verify"
}

cmd_verify() {
  require_env

  log "checking the catalogue against its own data before building on it"
  $PY -m eval.scripts.audit_hoip_catalogue

  log "committed feasibility ladder and target windows reproduce"
  $PY -m eval.scripts.freeze_hoip_catalogues --check
  $PY -m eval.scripts.freeze_hoip_specs --check

  log "exact valid-region geometry (enumerated, not sampled)"
  $PY -m eval.scripts.valid_region_geometry --suite hoip

  log "C1/C2/C3 toggles reach the optimizer (expect qEHVI / CDFRangeMultiOutputObjective, nonlinear=4)"
  $PY -m eval.scripts.verify_variants --suite hoip --task dense

  log "COMBOO port (the two-sided auxiliary-UCB baseline registered here)"
  $PY -m eval.scripts.verify_comboo_port

  log "unit tests"
  $PY -m pytest eval/tests -q || die "tests failed"
}

cmd_freeze() {
  require_env
  echo
  echo "This OVERWRITES eval/specs/hoip/*.json, which are committed."
  echo "Only do this if you intend to change the feasibility ladder or the"
  echo "calibration rule -- regenerating them invalidates comparison against"
  echo "results produced with the old catalogues/windows."
  read -r -p "Type 'overwrite' to proceed: " reply
  [ "$reply" = "overwrite" ] || { echo "aborted"; return 0; }
  $PY -m eval.scripts.freeze_hoip_catalogues
  $PY -m eval.scripts.freeze_hoip_specs
}

cmd_sweep() {
  require_env; warn_if_busy
  # dense / full / restricted catalogues x 3 widths x methods x seeds, including
  # COMBOO (M8) and Anubis (M9) at +60 cells over the base matrix.
  log "HOIP catalogue sweep"
  $PY -m eval.scripts.run_sweep --config "$CFG_MAIN" --workers "$WORKERS"
}

cmd_report() {
  require_env
  # Figures below carry the method's display name baked into pixels, same as
  # the .tex tables emit_tables produces — both must render under the
  # double-blind paper name, not the internal one, or a regenerated figure
  # silently reverts to the internal name.
  : "${PAPER_METHOD_NAME:?set PAPER_METHOD_NAME before emitting paper output}"
  log "tables from completed cells"
  $PY -m eval.scripts.report --suite hoip --protocol catalogue

  log "acquisition ranking AUC diagnostic"
  $PY -m eval.scripts.acquisition_auc --suite hoip --seeds 0 1 2 3 4

  log "acquisition plateau diagnostic"
  $PY -m eval.scripts.acquisition_plateau --suite hoip --task dense

  log "discovery curves, per catalog"
  $PY -m eval.scripts.plot_discovery_curves_grid --suite hoip --protocol catalogue \
      --tasks dense full restricted --width native \
      --out paper/figures/curves_hoip.png

  log "paper tables (.tex)"
  $PY -m eval.scripts.emit_tables --out paper/sections/tables
}

cmd_status() {
  [ -x "$PY" ] || { echo "environment not set up — run './eval/scripts/reproduce_matformbench.sh setup'"; return 0; }
  echo
  echo "  environment      $([ -x "$PY" ] && "$PY" --version 2>&1 || echo missing)"
  echo "  HOIP data        $([ -f eval/data/hoip/df_results.csv ] && echo present || echo MISSING)"
  echo "  frozen specs     $(ls eval/specs/hoip/*.json 2>/dev/null | wc -l)"
  echo
  echo "  catalogue sweep  $(find $RESULTS/catalogue -name run.json 2>/dev/null | wc -l) cells complete"
  echo "  errors           $(find $RESULTS -name error.json 2>/dev/null | wc -l)"
  echo
  echo "  per method:"
  find $RESULTS/catalogue -name run.json 2>/dev/null \
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
  cmd_verify
  cmd_sweep
  cmd_report
}

case "${1:-status}" in
  setup)   cmd_setup ;;
  verify)  cmd_verify ;;
  freeze)  cmd_freeze ;;
  sweep)   cmd_sweep ;;
  report)  cmd_report ;;
  status)  cmd_status ;;
  all)     cmd_all ;;
  *) sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//' ; exit 1 ;;
esac
