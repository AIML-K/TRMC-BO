#!/usr/bin/env bash
#
# docker run <image>                                # usage + status, all suites
# docker run <image> <suite> <subcommand> [args...]  # suite: hoip|olympus|matformbench|all
#
#   docker run trmcbo-reproduce hoip verify
#   docker run trmcbo-reproduce hoip sweep
#   docker run trmcbo-reproduce olympus verify
#   docker run trmcbo-reproduce matformbench report
#   docker run trmcbo-reproduce all status
#
# `setup` is deliberately not part of this flow: the image already did it at
# build time (MatFormBench cloned, both venvs installed, Olympus emulator
# weights already committed under eval/data/olympus/). Subcommands go straight
# to verify/sweep/report/etc, matching each reproduce_*.sh's own "next:" hints.
set -euo pipefail
cd /repro

usage() {
  echo "usage: docker run <image> {hoip|olympus|matformbench|all} {status|verify|sweep|report|...}"
  echo "no suite given -- showing status for all three:"
  echo
  ./reproduce_hoip.sh status
  ./reproduce_olympus.sh status
  ./reproduce_matformbench.sh status
}

SUITE="${1:-}"
[ $# -gt 0 ] && shift || true
CMD="${1:-status}"
[ $# -gt 0 ] && shift || true

case "$SUITE" in
  hoip)         exec ./reproduce_hoip.sh "$CMD" "$@" ;;
  olympus)      exec ./reproduce_olympus.sh "$CMD" "$@" ;;
  matformbench) exec ./reproduce_matformbench.sh "$CMD" "$@" ;;
  all)
    ./reproduce_hoip.sh "$CMD" "$@"
    ./reproduce_olympus.sh "$CMD" "$@"
    ./reproduce_matformbench.sh "$CMD" "$@"
    ;;
  ""|status)    usage ;;
  *)            usage; exit 1 ;;
esac
