#!/usr/bin/env bash
# Run a domain's live agent implement N times and collect the held-out results.
# On ONE pod the harness uses the whole GPU per job, so runs go sequentially here.
# For real parallelism, launch this on N separate single-GPU pods (one run each) — see the runbook.
#
# usage: scripts/cloud/fanout.sh <domain> <n> <bar>
#   domain ∈ occ | bev | occ-scaffold | bev-scaffold
# example: scripts/cloud/fanout.sh occ-scaffold 3 0.051
set -euo pipefail
DOMAIN="${1:?domain: occ|bev|occ-scaffold|bev-scaffold}"; N="${2:?n runs}"; BAR="${3:?bar}"
export PYTHONPATH="${PYTHONPATH:-.:../blueberry_ver2}"

case "$DOMAIN" in
  occ)          MOD=vo_lab.run_occ_implement ;;
  bev)          MOD=vo_lab.run_bev_implement ;;
  occ-scaffold) MOD=vo_lab.run_occ_scaffold_implement ;;
  bev-scaffold) MOD=vo_lab.run_bev_scaffold_implement ;;
  *) echo "unknown domain $DOMAIN"; exit 2 ;;
esac

echo "fanout: $DOMAIN x$N at bar $BAR  ($MOD)"
for r in $(seq 1 "$N"); do
  ROOT="./_${DOMAIN}_fanout_run${r}"; rm -rf "$ROOT"
  echo "--- run $r/$N -> $ROOT ---"
  python3 -c "import vo_lab; from ${MOD} import main; main(${BAR}, root='${ROOT}')" 2>&1 \
    | grep -aE "RESULT:|measured|miou" | tail -3 || echo "  run $r: see log"
done
echo "=== fanout done: $DOMAIN x$N ==="
