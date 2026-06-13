#!/bin/bash
# Robust auto-continuing audit pipeline: waits for the current batch, then runs M1/M3 reproductions,
# capturing results to a durable file + auto-committing after each phase (reboot-resilient).
cd /home/ws/devel/whasuk/LenaLab; set -a; . ./.env 2>/dev/null; set +a
R=artifacts/audit_repro_results.txt
cap(){ echo "$@" >> "$R"; }
commit(){ git add -A artifacts/ 2>/dev/null; git commit -q -m "audit auto-commit: $1" 2>/dev/null && echo "[committed] $1"; }

# Phase 0: wait for the in-flight batch (aug variance + contamination + C++ VO)
while pgrep -f "run_aug_variance|contamination_synthetic|run_cpp_vio_phase1" >/dev/null 2>&1; do sleep 90; done
cap "===== AUDIT REPRO RESULTS ($(date 2>/dev/null)) ====="
cap "--- gaussian+aug variance ---"
python3 -c "import json,statistics as s; d=json.load(open('artifacts/fidelity_ladder/aug_variance.json')); a=[r['ate'] for r in d if r['ate']]; print('  aug n=%d mean %.2f std %.2f  (vs rendered 27.35+/-1.49)'%(len(a),s.mean(a),s.stdev(a) if len(a)>1 else 0))" >> "$R" 2>&1
cap "--- contamination (Ep12 ~1.20%) + C++ Phase1 (Ep18a ~2.18%) ---"
grep -aiE "t_err|VERIFIED|1\.[0-9][0-9]|2\.[0-9][0-9]|reference|measured|verdict" /tmp/cpu_run.log 2>/dev/null | tail -10 >> "$R"
commit "aug + contamination + C++ reproduction results"

# Phase 1: M1 BA reproduction (Ep9 ~2.03%)
cap "--- M1 BA reproduction (Ep9, expect ~2.03%) ---"
timeout 2400 python3 scripts/salvage_m1.py 2>&1 | grep -aiE "t_err|2\.0[0-9]|%|verdict|measured|robust" | tail -6 >> "$R" 2>&1
commit "M1 BA reproduction"

# Phase 2: M3 VIO reproduction (Ep15 ~3.83%)
cap "--- M3 VIO reproduction (Ep15, expect ~3.83%) ---"
timeout 2400 python3 scripts/m3_derisk_vio.py 2>&1 | grep -aiE "t_err|3\.8|%|verdict|VIO|measured" | tail -6 >> "$R" 2>&1
commit "M3 VIO reproduction"

cap "===== ALL AUDIT REPRODUCTIONS DONE ====="
commit "all audit reproductions complete"
