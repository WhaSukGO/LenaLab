#!/usr/bin/env bash
# Pod SUPERVISOR = visibility + safety in one. Every ~90s it writes a human-readable snapshot to
# /tmp/lenalab_status.txt (so progress is visible ANY time without re-checking by hand) AND guards the
# run: auto-terminates the pod via the RunPod API on done / error / stale (no progress) / max-wall.
# Runs as a tracked background task, so it also notifies on exit.
#
# usage: pod_supervisor.sh <pod_id> <ssh_target> <result_log> <progress_log> <done_regex> [stale_min] [max_min] [grace_min]
set -uo pipefail
PID="$1"; SSHT="$2"; RLOG="$3"; PLOG="$4"; DONE_RE="$5"; STALE_MIN="${6:-25}"; MAX_MIN="${7:-100}"; GRACE_MIN="${8:-10}"
SALVAGE_CMD="${9:-}"   # optional: a shell command run ONCE when the run first finishes (before the grace/
                       # terminate) -- e.g. an scp of the agent's authored code back. ARTIFACT-SAFE.
KEY=$(grep '^RUNPOD_API_KEY=' "$(cd "$(dirname "$0")/../.." && pwd)/.env" | cut -d= -f2- | tr -d '\r')
SSH="ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o BatchMode=yes $SSHT"
STATUS=/tmp/lenalab_status.txt
term(){ curl -s -X DELETE -H "Authorization: Bearer $KEY" "https://rest.runpod.io/v1/pods/$PID" -o /dev/null -w "%{http_code}"; }
start=$(date +%s); lastp=""; lastchange=$start; done_at=0
while true; do
  now=$(date +%s); el=$(( (now-start)/60 ))
  probe=$($SSH "
    d=\$(grep -cE '$DONE_RE' $RLOG 2>/dev/null||echo 0); e=\$(grep -cE 'Traceback|Error|CUDA out of memory|RuntimeError' $RLOG 2>/dev/null||echo 0)
    g=\$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null|head -1)
    p=\$(tail -1 $PLOG 2>/dev/null)
    r=\$(grep -aE 'RESULT:|measured' $RLOG 2>/dev/null|tail -2|tr '\n' '~')
    echo \"D=\$d|E=\$e|G=\$g|P=\$p|R=\$r\"" 2>/dev/null)
  D=$(sed -n 's/.*D=\([0-9]*\).*/\1/p' <<<"$probe"); E=$(sed -n 's/.*E=\([0-9]*\).*/\1/p' <<<"$probe")
  G=$(sed -n 's/.*|G=\(.*\)|P=.*/\1/p' <<<"$probe"); P=$(sed -n 's/.*|P=\(.*\)|R=.*/\1/p' <<<"$probe"); R=$(sed -n 's/.*|R=\(.*\)/\1/p' <<<"$probe")
  D=${D:-0}; E=${E:-0}
  [ "$P" != "$lastp" ] && { lastp="$P"; lastchange=$now; }
  util=$(grep -oE '^[0-9]+' <<<"$G"); [ -n "$util" ] && [ "$util" -ge 5 ] && lastchange=$now  # GPU busy => not stale
  stale=$(( (now-lastchange)/60 ))
  state="running"; [ "$D" != "0" ] && state="DONE"; [ "$E" != "0" ] && state="ERROR"
  { echo "LenaLab cloud run — pod $PID (\$0.69/hr) — elapsed ${el}m"
    echo "state    : $state"
    echo "progress : ${P:-<no progress line yet>}"
    echo "GPU      : ${G:-?}"
    echo "guard    : stale ${stale}/${STALE_MIN}m | maxwall ${el}/${MAX_MIN}m | grace ${GRACE_MIN}m"
    echo "result   : ${R:-pending}"
    echo "(updated every ~90s by pod_supervisor.sh; cat this file any time)"
  } > "$STATUS"
  if [ "$E" != "0" ]; then echo "OUTCOME=ERROR http=$(term)" >> "$STATUS"; break; fi
  if [ "$D" != "0" ]; then
    [ "$done_at" = 0 ] && { done_at=$now; touch "/tmp/pod_supervisor_${PID}.success"
      [ -n "$SALVAGE_CMD" ] && eval "$SALVAGE_CMD" > "/tmp/pod_supervisor_${PID}.salvage.log" 2>&1; }
    [ $(( (now-done_at)/60 )) -ge "$GRACE_MIN" ] && { echo "OUTCOME=DONE+grace http=$(term)" >> "$STATUS"; break; }
  fi
  if [ "$stale" -ge "$STALE_MIN" ] && [ "$done_at" = 0 ]; then echo "OUTCOME=STALE http=$(term)" >> "$STATUS"; break; fi
  if [ "$el" -ge "$MAX_MIN" ]; then echo "OUTCOME=MAXWALL http=$(term)" >> "$STATUS"; break; fi
  sleep 90
done
touch "/tmp/pod_supervisor_${PID}.done"
