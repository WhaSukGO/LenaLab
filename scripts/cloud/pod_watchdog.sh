#!/usr/bin/env bash
# Pod staleness + cost watchdog (closes the gap native completion-notifications can't: a silent hang).
# Monitors a remote job log's ADVANCEMENT; terminates the pod via the RunPod API when the job finishes,
# errors, goes stale (no new log bytes for STALE_MIN), or exceeds MAX_MIN -- so a hung job can't bleed
# GPU cost unnoticed. Writes /tmp/pod_watchdog_<pod>.status (live) and .done (on exit).
#
# usage: pod_watchdog.sh <pod_id> <ssh_target> <remote_logpath> [done_regex] [stale_min] [max_min]
#   ssh_target e.g.: "root@1.2.3.4 -p 30781 -i /home/ws/.ssh/id_ed25519"
set -uo pipefail
PID="$1"; SSHT="$2"; LOG="$3"; DONE_RE="${4:-_DONE}"; STALE_MIN="${5:-20}"; MAX_MIN="${6:-120}"
KEY=$(grep '^RUNPOD_API_KEY=' "$(cd "$(dirname "$0")/../.." && pwd)/.env" | cut -d= -f2- | tr -d '\r')
SSH="ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o BatchMode=yes $SSHT"
STAT="/tmp/pod_watchdog_${PID}.status"; rm -f "/tmp/pod_watchdog_${PID}.done"
term(){ curl -s -X DELETE -H "Authorization: Bearer $KEY" "https://rest.runpod.io/v1/pods/$PID" -o /dev/null -w "%{http_code}"; }
GRACE_MIN="${7:-12}"                                       # after a successful finish, keep the pod this
start=$(date +%s); lastsize=-1; lastchange=$start; done_at=0  # long so I can salvage results, then kill
echo "watchdog start pod=$PID done=/$DONE_RE/ stale=${STALE_MIN}m max=${MAX_MIN}m grace=${GRACE_MIN}m" > "$STAT"
while true; do
  now=$(date +%s); el=$(( (now-start)/60 ))
  probe=$($SSH "s=\$(wc -c < $LOG 2>/dev/null||echo 0); d=\$(grep -cE '$DONE_RE' $LOG 2>/dev/null||echo 0); e=\$(grep -cE 'Traceback|Error|CUDA out of memory|RuntimeError' $LOG 2>/dev/null||echo 0); echo \"\$s \$d \$e\"; tail -1 $LOG 2>/dev/null" 2>/dev/null)
  size=$(echo "$probe" | head -1 | awk '{print $1}'); done=$(echo "$probe" | head -1 | awk '{print $2}'); err=$(echo "$probe" | head -1 | awk '{print $3}')
  last=$(echo "$probe" | tail -1); size=${size:-CONN}; done=${done:-0}; err=${err:-0}
  [ "$size" != "$lastsize" ] && { lastsize=$size; lastchange=$now; }
  stale=$(( (now-lastchange)/60 ))
  echo "[${el}m] size=$size stale=${stale}m done=$done err=$err | $last" >> "$STAT"
  # failures: terminate immediately (stop cost). success: signal + grace window, then terminate.
  if [ "$err" != "0" ]; then echo "RESULT=ERROR (${el}m); terminating http=$(term)" >> "$STAT"; break; fi
  if [ "$done" != "0" ]; then
    [ "$done_at" = 0 ] && { done_at=$now; touch "/tmp/pod_watchdog_${PID}.success"; echo "DONE seen (${el}m); ${GRACE_MIN}m grace to salvage" >> "$STAT"; }
    [ $(( (now-done_at)/60 )) -ge "$GRACE_MIN" ] && { echo "RESULT=DONE+grace; terminating http=$(term)" >> "$STAT"; break; }
  fi
  if [ "$stale" -ge "$STALE_MIN" ] && [ "$done_at" = 0 ]; then echo "RESULT=STALE (${stale}m); terminating http=$(term)" >> "$STAT"; break; fi
  if [ "$el" -ge "$MAX_MIN" ]; then echo "RESULT=MAXWALL (${el}m); terminating http=$(term)" >> "$STAT"; break; fi
  sleep 120
done
touch "/tmp/pod_watchdog_${PID}.done"
