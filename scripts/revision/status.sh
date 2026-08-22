#!/usr/bin/env bash
# One glance at what the revision experiments are doing on the run machine.
#
#   bash scripts/revision/status.sh
#
# Prints the tmux queue, the progress of every revision run, the GPU state and
# the tail of each experiment log. Read only: it starts and stops nothing.

set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

hr () { printf '%s\n' "------------------------------------------------------------"; }

hr
echo "TMUX SESSIONS"
tmux ls 2>/dev/null || echo "  no tmux server running"

hr
echo "QUEUE (last lines of the rev session)"
tmux capture-pane -p -t rev 2>/dev/null | grep -vE '^\s*$' | tail -6 \
    || echo "  session 'rev' not found"

hr
echo "REVISION RUNS"
printf '  %-46s %-9s %-8s %s\n' "RUN" "EPOCH" "ELAPSED" "LAST val_map"
found=0
for d in $(ls -dt runs/2026*_e[0-9]*/ 2>/dev/null); do
    found=1
    name=$(basename "$d")
    total=$(grep -A20 '^  train:' "$d/fingerprint.yaml" 2>/dev/null \
            | grep -m1 'epochs:' | tr -dc '0-9')
    done_ep=$(wc -l < "$d/log.jsonl" 2>/dev/null || echo 0)
    stamp=$(echo "$name" | sed -E 's/^([0-9]{4})([0-9]{2})([0-9]{2})_([0-9]{2})([0-9]{2})([0-9]{2}).*/\1-\2-\3 \4:\5:\6/')
    start=$(date -d "$stamp" +%s 2>/dev/null)
    if [ -n "${start:-}" ]; then
        last=$(stat -c %Y "$d/log.jsonl" 2>/dev/null || date +%s)
        mins=$(( (last - start) / 60 ))
    else
        mins="?"
    fi
    vmap=$(grep -o '"val_map": [0-9.]*' "$d/log.jsonl" 2>/dev/null | tail -1 | cut -d' ' -f2)
    printf '  %-46s %-9s %-8s %s\n' "$name" "${done_ep}/${total:-?}" "${mins}m" "${vmap:-.}"
done
[ "$found" = 0 ] && echo "  (no revision runs yet)"

hr
echo "GPU"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | while read -r line; do
    pid=${line%%,*}
    echo "  ${line}  <- $(ps -o args= -p "$pid" 2>/dev/null | cut -c1-70)"
done

hr
echo "EXPERIMENT LOGS"
for f in runs/revision_e*.log; do
    [ -e "$f" ] || continue
    printf '  %-26s %s\n' "$(basename "$f")" "$(grep -vE 'HF_TOKEN|^\s*$' "$f" | tail -1 | cut -c1-70)"
done
hr
