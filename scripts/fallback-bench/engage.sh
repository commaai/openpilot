#!/bin/bash
# retry rising edges until device reports enabled, then hold engage
# usage: engage.sh [timeout_s]
KEY=/home/batman/xx/private/key/id_rsa
DEV=comma@192.168.61.224
T=${1:-120}
START=$(date +%s)
while true; do
  L=$(ssh -i $KEY -o ConnectTimeout=5 -o StrictHostKeyChecking=no $DEV 'tail -1 /data/fallback_watch.jsonl' 2>/dev/null)
  EN=$(echo "$L" | grep -o '"enabled": [a-z]*' | awk '{print $2}')
  ST=$(echo "$L" | grep -o '"state": "[a-z]*"' | cut -d'"' -f4)
  echo "$(date +%T) enabled=$EN state=$ST"
  if [ "$EN" = "true" ] && [ "$ST" = "enabled" ]; then echo ENGAGED; exit 0; fi
  if [ $(( $(date +%s) - START )) -gt "$T" ]; then echo TIMEOUT; exit 1; fi
  if [ "$EN" != "true" ]; then
    echo disengage > /tmp/bench_cruise
    sleep 1.2
    echo engage > /tmp/bench_cruise
  fi
  sleep 5
done
