#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

# find process with name passed in (excluding this process)
for PID in $(pgrep -f $1); do
  if [ "$PID" != "$$" ]; then
    ps -p $PID -o args
    TRACE_PID=$PID
    break
  fi
done

if [ -z "$TRACE_PID" ]; then
  echo "could not find PID for $1"
  exit 1
fi

sudo env PATH=$PATH py-spy record -d 5 -o /tmp/perf$TRACE_PID.svg -p $TRACE_PID &&
google-chrome /tmp/perf$TRACE_PID.svg
