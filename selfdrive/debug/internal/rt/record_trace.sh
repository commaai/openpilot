#!/usr/bin/bash

cd /d/tracing

# setup tracer
echo "function" > current_tracer

echo "start tracing"
echo 1 > tracing_on

# do stuff
sleep 2
#/data/openpilot/scripts/restart_modem.sh
#sleep 3
#/data/openpilot/scripts/restart_modem.sh
sleep 5

# disable tracing
echo "done tracing"
echo 0 > tracing_on

# copy
echo "copy traces"
cp trace /tmp/trace.txt
cp per_cpu/cpu3/trace /tmp/trace_cpu3.txt
