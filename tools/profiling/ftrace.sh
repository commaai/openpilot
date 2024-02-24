#!/usr/bin/bash
set -e

cd /sys/kernel/tracing

echo 1 > tracing_on
echo boot > trace_clock
echo 1000 > buffer_size_kb

# /sys/kernel/tracing/available_events
echo 1 > events/irq/enable
echo 1 > events/sched/enable
echo 1 > events/kgsl/enable
echo 1 > events/camera/enable
echo 1 > events/workqueue/enable

echo > trace
sleep 5
echo 0 > tracing_on

cp trace /tmp/trace
chown comma: /tmp/trace
echo /tmp/trace
