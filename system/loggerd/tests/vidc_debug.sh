#!/usr/bin/env bash
set -e

cd /sys/kernel/debug/tracing
echo "" > trace
echo 1 > tracing_on
echo 1 > /sys/kernel/debug/tracing/events/msm_vidc/enable

echo 0xff > /sys/module/videobuf2_core/parameters/debug
echo 0x7fffffff > /sys/kernel/debug/msm_vidc/debug_level
echo 0xff > /sys/devices/platform/soc/aa00000.qcom,vidc/video4linux/video33/dev_debug

cat /sys/kernel/debug/tracing/trace_pipe
