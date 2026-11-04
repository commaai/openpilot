#!/usr/bin/env bash
set -e

cd /sys/kernel/debug/tracing
echo "" > trace
echo 1 > tracing_on
#echo Y > /sys/kernel/debug/camera_icp/a5_debug_q
echo 0x1 > /sys/kernel/debug/camera_icp/a5_debug_type
echo 1 > /sys/kernel/debug/tracing/events/camera/enable
echo 0xffffffff > /sys/kernel/debug/camera_icp/a5_debug_lvl
echo 1 > /sys/kernel/debug/tracing/events/camera/cam_icp_fw_dbg/enable

cat /sys/kernel/debug/tracing/trace_pipe
