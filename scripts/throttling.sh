#!/data/data/com.termux/files/usr/bin/bash
watch -n1 '
  cat /sys/kernel/debug/clk/pwrcl_clk/measure
  cat /sys/kernel/debug/clk/perfcl_clk/measure
  cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq
  cat /sys/class/kgsl/kgsl-3d0/gpuclk
  echo
  echo -n "CPU0 " ; cat /sys/devices/virtual/thermal/thermal_zone5/temp
  echo -n "CPU1 " ; cat /sys/devices/virtual/thermal/thermal_zone7/temp
  echo -n "CPU2 " ; cat /sys/devices/virtual/thermal/thermal_zone10/temp
  echo -n "CPU3 " ; cat /sys/devices/virtual/thermal/thermal_zone12/temp
  echo -n "MEM  " ; cat /sys/devices/virtual/thermal/thermal_zone2/temp
  echo -n "GPU  " ; cat /sys/devices/virtual/thermal/thermal_zone16/temp
  echo -n "BAT  " ; cat /sys/devices/virtual/thermal/thermal_zone29/temp
'

