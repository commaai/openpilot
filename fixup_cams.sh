#!/bin/bash
set -e
set -x

for cam-cpas crm_workq-cam_fd_worker crm_workq-cam_ife_worke crm_workq-cam_lrme_devi
crm_workq-cam_lrme_hw_w
qcom,cam170-cpas-cdm0
qcom,cam_virtual_cdm

sudo su -c "echo 40 | tee /sys/devices/virtual/workqueue/*cam*/cpumask"
echo 40 | sudo tee /sys/devices/virtual/workqueue/*cam*/cpumask
sudo cat /sys/devices/virtual/workqueue/*cam*/cpumask

for pid in $(ps -A | grep cam | awk '{print $1}'); do
  chrt -p $pid
  sudo chrt -f -p 1 $pid
  sudo taskset -pc 6 $pid || echo "failed on $pid"
done
