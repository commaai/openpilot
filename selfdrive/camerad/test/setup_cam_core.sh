#!/bin/bash

# camerad gets core 6

# move IRQs
CAM_IRQS="235 237 239 241 243"
for irq in $CAM_IRQS; do
  sudo su -c "echo 6 > /proc/irq/$irq/smp_affinity_list"
done

# workqueues
sudo su -c "echo 40 > /sys/devices/virtual/workqueue/qcom,cam170-cpas-cdm0/cpumask"
sudo su -c "echo 40 > /sys/devices/virtual/workqueue/qcom,cam_virtual_cdm/cpumask"

for pid in $(pgrep "cam"); do
  sudo chrt -f -p 50 $pid
done
