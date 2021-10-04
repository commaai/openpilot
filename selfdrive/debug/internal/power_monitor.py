#!/usr/bin/env python3
import os
import time
import sys
from datetime import datetime

def average(avg, sample):
  # Weighted avg between existing value and new sample
  return ((avg[0] * avg[1] + sample) / (avg[1] + 1), avg[1] + 1)


if __name__ == '__main__':
  try:
    if len(sys.argv) > 1 and sys.argv[1] == "--charge":
      print("not disabling charging")
    else:
      print("disabling charging")
      os.system('echo "0" > /sys/class/power_supply/battery/charging_enabled')

    voltage_average = (0., 0)  # average, count
    current_average = (0., 0)
    power_average = (0., 0)
    capacity_average = (0., 0)
    bat_temp_average = (0., 0)
    start_time = datetime.now()
    while 1:
      with open("/sys/class/power_supply/bms/voltage_now") as f:
        voltage = int(f.read()) / 1e6   # volts

      with open("/sys/class/power_supply/bms/current_now") as f:
        current = int(f.read()) / 1e3   # ma

      power = voltage * current

      with open("/sys/class/power_supply/bms/capacity_raw") as f:
        capacity = int(f.read()) / 1e2  # percent

      with open("/sys/class/power_supply/bms/temp") as f:
        bat_temp = int(f.read()) / 1e1  # celsius

      # compute averages
      voltage_average = average(voltage_average, voltage)
      current_average = average(current_average, current)
      power_average = average(power_average, power)
      capacity_average = average(capacity_average, capacity)
      bat_temp_average = average(bat_temp_average, bat_temp)

      print(f"{voltage:.2f} volts {current:12.2f} ma {power:12.2f} mW {capacity:8.2f}% battery {bat_temp:8.1f} degC")
      time.sleep(0.1)
  finally:
    stop_time = datetime.now()
    print("\n----------------------Average-----------------------------------")
    voltage = voltage_average[0]
    current = current_average[0]
    power = power_average[0]
    capacity = capacity_average[0]
    bat_temp = bat_temp_average[0]
    print(f"{voltage:.2f} volts {current:12.2f} ma {power:12.2f} mW {capacity:8.2f}% battery {bat_temp:8.1f} degC")
    print(f"  {(stop_time - start_time).total_seconds():.2f} Seconds     {voltage_average[1]} samples")
    print("----------------------------------------------------------------")

    # reenable charging
    os.system('echo "1" > /sys/class/power_supply/battery/charging_enabled')
    print("charging enabled\n")
