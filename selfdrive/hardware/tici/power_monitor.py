#!/usr/bin/env python3
import sys
import time

def average(avg, sample):
  # Weighted avg between existing value and new sample
  return ((avg[0] * avg[1] + sample) / (avg[1] + 1), avg[1] + 1)

if __name__ == '__main__':

  sample_time = None
  if len(sys.argv) > 1:
    sample_time = int(sys.argv[1])

  start_time = time.monotonic()
  try:
    voltage_average = (0, 0)  # average, count
    current_average = (0, 0)
    power_average = (0, 0)
    power_total_average = (0, 0)
    while sample_time is None or time.monotonic() - start_time < sample_time:
      with open("/sys/bus/i2c/devices/0-0040/hwmon/hwmon1/in1_input") as f:
        voltage_total = int(f.read()) / 1000.

      with open("/sys/bus/i2c/devices/0-0040/hwmon/hwmon1/curr1_input") as f:
        current_total = int(f.read())

      with open("/sys/class/power_supply/bms/voltage_now") as f:
        voltage = int(f.read()) / 1e6   # volts

      with open("/sys/class/power_supply/bms/current_now") as f:
        current = int(f.read()) / 1e3   # ma

      power = voltage*current
      power_total = voltage_total*current_total

      # compute averages
      voltage_average = average(voltage_average, voltage)
      current_average = average(current_average, current)
      power_average = average(power_average, power)
      power_total_average = average(power_total_average, power_total)

      print(f"{power:12.2f} mW {power_total:12.2f} mW {power_total - power:12.2f} mW")
      time.sleep(0.25)
  finally:
    stop_time = time.monotonic()
    print("\n----------------------Average-----------------------------------")
    voltage = voltage_average[0]
    current = current_average[0]
    power = power_average[0]
    print(f"{voltage:.2f} volts {current:12.2f} ma {power:12.2f} mW {power_total:12.2f} mW")
    print(f"  {stop_time - start_time:.2f} Seconds     {voltage_average[1]} samples")
    print("----------------------------------------------------------------")
