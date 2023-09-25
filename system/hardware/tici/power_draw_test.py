#!/usr/bin/env python3
import os
import time
import numpy as np
from openpilot.system.hardware.tici.hardware import Tici
from openpilot.system.hardware.tici.pins import GPIO
from openpilot.common.gpio import gpio_init, gpio_set, gpio_export

def read_power():
  with open("/sys/bus/i2c/devices/0-0040/hwmon/hwmon1/in1_input") as f:
    voltage_total = int(f.read()) / 1000.

  with open("/sys/bus/i2c/devices/0-0040/hwmon/hwmon1/curr1_input") as f:
    current_total = int(f.read())

  with open("/sys/class/power_supply/bms/voltage_now") as f:
    voltage = int(f.read()) / 1e6   # volts

  with open("/sys/class/power_supply/bms/current_now") as f:
    current = int(f.read()) / 1e3   # ma

  power_som = voltage*current
  power_total = voltage_total*current_total

  return power_total, power_som

def read_power_avg():
  pwrs = []
  for _ in range(100):
    pwrs.append(read_power())
    time.sleep(0.01)
  power_total, power_som = np.mean([x[0] for x in pwrs]), np.mean([x[1] for x in pwrs])
  return "total %7.2f mW  SOM %7.2f mW" % (power_total, power_som)


if __name__ == "__main__":
  gpio_export(GPIO.CAM0_AVDD_EN)
  gpio_export(GPIO.CAM0_RSTN)
  gpio_export(GPIO.CAM1_RSTN)
  gpio_export(GPIO.CAM2_RSTN)
  print("hello")
  os.system('kill $(pgrep -f "manager.py")')
  os.system('kill $(pgrep -f "python -m selfdrive.athena.manage_athenad")')
  os.system('kill $(pgrep -f "selfdrive.athena.athenad")')
  # stopping weston turns off lcd3v3
  os.system("sudo service weston stop")
  os.system("sudo service ModemManager stop")
  print("services stopped")

  t = Tici()
  t.initialize_hardware()
  t.set_power_save(True)
  t.set_screen_brightness(0)
  gpio_init(GPIO.STM_RST_N, True)
  gpio_init(GPIO.HUB_RST_N, True)
  gpio_init(GPIO.GNSS_PWR_EN, True)
  gpio_init(GPIO.LTE_RST_N, True)
  gpio_init(GPIO.LTE_PWRKEY, True)
  gpio_init(GPIO.CAM0_AVDD_EN, True)
  gpio_init(GPIO.CAM0_RSTN, True)
  gpio_init(GPIO.CAM1_RSTN, True)
  gpio_init(GPIO.CAM2_RSTN, True)


  os.system("sudo su -c 'echo 0 > /sys/kernel/debug/regulator/camera_rear_ldo/enable'")  # cam 1v2 off
  gpio_set(GPIO.CAM0_AVDD_EN, False)    # cam 2v8 off
  gpio_set(GPIO.LTE_RST_N, True)        # quectel off
  gpio_set(GPIO.GNSS_PWR_EN, False)    # gps off
  gpio_set(GPIO.STM_RST_N, True)        # panda off
  gpio_set(GPIO.HUB_RST_N, False)       # hub off
  # cameras in reset
  gpio_set(GPIO.CAM0_RSTN, False)
  gpio_set(GPIO.CAM1_RSTN, False)
  gpio_set(GPIO.CAM2_RSTN, False)
  time.sleep(8)

  print("baseline: ", read_power_avg())
  gpio_set(GPIO.CAM0_AVDD_EN, True)
  time.sleep(2)
  print("cam avdd: ", read_power_avg())
  os.system("sudo su -c 'echo 1 > /sys/kernel/debug/regulator/camera_rear_ldo/enable'")
  time.sleep(2)
  print("cam dvdd: ", read_power_avg())
  gpio_set(GPIO.CAM0_RSTN, True)
  gpio_set(GPIO.CAM1_RSTN, True)
  gpio_set(GPIO.CAM2_RSTN, True)
  time.sleep(2)
  print("cams up:  ", read_power_avg())
  gpio_set(GPIO.HUB_RST_N, True)
  time.sleep(2)
  print("usb hub:  ", read_power_avg())
  gpio_set(GPIO.STM_RST_N, False)
  time.sleep(5)
  print("panda:    ", read_power_avg())
  gpio_set(GPIO.GNSS_PWR_EN, True)
  time.sleep(5)
  print("gps:      ", read_power_avg())
  gpio_set(GPIO.LTE_RST_N, False)
  time.sleep(1)
  gpio_set(GPIO.LTE_PWRKEY, True)
  time.sleep(1)
  gpio_set(GPIO.LTE_PWRKEY, False)
  time.sleep(5)
  print("quectel:  ", read_power_avg())

