#!/usr/bin/env python3
import os
import time
import numpy as np
from selfdrive.hardware.tici.hardware import Tici
from selfdrive.hardware.tici.pins import *
from common.gpio import gpio_init, gpio_set

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
  for i in range(10):
    pwrs.append(read_power())
    time.sleep(0.01)
  return np.mean([x[0] for x in pwrs]), np.mean([x[1] for x in pwrs])


def gpio_export(pin):
  try:
    with open(f"/sys/class/gpio/export", 'w') as f:
      f.write(str(pin))
  except Exception as e:
    print(f"Failed to export gpio {pin}")


if __name__ == "__main__":
  print("hello")
  os.system("nmcli radio wifi off")
  os.system('kill $(pgrep -f "python -m selfdrive.athena.manage_athenad")')
  os.system('kill $(pgrep -f "selfdrive.athena.athenad")')
  os.system("sudo service weston stop")
  os.system("sudo service ModemManager stop")
  print("services stopped")

  t = Tici()
  t.initialize_hardware()
  t.set_power_save(True)
  t.set_screen_brightness(0)
  gpio_init(GPIO_STM_RST_N, True)
  gpio_init(GPIO_HUB_RST_N, True)
  gpio_init(GPIO_UBLOX_PWR_EN, True)
  gpio_init(GPIO_LTE_RST_N, True)
  gpio_init(GPIO_LTE_PWRKEY, True)

  # cameras
  gpio_export(GPIO_CAM0_DVDD_EN)
  gpio_export(GPIO_CAM0_AVDD_EN)
  gpio_init(GPIO_CAM0_DVDD_EN, True)
  gpio_init(GPIO_CAM0_AVDD_EN, True)

  print("on")
  gpio_set(GPIO_STM_RST_N, False)       # this turns panda on
  gpio_set(GPIO_HUB_RST_N, True)        # this turns hub on

  # lcd off (crashes kernel) -- disabling weston does this
  #os.system("sudo su -c 'echo 0 > /sys/kernel/debug/regulator/lcd3v3/enable'")

  """
  gpio_set(GPIO_LTE_RST_N, False)        # this turns quectel on
  time.sleep(1)
  gpio_set(GPIO_LTE_PWRKEY, True)
  time.sleep(1)
  gpio_set(GPIO_LTE_PWRKEY, False)
  """

  # off
  gpio_set(GPIO_LTE_RST_N, True)        # this turns quectel off
  gpio_set(GPIO_UBLOX_PWR_EN, False)    # this turns gps off
  gpio_set(GPIO_CAM0_DVDD_EN, False)
  gpio_set(GPIO_CAM0_AVDD_EN, False)
  time.sleep(2)
  os.system("lsusb")

  print("baseline", read_power_avg())
  gpio_set(GPIO_STM_RST_N, True)
  gpio_set(GPIO_HUB_RST_N, False)
  time.sleep(3)
  print("panda off", read_power_avg())
  os.system("lsusb")



