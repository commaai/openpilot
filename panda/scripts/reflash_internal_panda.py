#!/usr/bin/env python3
import glob
import os
import time
from panda import Panda, PandaDFU

class GPIO:
  STM_RST_N = 124
  STM_BOOT0 = 134


def tlmm_base() -> int:
  # the SoC GPIO controller's sysfs number base: 0 on the downstream kernel, assigned at boot on the upstream one
  for chip in glob.glob("/sys/class/gpio/gpiochip*"):
    with open(os.path.join(chip, "label")) as f:
      if f.read().strip() == "3400000.pinctrl":
        with open(os.path.join(chip, "base")) as b:
          return int(b.read())
  raise RuntimeError("SoC GPIO controller not found")

def gpio_init(pin, output):
  with open(f"/sys/class/gpio/gpio{tlmm_base() + pin}/direction", 'wb') as f:
    f.write(b"out" if output else b"in")

def gpio_set(pin, high):
  with open(f"/sys/class/gpio/gpio{tlmm_base() + pin}/value", 'wb') as f:
    f.write(b"1" if high else b"0")


if __name__ == "__main__":
  for pin in (GPIO.STM_RST_N, GPIO.STM_BOOT0):
    gpio_init(pin, True)

  # flash bootstub
  print("resetting into DFU")
  gpio_set(GPIO.STM_RST_N, 1)
  gpio_set(GPIO.STM_BOOT0, 1)
  time.sleep(0.2)
  gpio_set(GPIO.STM_RST_N, 0)
  gpio_set(GPIO.STM_BOOT0, 0)
  # give the bootloader time to start before any SPI traffic
  time.sleep(0.01)

  # bootstub flash takes 2s and is limited by the 255 byte flashing chunk size
  print("flashing bootstub")
  assert Panda.wait_for_dfu(None, 5)
  PandaDFU(None).recover()

  print("flashing app")
  assert Panda.wait_for_panda(None, 5)
  p = Panda()
  assert p.bootstub
  p.flash()
