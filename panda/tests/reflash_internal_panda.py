#!/usr/bin/env python3
import time
from panda import Panda, PandaDFU

class GPIO:
  STM_RST_N = 124
  STM_BOOT0 = 134
  HUB_RST_N = 30


def gpio_init(pin, output):
  with open(f"/sys/class/gpio/gpio{pin}/direction", 'wb') as f:
    f.write(b"out" if output else b"in")

def gpio_set(pin, high):
  with open(f"/sys/class/gpio/gpio{pin}/value", 'wb') as f:
    f.write(b"1" if high else b"0")


if __name__ == "__main__":
  for pin in (GPIO.STM_RST_N, GPIO.STM_BOOT0, GPIO.HUB_RST_N):
    gpio_init(pin, True)

  # reset USB hub
  gpio_set(GPIO.HUB_RST_N, 0)
  time.sleep(0.5)
  gpio_set(GPIO.HUB_RST_N, 1)

  # flash bootstub
  print("resetting into DFU")
  gpio_set(GPIO.STM_RST_N, 1)
  gpio_set(GPIO.STM_BOOT0, 1)
  time.sleep(1)
  gpio_set(GPIO.STM_RST_N, 0)
  gpio_set(GPIO.STM_BOOT0, 0)
  time.sleep(1)

  print("flashing bootstub")
  PandaDFU(None).recover()

  gpio_set(GPIO.STM_RST_N, 1)
  time.sleep(0.5)
  gpio_set(GPIO.STM_RST_N, 0)
  time.sleep(1)

  print("flashing app")
  p = Panda()
  assert p.bootstub
  p.flash()
