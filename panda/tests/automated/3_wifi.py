from __future__ import print_function
import os
import time
from panda import Panda
from helpers import connect_wifi, test_white, test_white_and_grey, panda_color_to_serial
import requests

@test_white_and_grey
@panda_color_to_serial
def test_get_serial(serial=None):
  p = Panda(serial)
  print(p.get_serial())

@test_white_and_grey
@panda_color_to_serial
def test_get_serial_in_flash_mode(serial=None):
  p = Panda(serial)
  p.reset(enter_bootstub=True)
  assert(p.bootstub)
  print(p.get_serial())
  p.reset()

@test_white
@panda_color_to_serial
def test_connect_wifi(serial=None):
  connect_wifi(serial)

@test_white
@panda_color_to_serial
def test_flash_wifi(serial=None):
  connect_wifi(serial)
  assert Panda.flash_ota_wifi(release=False), "OTA Wifi Flash Failed"
  connect_wifi(serial)

@test_white
@panda_color_to_serial
def test_wifi_flash_st(serial=None):
  connect_wifi(serial)
  assert Panda.flash_ota_st(), "OTA ST Flash Failed"
  connected = False
  st = time.time()
  while not connected and (time.time() - st) < 20:
    try:
      p = Panda(serial=serial)
      p.get_serial()
      connected = True
    except:
      time.sleep(1)

  if not connected:
    assert False, "Panda failed to connect on USB after flashing"

@test_white
@panda_color_to_serial
def test_webpage_fetch(serial=None):
  connect_wifi(serial)
  r = requests.get("http://192.168.0.10/")
  print(r.text)

  assert "This is your comma.ai panda" in r.text
