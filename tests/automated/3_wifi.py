from __future__ import print_function
import os
from panda import Panda
from helpers import connect_wifi
import requests

def test_get_serial():
  p = Panda()
  print(p.get_serial())

def test_get_serial_in_flash_mode():
  p = Panda()
  p.reset(enter_bootstub=True)
  assert(p.bootstub)
  print(p.get_serial())
  p.reset()

def test_connect_wifi():
  connect_wifi()

def test_flash_wifi():
  Panda.flash_ota_wifi()
  connect_wifi()

def test_wifi_flash_st():
  Panda.flash_ota_st()

def test_webpage_fetch():
  r = requests.get("http://192.168.0.10/")
  print(r.text)

  assert "This is your comma.ai panda" in r.text

