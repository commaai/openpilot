import time
from panda import Panda
from .helpers import connect_wifi, test_white, test_all_pandas, panda_type_to_serial, panda_connect_and_init
import requests

@test_all_pandas
@panda_connect_and_init
def test_get_serial(p):
  print(p.get_serial())

@test_all_pandas
@panda_connect_and_init
def test_get_serial_in_flash_mode(p):
  p.reset(enter_bootstub=True)
  assert(p.bootstub)
  print(p.get_serial())
  p.reset()

@test_white
@panda_type_to_serial
def test_connect_wifi(serials=None):
  connect_wifi(serials[0])

@test_white
@panda_type_to_serial
def test_flash_wifi(serials=None):
  connect_wifi(serials[0])
  assert Panda.flash_ota_wifi(release=False), "OTA Wifi Flash Failed"
  connect_wifi(serials[0])

@test_white
@panda_type_to_serial
def test_wifi_flash_st(serials=None):
  connect_wifi(serials[0])
  assert Panda.flash_ota_st(), "OTA ST Flash Failed"
  connected = False
  st = time.time()
  while not connected and (time.time() - st) < 20:
    try:
      p = Panda(serial=serials[0])
      p.get_serial()
      connected = True
    except:
      time.sleep(1)

  if not connected:
    assert False, "Panda failed to connect on USB after flashing"

@test_white
@panda_type_to_serial
def test_webpage_fetch(serials=None):
  connect_wifi(serials[0])
  r = requests.get("http://192.168.0.10/")
  print(r.text)

  assert "This is your comma.ai panda" in r.text
