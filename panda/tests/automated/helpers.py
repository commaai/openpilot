import os
import sys
import time
import random
import subprocess
import requests
from functools import wraps
from panda import Panda
from nose.tools import timed, assert_equal, assert_less, assert_greater
from parameterized import parameterized, param

test_white_and_grey = parameterized([param(panda_color="White"),
                                     param(panda_color="Grey")])
test_white = parameterized([param(panda_color="White")])
test_grey = parameterized([param(panda_color="Grey")])
test_two_panda = parameterized([param(panda_color=["Grey", "White"]),
                                param(panda_color=["White", "Grey"])])

_serials = {}
def get_panda_serial(is_grey=None):
  global _serials
  if is_grey not in _serials:
    for serial in Panda.list():
      p = Panda(serial=serial)
      if is_grey is None or p.is_grey() == is_grey:
        _serials[is_grey] = serial
        return serial
    raise IOError("Panda not found. is_grey: {}".format(is_grey))
  else:
    return _serials[is_grey]

def connect_wo_esp(serial=None):
  # connect to the panda
  p = Panda(serial=serial)

  # power down the ESP
  p.set_esp_power(False)

  # clear old junk
  while len(p.can_recv()) > 0:
    pass

  return p

def connect_wifi(serial=None):
  p = Panda(serial=serial)
  p.set_esp_power(True)
  dongle_id, pw = p.get_serial()
  assert(dongle_id.isalnum())
  _connect_wifi(dongle_id, pw)

FNULL = open(os.devnull, 'w')
def _connect_wifi(dongle_id, pw, insecure_okay=False):
  ssid = str("panda-" + dongle_id)

  r = subprocess.call(["ping", "-W", "4", "-c", "1", "192.168.0.10"], stdout=FNULL, stderr=subprocess.STDOUT)
  if not r:
    #Can already ping, try connecting on wifi
    try:
      p = Panda("WIFI")
      p.get_serial()
      print("Already connected")
      return
    except:
      pass

  print("WIFI: connecting to %s" % ssid)

  while 1:
    if sys.platform == "darwin":
      os.system("networksetup -setairportnetwork en0 %s %s" % (ssid, pw))
    else:
      wlan_interface = subprocess.check_output(["sh", "-c", "iw dev | awk '/Interface/ {print $2}'"]).strip()
      cnt = 0
      MAX_TRIES = 10
      while cnt < MAX_TRIES:
        print("WIFI: scanning %d" % cnt)
        os.system("iwlist %s scanning > /dev/null" % wlan_interface)
        os.system("nmcli device wifi rescan")
        wifi_scan = filter(lambda x: ssid in x, subprocess.check_output(["nmcli","dev", "wifi", "list"]).split("\n"))
        if len(wifi_scan) != 0:
          break
        time.sleep(0.1)
        # MAX_TRIES tries, ~10 seconds max
        cnt += 1
      assert cnt < MAX_TRIES
      if "-pair" in wifi_scan[0]:
        os.system("nmcli d wifi connect %s-pair" % (ssid))
        connect_cnt = 0
        MAX_TRIES = 20
        while connect_cnt < MAX_TRIES:
          connect_cnt += 1
          r = subprocess.call(["ping", "-W", "4", "-c", "1", "192.168.0.10"], stdout=FNULL, stderr=subprocess.STDOUT)
          if r:
            print("Waiting for panda to ping...")
            time.sleep(0.1)
          else:
            break
        if insecure_okay:
          break
        # fetch webpage
        print("connecting to insecure network to secure")
        try:
          r = requests.get("http://192.168.0.10/")
        except requests.ConnectionError:
          r = requests.get("http://192.168.0.10/")
        assert r.status_code==200

        print("securing")
        try:
          r = requests.get("http://192.168.0.10/secure", timeout=0.01)
        except requests.exceptions.Timeout:
          print("timeout http request to secure")
          pass
      else:
        ret = os.system("nmcli d wifi connect %s password %s" % (ssid, pw))
        if os.WEXITSTATUS(ret) == 0:
          #check ping too
          ping_ok = False
          connect_cnt = 0
          MAX_TRIES = 10
          while connect_cnt < MAX_TRIES:
            connect_cnt += 1
            r = subprocess.call(["ping", "-W", "4", "-c", "1", "192.168.0.10"], stdout=FNULL, stderr=subprocess.STDOUT)
            if r:
              print("Waiting for panda to ping...")
              time.sleep(0.1)
            else:
              ping_ok = True
              break
          if ping_ok:
            break

  # TODO: confirm that it's connected to the right panda

def time_many_sends(p, bus, precv=None, msg_count=100, msg_id=None, two_pandas=False):
  if precv == None:
    precv = p
  if msg_id == None:
    msg_id = random.randint(0x100, 0x200)
  if p == precv and two_pandas:
    raise ValueError("Cannot have two pandas that are the same panda")

  st = time.time()
  p.can_send_many([(msg_id, 0, "\xaa"*8, bus)]*msg_count)
  r = []
  r_echo = []
  r_len_expected = msg_count if two_pandas else msg_count*2
  r_echo_len_exected = msg_count if two_pandas else 0

  while len(r) < r_len_expected and (time.time() - st) < 5:
    r.extend(precv.can_recv())
  et = time.time()
  if two_pandas:
    while len(r_echo) < r_echo_len_exected and (time.time() - st) < 10:
      r_echo.extend(p.can_recv())

  sent_echo = filter(lambda x: x[3] == 0x80 | bus and x[0] == msg_id, r)
  sent_echo.extend(filter(lambda x: x[3] == 0x80 | bus and x[0] == msg_id, r_echo))
  resp = filter(lambda x: x[3] == bus and x[0] == msg_id, r)

  leftovers = filter(lambda x: (x[3] != 0x80 | bus and x[3] != bus) or x[0] != msg_id, r)
  assert_equal(len(leftovers), 0)

  assert_equal(len(resp), msg_count)
  assert_equal(len(sent_echo), msg_count)

  et = (et-st)*1000.0
  comp_kbps = (1+11+1+1+1+4+8*8+15+1+1+1+7)*msg_count / et

  return comp_kbps


def panda_color_to_serial(fn):
  @wraps(fn)
  def wrapper(panda_color=None, **kwargs):
    pandas_is_grey = []
    if panda_color is not None:
      if not isinstance(panda_color, list):
        panda_color = [panda_color]
      panda_color = [s.lower() for s in panda_color]
    for p in panda_color:
      if p is None:
        pandas_is_grey.append(None)
      elif p in ["grey", "gray"]:
        pandas_is_grey.append(True)
      elif p in ["white"]:
        pandas_is_grey.append(False)
      else:
        raise ValueError("Invalid Panda Color {}".format(p))
    return fn(*[get_panda_serial(is_grey) for is_grey in pandas_is_grey], **kwargs)
  return wrapper
