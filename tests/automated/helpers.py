import os
import sys
import time
import random
import subprocess
from panda import Panda
from nose.tools import timed, assert_equal, assert_less, assert_greater

def connect_wo_esp():
  # connect to the panda
  p = Panda()

  # power down the ESP
  p.set_esp_power(False)

  # clear old junk
  while len(p.can_recv()) > 0:
    pass

  return p

def connect_wifi():
  p = Panda()
  dongle_id, pw = p.get_serial()
  assert(dongle_id.isalnum())
  _connect_wifi(dongle_id, pw)

def _connect_wifi(dongle_id, pw):
  ssid = str("panda-" + dongle_id)

  print("WIFI: connecting to %s" % ssid)

  if sys.platform == "darwin":
    os.system("networksetup -setairportnetwork en0 %s %s" % (ssid, pw))
  else:
    cnt = 0
    MAX_TRIES = 10
    while cnt < MAX_TRIES:
      print "WIFI: scanning %d" % cnt
      os.system("nmcli device wifi rescan")
      wifi_scan = filter(lambda x: ssid in x, subprocess.check_output(["nmcli","dev", "wifi", "list"]).split("\n"))
      if len(wifi_scan) != 0:
        break
      time.sleep(0.1)
      # MAX_TRIES tries, ~10 seconds max
      cnt += 1
    assert cnt < MAX_TRIES
    os.system("nmcli d wifi connect %s password %s" % (ssid, pw))
  
  # TODO: confirm that it's connected to the right panda

def time_many_sends(p, bus, precv=None, msg_count=100, msg_id=None):
  if precv == None:
    precv = p
  if msg_id == None:
    msg_id = random.randint(0x100, 0x200)

  st = time.time()
  p.can_send_many([(msg_id, 0, "\xaa"*8, bus)]*msg_count)
  r = []

  while len(r) < (msg_count*2) and (time.time() - st) < 3:
    r.extend(precv.can_recv())

  sent_echo = filter(lambda x: x[3] == 0x80 | bus and x[0] == msg_id, r)
  loopback_resp = filter(lambda x: x[3] == bus and x[0] == msg_id, r)

  assert_equal(len(sent_echo), msg_count)
  assert_equal(len(loopback_resp), msg_count)

  et = (time.time()-st)*1000.0
  comp_kbps = (1+11+1+1+1+4+8*8+15+1+1+1+7)*msg_count / et

  return comp_kbps

