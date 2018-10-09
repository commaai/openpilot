import os
import sys
import time
import random
import subprocess
import requests
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

def _connect_wifi(dongle_id, pw, insecure_okay=False):
  ssid = str("panda-" + dongle_id)

  print("WIFI: connecting to %s" % ssid)

  while 1:
    if sys.platform == "darwin":
      os.system("networksetup -setairportnetwork en0 %s %s" % (ssid, pw))
    else:
      wlan_interface = subprocess.check_output(["sh", "-c", "iw dev | awk '/Interface/ {print $2}'"]).strip()
      cnt = 0
      MAX_TRIES = 10
      while cnt < MAX_TRIES:
        print "WIFI: scanning %d" % cnt
        os.system("sudo iwlist %s scanning > /dev/null" % wlan_interface)
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
        if insecure_okay:
          break
        # fetch webpage
        print "connecting to insecure network to secure"
        r = requests.get("http://192.168.0.10/")
        assert r.status_code==200

        print "securing"
        try:
          r = requests.get("http://192.168.0.10/secure", timeout=0.01)
        except requests.exceptions.Timeout:
          pass
      else:
        os.system("nmcli d wifi connect %s password %s" % (ssid, pw))
        break
  
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

