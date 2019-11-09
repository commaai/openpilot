import os
import sys
import time
import random
import subprocess
import requests
import _thread
from functools import wraps
from panda import Panda
from nose.tools import assert_equal
from parameterized import parameterized, param

SPEED_NORMAL = 500
SPEED_GMLAN = 33.3

test_all_types = parameterized([
    param(panda_type=Panda.HW_TYPE_WHITE_PANDA),
    param(panda_type=Panda.HW_TYPE_GREY_PANDA),
    param(panda_type=Panda.HW_TYPE_BLACK_PANDA)
  ])
test_all_pandas = parameterized(
    Panda.list()
  )
test_white_and_grey = parameterized([
    param(panda_type=Panda.HW_TYPE_WHITE_PANDA),
    param(panda_type=Panda.HW_TYPE_GREY_PANDA)
  ])
test_white = parameterized([
    param(panda_type=Panda.HW_TYPE_WHITE_PANDA)
  ])
test_grey = parameterized([
    param(panda_type=Panda.HW_TYPE_GREY_PANDA)
  ])
test_two_panda = parameterized([
    param(panda_type=[Panda.HW_TYPE_GREY_PANDA, Panda.HW_TYPE_WHITE_PANDA]),
    param(panda_type=[Panda.HW_TYPE_WHITE_PANDA, Panda.HW_TYPE_GREY_PANDA]),
    param(panda_type=[Panda.HW_TYPE_BLACK_PANDA, Panda.HW_TYPE_BLACK_PANDA])
  ])
test_two_black_panda = parameterized([
    param(panda_type=[Panda.HW_TYPE_BLACK_PANDA, Panda.HW_TYPE_BLACK_PANDA])
  ])

def connect_wifi(serial=None):
  p = Panda(serial=serial)
  p.set_esp_power(True)
  dongle_id, pw = p.get_serial()
  assert(dongle_id.isalnum())
  _connect_wifi(dongle_id, pw)

FNULL = open(os.devnull, 'w')
def _connect_wifi(dongle_id, pw, insecure_okay=False):
  ssid = "panda-" + dongle_id

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
      wlan_interface = subprocess.check_output(["sh", "-c", "iw dev | awk '/Interface/ {print $2}'"]).strip().decode('utf8')
      cnt = 0
      MAX_TRIES = 10
      while cnt < MAX_TRIES:
        print("WIFI: scanning %d" % cnt)
        os.system("iwlist %s scanning > /dev/null" % wlan_interface)
        os.system("nmcli device wifi rescan")
        wifi_networks = [x.decode("utf8") for x in subprocess.check_output(["nmcli","dev", "wifi", "list"]).split(b"\n")]
        wifi_scan = [x for x in wifi_networks if ssid in x]
        if len(wifi_scan) != 0:
          break
        time.sleep(0.1)
        # MAX_TRIES tries, ~10 seconds max
        cnt += 1
      assert cnt < MAX_TRIES
      if "-pair" in wifi_scan[0]:
        os.system("nmcli d wifi connect %s-pair" % (ssid))
        connect_cnt = 0
        MAX_TRIES = 100
        while connect_cnt < MAX_TRIES:
          connect_cnt += 1
          r = subprocess.call(["ping", "-W", "4", "-c", "1", "192.168.0.10"], stdout=FNULL, stderr=subprocess.STDOUT)
          if r:
            print("Waiting for panda to ping...")
            time.sleep(0.5)
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
  p.can_send_many([(msg_id, 0, b"\xaa"*8, bus)]*msg_count)
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

  sent_echo = [x for x in r if x[3] == 0x80 | bus and x[0] == msg_id]
  sent_echo.extend([x for x in r_echo if x[3] == 0x80 | bus and x[0] == msg_id])
  resp = [x for x in r if x[3] == bus and x[0] == msg_id]

  leftovers = [x for x in r if (x[3] != 0x80 | bus and x[3] != bus) or x[0] != msg_id]
  assert_equal(len(leftovers), 0)

  assert_equal(len(resp), msg_count)
  assert_equal(len(sent_echo), msg_count)

  et = (et-st)*1000.0
  comp_kbps = (1+11+1+1+1+4+8*8+15+1+1+1+7)*msg_count / et

  return comp_kbps

_panda_serials = None
def panda_type_to_serial(fn):
  @wraps(fn)
  def wrapper(panda_type=None, **kwargs):
    # Change panda_types to a list
    if panda_type is not None:
      if not isinstance(panda_type, list):
        panda_type = [panda_type]

    # If not done already, get panda serials and their type
    global _panda_serials
    if _panda_serials == None:
      _panda_serials = []
      for serial in Panda.list():
        p = Panda(serial=serial)
        _panda_serials.append((serial, p.get_type()))
        p.close()

    # Find a panda with the correct types and add the corresponding serial
    serials = []
    for p_type in panda_type:
      found = False
      for serial, pt in _panda_serials:
        # Never take the same panda twice
        if (pt == p_type) and (serial not in serials):
          serials.append(serial)
          found = True
          break
      if not found:
        raise IOError("No unused panda found for type: {}".format(p_type))
    return fn(serials, **kwargs)
  return wrapper

def heartbeat_thread(p):
  while True:
    try:
      p.send_heartbeat()
      time.sleep(1)
    except:
      break

def panda_connect_and_init(fn):
  @wraps(fn)
  def wrapper(panda_serials=None, **kwargs):
    # Change panda_serials to a list
    if panda_serials is not None:
      if not isinstance(panda_serials, list):
        panda_serials = [panda_serials]

    # Connect to pandas
    pandas = []
    for panda_serial in panda_serials:
      pandas.append(Panda(serial=panda_serial))

    # Initialize pandas
    for panda in pandas:
      panda.set_can_loopback(False)
      panda.set_gmlan(None)
      panda.set_esp_power(False)
      for bus, speed in [(0, SPEED_NORMAL), (1, SPEED_NORMAL), (2, SPEED_NORMAL), (3, SPEED_GMLAN)]:
        panda.set_can_speed_kbps(bus, speed)
      clear_can_buffers(panda)
      _thread.start_new_thread(heartbeat_thread, (panda,))

    # Run test function
    ret = fn(*pandas, **kwargs)

    # Close all connections
    for panda in pandas:
      panda.close()

    # Return test function result
    return ret
  return wrapper

def clear_can_buffers(panda):
  # clear tx buffers
  for i in range(4):
    panda.can_clear(i)

  # clear rx buffers
  panda.can_clear(0xFFFF)
  r = [1]
  st = time.time()
  while len(r) > 0:
    r = panda.can_recv()
    time.sleep(0.05)
    if (time.time() - st) > 10:
      print("Unable to clear can buffers for panda ", panda.get_serial())
      assert False
