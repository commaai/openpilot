import time
import random
import _thread
import faulthandler
from functools import wraps
from panda import Panda
from panda_jungle import PandaJungle  # pylint: disable=import-error
from nose.tools import assert_equal
from parameterized import parameterized, param
from .timeout import run_with_timeout
from .wifi_helpers import _connect_wifi

SPEED_NORMAL = 500
SPEED_GMLAN = 33.3
BUS_SPEEDS = [(0, SPEED_NORMAL), (1, SPEED_NORMAL), (2, SPEED_NORMAL), (3, SPEED_GMLAN)]
TIMEOUT = 30
GEN2_HW_TYPES = [Panda.HW_TYPE_BLACK_PANDA, Panda.HW_TYPE_UNO]
GPS_HW_TYPES = [Panda.HW_TYPE_GREY_PANDA, Panda.HW_TYPE_BLACK_PANDA, Panda.HW_TYPE_UNO]

# Enable fault debug
faulthandler.enable(all_threads=False)

# Connect to Panda Jungle
panda_jungle = PandaJungle()

# Find all panda's connected
_panda_serials = None
def init_panda_serials():
  global panda_jungle, _panda_serials
  _panda_serials = []
  panda_jungle.set_panda_power(True)
  time.sleep(5)
  for serial in Panda.list():
    p = Panda(serial=serial)
    _panda_serials.append((serial, p.get_type()))
    p.close()
  print('Found', str(len(_panda_serials)), 'pandas')
init_panda_serials()

# Panda providers
test_all_types = parameterized([
    param(panda_type=Panda.HW_TYPE_WHITE_PANDA),
    param(panda_type=Panda.HW_TYPE_GREY_PANDA),
    param(panda_type=Panda.HW_TYPE_BLACK_PANDA)
  ])
test_all_pandas = parameterized(
    list(map(lambda x: x[0], _panda_serials))
  )
test_all_gen2_pandas = parameterized(
    list(map(lambda x: x[0], filter(lambda x: x[1] in GEN2_HW_TYPES, _panda_serials)))
  )
test_all_gps_pandas = parameterized(
    list(map(lambda x: x[0], filter(lambda x: x[1] in GPS_HW_TYPES, _panda_serials)))
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
test_black = parameterized([
    param(panda_type=Panda.HW_TYPE_BLACK_PANDA)
  ])

def connect_wifi(serial=None):
  p = Panda(serial=serial)
  p.set_esp_power(True)
  dongle_id, pw = p.get_serial()
  assert(dongle_id.isalnum())
  _connect_wifi(dongle_id, pw)

def time_many_sends(p, bus, p_recv=None, msg_count=100, msg_id=None, two_pandas=False):
  if p_recv == None:
    p_recv = p
  if msg_id == None:
    msg_id = random.randint(0x100, 0x200)
  if p == p_recv and two_pandas:
    raise ValueError("Cannot have two pandas that are the same panda")

  start_time = time.time()
  p.can_send_many([(msg_id, 0, b"\xaa"*8, bus)]*msg_count)
  r = []
  r_echo = []
  r_len_expected = msg_count if two_pandas else msg_count*2
  r_echo_len_exected = msg_count if two_pandas else 0

  while len(r) < r_len_expected and (time.time() - start_time) < 5:
    r.extend(p_recv.can_recv())
  end_time = time.time()
  if two_pandas:
    while len(r_echo) < r_echo_len_exected and (time.time() - start_time) < 10:
      r_echo.extend(p.can_recv())

  sent_echo = [x for x in r if x[3] == 0x80 | bus and x[0] == msg_id]
  sent_echo.extend([x for x in r_echo if x[3] == 0x80 | bus and x[0] == msg_id])
  resp = [x for x in r if x[3] == bus and x[0] == msg_id]

  leftovers = [x for x in r if (x[3] != 0x80 | bus and x[3] != bus) or x[0] != msg_id]
  assert_equal(len(leftovers), 0)

  assert_equal(len(resp), msg_count)
  assert_equal(len(sent_echo), msg_count)

  end_time = (end_time-start_time)*1000.0
  comp_kbps = (1+11+1+1+1+4+8*8+15+1+1+1+7)*msg_count / end_time

  return comp_kbps

def reset_pandas():
  panda_jungle.set_panda_power(False)
  time.sleep(2)
  panda_jungle.set_panda_power(True)
  time.sleep(5)

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
      init_panda_serials()

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

def start_heartbeat_thread(p):
  def heartbeat_thread(p):
    while True:
      try:
        p.send_heartbeat()
        time.sleep(1)
      except:
        break
  _thread.start_new_thread(heartbeat_thread, (p,))

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

    # Initialize jungle
    clear_can_buffers(panda_jungle)
    panda_jungle.set_can_loopback(False)
    panda_jungle.set_obd(False)
    panda_jungle.set_harness_orientation(PandaJungle.HARNESS_ORIENTATION_1)
    for bus, speed in BUS_SPEEDS:
        panda_jungle.set_can_speed_kbps(bus, speed)

    # Initialize pandas
    for panda in pandas:
      panda.set_can_loopback(False)
      panda.set_gmlan(None)
      panda.set_esp_power(False)
      panda.set_power_save(False)
      for bus, speed in BUS_SPEEDS:
        panda.set_can_speed_kbps(bus, speed)
      clear_can_buffers(panda)
      panda.set_power_save(False)

    try:
      run_with_timeout(TIMEOUT, fn, *pandas, **kwargs)

      # Check if the pandas did not throw any faults while running test
      for panda in pandas:
        panda.reconnect()
        assert panda.health()['fault_status'] == 0
    except Exception as e:
      raise e
    finally:
      # Close all connections
      for panda in pandas:
        panda.close()        
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
