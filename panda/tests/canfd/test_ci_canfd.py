import os
import time
import random
import _thread
from collections import defaultdict
from panda import Panda
from panda_jungle import PandaJungle  # pylint: disable=import-error

H7_HW_TYPES = [Panda.HW_TYPE_RED_PANDA]
JUNGLE_SERIAL = os.getenv("JUNGLE")
H7_PANDAS_EXCLUDE = [] # type: ignore
if os.getenv("H7_PANDAS_EXCLUDE"):
  H7_PANDAS_EXCLUDE = os.getenv("H7_PANDAS_EXCLUDE").strip().split(" ") # type: ignore

#TODO: REMOVE, temporary list of CAN FD lengths, one in panda python lib MUST be used
DLC_TO_LEN = [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20, 24, 32, 48]

_panda_serials = []

def start_heartbeat_thread(p):
  def heartbeat_thread(p):
    while True:
      try:
        p.send_heartbeat()
        time.sleep(.5)
      except Exception:
        break
  _thread.start_new_thread(heartbeat_thread, (p,))

def panda_init(serial, enable_canfd=False):
  p = Panda(serial=serial)
  start_heartbeat_thread(p)
  p.set_power_save(False)
  for bus in range(3):
    if enable_canfd:
      p.set_can_data_speed_kbps(bus, 2000)
  p.set_safety_mode(Panda.SAFETY_ALLOUTPUT)
  return p

if JUNGLE_SERIAL:
  panda_jungle = PandaJungle(JUNGLE_SERIAL)
  panda_jungle.set_panda_power(False)
  time.sleep(2)
  panda_jungle.set_panda_power(True)
  time.sleep(4)

for serial in Panda.list():
  if serial not in H7_PANDAS_EXCLUDE:
    p = Panda(serial=serial)
    if p.get_type() in H7_HW_TYPES:
      assert p.recover(timeout=30)
      _panda_serials.append(serial)
    p.close()

if len(_panda_serials) < 2:
  print("Minimum two H7 type pandas should be connected.")
  assert False


def canfd_test(p_send, p_recv):
  for _ in range(100):
    sent_msgs = defaultdict(set)
    to_send = []
    for _ in range(200):
      bus = random.randrange(3)
      for dlc in range(len(DLC_TO_LEN)):
        address = random.randrange(1, 1<<29)
        data = bytes([random.getrandbits(8) for _ in range(DLC_TO_LEN[dlc])])
        to_send.append([address, 0, data, bus])
        sent_msgs[bus].add((address, data))

    p_send.can_send_many(to_send, timeout=0)

    start_time = time.time()
    while time.time() - start_time < 1:
      incoming = p_recv.can_recv()
      for msg in incoming:
        address, _, data, bus = msg
        k = (address, bytes(data))
        assert k in sent_msgs[bus], f"message {k} was never sent on bus {bus}"
        sent_msgs[bus].discard(k)

    for bus in range(3):
      assert not len(sent_msgs[bus]), f"loop : bus {bus} missing {len(sent_msgs[bus])} messages"

  # Set back to silent mode
  p_send.set_safety_mode(Panda.SAFETY_SILENT)
  p_recv.set_safety_mode(Panda.SAFETY_SILENT)
  p_send.set_power_save(True)
  p_recv.set_power_save(True)
  print("Got all messages intact")


p_send = panda_init(_panda_serials[0], enable_canfd=False)
p_recv = panda_init(_panda_serials[1], enable_canfd=True)

# Check that sending panda CAN FD and BRS are turned off
for bus in range(3):
  canfd, brs = p_send.get_canfd_status(bus)
  assert not canfd
  assert not brs

# Receiving panda sends dummy CAN FD message that should enable CAN FD on sender side
for bus in range(3):
  p_recv.can_send(0x200, b"dummymessage", bus)
p_recv.can_recv()

# Check if all tested buses on sending panda have swithed to CAN FD with BRS
for bus in range(3):
  canfd, brs = p_send.get_canfd_status(bus)
  assert canfd
  assert brs

canfd_test(p_send, p_recv)
