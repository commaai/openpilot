#!/usr/bin/env python3
import os
import time
import random
from collections import defaultdict
from opendbc.car.structs import CarParams
from panda import Panda, calculate_checksum, DLC_TO_LEN
from panda import PandaJungle
from panda.tests.hitl.helpers import time_many_sends

H7_HW_TYPES = [Panda.HW_TYPE_RED_PANDA, Panda.HW_TYPE_RED_PANDA_V2]
JUNGLE_SERIAL = os.getenv("JUNGLE")
H7_PANDAS_EXCLUDE = [] # type: ignore
if os.getenv("H7_PANDAS_EXCLUDE"):
  H7_PANDAS_EXCLUDE = os.getenv("H7_PANDAS_EXCLUDE").strip().split(" ") # type: ignore

def panda_reset():
  panda_serials = []

  panda_jungle = PandaJungle(JUNGLE_SERIAL)
  panda_jungle.set_can_silent(True)
  panda_jungle.set_panda_power(False)
  time.sleep(1)
  panda_jungle.set_panda_power(True)
  time.sleep(4)

  for serial in Panda.list():
    if serial not in H7_PANDAS_EXCLUDE:
      with Panda(serial=serial) as p:
        if p.get_type() in H7_HW_TYPES:
          p.reset()
          panda_serials.append(serial)

  print("test pandas", panda_serials)
  assert len(panda_serials) == 2, "Two H7 pandas required"

  return panda_serials

def panda_init(serial, enable_canfd=False, enable_non_iso=False):
  p = Panda(serial=serial)
  p.set_power_save(False)
  for bus in range(3):
    p.set_can_speed_kbps(0, 500)
    if enable_canfd:
      p.set_can_data_speed_kbps(bus, 2000)
    if enable_non_iso:
      p.set_canfd_non_iso(bus, True)
  p.set_safety_mode(CarParams.SafetyModel.allOutput)
  return p

def test_canfd_throughput(p, p_recv=None):
  two_pandas = p_recv is not None
  p.set_safety_mode(CarParams.SafetyModel.allOutput)
  if two_pandas:
    p_recv.set_safety_mode(CarParams.SafetyModel.allOutput)
  # enable output mode
  else:
    p.set_can_loopback(True)

  tests = [
    [500, 1000, 2000], # speeds
    [93, 87, 78], # saturation thresholds
  ]

  for i in range(len(tests[0])):
    # set bus 0 data speed to speed
    p.set_can_data_speed_kbps(0, tests[0][i])
    if p_recv is not None:
      p_recv.set_can_data_speed_kbps(0, tests[0][i])
    time.sleep(0.05)

    comp_kbps = time_many_sends(p, 0, p_recv=p_recv, msg_count=400, two_pandas=two_pandas, msg_len=64)

    # bit count from https://en.wikipedia.org/wiki/CAN_bus
    saturation_pct = (comp_kbps / tests[0][i]) * 100.0
    assert saturation_pct > tests[1][i]
    assert saturation_pct < 100

def canfd_test(p_send, p_recv):
  for n in range(100):
    sent_msgs = defaultdict(set)
    to_send = []
    for _ in range(200):
      bus = random.randrange(3)
      for dlc in range(len(DLC_TO_LEN)):
        address = random.randrange(1, 1<<29)
        data = bytearray(random.getrandbits(8) for _ in range(DLC_TO_LEN[dlc]))
        if len(data) >= 2:
          data[0] = calculate_checksum(data[1:] + bytes(str(address), encoding="utf-8"))
        to_send.append([address, data, bus])
        sent_msgs[bus].add((address, bytes(data)))

    p_send.can_send_many(to_send, timeout=0)

    start_time = time.monotonic()
    while (time.monotonic() - start_time < 1) and any(len(x) > 0 for x in sent_msgs.values()):
      incoming = p_recv.can_recv()
      for msg in incoming:
        address, data, bus = msg
        if len(data) >= 2:
          assert calculate_checksum(data[1:] + bytes(str(address), encoding="utf-8")) == data[0]
        k = (address, bytes(data))
        assert k in sent_msgs[bus], f"message {k} was never sent on bus {bus}"
        sent_msgs[bus].discard(k)

    for bus in range(3):
      assert not len(sent_msgs[bus]), f"loop {n}: bus {bus} missing {len(sent_msgs[bus])} messages"

def setup_test(enable_non_iso=False):
  panda_serials = panda_reset()

  p_send = panda_init(panda_serials[0], enable_canfd=False, enable_non_iso=enable_non_iso)
  p_recv = panda_init(panda_serials[1], enable_canfd=True, enable_non_iso=enable_non_iso)

  # Check that sending panda CAN FD and BRS are turned off
  for bus in range(3):
    health = p_send.can_health(bus)
    assert not health["canfd_enabled"]
    assert not health["brs_enabled"]
    assert health["canfd_non_iso"] == enable_non_iso

  # Receiving panda sends dummy CAN FD message that should enable CAN FD on sender side
  for bus in range(3):
    p_recv.can_send(0x200, b"dummymessage", bus)
  p_recv.can_recv()
  p_send.can_recv()

  # Check if all tested buses on sending panda have swithed to CAN FD with BRS
  for bus in range(3):
    health = p_send.can_health(bus)
    assert health["canfd_enabled"]
    assert health["brs_enabled"]
    assert health["canfd_non_iso"] == enable_non_iso

  return p_send, p_recv

def main():
  print("[TEST CAN-FD]")
  p_send, p_recv = setup_test()
  canfd_test(p_send, p_recv)

  print("[TEST CAN-FD non-ISO]")
  p_send, p_recv = setup_test(enable_non_iso=True)
  canfd_test(p_send, p_recv)

  print("[TEST CAN-FD THROUGHPUT]")
  panda_serials = panda_reset()
  p_send = panda_init(panda_serials[0], enable_canfd=True)
  p_recv = panda_init(panda_serials[1], enable_canfd=True)
  test_canfd_throughput(p_send, p_recv)

if __name__ == "__main__":
  main()
