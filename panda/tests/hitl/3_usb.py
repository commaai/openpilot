import time
import pytest
from flaky import flaky

from opendbc.car.structs import CarParams
from panda import Panda
from panda.tests.hitl.helpers import time_many_sends

pytestmark = [
  pytest.mark.test_panda_types((Panda.HW_TYPE_RED_PANDA, ))
]

def test_can_loopback(p):
  p.set_safety_mode(CarParams.SafetyModel.allOutput)
  p.set_can_loopback(True)

  for bus in (0, 1, 2):
    # set bus 0 speed to 5000
    p.set_can_speed_kbps(bus, 500)

    # send a message on bus 0
    p.can_send(0x1aa, b"message", bus)

    # confirm receive both on loopback and send receipt
    time.sleep(0.05)
    r = p.can_recv()
    sr = [x for x in r if x[2] == 0x80 | bus]
    lb = [x for x in r if x[2] == bus]
    assert len(sr) == 1
    assert len(lb) == 1

    # confirm data is correct
    assert 0x1aa == sr[0][0] == lb[0][0]
    assert b"message" == sr[0][1] == lb[0][1]

def test_reliability(p):
  MSG_COUNT = 100

  p.set_safety_mode(CarParams.SafetyModel.allOutput)
  p.set_can_loopback(True)
  p.set_can_speed_kbps(0, 1000)

  addrs = list(range(100, 100 + MSG_COUNT))
  ts = [(j, b"\xaa" * 8, 0) for j in addrs]

  for _ in range(100):
    st = time.monotonic()

    p.can_send_many(ts)

    r = []
    while len(r) < 200 and (time.monotonic() - st) < 0.5:
      r.extend(p.can_recv())

    sent_echo = [x for x in r if x[2] == 0x80]
    loopback_resp = [x for x in r if x[2] == 0]

    assert sorted([x[0] for x in loopback_resp]) == addrs
    assert sorted([x[0] for x in sent_echo]) == addrs
    assert len(r) == 200

    # take sub 20ms
    et = (time.monotonic() - st) * 1000.0
    assert et < 20

@flaky(max_runs=6, min_passes=1)
def test_throughput(p):
  # enable output mode
  p.set_safety_mode(CarParams.SafetyModel.allOutput)

  # enable CAN loopback mode
  p.set_can_loopback(True)

  for speed in [10, 20, 50, 100, 125, 250, 500, 1000]:
    # set bus 0 speed to speed
    p.set_can_speed_kbps(0, speed)
    time.sleep(0.05)

    comp_kbps = time_many_sends(p, 0)

    # bit count from https://en.wikipedia.org/wiki/CAN_bus
    saturation_pct = (comp_kbps / speed) * 100.0
    assert saturation_pct > 80
    assert saturation_pct < 100

    print("loopback 100 messages at speed %d, comp speed is %.2f, percent %.2f" % (speed, comp_kbps, saturation_pct))

# this will fail if you have hardware serial connected
def test_serial_debug(p):
  _ = p.serial_read(Panda.SERIAL_DEBUG)  # junk
  p.call_control_api(0x01)
  assert p.serial_read(Panda.SERIAL_DEBUG).startswith(b"NO HANDLER")
