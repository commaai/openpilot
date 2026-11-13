import time
import itertools

from opendbc.car.structs import CarParams
from panda.tests.hitl.base import PandaTestCase


class TestComms(PandaTestCase):
  def test_can_loopback(self):
    p = self.p
    p.set_safety_mode(CarParams.SafetyModel.allOutput)
    p.set_can_loopback(True)

    for bus, speed in itertools.product(range(3), (10, 20, 50, 100, 125, 250, 500, 1000)):
      p.set_can_speed_kbps(bus, speed)
      messages = [(addr, b"message", bus) for addr in range(0x100, 0x164)]
      start = time.monotonic()
      p.can_send_many(messages)
      received = []
      deadline = time.monotonic() + 5
      while len(received) < 2 * len(messages) and time.monotonic() < deadline:
        received.extend(p.can_recv())
      if speed < 50:  # rates the UP cannot generate
        saturation = len(messages) * (47 + 8 * len(messages[0][1])) / ((time.monotonic() - start) * speed * 1000)
        assert 0.8 < saturation < 1, (bus, speed, saturation)
      expected = messages + [(addr, dat, bus | 0x80) for addr, dat, bus in messages]
      assert sorted(received) == sorted(expected), (bus, speed, len(received))

  def test_reliability(self):
    p = self.p
    MSG_COUNT = 100

    p.set_safety_mode(CarParams.SafetyModel.allOutput)
    p.set_can_loopback(True)
    p.set_can_speed_kbps(0, 1000)

    addrs = list(range(100, 100 + MSG_COUNT))
    ts = [(j, b"\xaa" * 8, 0) for j in addrs]

    latencies = []
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

      latencies.append((time.monotonic() - st) * 1000.0)

    # Bound sustained throughput without treating a host scheduling pause as frame loss.
    assert sum(latencies) / len(latencies) < 20, latencies

  def test_debug_logs(self):
    p = self.p
    _ = p.debug_read()  # junk
    p.call_control_api(0x01)
    assert p.debug_read().startswith(b"NO HANDLER")
