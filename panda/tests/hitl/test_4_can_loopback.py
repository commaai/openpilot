import os
import time
import random
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

from opendbc.car.structs import CarParams
from panda import Panda
from panda.tests.hitl.base import PandaTestCase, panda_test
from panda.tests.hitl.helpers import get_random_can_messages, clear_can_buffers

class TestCanLoopback(PandaTestCase):
  def _exchange(self, sender, receiver, messages):
    start = time.monotonic()
    sender.can_send_many(messages)
    received = []
    while len(received) < len(messages) and time.monotonic() - start < 1:
      received.extend(receiver.can_recv())
    elapsed = time.monotonic() - start
    assert sorted(received) == sorted(messages), (len(messages), len(received), received[:8])
    if sender is self.p:
      assert sorted(self.p.can_recv()) == sorted((addr, dat, bus | 0x80) for addr, dat, bus in messages)
    return elapsed

  @panda_test(timeout=300)
  def test_physical_bulk(self):
    self.p.set_safety_mode(CarParams.SafetyModel.allOutput)
    for bus in range(3):
      self.p.set_can_speed_kbps(bus, 1000)
      self.up.set_can_speed_kbps(bus, 1000)
    for sender, receiver in ((self.p, self.up), (self.up, self.p)):
      # Bound each burst to the UP's receive capacity; never retry missing frames.
      for start in range(0, 40000, 8):
        messages = [(0x123, i.to_bytes(4, "little") + b"test", 0 if i < 10000 else i % 3) for i in range(start, start + 8)]
        self._exchange(sender, receiver, messages)
      assert not receiver.can_recv()

  @panda_test(timeout=120)
  def test_latency(self):
    self.p.set_safety_mode(CarParams.SafetyModel.allOutput)
    for speed in (50, 100, 125, 250, 500, 1000):
      for bus in range(3):
        self.p.set_can_speed_kbps(bus, speed)
        self.up.set_can_speed_kbps(bus, speed)
      for sender, receiver in ((self.p, self.up), (self.up, self.p)):
        for bus in range(3):
          latencies = [self._exchange(sender, receiver, [(0x123, i.to_bytes(4, "little") + b"test", bus)]) for i in range(20)]
          # End-to-end bounds include the UP RPC and CAN serialization time.
          wire_time = 135 / (speed * 1000)  # worst-case stuffed 8-byte standard frame
          assert max(latencies) < wire_time + 0.05, (speed, bus, max(latencies))
          assert sum(latencies) / len(latencies) < wire_time + 0.01, (speed, bus, latencies)

  def test_bitrates(self):
    p, up = self.p, self.up
    p.set_safety_mode(CarParams.SafetyModel.allOutput)
    flipped = p.health()["car_harness_status"] == Panda.HARNESS_STATUS_FLIPPED
    # The UP cannot clock 10 or 20 kbps; internal loopback covers those rates.
    for speed in (50, 100, 125, 250, 500, 1000):
      for bus in range(3):
        p.set_can_speed_kbps(bus, speed)
        up.set_can_speed_kbps(bus, speed)
      time.sleep(0.1)
      for bus in range(3):
        can_number = 2 - bus if flipped else bus
        message = (0x123, b"\xaa" * 8, bus)
        target = p.can_health(can_number)["total_tx_cnt"] + 600
        p.can_send_many([message] * 600, timeout=5000)
        # Time queued transmissions, excluding host packing and reading back frames.
        start = time.monotonic()
        remaining = target - p.can_health(can_number)["total_tx_cnt"]
        wire_time = remaining * 111 / (speed * 1000)  # 8-byte standard frames, excluding stuffing
        time.sleep(0.75 * wire_time)
        while p.can_health(can_number)["total_tx_cnt"] < target and time.monotonic() - start < 5:
          time.sleep(0.001)
        saturation = wire_time / (time.monotonic() - start)
        assert 0.8 < saturation < 1, (speed, bus, saturation)
        assert p.can_recv() == [(message[0], message[1], bus | 0x80)] * 600
        # The UP can drop frames under saturation; validate what it did receive.
        received = up.can_recv()
        assert received and all(m == message for m in received), (speed, bus, received)
        up.can_send_many([message])
        received = []
        deadline = time.monotonic() + 1
        while not received and time.monotonic() < deadline:
          received.extend(p.can_recv())
          time.sleep(0.001)
        assert received == [message], (speed, bus, received)

  @panda_test(expect_can_error=True)
  def test_gen2_loopback(self):
    p = self.p
    up = self.up
    def test(p_send, p_recv, address=None):
      for bus in range(4):
        obd = False
        if bus == 3:
          obd = True
          bus = 1

        # Clear buses
        clear_can_buffers(p_send)
        clear_can_buffers(p_recv)

        # Send a random string
        addr = address if address else random.randint(1, 2000)
        string = b"test" + os.urandom(4)
        p.set_obd(obd)
        # The UP exposes the OBD pair as bus 3 instead of switching bus 1.
        send_bus = 3 if obd and p_send is up else bus
        recv_bus = 3 if obd and p_recv is up else bus
        time.sleep(0.2)
        p_send.can_send(addr, string, send_bus)
        time.sleep(0.2)

        content = p_recv.can_recv()

        # Check amount of messages
        assert len(content) == 1

        # Check content
        assert content[0][0] == addr and content[0][1] == string

        # Check bus
        assert content[0][2] == recv_bus

        print("Bus:", bus, "address:", addr, "OBD:", obd, "OK")

    # Run tests in both directions
    p.set_safety_mode(CarParams.SafetyModel.allOutput)
    test(p, up)
    test(up, p)

    # Test extended frame address with ELM327 mode
    p.set_safety_mode(CarParams.SafetyModel.elm327)
    test(p, up, 0x18DB33F1)
    test(up, p, 0x18DB33F1)

    p.set_obd(False)

  def test_bulk_write(self):
    p = self.p
    p.set_safety_mode(CarParams.SafetyModel.allOutput)
    p.set_can_loopback(True)
    # Send far more than the panda's TX buffers can hold.
    NUM_MESSAGES_PER_BUS = 10000

    def flood_tx(panda):
      print('Sending!')
      msg = b"\xaa" * 8
      packet = []
      # start with many messages on a single bus (higher contention for single TX ring buffer)
      packet += [[0xaa, msg, 0]] * NUM_MESSAGES_PER_BUS
      # end with many messages on multiple buses
      packet += [[0xaa, msg, 0], [0xaa, msg, 1], [0xaa, msg, 2]] * NUM_MESSAGES_PER_BUS

      panda.can_send_many(packet, timeout=10000)
      print(f"Done sending {4 * NUM_MESSAGES_PER_BUS} messages!", time.monotonic())
      print(panda.health())

    # Drain looped-back frames and receipts while the writer fills the TX queues.
    with ThreadPoolExecutor(max_workers=1) as executor:
      sender = executor.submit(flood_tx, p)
      received = []
      deadline = time.monotonic() + 30
      while len(received) < 8 * NUM_MESSAGES_PER_BUS and time.monotonic() < deadline:
        received.extend(p.can_recv())
      sender.result()

    for bus, count in ((0, 2 * NUM_MESSAGES_PER_BUS), (1, NUM_MESSAGES_PER_BUS), (2, NUM_MESSAGES_PER_BUS)):
      for receipt in (0, 0x80):
        actual = received.count((0xaa, b"\xaa" * 8, bus | receipt))
        assert actual == count, (bus, receipt, actual, count)
    assert len(received) == 8 * NUM_MESSAGES_PER_BUS

  def test_message_integrity(self):
    p = self.p
    p.set_safety_mode(CarParams.SafetyModel.allOutput)
    p.set_can_loopback(True)
    for i in range(250):
      sent_msgs = defaultdict(set)
      for _ in range(random.randrange(10)):
        to_send = get_random_can_messages(random.randrange(100))
        for m in to_send:
          sent_msgs[m[2]].add((m[0], m[1]))
        p.can_send_many(to_send, timeout=0)

      start_time = time.monotonic()
      while time.monotonic() - start_time < 2 and any(len(sent_msgs[bus]) for bus in range(3)):
        recvd = p.can_recv()
        for msg in recvd:
          if msg[2] >= 128:
            k = (msg[0], bytes(msg[1]))
            bus = msg[2]-128
            assert k in sent_msgs[bus], f"message {k} was never sent on bus {bus}"
            sent_msgs[msg[2]-128].discard(k)

      # if a set isn't empty, messages got dropped
      for bus in range(3):
        assert not len(sent_msgs[bus]), f"loop {i}: bus {bus} missing {len(sent_msgs[bus])} messages"

    print("Got all messages intact")
