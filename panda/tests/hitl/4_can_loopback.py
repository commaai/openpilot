import os
import time
import pytest
import random
import threading
from flaky import flaky
from collections import defaultdict

from opendbc.car.structs import CarParams
from panda.tests.hitl.helpers import time_many_sends, get_random_can_messages, clear_can_buffers

@flaky(max_runs=3, min_passes=1)
@pytest.mark.timeout(35)
def test_send_recv(p, panda_jungle):
  def test(p_send, p_recv):
    for bus in (0, 1, 2):
      for speed in (10, 20, 50, 100, 125, 250, 500, 1000):
        clear_can_buffers(p_send, speed)
        clear_can_buffers(p_recv, speed)

        comp_kbps = time_many_sends(p_send, bus, p_recv, two_pandas=True)

        saturation_pct = (comp_kbps / speed) * 100.0
        assert 80 < saturation_pct < 100

        print(f"two pandas bus {bus}, 100 messages at speed {speed:4d}, comp speed is {comp_kbps:7.2f}, {saturation_pct:6.2f}%")

  # Run tests in both directions
  p.set_safety_mode(CarParams.SafetyModel.allOutput)
  test(p, panda_jungle)
  test(panda_jungle, p)


@flaky(max_runs=6, min_passes=1)
@pytest.mark.timeout(30)
def test_latency(p, panda_jungle):
  def test(p_send, p_recv):
    for bus in (0, 1, 2):
      for speed in (10, 20, 50, 100, 125, 250, 500, 1000):
        clear_can_buffers(p_send, speed)
        clear_can_buffers(p_recv, speed)

        latencies = []
        comp_kbps_list = []
        saturation_pcts = []

        num_messages = 100

        for _ in range(num_messages):
          st = time.monotonic()
          p_send.can_send(0x1ab, b"message", bus)
          r = []
          while len(r) < 1 and (time.monotonic() - st) < 5:
            r = p_recv.can_recv()
          et = time.monotonic()
          r_echo = []
          while len(r_echo) < 1 and (time.monotonic() - st) < 10:
            r_echo = p_send.can_recv()

          if len(r) == 0 or len(r_echo) == 0:
            print(f"r: {r}, r_echo: {r_echo}")

          assert len(r) == 1
          assert len(r_echo) == 1

          et = (et - st) * 1000.0
          comp_kbps = (1 + 11 + 1 + 1 + 1 + 4 + 8 * 8 + 15 + 1 + 1 + 1 + 7) / et
          latency = et - ((1 + 11 + 1 + 1 + 1 + 4 + 8 * 8 + 15 + 1 + 1 + 1 + 7) / speed)

          assert latency < 5.0

          saturation_pct = (comp_kbps / speed) * 100.0
          latencies.append(latency)
          comp_kbps_list.append(comp_kbps)
          saturation_pcts.append(saturation_pct)

        average_latency = sum(latencies) / num_messages
        assert average_latency < 1.0
        average_comp_kbps = sum(comp_kbps_list) / num_messages
        average_saturation_pct = sum(saturation_pcts) / num_messages

        print("two pandas bus {}, {} message average at speed {:4d}, latency is {:5.3f}ms, comp speed is {:7.2f}, percent {:6.2f}"
              .format(bus, num_messages, speed, average_latency, average_comp_kbps, average_saturation_pct))

  # Run tests in both directions
  p.set_safety_mode(CarParams.SafetyModel.allOutput)
  test(p, panda_jungle)
  test(panda_jungle, p)


@pytest.mark.panda_expect_can_error
def test_gen2_loopback(p, panda_jungle):
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
      p_send.set_obd(obd)
      p_recv.set_obd(obd)
      time.sleep(0.2)
      p_send.can_send(addr, string, bus)
      time.sleep(0.2)

      content = p_recv.can_recv()

      # Check amount of messages
      assert len(content) == 1

      # Check content
      assert content[0][0] == addr and content[0][1] == string

      # Check bus
      assert content[0][2] == bus

      print("Bus:", bus, "address:", addr, "OBD:", obd, "OK")

  # Run tests in both directions
  p.set_safety_mode(CarParams.SafetyModel.allOutput)
  test(p, panda_jungle)
  test(panda_jungle, p)

  # Test extended frame address with ELM327 mode
  p.set_safety_mode(CarParams.SafetyModel.elm327)
  test(p, panda_jungle, 0x18DB33F1)
  test(panda_jungle, p, 0x18DB33F1)

  # TODO: why it's not being reset by fixtures reinit?
  p.set_obd(False)
  panda_jungle.set_obd(False)

def test_bulk_write(p, panda_jungle):
  # The TX buffers on pandas is 0x100 in length.
  NUM_MESSAGES_PER_BUS = 10000

  def flood_tx(panda):
    print('Sending!')
    msg = b"\xaa" * 8
    packet = []
    # start with many messages on a single bus (higher contention for single TX ring buffer)
    packet += [[0xaa, msg, 0]] * NUM_MESSAGES_PER_BUS
    # end with many messages on multiple buses
    packet += [[0xaa, msg, 0], [0xaa, msg, 1], [0xaa, msg, 2]] * NUM_MESSAGES_PER_BUS

    # Disable timeout
    panda.set_safety_mode(CarParams.SafetyModel.allOutput)
    panda.can_send_many(packet, timeout=0)
    print(f"Done sending {4 * NUM_MESSAGES_PER_BUS} messages!", time.monotonic())
    print(panda.health())

  # Start transmisson
  threading.Thread(target=flood_tx, args=(p,)).start()

  # Receive as much as we can in a few second time period
  rx = []
  old_len = 0
  start_time = time.monotonic()
  while time.monotonic() - start_time < 5 or len(rx) > old_len:
    old_len = len(rx)
    rx.extend(panda_jungle.can_recv())
  print(f"Received {len(rx)} messages", time.monotonic())

  # All messages should have been received
  if len(rx) != 4 * NUM_MESSAGES_PER_BUS:
    raise Exception("Did not receive all messages!")

def test_message_integrity(p):
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
