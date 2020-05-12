import os
import time
import random
import threading
from panda import Panda
from nose.tools import assert_equal, assert_less, assert_greater
from .helpers import panda_jungle, start_heartbeat_thread, reset_pandas, time_many_sends, test_all_pandas, test_all_gen2_pandas, clear_can_buffers, panda_connect_and_init

# Reset the pandas before running tests
def aaaa_reset_before_tests():
  reset_pandas()

@test_all_pandas
@panda_connect_and_init
def test_send_recv(p):
  def test(p_send, p_recv):
    p_send.set_can_loopback(False)
    p_recv.set_can_loopback(False)

    p_send.can_send_many([(0x1ba, 0, b"message", 0)]*2)
    time.sleep(0.05)
    p_recv.can_recv()
    p_send.can_recv()

    busses = [0,1,2]

    for bus in busses:
      for speed in [100, 250, 500, 750, 1000]:
        p_send.set_can_speed_kbps(bus, speed)
        p_recv.set_can_speed_kbps(bus, speed)
        time.sleep(0.05)

        clear_can_buffers(p_send)
        clear_can_buffers(p_recv)

        comp_kbps = time_many_sends(p_send, bus, p_recv, two_pandas=True)

        saturation_pct = (comp_kbps/speed) * 100.0
        assert_greater(saturation_pct, 80)
        assert_less(saturation_pct, 100)

        print("two pandas bus {}, 100 messages at speed {:4d}, comp speed is {:7.2f}, percent {:6.2f}".format(bus, speed, comp_kbps, saturation_pct))

  # Start heartbeat
  start_heartbeat_thread(p)

  # Set safety mode and power saving
  p.set_safety_mode(Panda.SAFETY_ALLOUTPUT)
  p.set_power_save(False)

  try:
    # Run tests in both directions
    test(p, panda_jungle)
    test(panda_jungle, p)
  except Exception as e:
    # Raise errors again, we don't want them to get lost
    raise e
  finally:
    # Set back to silent mode
    p.set_safety_mode(Panda.SAFETY_SILENT)

@test_all_pandas
@panda_connect_and_init
def test_latency(p):
  def test(p_send, p_recv):
    p_send.set_can_loopback(False)
    p_recv.set_can_loopback(False)

    p_send.set_can_speed_kbps(0, 100)
    p_recv.set_can_speed_kbps(0, 100)
    time.sleep(0.05)

    p_send.can_send_many([(0x1ba, 0, b"testmsg", 0)]*10)
    time.sleep(0.05)
    p_recv.can_recv()
    p_send.can_recv()

    busses = [0,1,2]

    for bus in busses:
      for speed in [100, 250, 500, 750, 1000]:
        p_send.set_can_speed_kbps(bus, speed)
        p_recv.set_can_speed_kbps(bus, speed)
        time.sleep(0.1)

        # clear can buffers
        clear_can_buffers(p_send)
        clear_can_buffers(p_recv)

        latencies = []
        comp_kbps_list = []
        saturation_pcts = []

        num_messages = 100

        for i in range(num_messages):
          st = time.time()
          p_send.can_send(0x1ab, b"message", bus)
          r = []
          while len(r) < 1 and (time.time() - st) < 5:
            r = p_recv.can_recv()
          et = time.time()
          r_echo = []
          while len(r_echo) < 1 and (time.time() - st) < 10:
            r_echo = p_send.can_recv()

          if len(r) == 0 or len(r_echo) == 0:
            print("r: {}, r_echo: {}".format(r, r_echo))

          assert_equal(len(r),1)
          assert_equal(len(r_echo),1)

          et = (et - st)*1000.0
          comp_kbps = (1+11+1+1+1+4+8*8+15+1+1+1+7) / et
          latency = et - ((1+11+1+1+1+4+8*8+15+1+1+1+7) / speed)

          assert_less(latency, 5.0)

          saturation_pct = (comp_kbps/speed) * 100.0
          latencies.append(latency)
          comp_kbps_list.append(comp_kbps)
          saturation_pcts.append(saturation_pct)

        average_latency = sum(latencies)/num_messages
        assert_less(average_latency, 1.0)
        average_comp_kbps = sum(comp_kbps_list)/num_messages
        average_saturation_pct = sum(saturation_pcts)/num_messages

        print("two pandas bus {}, {} message average at speed {:4d}, latency is {:5.3f}ms, comp speed is {:7.2f}, percent {:6.2f}"\
              .format(bus, num_messages, speed, average_latency, average_comp_kbps, average_saturation_pct))

  # Start heartbeat
  start_heartbeat_thread(p)

  # Set safety mode and power saving
  p.set_safety_mode(Panda.SAFETY_ALLOUTPUT)
  p.set_power_save(False)

  try:
    # Run tests in both directions
    test(p, panda_jungle)
    test(panda_jungle, p)
  except Exception as e:
    # Raise errors again, we don't want them to get lost
    raise e
  finally:
    # Set back to silent mode
    p.set_safety_mode(Panda.SAFETY_SILENT)


@test_all_gen2_pandas
@panda_connect_and_init
def test_gen2_loopback(p):
  def test(p_send, p_recv):
    for bus in range(4):
      obd = False
      if bus == 3:
        obd = True
        bus = 1
      
      # Clear buses
      clear_can_buffers(p_send)
      clear_can_buffers(p_recv)

      # Send a random string
      addr = random.randint(1, 2000)
      string = b"test"+os.urandom(4)
      p_send.set_obd(obd)
      p_recv.set_obd(obd)
      time.sleep(0.2)
      p_send.can_send(addr, string, bus)
      time.sleep(0.2)

      content = p_recv.can_recv()

      # Check amount of messages
      assert len(content) == 1

      # Check content
      assert content[0][0] == addr and content[0][2] == string
      
      # Check bus
      assert content[0][3] == bus

      print("Bus:", bus, "OBD:", obd, "OK")

  # Start heartbeat
  start_heartbeat_thread(p)

  # Set safety mode and power saving
  p.set_safety_mode(Panda.SAFETY_ALLOUTPUT)
  p.set_power_save(False)

  try:
    # Run tests in both directions
    test(p, panda_jungle)
    test(panda_jungle, p)
  except Exception as e:
    # Raise errors again, we don't want them to get lost
    raise e
  finally:
    # Set back to silent mode
    p.set_safety_mode(Panda.SAFETY_SILENT)

@test_all_pandas
@panda_connect_and_init
def test_bulk_write(p):
  # The TX buffers on pandas is 0x100 in length.
  NUM_MESSAGES_PER_BUS = 10000

  def flood_tx(panda):
    print('Sending!')
    msg = b"\xaa"*4
    packet = [[0xaa, None, msg, 0], [0xaa, None, msg, 1], [0xaa, None, msg, 2]] * NUM_MESSAGES_PER_BUS

    # Disable timeout
    panda.can_send_many(packet, timeout=0)
    print(f"Done sending {3*NUM_MESSAGES_PER_BUS} messages!")
  
  # Start heartbeat
  start_heartbeat_thread(p)

  # Set safety mode and power saving
  p.set_safety_mode(Panda.SAFETY_ALLOUTPUT)
  p.set_power_save(False)
  
  # Start transmisson
  threading.Thread(target=flood_tx, args=(p,)).start()

  # Receive as much as we can in a few second time period
  rx = []
  old_len = 0
  start_time = time.time()
  while time.time() - start_time < 2 or len(rx) > old_len:
    old_len = len(rx)
    rx.extend(panda_jungle.can_recv())
  print(f"Received {len(rx)} messages")

  # All messages should have been received
  if len(rx) != 3*NUM_MESSAGES_PER_BUS:
    Exception("Did not receive all messages!")

  # Set back to silent mode
  p.set_safety_mode(Panda.SAFETY_SILENT)
