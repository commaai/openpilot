
import os
import time
import random
from panda import Panda
from nose.tools import assert_equal, assert_less, assert_greater
from .helpers import time_many_sends, test_two_panda, test_two_black_panda, panda_type_to_serial, clear_can_buffers, panda_connect_and_init

@test_two_panda
@panda_type_to_serial
@panda_connect_and_init
def test_send_recv(p_send, p_recv):
  p_send.set_safety_mode(Panda.SAFETY_ALLOUTPUT)
  p_recv.set_safety_mode(Panda.SAFETY_ALLOUTPUT)
  p_send.set_can_loopback(False)
  p_recv.set_can_loopback(False)

  assert not p_send.legacy
  assert not p_recv.legacy

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

      comp_kbps = time_many_sends(p_send, bus, p_recv, two_pandas=True)

      saturation_pct = (comp_kbps/speed) * 100.0
      assert_greater(saturation_pct, 80)
      assert_less(saturation_pct, 100)

      print("two pandas bus {}, 100 messages at speed {:4d}, comp speed is {:7.2f}, percent {:6.2f}".format(bus, speed, comp_kbps, saturation_pct))

@test_two_panda
@panda_type_to_serial
@panda_connect_and_init
def test_latency(p_send, p_recv):
  p_send.set_safety_mode(Panda.SAFETY_ALLOUTPUT)
  p_recv.set_safety_mode(Panda.SAFETY_ALLOUTPUT)
  p_send.set_can_loopback(False)
  p_recv.set_can_loopback(False)

  assert not p_send.legacy
  assert not p_recv.legacy

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

      #clear can buffers
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

@test_two_black_panda
@panda_type_to_serial
@panda_connect_and_init
def test_black_loopback(panda0, panda1):
  # disable safety modes
  panda0.set_safety_mode(Panda.SAFETY_ALLOUTPUT)
  panda1.set_safety_mode(Panda.SAFETY_ALLOUTPUT)

  # disable loopback
  panda0.set_can_loopback(False)
  panda1.set_can_loopback(False)

  # clear stuff
  panda0.can_send_many([(0x1ba, 0, b"testmsg", 0)]*10)
  time.sleep(0.05)
  panda0.can_recv()
  panda1.can_recv()

  # test array (send bus, sender obd, reciever obd, expected busses)
  test_array = [
    (0, False, False, [0]),
    (1, False, False, [1]),
    (2, False, False, [2]),
    (0, False, True, [0, 1]),
    (1, False, True, []),
    (2, False, True, [2]),
    (0, True, False, [0]),
    (1, True, False, [0]),
    (2, True, False, [2]),
    (0, True, True, [0, 1]),
    (1, True, True, [0, 1]),
    (2, True, True, [2])
  ]

  # test functions
  def get_test_string():
      return b"test"+os.urandom(10)

  def _test_buses(send_panda, recv_panda, _test_array):
    for send_bus, send_obd, recv_obd, recv_buses in _test_array:
      print("\nSend bus:", send_bus, " Send OBD:", send_obd, " Recv OBD:", recv_obd)

      # set OBD on pandas
      send_panda.set_gmlan(True if send_obd else None)
      recv_panda.set_gmlan(True if recv_obd else None)

      # clear buffers
      clear_can_buffers(send_panda)
      clear_can_buffers(recv_panda)

      # send the characters
      at = random.randint(1, 2000)
      st = get_test_string()[0:8]
      send_panda.can_send(at, st, send_bus)
      time.sleep(0.1)

      # check for receive
      cans_echo = send_panda.can_recv()
      cans_loop = recv_panda.can_recv()

      loop_buses = []
      for loop in cans_loop:
        print("  Loop on bus", str(loop[3]))
        loop_buses.append(loop[3])
      if len(cans_loop) == 0:
        print("  No loop")

      # test loop buses
      recv_buses.sort()
      loop_buses.sort()
      assert recv_buses == loop_buses
      print("  TEST PASSED")
    print("\n")

  # test both orientations
  print("***************** TESTING (0 --> 1) *****************")
  _test_buses(panda0, panda1, test_array)
  print("***************** TESTING (1 --> 0) *****************")
  _test_buses(panda1, panda0, test_array)
