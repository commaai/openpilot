from __future__ import print_function
import time
from panda import Panda
from nose.tools import assert_equal, assert_less, assert_greater
from helpers import time_many_sends, test_two_panda, panda_color_to_serial

@test_two_panda
@panda_color_to_serial
def test_send_recv(serial_sender=None, serial_reciever=None):
  p_send = Panda(serial_sender)
  p_recv = Panda(serial_reciever)

  p_send.set_safety_mode(Panda.SAFETY_ALLOUTPUT)
  p_send.set_can_loopback(False)

  # send heartbeat
  p_send.send_heartbeat()

  p_recv.set_can_loopback(False)

  assert not p_send.legacy
  assert not p_recv.legacy

  p_send.can_send_many([(0x1ba, 0, "message", 0)]*2)
  time.sleep(0.05)
  p_recv.can_recv()
  p_send.can_recv()

  busses = [0,1,2]

  for bus in busses:
    for speed in [100, 250, 500, 750, 1000]:
      # send heartbeat
      p_send.send_heartbeat()

      p_send.set_can_speed_kbps(bus, speed)
      p_recv.set_can_speed_kbps(bus, speed)
      time.sleep(0.05)

      comp_kbps = time_many_sends(p_send, bus, p_recv, two_pandas=True)

      saturation_pct = (comp_kbps/speed) * 100.0
      assert_greater(saturation_pct, 80)
      assert_less(saturation_pct, 100)

      print("two pandas bus {}, 100 messages at speed {:4d}, comp speed is {:7.2f}, percent {:6.2f}".format(bus, speed, comp_kbps, saturation_pct))

@test_two_panda
@panda_color_to_serial
def test_latency(serial_sender=None, serial_reciever=None):
  p_send = Panda(serial_sender)
  p_recv = Panda(serial_reciever)

  # send heartbeat
  p_send.send_heartbeat()
  p_recv.send_heartbeat()

  p_send.set_safety_mode(Panda.SAFETY_ALLOUTPUT)
  p_send.set_can_loopback(False)

  p_recv.set_can_loopback(False)

  assert not p_send.legacy
  assert not p_recv.legacy

  p_send.set_can_speed_kbps(0, 100)
  p_recv.set_can_speed_kbps(0, 100)
  time.sleep(0.05)

  p_send.can_send_many([(0x1ba, 0, "testmsg", 0)]*10)
  time.sleep(0.05)
  p_recv.can_recv()
  p_send.can_recv()

  # send heartbeat
  p_send.send_heartbeat()  
  p_recv.send_heartbeat()

  busses = [0,1,2]

  for bus in busses:
    for speed in [100, 250, 500, 750, 1000]:
      # send heartbeat
      p_send.send_heartbeat() 
      p_recv.send_heartbeat()

      p_send.set_can_speed_kbps(bus, speed)
      p_recv.set_can_speed_kbps(bus, speed)
      time.sleep(0.1)
      #clear can buffers
      r = [1]
      while len(r) > 0:
        r = p_send.can_recv()
      r = [1]
      while len(r) > 0:
        r = p_recv.can_recv()
      time.sleep(0.05)

      latencies = []
      comp_kbps_list = []
      saturation_pcts = []

      num_messages = 100

      for i in range(num_messages):
        st = time.time()
        p_send.can_send(0x1ab, "message", bus)
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
