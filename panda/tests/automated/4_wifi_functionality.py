from __future__ import print_function
import time
from panda import Panda
from helpers import time_many_sends, connect_wifi, test_white, panda_color_to_serial
from nose.tools import timed, assert_equal, assert_less, assert_greater

@test_white
@panda_color_to_serial
def test_get_serial_wifi(serial=None):
  connect_wifi(serial)

  p = Panda("WIFI")
  print(p.get_serial())

@test_white
@panda_color_to_serial
def test_throughput(serial=None):
  connect_wifi(serial)
  p = Panda(serial)

  # enable output mode
  p.set_safety_mode(Panda.SAFETY_ALLOUTPUT)

  # send heartbeat
  p.send_heartbeat()

  # enable CAN loopback mode
  p.set_can_loopback(True)

  p = Panda("WIFI")

  for speed in [100,250,500,750,1000]:
    # send heartbeat
    p.send_heartbeat()

    # set bus 0 speed to speed
    p.set_can_speed_kbps(0, speed)
    time.sleep(0.1)

    comp_kbps = time_many_sends(p, 0)

    # bit count from https://en.wikipedia.org/wiki/CAN_bus
    saturation_pct = (comp_kbps/speed) * 100.0
    #assert_greater(saturation_pct, 80)
    #assert_less(saturation_pct, 100)

    print("WIFI loopback 100 messages at speed %d, comp speed is %.2f, percent %.2f" % (speed, comp_kbps, saturation_pct))

@test_white
@panda_color_to_serial
def test_recv_only(serial=None):
  connect_wifi(serial)
  p = Panda(serial)
  p.set_safety_mode(Panda.SAFETY_ALLOUTPUT)

  # send heartbeat
  p.send_heartbeat()

  p.set_can_loopback(True)
  pwifi = Panda("WIFI")

  # TODO: msg_count=1000 drops packets, is this fixable?
  for msg_count in [10,100,200]:
    # send heartbeat
    p.send_heartbeat()

    speed = 500
    p.set_can_speed_kbps(0, speed)
    comp_kbps = time_many_sends(p, 0, pwifi, msg_count)
    saturation_pct = (comp_kbps/speed) * 100.0

    print("HT WIFI loopback %d messages at speed %d, comp speed is %.2f, percent %.2f" % (msg_count, speed, comp_kbps, saturation_pct))
