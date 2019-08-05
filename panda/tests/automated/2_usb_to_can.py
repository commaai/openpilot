from __future__ import print_function
import os
import sys
import time
from panda import Panda
from nose.tools import assert_equal, assert_less, assert_greater
from helpers import time_many_sends, connect_wo_esp, test_white_and_grey, panda_color_to_serial

SPEED_NORMAL = 500
SPEED_GMLAN = 33.3

@test_white_and_grey
@panda_color_to_serial
def test_can_loopback(serial=None):
  p = connect_wo_esp(serial)

  # enable output mode
  p.set_safety_mode(Panda.SAFETY_ALLOUTPUT)

  # enable CAN loopback mode
  p.set_can_loopback(True)

  if p.legacy:
    busses = [0,1]
  else:
    busses = [0,1,2]

  for bus in busses:
    # send heartbeat
    p.send_heartbeat()

    # set bus 0 speed to 250
    p.set_can_speed_kbps(bus, 250)

    # send a message on bus 0
    p.can_send(0x1aa, "message", bus)

    # confirm receive both on loopback and send receipt
    time.sleep(0.05)
    r = p.can_recv()
    sr = filter(lambda x: x[3] == 0x80 | bus, r)
    lb = filter(lambda x: x[3] == bus, r)
    assert len(sr) == 1
    assert len(lb) == 1

    # confirm data is correct
    assert 0x1aa == sr[0][0] == lb[0][0]
    assert "message" == sr[0][2] == lb[0][2]

@test_white_and_grey
@panda_color_to_serial
def test_safety_nooutput(serial=None):
  p = connect_wo_esp(serial)

  # enable output mode
  p.set_safety_mode(Panda.SAFETY_NOOUTPUT)

  # send heartbeat
  p.send_heartbeat()

  # enable CAN loopback mode
  p.set_can_loopback(True)

  # send a message on bus 0
  p.can_send(0x1aa, "message", 0)

  # confirm receive nothing
  time.sleep(0.05)
  r = p.can_recv()
  assert len(r) == 0

@test_white_and_grey
@panda_color_to_serial
def test_reliability(serial=None):
  p = connect_wo_esp(serial)

  LOOP_COUNT = 100
  MSG_COUNT = 100

  # enable output mode
  p.set_safety_mode(Panda.SAFETY_ALLOUTPUT)
  p.set_can_loopback(True)
  p.set_can_speed_kbps(0, 1000)

  # send heartbeat
  p.send_heartbeat()

  addrs = range(100, 100+MSG_COUNT)
  ts = [(j, 0, "\xaa"*8, 0) for j in addrs]

  # 100 loops
  for i in range(LOOP_COUNT):
    # send heartbeat
    p.send_heartbeat()

    st = time.time()

    p.can_send_many(ts)

    r = []
    while len(r) < 200 and (time.time() - st) < 0.5:
      r.extend(p.can_recv())

    sent_echo = filter(lambda x: x[3] == 0x80, r)
    loopback_resp = filter(lambda x: x[3] == 0, r)

    assert_equal(sorted(map(lambda x: x[0], loopback_resp)), addrs)
    assert_equal(sorted(map(lambda x: x[0], sent_echo)), addrs)
    assert_equal(len(r), 200)

    # take sub 20ms
    et = (time.time()-st)*1000.0
    assert_less(et, 20)

    sys.stdout.write("P")
    sys.stdout.flush()

@test_white_and_grey
@panda_color_to_serial
def test_throughput(serial=None):
  p = connect_wo_esp(serial)

  # enable output mode
  p.set_safety_mode(Panda.SAFETY_ALLOUTPUT)

  # send heartbeat
  p.send_heartbeat()

  # enable CAN loopback mode
  p.set_can_loopback(True)

  for speed in [100,250,500,750,1000]:
    # set bus 0 speed to speed
    p.set_can_speed_kbps(0, speed)
    time.sleep(0.05)

    # send heartbeat
    p.send_heartbeat()

    comp_kbps = time_many_sends(p, 0)

    # bit count from https://en.wikipedia.org/wiki/CAN_bus
    saturation_pct = (comp_kbps/speed) * 100.0
    assert_greater(saturation_pct, 80)
    assert_less(saturation_pct, 100)

    print("loopback 100 messages at speed %d, comp speed is %.2f, percent %.2f" % (speed, comp_kbps, saturation_pct))

@test_white_and_grey
@panda_color_to_serial
def test_gmlan(serial=None):
  p = connect_wo_esp(serial)

  if p.legacy:
    return

  # enable output mode
  p.set_safety_mode(Panda.SAFETY_ALLOUTPUT)

  # send heartbeat
  p.send_heartbeat()

  # enable CAN loopback mode
  p.set_can_loopback(True)

  p.set_can_speed_kbps(1, SPEED_NORMAL)
  p.set_can_speed_kbps(2, SPEED_NORMAL)
  p.set_can_speed_kbps(3, SPEED_GMLAN)

  # set gmlan on CAN2
  for bus in [Panda.GMLAN_CAN2, Panda.GMLAN_CAN3, Panda.GMLAN_CAN2, Panda.GMLAN_CAN3]:
    # send heartbeat
    p.send_heartbeat()

    p.set_gmlan(bus)
    comp_kbps_gmlan = time_many_sends(p, 3)
    assert_greater(comp_kbps_gmlan, 0.8 * SPEED_GMLAN)
    assert_less(comp_kbps_gmlan, 1.0 * SPEED_GMLAN)

    p.set_gmlan(None)
    comp_kbps_normal = time_many_sends(p, bus)
    assert_greater(comp_kbps_normal, 0.8 * SPEED_NORMAL)
    assert_less(comp_kbps_normal, 1.0 * SPEED_NORMAL)

    print("%d: %.2f kbps vs %.2f kbps" % (bus, comp_kbps_gmlan, comp_kbps_normal))

@test_white_and_grey
@panda_color_to_serial
def test_gmlan_bad_toggle(serial=None):
  p = connect_wo_esp(serial)

  if p.legacy:
    return

  # enable output mode
  p.set_safety_mode(Panda.SAFETY_ALLOUTPUT)

  # send heartbeat
  p.send_heartbeat()

  # enable CAN loopback mode
  p.set_can_loopback(True)

  # GMLAN_CAN2
  for bus in [Panda.GMLAN_CAN2, Panda.GMLAN_CAN3]:
    # send heartbeat
    p.send_heartbeat()

    p.set_gmlan(bus)
    comp_kbps_gmlan = time_many_sends(p, 3)
    assert_greater(comp_kbps_gmlan, 0.6 * SPEED_GMLAN)
    assert_less(comp_kbps_gmlan, 1.0 * SPEED_GMLAN)

  # normal
  for bus in [Panda.GMLAN_CAN2, Panda.GMLAN_CAN3]:
    # send heartbeat
    p.send_heartbeat()

    p.set_gmlan(None)
    comp_kbps_normal = time_many_sends(p, bus)
    assert_greater(comp_kbps_normal, 0.6 * SPEED_NORMAL)
    assert_less(comp_kbps_normal, 1.0 * SPEED_NORMAL)


# this will fail if you have hardware serial connected
@test_white_and_grey
@panda_color_to_serial
def test_serial_debug(serial=None):
  p = connect_wo_esp(serial)
  junk = p.serial_read(Panda.SERIAL_DEBUG)
  p.call_control_api(0xc0)
  assert(p.serial_read(Panda.SERIAL_DEBUG).startswith("can "))
