from __future__ import print_function
import sys
import time
from helpers import time_many_sends, connect_wifi, test_white, panda_color_to_serial
from panda import Panda, PandaWifiStreaming
from nose.tools import timed, assert_equal, assert_less, assert_greater

@test_white
@panda_color_to_serial
def test_udp_doesnt_drop(serial=None):
  connect_wifi(serial)

  p = Panda(serial)
  p.set_safety_mode(Panda.SAFETY_ALLOUTPUT)
  p.set_can_loopback(True)

  pwifi = PandaWifiStreaming()
  while 1:
    if len(pwifi.can_recv()) == 0:
      break

  for msg_count in [1, 100]:
    saturation_pcts = []
    for i in range({1: 0x80, 100: 0x20}[msg_count]):
      pwifi.kick()

      speed = 500
      p.set_can_speed_kbps(0, speed)
      comp_kbps = time_many_sends(p, 0, pwifi, msg_count=msg_count, msg_id=0x100+i)
      saturation_pct = (comp_kbps/speed) * 100.0

      if msg_count == 1:
        sys.stdout.write(".")
        sys.stdout.flush()
      else:
        print("UDP WIFI loopback %d messages at speed %d, comp speed is %.2f, percent %.2f" % (msg_count, speed, comp_kbps, saturation_pct))
        assert_greater(saturation_pct, 20) #sometimes the wifi can be slow...
        assert_less(saturation_pct, 100)
        saturation_pcts.append(saturation_pct)
    if len(saturation_pcts) > 0:
      assert_greater(sum(saturation_pcts)/len(saturation_pcts), 60)

  time.sleep(5)
  usb_ok_cnt = 0
  REQ_USB_OK_CNT = 500
  st = time.time()
  msg_id = 0x1bb
  bus = 0
  last_missing_msg = 0
  while usb_ok_cnt < REQ_USB_OK_CNT and (time.time() - st) < 40:
    p.can_send(msg_id, "message", bus)
    time.sleep(0.01)
    r = [1]
    missing = True
    while len(r) > 0:
      r = p.can_recv()
      r = filter(lambda x: x[3] == bus and x[0] == msg_id, r)
      if len(r) > 0:
        missing = False
        usb_ok_cnt += len(r)
      if missing:
        last_missing_msg = time.time()
  et = time.time() - st
  last_missing_msg = last_missing_msg - st
  print("waited {} for panda to recv can on usb, {} msgs, last missing at {}".format(et, usb_ok_cnt, last_missing_msg))
  assert usb_ok_cnt >= REQ_USB_OK_CNT, "Unable to recv can on USB after UDP"
