#!/usr/bin/env python2
import unittest
import numpy as np
import libpandasafety_py

MAX_BRAKE = 255

class TestHondaSafety(unittest.TestCase):
  @classmethod
  def setUp(cls):
    cls.safety = libpandasafety_py.libpandasafety
    cls.safety.safety_set_mode(4, 0)
    cls.safety.init_tests_honda()

  def _send_msg(self, bus, addr, length):
    to_send = libpandasafety_py.ffi.new('CAN_FIFOMailBox_TypeDef *')
    to_send[0].RIR = addr << 21
    to_send[0].RDTR = length
    to_send[0].RDTR = bus << 4

    return to_send

  def test_fwd_hook(self):
    buss = range(0x0, 0x3)
    msgs = range(0x1, 0x800)
    is_panda_black = self.safety.get_hw_type() == 3  # black panda
    bus_rdr_cam = 2 if is_panda_black else 1
    bus_rdr_car = 0 if is_panda_black else 2
    bus_pt = 1 if is_panda_black else 0

    blocked_msgs = [0xE4, 0x33D]
    for b in buss:
      for m in msgs:
        if b == bus_pt:
          fwd_bus = -1
        elif b == bus_rdr_cam:
          fwd_bus = -1 if m in blocked_msgs else bus_rdr_car
        elif b == bus_rdr_car:
          fwd_bus = bus_rdr_cam

        # assume len 8
        self.assertEqual(fwd_bus, self.safety.safety_fwd_hook(b, self._send_msg(b, m, 8)))


if __name__ == "__main__":
  unittest.main()
