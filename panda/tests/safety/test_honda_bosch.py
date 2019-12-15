#!/usr/bin/env python3
import unittest
import libpandasafety_py  # pylint: disable=import-error
from panda import Panda

MAX_BRAKE = 255

class TestHondaSafety(unittest.TestCase):
  @classmethod
  def setUp(cls):
    cls.safety = libpandasafety_py.libpandasafety
    cls.safety.safety_set_mode(Panda.SAFETY_HONDA_BOSCH, 0)
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
    #has_relay = self.safety.get_hw_type() == 3  # black panda
    has_relay = self.safety.board_has_relay()
    bus_rdr_cam = 2 if has_relay else 1
    bus_rdr_car = 0 if has_relay else 2
    bus_pt = 1 if has_relay else 0

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
