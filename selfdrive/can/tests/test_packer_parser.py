#!/usr/bin/env python3

import unittest

from selfdrive.can.parser import CANParser
from selfdrive.can.packer import CANPacker

from selfdrive.boardd.boardd import can_list_to_can_capnp

from selfdrive.car.honda.interface import CarInterface as HondaInterface
from selfdrive.car.honda.values import CAR as HONDA_CAR
from selfdrive.car.honda.values import DBC as HONDA_DBC

from selfdrive.car.subaru.interface import CarInterface as SubaruInterface
from selfdrive.car.subaru.values import CAR as SUBARU_CAR
from selfdrive.car.subaru.values import DBC as SUBARU_DBC

class TestCanParserPacker(unittest.TestCase):
  def test_civic(self):
    CP = HondaInterface.get_params(HONDA_CAR.CIVIC)

    signals = [
      ("STEER_TORQUE", "STEERING_CONTROL", 0),
      ("STEER_TORQUE_REQUEST", "STEERING_CONTROL", 0),
    ]
    checks = []

    parser = CANParser(HONDA_DBC[CP.carFingerprint]['pt'], signals, checks, 0)
    packer = CANPacker(HONDA_DBC[CP.carFingerprint]['pt'])

    idx = 0

    for steer in range(0, 255):
      for active in [1, 0]:
        values = {
          "STEER_TORQUE": steer,
          "STEER_TORQUE_REQUEST": active,
        }

        msgs = packer.make_can_msg("STEERING_CONTROL", 0, values, idx)
        bts = can_list_to_can_capnp([msgs])

        parser.update_string(bts)

        self.assertAlmostEqual(parser.vl["STEERING_CONTROL"]["STEER_TORQUE"], steer)
        self.assertAlmostEqual(parser.vl["STEERING_CONTROL"]["STEER_TORQUE_REQUEST"], active)
        self.assertAlmostEqual(parser.vl["STEERING_CONTROL"]["COUNTER"], idx % 4)

        idx += 1

  def test_subaru(self):
    # Subuaru is little endian
    CP = SubaruInterface.get_params(SUBARU_CAR.IMPREZA)

    signals = [
      ("Counter", "ES_LKAS", 0),
      ("LKAS_Output", "ES_LKAS", 0),
      ("LKAS_Request", "ES_LKAS", 0),
      ("SET_1", "ES_LKAS", 0),

    ]

    checks = []

    parser = CANParser(SUBARU_DBC[CP.carFingerprint]['pt'], signals, checks, 0)
    packer = CANPacker(SUBARU_DBC[CP.carFingerprint]['pt'])

    idx = 0

    for steer in range(0, 255):
      for active in [1, 0]:
        values = {
          "Counter": idx,
          "LKAS_Output": steer,
          "LKAS_Request": active,
          "SET_1": 1
        }

        msgs = packer.make_can_msg("ES_LKAS", 0, values)
        bts = can_list_to_can_capnp([msgs])
        parser.update_string(bts)

        self.assertAlmostEqual(parser.vl["ES_LKAS"]["LKAS_Output"], steer)
        self.assertAlmostEqual(parser.vl["ES_LKAS"]["LKAS_Request"], active)
        self.assertAlmostEqual(parser.vl["ES_LKAS"]["SET_1"], 1)
        self.assertAlmostEqual(parser.vl["ES_LKAS"]["Counter"], idx % 16)

        idx += 1


if __name__ == "__main__":
  unittest.main()
