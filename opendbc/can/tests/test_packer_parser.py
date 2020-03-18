#!/usr/bin/env python3

import unittest

from opendbc.can.parser import CANParser
from opendbc.can.packer import CANPacker
import cereal.messaging as messaging


# Python implementation so we don't have to depend on boardd
def can_list_to_can_capnp(can_msgs, msgtype='can'):
  dat = messaging.new_message()
  dat.init(msgtype, len(can_msgs))

  for i, can_msg in enumerate(can_msgs):
    if msgtype == 'sendcan':
      cc = dat.sendcan[i]
    else:
      cc = dat.can[i]

    cc.address = can_msg[0]
    cc.busTime = can_msg[1]
    cc.dat = bytes(can_msg[2])
    cc.src = can_msg[3]

  return dat.to_bytes()


class TestCanParserPacker(unittest.TestCase):
  def test_civic(self):

    dbc_file = "honda_civic_touring_2016_can_generated"

    signals = [
      ("STEER_TORQUE", "STEERING_CONTROL", 0),
      ("STEER_TORQUE_REQUEST", "STEERING_CONTROL", 0),
    ]
    checks = []

    parser = CANParser(dbc_file, signals, checks, 0)
    packer = CANPacker(dbc_file)

    idx = 0

    for steer in range(-256, 255):
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

    dbc_file = "subaru_global_2017"

    signals = [
      ("Counter", "ES_LKAS", 0),
      ("LKAS_Output", "ES_LKAS", 0),
      ("LKAS_Request", "ES_LKAS", 0),
      ("SET_1", "ES_LKAS", 0),

    ]

    checks = []

    parser = CANParser(dbc_file, signals, checks, 0)
    packer = CANPacker(dbc_file)

    idx = 0

    for steer in range(-256, 255):
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
