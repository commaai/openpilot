#!/usr/bin/env python3
import unittest
import random

import cereal.messaging as messaging
from opendbc.can.parser import CANParser
from opendbc.can.packer import CANPacker

# Python implementation so we don't have to depend on boardd
def can_list_to_can_capnp(can_msgs, msgtype='can', logMonoTime=None):
  dat = messaging.new_message()
  dat.init(msgtype, len(can_msgs))

  if logMonoTime is not None:
    dat.logMonoTime = logMonoTime

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
  def test_packer(self):
    packer = CANPacker("test")

    for b in range(6):
      for i in range(256):
        values = {"COUNTER": i}
        addr, _, dat, bus = packer.make_can_msg("CAN_FD_MESSAGE", b, values)
        self.assertEqual(addr, 245)
        self.assertEqual(bus, b)
        self.assertEqual(dat[0], i)

  def test_packer_parser(self):

    signals = [
      ("COUNTER", "STEERING_CONTROL"),
      ("CHECKSUM", "STEERING_CONTROL"),
      ("STEER_TORQUE", "STEERING_CONTROL"),
      ("STEER_TORQUE_REQUEST", "STEERING_CONTROL"),

      ("COUNTER", "CAN_FD_MESSAGE"),
      ("64_BIT_LE", "CAN_FD_MESSAGE"),
      ("64_BIT_BE", "CAN_FD_MESSAGE"),
      ("SIGNED", "CAN_FD_MESSAGE"),
    ]
    checks = [("STEERING_CONTROL", 0), ("CAN_FD_MESSAGE", 0)]

    packer = CANPacker("test")
    parser = CANParser("test", signals, checks, 0)

    idx = 0

    for steer in range(-256, 255):
      for active in (1, 0):
        v1 = {
          "STEER_TORQUE": steer,
          "STEER_TORQUE_REQUEST": active,
        }
        m1 = packer.make_can_msg("STEERING_CONTROL", 0, v1, idx)

        v2 = {
          "COUNTER": idx % 256,
          "SIGNED": steer,
          "64_BIT_LE": random.randint(0, 100),
          "64_BIT_BE": random.randint(0, 100),
        }
        m2 = packer.make_can_msg("CAN_FD_MESSAGE", 0, v2)

        bts = can_list_to_can_capnp([m1, m2])
        parser.update_string(bts)

        for key, val in v1.items():
          self.assertAlmostEqual(parser.vl["STEERING_CONTROL"][key], val)

        for key, val in v2.items():
          self.assertAlmostEqual(parser.vl["CAN_FD_MESSAGE"][key], val)

        # also check address
        for sig in ("STEER_TORQUE", "STEER_TORQUE_REQUEST", "COUNTER", "CHECKSUM"):
          self.assertEqual(parser.vl["STEERING_CONTROL"][sig], parser.vl[228][sig])

        idx += 1

  def test_scale_offset(self):
    """Test that both scale and offset are correctly preserved"""
    dbc_file = "honda_civic_touring_2016_can_generated"

    signals = [
      ("USER_BRAKE", "VSA_STATUS"),
    ]
    checks = [("VSA_STATUS", 50)]

    parser = CANParser(dbc_file, signals, checks, 0)
    packer = CANPacker(dbc_file)

    idx = 0
    for brake in range(0, 100):
      values = {"USER_BRAKE": brake}
      msgs = packer.make_can_msg("VSA_STATUS", 0, values, idx)
      bts = can_list_to_can_capnp([msgs])

      parser.update_string(bts)

      self.assertAlmostEqual(parser.vl["VSA_STATUS"]["USER_BRAKE"], brake)
      idx += 1

  def test_subaru(self):
    # Subuaru is little endian

    dbc_file = "subaru_global_2017_generated"

    signals = [
      ("Counter", "ES_LKAS"),
      ("LKAS_Output", "ES_LKAS"),
      ("LKAS_Request", "ES_LKAS"),
      ("SET_1", "ES_LKAS"),
    ]
    checks = [("ES_LKAS", 50)]

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

  def test_bus_timeout(self):
    """Test CAN bus timeout detection"""
    dbc_file = "honda_civic_touring_2016_can_generated"

    freq = 100
    checks = [("VSA_STATUS", freq), ("STEER_MOTOR_TORQUE", freq/2)]

    parser = CANParser(dbc_file, [], checks, 0)
    packer = CANPacker(dbc_file)

    i = 0
    def send_msg(blank=False):
      nonlocal i
      i += 1
      t = i*((1 / freq) * 1e9)

      if blank:
        msgs = []
      else:
        msgs = [packer.make_can_msg("VSA_STATUS", 0, {}), ]

      can = can_list_to_can_capnp(msgs, logMonoTime=t)
      parser.update_strings([can, ])

    # all good, no timeout
    for _ in range(1000):
      send_msg()
      self.assertFalse(parser.bus_timeout, str(_))

    # timeout after 10 blank msgs
    for n in range(200):
      send_msg(blank=True)
      self.assertEqual(n >= 10, parser.bus_timeout)

    # no timeout immediately after seen again
    send_msg()
    self.assertFalse(parser.bus_timeout)


  def test_updated(self):
    """Test updated value dict"""
    dbc_file = "honda_civic_touring_2016_can_generated"

    signals = [("USER_BRAKE", "VSA_STATUS")]
    checks = [("VSA_STATUS", 50)]

    parser = CANParser(dbc_file, signals, checks, 0)
    packer = CANPacker(dbc_file)

    # Make sure nothing is updated
    self.assertEqual(len(parser.vl_all["VSA_STATUS"]["USER_BRAKE"]), 0)

    idx = 0
    for _ in range(10):
      # Ensure CANParser holds the values of any duplicate messages over multiple frames
      user_brake_vals = [random.randrange(100) for _ in range(random.randrange(5, 10))]
      half_idx = len(user_brake_vals) // 2
      can_msgs = [[], []]
      for frame, brake_vals in enumerate((user_brake_vals[:half_idx], user_brake_vals[half_idx:])):
        for user_brake in brake_vals:
          values = {"USER_BRAKE": user_brake}
          can_msgs[frame].append(packer.make_can_msg("VSA_STATUS", 0, values, idx))
          idx += 1

      can_strings = [can_list_to_can_capnp(msgs) for msgs in can_msgs]
      parser.update_strings(can_strings)
      vl_all = parser.vl_all["VSA_STATUS"]["USER_BRAKE"]

      self.assertEqual(vl_all, user_brake_vals)
      if len(user_brake_vals):
        self.assertEqual(vl_all[-1], parser.vl["VSA_STATUS"]["USER_BRAKE"])

if __name__ == "__main__":
  unittest.main()
