#!/usr/bin/env python3
import unittest
import random

import cereal.messaging as messaging
from opendbc.can.parser import CANParser
from opendbc.can.packer import CANPacker
from opendbc.can.tests import TEST_DBC

MAX_BAD_COUNTER = 5


# Python implementation so we don't have to depend on boardd
def can_list_to_can_capnp(can_msgs, msgtype='can', logMonoTime=None):
  dat = messaging.new_message(msgtype, len(can_msgs))

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
    packer = CANPacker(TEST_DBC)

    for b in range(6):
      for i in range(256):
        values = {"COUNTER": i}
        addr, _, dat, bus = packer.make_can_msg("CAN_FD_MESSAGE", b, values)
        self.assertEqual(addr, 245)
        self.assertEqual(bus, b)
        self.assertEqual(dat[0], i)

  def test_packer_counter(self):
    msgs = [("CAN_FD_MESSAGE", 0), ]
    packer = CANPacker(TEST_DBC)
    parser = CANParser(TEST_DBC, msgs, 0)

    # packer should increment the counter
    for i in range(1000):
      msg = packer.make_can_msg("CAN_FD_MESSAGE", 0, {})
      dat = can_list_to_can_capnp([msg, ])
      parser.update_strings([dat])
      self.assertEqual(parser.vl["CAN_FD_MESSAGE"]["COUNTER"], i % 256)

    # setting COUNTER should override
    for _ in range(100):
      cnt = random.randint(0, 255)
      msg = packer.make_can_msg("CAN_FD_MESSAGE", 0, {
        "COUNTER": cnt,
        "SIGNED": 0
      })
      dat = can_list_to_can_capnp([msg, ])
      parser.update_strings([dat])
      self.assertEqual(parser.vl["CAN_FD_MESSAGE"]["COUNTER"], cnt)

    # then, should resume counting from the override value
    cnt = parser.vl["CAN_FD_MESSAGE"]["COUNTER"]
    for i in range(100):
      msg = packer.make_can_msg("CAN_FD_MESSAGE", 0, {})
      dat = can_list_to_can_capnp([msg, ])
      parser.update_strings([dat])
      self.assertEqual(parser.vl["CAN_FD_MESSAGE"]["COUNTER"], (cnt + i) % 256)

  def test_parser_can_valid(self):
    msgs = [("CAN_FD_MESSAGE", 10), ]
    packer = CANPacker(TEST_DBC)
    parser = CANParser(TEST_DBC, msgs, 0)

    # shouldn't be valid initially
    self.assertFalse(parser.can_valid)

    # not valid until the message is seen
    for _ in range(100):
      dat = can_list_to_can_capnp([])
      parser.update_strings([dat])
      self.assertFalse(parser.can_valid)

    # valid once seen
    for i in range(1, 100):
      t = int(0.01 * i * 1e9)
      msg = packer.make_can_msg("CAN_FD_MESSAGE", 0, {})
      dat = can_list_to_can_capnp([msg, ], logMonoTime=t)
      parser.update_strings([dat])
      self.assertTrue(parser.can_valid)

  def test_parser_counter_can_valid(self):
    """
    Tests number of allowed bad counters + ensures CAN stays invalid
    while receiving invalid messages + that we can recover
    """
    msgs = [
      ("STEERING_CONTROL", 0),
    ]
    packer = CANPacker("honda_civic_touring_2016_can_generated")
    parser = CANParser("honda_civic_touring_2016_can_generated", msgs, 0)

    msg = packer.make_can_msg("STEERING_CONTROL", 0, {"COUNTER": 0})
    bts = can_list_to_can_capnp([msg])

    # bad static counter, invalid once it's seen MAX_BAD_COUNTER messages
    for idx in range(0x1000):
      parser.update_strings([bts])
      self.assertEqual((idx + 1) < MAX_BAD_COUNTER, parser.can_valid)

    # one to recover
    msg = packer.make_can_msg("STEERING_CONTROL", 0, {"COUNTER": 1})
    bts = can_list_to_can_capnp([msg])
    parser.update_strings([bts])
    self.assertTrue(parser.can_valid)

  def test_parser_no_partial_update(self):
    """
    Ensure that the CANParser doesn't partially update messages with invalid signals (COUNTER/CHECKSUM).
    Previously, the signal update loop would only break once it got to one of these invalid signals,
    after already updating most/all of the signals.
    """
    msgs = [
      ("STEERING_CONTROL", 0),
    ]
    packer = CANPacker("honda_civic_touring_2016_can_generated")
    parser = CANParser("honda_civic_touring_2016_can_generated", msgs, 0)

    def rx_steering_msg(values, bad_checksum=False):
      msg = packer.make_can_msg("STEERING_CONTROL", 0, values)
      if bad_checksum:
        # add 1 to checksum
        msg[2] = bytearray(msg[2])
        msg[2][4] = (msg[2][4] & 0xF0) | ((msg[2][4] & 0x0F) + 1)

      bts = can_list_to_can_capnp([msg])
      parser.update_strings([bts])

    rx_steering_msg({"STEER_TORQUE": 100}, bad_checksum=False)
    self.assertEqual(parser.vl["STEERING_CONTROL"]["STEER_TORQUE"], 100)
    self.assertEqual(parser.vl_all["STEERING_CONTROL"]["STEER_TORQUE"], [100])

    for _ in range(5):
      rx_steering_msg({"STEER_TORQUE": 200}, bad_checksum=True)
      self.assertEqual(parser.vl["STEERING_CONTROL"]["STEER_TORQUE"], 100)
      self.assertEqual(parser.vl_all["STEERING_CONTROL"]["STEER_TORQUE"], [])

    # Even if CANParser doesn't update instantaneous vl, make sure it didn't add invalid values to vl_all
    rx_steering_msg({"STEER_TORQUE": 300}, bad_checksum=False)
    self.assertEqual(parser.vl["STEERING_CONTROL"]["STEER_TORQUE"], 300)
    self.assertEqual(parser.vl_all["STEERING_CONTROL"]["STEER_TORQUE"], [300])

  def test_packer_parser(self):
    msgs = [
      ("Brake_Status", 0),
      ("CAN_FD_MESSAGE", 0),
      ("STEERING_CONTROL", 0),
    ]
    packer = CANPacker(TEST_DBC)
    parser = CANParser(TEST_DBC, msgs, 0)

    for steer in range(-256, 255):
      for active in (1, 0):
        values = {
          "STEERING_CONTROL": {
            "STEER_TORQUE": steer,
            "STEER_TORQUE_REQUEST": active,
          },
          "Brake_Status": {
            "Signal1": 61042322657536.0,
          },
          "CAN_FD_MESSAGE": {
            "SIGNED": steer,
            "64_BIT_LE": random.randint(0, 100),
            "64_BIT_BE": random.randint(0, 100),
          },
        }

        msgs = [packer.make_can_msg(k, 0, v) for k, v in values.items()]
        bts = can_list_to_can_capnp(msgs)
        parser.update_strings([bts])

        for k, v in values.items():
          for key, val in v.items():
            self.assertAlmostEqual(parser.vl[k][key], val)

        # also check address
        for sig in ("STEER_TORQUE", "STEER_TORQUE_REQUEST", "COUNTER", "CHECKSUM"):
          self.assertEqual(parser.vl["STEERING_CONTROL"][sig], parser.vl[228][sig])

  def test_scale_offset(self):
    """Test that both scale and offset are correctly preserved"""
    dbc_file = "honda_civic_touring_2016_can_generated"
    msgs = [("VSA_STATUS", 50)]
    parser = CANParser(dbc_file, msgs, 0)
    packer = CANPacker(dbc_file)

    for brake in range(100):
      values = {"USER_BRAKE": brake}
      msgs = packer.make_can_msg("VSA_STATUS", 0, values)
      bts = can_list_to_can_capnp([msgs])

      parser.update_strings([bts])

      self.assertAlmostEqual(parser.vl["VSA_STATUS"]["USER_BRAKE"], brake)

  def test_subaru(self):
    # Subaru is little endian

    dbc_file = "subaru_global_2017_generated"

    msgs = [("ES_LKAS", 50)]

    parser = CANParser(dbc_file, msgs, 0)
    packer = CANPacker(dbc_file)

    idx = 0
    for steer in range(-256, 255):
      for active in [1, 0]:
        values = {
          "LKAS_Output": steer,
          "LKAS_Request": active,
          "SET_1": 1
        }

        msgs = packer.make_can_msg("ES_LKAS", 0, values)
        bts = can_list_to_can_capnp([msgs])
        parser.update_strings([bts])

        self.assertAlmostEqual(parser.vl["ES_LKAS"]["LKAS_Output"], steer)
        self.assertAlmostEqual(parser.vl["ES_LKAS"]["LKAS_Request"], active)
        self.assertAlmostEqual(parser.vl["ES_LKAS"]["SET_1"], 1)
        self.assertAlmostEqual(parser.vl["ES_LKAS"]["COUNTER"], idx % 16)
        idx += 1

  def test_bus_timeout(self):
    """Test CAN bus timeout detection"""
    dbc_file = "honda_civic_touring_2016_can_generated"

    freq = 100
    msgs = [("VSA_STATUS", freq), ("STEER_MOTOR_TORQUE", freq/2)]

    parser = CANParser(dbc_file, msgs, 0)
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
    msgs = [("VSA_STATUS", 50)]
    parser = CANParser(dbc_file, msgs, 0)
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
          can_msgs[frame].append(packer.make_can_msg("VSA_STATUS", 0, values))
          idx += 1

      can_strings = [can_list_to_can_capnp(msgs) for msgs in can_msgs]
      parser.update_strings(can_strings)
      vl_all = parser.vl_all["VSA_STATUS"]["USER_BRAKE"]

      self.assertEqual(vl_all, user_brake_vals)
      if len(user_brake_vals):
        self.assertEqual(vl_all[-1], parser.vl["VSA_STATUS"]["USER_BRAKE"])

  def test_timestamp_nanos(self):
    """Test message timestamp dict"""
    dbc_file = "honda_civic_touring_2016_can_generated"

    msgs = [
      ("VSA_STATUS", 50),
      ("POWERTRAIN_DATA", 100),
    ]

    parser = CANParser(dbc_file, msgs, 0)
    packer = CANPacker(dbc_file)

    # Check the default timestamp is zero
    for msg in ("VSA_STATUS", "POWERTRAIN_DATA"):
      ts_nanos = parser.ts_nanos[msg].values()
      self.assertEqual(set(ts_nanos), {0})

    # Check:
    # - timestamp is only updated for correct messages
    # - timestamp is correct for multiple runs
    # - timestamp is from the latest message if updating multiple strings
    for _ in range(10):
      can_strings = []
      log_mono_time = 0
      for i in range(10):
        log_mono_time = int(0.01 * i * 1e+9)
        can_msg = packer.make_can_msg("VSA_STATUS", 0, {})
        can_strings.append(can_list_to_can_capnp([can_msg], logMonoTime=log_mono_time))
      parser.update_strings(can_strings)

      ts_nanos = parser.ts_nanos["VSA_STATUS"].values()
      self.assertEqual(set(ts_nanos), {log_mono_time})
      ts_nanos = parser.ts_nanos["POWERTRAIN_DATA"].values()
      self.assertEqual(set(ts_nanos), {0})

  def test_nonexistent_messages(self):
    # Ensure we don't allow messages not in the DBC
    existing_messages = ("STEERING_CONTROL", 228, "CAN_FD_MESSAGE", 245)

    for msg in existing_messages:
      CANParser(TEST_DBC, [(msg, 0)])
      with self.assertRaises(RuntimeError):
        new_msg = msg + "1" if isinstance(msg, str) else msg + 1
        CANParser(TEST_DBC, [(new_msg, 0)])

  def test_track_all_signals(self):
    parser = CANParser("toyota_nodsu_pt_generated", [("ACC_CONTROL", 0)])
    self.assertEqual(parser.vl["ACC_CONTROL"], {
      "ACCEL_CMD": 0,
      "ALLOW_LONG_PRESS": 0,
      "ACC_MALFUNCTION": 0,
      "RADAR_DIRTY": 0,
      "DISTANCE": 0,
      "MINI_CAR": 0,
      "ACC_TYPE": 0,
      "CANCEL_REQ": 0,
      "ACC_CUT_IN": 0,
      "LEAD_VEHICLE_STOPPED": 0,
      "PERMIT_BRAKING": 0,
      "RELEASE_STANDSTILL": 0,
      "ITS_CONNECT_LEAD": 0,
      "ACCEL_CMD_ALT": 0,
      "CHECKSUM": 0,
    })

  def test_disallow_duplicate_messages(self):
    CANParser("toyota_nodsu_pt_generated", [("ACC_CONTROL", 5)])

    with self.assertRaises(RuntimeError):
      CANParser("toyota_nodsu_pt_generated", [("ACC_CONTROL", 5), ("ACC_CONTROL", 10)])

    with self.assertRaises(RuntimeError):
      CANParser("toyota_nodsu_pt_generated", [("ACC_CONTROL", 10), ("ACC_CONTROL", 10)])

  def test_allow_undefined_msgs(self):
    # TODO: we should throw an exception for these, but we need good
    #  discovery tests in openpilot first
    packer = CANPacker("toyota_nodsu_pt_generated")

    self.assertEqual(packer.make_can_msg("ACC_CONTROL", 0, {"UNKNOWN_SIGNAL": 0}),
                     [835, 0, b'\x00\x00\x00\x00\x00\x00\x00N', 0])
    self.assertEqual(packer.make_can_msg("UNKNOWN_MESSAGE", 0, {"UNKNOWN_SIGNAL": 0}),
                     [0, 0, b'', 0])
    self.assertEqual(packer.make_can_msg(0, 0, {"UNKNOWN_SIGNAL": 0}),
                     [0, 0, b'', 0])


if __name__ == "__main__":
  unittest.main()
