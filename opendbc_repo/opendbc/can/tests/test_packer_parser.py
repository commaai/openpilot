import pytest
import random

from opendbc.can.parser import CANParser
from opendbc.can.packer import CANPacker
from opendbc.can.tests import TEST_DBC

MAX_BAD_COUNTER = 5


class TestCanParserPacker:
  def test_packer(self):
    packer = CANPacker(TEST_DBC)

    for b in range(6):
      for i in range(256):
        values = {"COUNTER": i}
        addr, dat, bus = packer.make_can_msg("CAN_FD_MESSAGE", b, values)
        assert addr == 245
        assert bus == b
        assert dat[0] == i

  def test_packer_counter(self):
    msgs = [("CAN_FD_MESSAGE", 0), ]
    packer = CANPacker(TEST_DBC)
    parser = CANParser(TEST_DBC, msgs, 0)

    # packer should increment the counter
    for i in range(1000):
      msg = packer.make_can_msg("CAN_FD_MESSAGE", 0, {})
      parser.update_strings([0, [msg]])
      assert parser.vl["CAN_FD_MESSAGE"]["COUNTER"] == (i % 256)

    # setting COUNTER should override
    for _ in range(100):
      cnt = random.randint(0, 255)
      msg = packer.make_can_msg("CAN_FD_MESSAGE", 0, {
        "COUNTER": cnt,
        "SIGNED": 0
      })
      parser.update_strings([0, [msg]])
      assert parser.vl["CAN_FD_MESSAGE"]["COUNTER"] == cnt

    # then, should resume counting from the override value
    cnt = parser.vl["CAN_FD_MESSAGE"]["COUNTER"]
    for i in range(100):
      msg = packer.make_can_msg("CAN_FD_MESSAGE", 0, {})
      parser.update_strings([0, [msg]])
      assert parser.vl["CAN_FD_MESSAGE"]["COUNTER"] == ((cnt + i) % 256)

  def test_parser_can_valid(self):
    msgs = [("CAN_FD_MESSAGE", 10), ]
    packer = CANPacker(TEST_DBC)
    parser = CANParser(TEST_DBC, msgs, 0)

    # shouldn't be valid initially
    assert not parser.can_valid

    # not valid until the message is seen
    for _ in range(100):
      parser.update_strings([0, []])
      assert not parser.can_valid

    # valid once seen
    for i in range(1, 100):
      t = int(0.01 * i * 1e9)
      msg = packer.make_can_msg("CAN_FD_MESSAGE", 0, {})
      parser.update_strings([t, [msg]])
      assert parser.can_valid

  def test_parser_updated_list(self):
    msgs = [("CAN_FD_MESSAGE", 10), ]
    parser = CANParser(TEST_DBC, msgs, 0)
    packer = CANPacker(TEST_DBC)

    msg = packer.make_can_msg("CAN_FD_MESSAGE", 0, {})
    ret = parser.update_strings([0, [msg]])
    assert ret == {245}

    ret = parser.update_strings([])
    assert len(ret) == 0

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

    # bad static counter, invalid once it's seen MAX_BAD_COUNTER messages
    for idx in range(0x1000):
      parser.update_strings([0, [msg]])
      assert ((idx + 1) < MAX_BAD_COUNTER) == parser.can_valid

    # one to recover
    msg = packer.make_can_msg("STEERING_CONTROL", 0, {"COUNTER": 1})
    parser.update_strings([0, [msg]])
    assert parser.can_valid

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
        dat = bytearray(msg[1])
        dat[4] = (dat[4] & 0xF0) | ((dat[4] & 0x0F) + 1)
        msg = (msg[0], bytes(dat), msg[2])

      parser.update_strings([0, [msg]])

    rx_steering_msg({"STEER_TORQUE": 100}, bad_checksum=False)
    assert parser.vl["STEERING_CONTROL"]["STEER_TORQUE"] == 100
    assert parser.vl_all["STEERING_CONTROL"]["STEER_TORQUE"] == [100]

    for _ in range(5):
      rx_steering_msg({"STEER_TORQUE": 200}, bad_checksum=True)
      assert parser.vl["STEERING_CONTROL"]["STEER_TORQUE"] == 100
      assert parser.vl_all["STEERING_CONTROL"]["STEER_TORQUE"] == []

    # Even if CANParser doesn't update instantaneous vl, make sure it didn't add invalid values to vl_all
    rx_steering_msg({"STEER_TORQUE": 300}, bad_checksum=False)
    assert parser.vl["STEERING_CONTROL"]["STEER_TORQUE"] == 300
    assert parser.vl_all["STEERING_CONTROL"]["STEER_TORQUE"] == [300]

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
        parser.update_strings([0, msgs])

        for k, v in values.items():
          for key, val in v.items():
            assert parser.vl[k][key] == pytest.approx(val)

        # also check address
        for sig in ("STEER_TORQUE", "STEER_TORQUE_REQUEST", "COUNTER", "CHECKSUM"):
          assert parser.vl["STEERING_CONTROL"][sig] == parser.vl[228][sig]

  def test_scale_offset(self):
    """Test that both scale and offset are correctly preserved"""
    dbc_file = "honda_civic_touring_2016_can_generated"
    msgs = [("VSA_STATUS", 50)]
    parser = CANParser(dbc_file, msgs, 0)
    packer = CANPacker(dbc_file)

    for brake in range(100):
      values = {"USER_BRAKE": brake}
      msgs = packer.make_can_msg("VSA_STATUS", 0, values)
      parser.update_strings([0, [msgs]])

      assert parser.vl["VSA_STATUS"]["USER_BRAKE"] == pytest.approx(brake)

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
        parser.update_strings([0, [msgs]])

        assert parser.vl["ES_LKAS"]["LKAS_Output"] == pytest.approx(steer)
        assert parser.vl["ES_LKAS"]["LKAS_Request"] == pytest.approx(active)
        assert parser.vl["ES_LKAS"]["SET_1"] == pytest.approx(1)
        assert parser.vl["ES_LKAS"]["COUNTER"] == pytest.approx(idx % 16)
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

      parser.update_strings([t, msgs])

    # all good, no timeout
    for _ in range(1000):
      send_msg()
      assert not parser.bus_timeout, str(_)

    # timeout after 10 blank msgs
    for n in range(200):
      send_msg(blank=True)
      assert (n >= 10) == parser.bus_timeout

    # no timeout immediately after seen again
    send_msg()
    assert not parser.bus_timeout

  def test_updated(self):
    """Test updated value dict"""
    dbc_file = "honda_civic_touring_2016_can_generated"
    msgs = [("VSA_STATUS", 50)]
    parser = CANParser(dbc_file, msgs, 0)
    packer = CANPacker(dbc_file)

    # Make sure nothing is updated
    assert len(parser.vl_all["VSA_STATUS"]["USER_BRAKE"]) == 0

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

      parser.update_strings([[0, m] for m in can_msgs])
      vl_all = parser.vl_all["VSA_STATUS"]["USER_BRAKE"]

      assert vl_all == user_brake_vals
      if len(user_brake_vals):
        assert vl_all[-1] == parser.vl["VSA_STATUS"]["USER_BRAKE"]

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
      assert set(ts_nanos) == {0}

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
        can_strings.append((log_mono_time, [can_msg]))
      parser.update_strings(can_strings)

      ts_nanos = parser.ts_nanos["VSA_STATUS"].values()
      assert set(ts_nanos) == {log_mono_time}
      ts_nanos = parser.ts_nanos["POWERTRAIN_DATA"].values()
      assert set(ts_nanos) == {0}

  def test_nonexistent_messages(self):
    # Ensure we don't allow messages not in the DBC
    existing_messages = ("STEERING_CONTROL", 228, "CAN_FD_MESSAGE", 245)

    for msg in existing_messages:
      CANParser(TEST_DBC, [(msg, 0)])
      with pytest.raises(RuntimeError):
        new_msg = msg + "1" if isinstance(msg, str) else msg + 1
        CANParser(TEST_DBC, [(new_msg, 0)])

  def test_track_all_signals(self):
    parser = CANParser("toyota_nodsu_pt_generated", [("ACC_CONTROL", 0)])
    assert parser.vl["ACC_CONTROL"] == {
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
    }

  def test_disallow_duplicate_messages(self):
    CANParser("toyota_nodsu_pt_generated", [("ACC_CONTROL", 5)])

    with pytest.raises(RuntimeError):
      CANParser("toyota_nodsu_pt_generated", [("ACC_CONTROL", 5), ("ACC_CONTROL", 10)])

    with pytest.raises(RuntimeError):
      CANParser("toyota_nodsu_pt_generated", [("ACC_CONTROL", 10), ("ACC_CONTROL", 10)])

  def test_allow_undefined_msgs(self):
    # TODO: we should throw an exception for these, but we need good
    #  discovery tests in openpilot first
    packer = CANPacker("toyota_nodsu_pt_generated")

    assert packer.make_can_msg("ACC_CONTROL", 0, {"UNKNOWN_SIGNAL": 0}) == (835, b'\x00\x00\x00\x00\x00\x00\x00N', 0)
    assert packer.make_can_msg("UNKNOWN_MESSAGE", 0, {"UNKNOWN_SIGNAL": 0}) == (0, b'', 0)
    assert packer.make_can_msg(0, 0, {"UNKNOWN_SIGNAL": 0}) == (0, b'', 0)
