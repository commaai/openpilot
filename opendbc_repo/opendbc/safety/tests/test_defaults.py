#!/usr/bin/env python3
import unittest

import opendbc.safety.tests.common as common
from opendbc.car.structs import CarParams
from opendbc.safety.tests.libsafety.libsafety_py import ffi


class TestDefaultRxHookBase(common.SafetyTest):
  FWD_BUS_LOOKUP = {}

  def test_rx_hook(self):
    # default rx hook allows all msgs
    for bus in range(4):
      for addr in self.SCANNED_ADDRS:
        self.assertTrue(self._rx(common.make_msg(bus, addr, 8)), f"failed RX {addr=}")


class TestNoOutput(TestDefaultRxHookBase):
  SAFETY_MODEL = CarParams.SafetyModel.noOutput

  TX_MSGS = []


class TestSilent(TestNoOutput):
  """SILENT uses same hooks as NOOUTPUT"""
  SAFETY_MODEL = CarParams.SafetyModel.silent


class TestAllOutput(TestDefaultRxHookBase):
  SAFETY_MODEL = CarParams.SafetyModel.allOutput

  # Allow all messages
  TX_MSGS = [[addr, bus] for addr in common.SafetyTest.SCANNED_ADDRS
             for bus in range(4)]

  def test_spam_can_buses(self):
    # asserts tx allowed for all scanned addrs
    for bus in range(4):
      for addr in self.SCANNED_ADDRS:
        should_tx = [addr, bus] in self.TX_MSGS
        self.assertEqual(should_tx, self._tx(common.make_msg(bus, addr, 8)), f"allowed TX {addr=} {bus=}")

  def test_default_controls_not_allowed(self):
    # controls always allowed
    self.assertTrue(self.safety.get_controls_allowed())

  def test_tx_hook_on_wrong_safety_mode(self):
    # No point, since we allow all messages
    pass


class TestAllOutputPassthrough(TestAllOutput):
  SAFETY_PARAM = 1

  FWD_BLACKLISTED_ADDRS = {}
  FWD_BUS_LOOKUP = {0: 2, 2: 0}


class TestSafetyFramework(common.SafetyTestBase):
  SAFETY_MODEL = CarParams.SafetyModel.noOutput

  def test_unsupported_safety_mode(self):
    self.safety.set_controls_allowed(True)
    self.assertEqual(self.safety.set_safety_hooks(0xFFFF, 0), -1)
    self.assertFalse(self.safety.get_controls_allowed())
    self.assertFalse(self.safety.safety_tx_hook(common.make_msg(0, 0x123)))


class TestSafetyRedundancy(common.SafetyTestBase):
  SAFETY_MODEL = CarParams.SafetyModel.allOutput

  def _flip_b(self, bit):
    state = common.state_buffer("safety_b")
    state[bit // 8] = bytes([state[bit // 8][0] ^ (1 << (bit % 8))])

  def _assert_fault(self):
    self.assertTrue(self.safety.safety_redundancy_fault)
    self.assertEqual(common.state_buffer("safety_a")[:], common.state_buffer("safety_b")[:])
    self.safety.safety_redundancy_fault = False

  def test_rx_decision_mismatch(self):
    # wrong counters in B fail its rx check, CLU11 is a hyundai rx check with a counter and no checksum
    self.safety.set_safety_hooks(CarParams.SafetyModel.hyundai, 0)
    self.safety.set_controls_allowed(True)
    for i in range(self.safety.get_rx_checks_len()):
      self._flip_b(ffi.offsetof("SafetyState", "rx_status", i, "wrong_counters") * 8 + 3)
    self.assertTrue(self.safety.safety_rx_hook(common.make_msg(0, 0x4F1, 4)))
    self._assert_fault()

  def test_tx_decision_mismatch(self):
    self.safety.set_controls_allowed(True)
    self._flip_b(ffi.offsetof("SafetyState", "relay_malfunction") * 8)
    self.assertTrue(self.safety.safety_tx_hook(common.make_msg(0, 0x123)))
    self._assert_fault()

  def test_fwd_decision_mismatch(self):
    self.safety.set_safety_hooks(CarParams.SafetyModel.allOutput, 1)
    self.assertEqual(self.safety.safety_fwd_hook(0, 0x123), 2)
    self._flip_b(ffi.offsetof("SafetyState", "relay_malfunction") * 8)
    self.assertEqual(self.safety.safety_fwd_hook(0, 0x123), 2)
    self._assert_fault()

  def test_state_mismatch(self):
    # the tick rewrites safety_rx_checks_invalid in both copies, and a nonzero rx_checks_len makes it read the NULL rx_checks
    rewritten = ffi.offsetof("SafetyCopy", "state") + ffi.offsetof("SafetyState", "safety_rx_checks_invalid")
    rx_checks_len = ffi.offsetof("SafetyCopy", "setup") + ffi.offsetof("SafetySetup", "current_safety_config") + ffi.offsetof("safety_config", "rx_checks_len")
    for byte in range(ffi.sizeof("SafetyCopy")):
      if byte != rewritten and not (rx_checks_len <= byte < rx_checks_len + ffi.sizeof("int")):
        self.safety.set_controls_allowed(True)
        self._flip_b(byte * 8)
        self.safety.safety_tick()
        self._assert_fault()

  def test_same_time(self):
    ford_lat = common.CANPackerSafety("ford_lincoln_base_pt").make_can_msg_safety("LateralMotionControl", 0, {"LatCtl_D_Rq": 1})
    steering = {
      CarParams.SafetyModel.hyundai: (common.make_msg(0, 0x340), lambda s: s.ts_torque_check_last),
      CarParams.SafetyModel.tesla: (common.make_msg(0, 0x488, dat=b"\x40\x00\x20\x00"), lambda s: s.ts_angle_check_last),
      CarParams.SafetyModel.ford: (ford_lat, lambda s: s.curvature_state.ts_check_last),
    }
    for mode, (msg, ts_last) in steering.items():
      with self.subTest(mode=mode):
        self.safety.set_safety_hooks(mode, 0)
        self.safety.set_timer(1000000)
        self.safety.set_timer_step(1)
        self.safety.safety_rx_hook(common.make_msg(0, 0x4F1, 4))
        self.safety.set_controls_allowed(True)
        self.safety.safety_tx_hook(msg)
        self.assertEqual(ts_last(self.safety.safety_a.state), 1000002)


class TestHeartbeat(common.SafetyTestBase):
  SAFETY_MODEL = CarParams.SafetyModel.allOutput

  def test_engaged_mismatch(self):
    for engaged in (True, False):
      self.safety.safety_heartbeat(engaged)
      self.safety.set_controls_allowed(True)
      for _ in range(3):
        self.assertTrue(self.safety.get_controls_allowed())
        self.safety.safety_heartbeat_check()
      self.assertEqual(engaged, self.safety.get_controls_allowed())


class TestModeCount(common.SafetyTestBase):
  SAFETY_MODEL = CarParams.SafetyModel.toyota
  SAFETY_PARAM = 73

  def test_relay_malfunction_after_two_ticks(self):
    # the relay malfunction check starts once safety_mode_cnt passes 1
    self.safety.set_safety_hooks(self.SAFETY_MODEL, self.SAFETY_PARAM)
    for _ in range(2):
      self.safety.safety_rx_hook(common.make_msg(0, 0x2E4, 8))
      self.assertFalse(self.safety.get_relay_malfunction())
      self.safety.safety_tick()
    self.safety.safety_rx_hook(common.make_msg(0, 0x2E4, 8))
    self.assertTrue(self.safety.get_relay_malfunction())


if __name__ == "__main__":
  unittest.main()
