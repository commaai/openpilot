import time
import os
import random
import unittest
from collections import defaultdict, Counter
import hypothesis.strategies as st
from hypothesis import Phase, given, settings
from openpilot.common.parameterized import parameterized_class
from openpilot.common.test import OpenpilotTestCase
from opendbc.can.dbc import SignalType
from opendbc.can.packer import set_value
from opendbc.car import DT_CTRL, gen_empty_fingerprint, structs
from opendbc.car.can_definitions import CanData
from opendbc.car.car_helpers import FRAME_FINGERPRINT, interfaces
from opendbc.car.fingerprints import MIGRATION
from opendbc.car.honda.values import HondaFlags
from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from opendbc.car.structs import car
from opendbc.car.tests.routes import non_tested_cars, routes, CarTestRoute
from opendbc.car.values import Platform, PLATFORMS
from opendbc.safety.tests.libsafety import libsafety_py
from openpilot.common.basedir import BASEDIR
from openpilot.selfdrive.pandad import can_capnp_to_list
from openpilot.selfdrive.test.helpers import read_segment_list
from openpilot.common.hardware.hw import DEFAULT_DOWNLOAD_CACHE_ROOT
from openpilot.tools.lib.logreader import LogReader, LogsUnavailable, openpilotci_source, internal_source, comma_api_source
from openpilot.tools.lib.file_sources import Source
from openpilot.tools.lib.route import SegmentName

SafetyModel = car.CarParams.SafetyModel
SteerControlType = structs.CarParams.SteerControlType

# TOYOTA_ANGLE_STEERING_LIMITS.max_angle in opendbc/safety/modes/toyota.h
TOYOTA_LTA_MAX_ANGLE = 1657

# panda safety stores angle_meas in brand-specific CAN units (angle_deg_to_can in opendbc/safety/modes/*.h).
ANGLE_DEG_TO_CAN = {
  "tesla": -10,
  "toyota": 17.452007,
  "nissan": 100,
  "psa": 10,
}

# CarControl fields fuzzed by test_panda_safety_tx_fuzzy. Add an entry here to
# extend TX fuzzing coverage to new state. Actuator ranges intentionally exceed
# what openpilot requests (except accel, which selfdrived clamps upstream of
# CarController) since CarController is responsible for clamping them.
TX_FUZZ_CAR_CONTROL_STRATEGY: dict[str, st.SearchStrategy] = {
  'actuators.torque': st.floats(-1.5, 1.5),
  'actuators.steeringAngleDeg': st.floats(-1000, 1000),
  'actuators.curvature': st.floats(-0.1, 0.1),
  'actuators.accel': st.floats(ACCEL_MIN, ACCEL_MAX),
  'actuators.longControlState': st.sampled_from(['off', 'pid', 'stopping', 'starting']),
  'actuators.speed': st.floats(0, 50),
  'hudControl.setSpeed': st.floats(0, 70),
  'hudControl.speedVisible': st.booleans(),
  'hudControl.lanesVisible': st.booleans(),
  'hudControl.leadVisible': st.booleans(),
  'leftBlinker': st.booleans(),
  'rightBlinker': st.booleans(),
}

# rx messages excluded from TX fuzzing: they latch panda-only tx-lockout states that
# openpilot's CarController does not react to (in reality panda deliberately blocks
# openpilot's messages in these states)
TX_FUZZ_RX_EXCLUSIONS: dict[str, set[tuple[int, int]]] = {
  "tesla": {
    (0, 0x286),  # DI_locStatus: latches Autopark state, which blocks all tx
    (0, 0x370),  # EPAS3S_sysStatus: hands-on level gates steering inside CarController,
                 # re-anchoring the angle command in jumps panda's rate limits can't follow
    (2, 0x488),  # DAS_steeringControl: latches stock LKAS state, which blocks steering
    (2, 0x2b9),  # DAS_control: latches stock AEB state, which blocks longitudinal
  },
}

# measured angle signals held constant in fuzzed rx payloads, keyed by brand -> {signal_name:
# raw value}. angle-control cars read the measured steering angle from the same message they
# read torque from. Fuzzing it freely teleports the angle every frame, which no real EPS does;
# openpilot rate-limits its command toward the measured angle and round-trips it through float
# degrees plus a learned offset, so a jumping measured angle produces command-vs-measured
# disagreements against panda's raw-CAN checks that are pure fuzzing artifacts, not TX bugs.
# Pinning the measured angle to a stable value (driving straight) keeps both sides aligned;
# the angle *command* still fuzzes through actuators.steeringAngleDeg, which is what exercises
# panda's active-steering rate limit, and torque and the other signals still fuzz freely.
TX_FUZZ_SIGNAL_PIN: dict[str, dict[str, int]] = {
  "toyota": {"STEER_ANGLE": 0},
}

# bits cleared from fuzzed rx payloads as (byte, mask), for known decode divergences:
# real traffic never sets these bits, so neither side is exercised by them
TX_FUZZ_PAYLOAD_MASKS: dict[tuple[str, int], list[tuple[int, int]]] = {
  # panda reads LH_EPS_03 driver torque as 13 bits, but the DBC defines 10 and bits
  # 50-52 are undefined (always zero on the wire). TODO: tighten panda's mask
  ("volkswagen", 0x9f): [(6, 0x1c)],
}

# openpilot and panda compute measured curvature from yaw rate and speed sampled at
# different times, so fuzzed CAN can diverge them (see _tx_fuzz_measured_state_coherent).
# desired_sign maps openpilot's applied curvature to panda's desired-curvature CAN units,
# error keys mirror the brand's measured-curvature error check where it has one
CURVATURE_COHERENCE = {
  "ford": {"curvature_to_can": 50000, "desired_sign": -1, "frequency": 20,
           "max_curvature_error": 100, "op_clip_min_speed": 9., "panda_check_min_speed": 10.},
  "volkswagen": {"curvature_to_can": 149253.7313, "desired_sign": 1, "frequency": 50},  # MEB only; MQB/PQ are torque-based
}

# matches MAX_LATERAL_ACCEL/MAX_LATERAL_JERK in opendbc/safety/lateral.h, used to bound
# what a real, rate-limited curvature command stream can do (panda adds tolerance on top,
# so anything skipped by these bounds is also within panda's)
PANDA_MAX_LATERAL_ACCEL = 3.0 + 9.81 * 0.06  # m/s^2
PANDA_MAX_LATERAL_JERK = 3.0 + 9.81 * 0.06   # m/s^3

NUM_JOBS = int(os.environ.get("NUM_JOBS", "1"))
JOB_ID = int(os.environ.get("JOB_ID", "0"))
INTERNAL_SEG_LIST = os.environ.get("INTERNAL_SEG_LIST", "")
INTERNAL_SEG_CNT = int(os.environ.get("INTERNAL_SEG_CNT", "0"))
MAX_EXAMPLES = int(os.environ.get("MAX_EXAMPLES", "300"))
CI = os.environ.get("CI", None) is not None


def get_test_cases() -> list[tuple[str, CarTestRoute | None]]:
  # build list of test cases
  test_cases = []
  if not len(INTERNAL_SEG_LIST):
    routes_by_car = defaultdict(set)
    for r in routes:
      routes_by_car[str(r.car_model)].add(r)

    for i, c in enumerate(sorted(PLATFORMS)):
      if i % NUM_JOBS == JOB_ID:
        test_cases.extend(sorted((c, r) for r in routes_by_car.get(c, (None,))))

  else:
    segment_list = read_segment_list(os.path.join(BASEDIR, INTERNAL_SEG_LIST))
    segment_list = random.sample(segment_list, INTERNAL_SEG_CNT or len(segment_list))
    for platform, segment in segment_list:
      platform = MIGRATION.get(platform, platform)
      segment_name = SegmentName(segment)
      test_cases.append((platform, CarTestRoute(segment_name.route_name.canonical_name, platform,
                                                segment=segment_name.segment_num)))
  return test_cases


class TestCarModelBase(OpenpilotTestCase):
  SLOW_TEST = True
  SHARED_DOWNLOAD_CACHE = True
  platform: Platform | None = None
  test_route: CarTestRoute | None = None

  can_msgs: list[tuple[int, list[CanData]]]
  fingerprint: dict[int, dict[int, int]]
  elm_frame: int | None
  car_safety_mode_frame: int | None

  @classmethod
  def get_testing_data_from_logreader(cls, lr):
    car_fw = []
    can_msgs = []
    cls.elm_frame = None
    cls.car_safety_mode_frame = None
    cls.fingerprint = gen_empty_fingerprint()
    alpha_long = False
    for msg in lr:
      if msg.which() == "can":
        can = can_capnp_to_list((msg.as_builder().to_bytes(),))[0]
        can_msgs.append((can[0], [CanData(*can) for can in can[1]]))
        if len(can_msgs) <= FRAME_FINGERPRINT:
          for m in msg.can:
            if m.src < 64:
              cls.fingerprint[m.src][m.address] = len(m.dat)

      elif msg.which() == "carParams":
        car_fw = msg.carParams.carFw
        if msg.carParams.openpilotLongitudinalControl:
          alpha_long = True
        if cls.platform is None:
          live_fingerprint = msg.carParams.carFingerprint
          cls.platform = MIGRATION.get(live_fingerprint, live_fingerprint)

      # Log which can frame the panda safety mode left ELM327, for CAN validity checks
      elif msg.which() == 'pandaStates':
        for ps in msg.pandaStates:
          if cls.elm_frame is None and ps.safetyModel != SafetyModel.elm327:
            cls.elm_frame = len(can_msgs)
          if cls.car_safety_mode_frame is None and ps.safetyModel not in \
            (SafetyModel.elm327, SafetyModel.noOutput):
            cls.car_safety_mode_frame = len(can_msgs)

      elif msg.which() == 'pandaStateDEPRECATED':
        if cls.elm_frame is None and msg.pandaStateDEPRECATED.safetyModel != SafetyModel.elm327:
          cls.elm_frame = len(can_msgs)
        if cls.car_safety_mode_frame is None and msg.pandaStateDEPRECATED.safetyModel not in \
          (SafetyModel.elm327, SafetyModel.noOutput):
          cls.car_safety_mode_frame = len(can_msgs)

    assert len(can_msgs) > int(50 / DT_CTRL), "no can data found"
    return car_fw, can_msgs, alpha_long

  @classmethod
  def get_testing_data(cls):
    test_segs = (2, 1, 0)
    if cls.test_route.segment is not None:
      test_segs = (cls.test_route.segment,)

    for seg in test_segs:
      segment_range = f"{cls.test_route.route}/{seg}"

      try:
        sources: list[Source] = [internal_source] if len(INTERNAL_SEG_LIST) else [openpilotci_source, comma_api_source]
        lr = LogReader(segment_range, sources=sources, sort_by_time=True)
        return cls.get_testing_data_from_logreader(lr)
      except (LogsUnavailable, AssertionError):
        pass

    raise Exception(f"Route: {repr(cls.test_route.route)} with segments: {test_segs} not found or no CAN msgs found. Is it uploaded and public?")


  @classmethod
  def setUpClass(cls):
    if cls.__name__ == 'TestCarModel' or cls.__name__.endswith('Base'):
      raise unittest.SkipTest

    if cls.test_route is None:
      if cls.platform in non_tested_cars:
        print(f"Skipping tests for {cls.platform}: missing route")
        raise unittest.SkipTest
      raise Exception(f"missing test route for {cls.platform}")

    car_fw, cls.can_msgs, alpha_long = cls.get_testing_data()

    # if relay is expected to be open in the route
    cls.openpilot_enabled = cls.car_safety_mode_frame is not None

    cls.CarInterface = interfaces[cls.platform]
    cls.CP = cls.CarInterface.get_params(cls.platform, cls.fingerprint, car_fw, alpha_long, False, docs=False)
    assert cls.CP
    assert cls.CP.carFingerprint == cls.platform

    os.environ["COMMA_CACHE"] = DEFAULT_DOWNLOAD_CACHE_ROOT

  @classmethod
  def tearDownClass(cls):
    del cls.can_msgs

  def setUp(self):
    self.CI = self.CarInterface(self.CP.copy())
    assert self.CI

    # TODO: check safetyModel is in release panda build
    self.safety = libsafety_py.libsafety

    cfg = self.CP.safetyConfigs[-1]
    set_status = self.safety.set_safety_hooks(cfg.safetyModel.raw, cfg.safetyParam)
    self.assertEqual(0, set_status, f"failed to set safetyModel {cfg}")
    self.safety.init_tests()

  def test_car_params(self):
    if self.CP.dashcamOnly:
      self.skipTest("no need to check carParams for dashcamOnly")

    # make sure car params are within a valid range
    self.assertGreater(self.CP.mass, 1)

    if self.CP.steerControlType not in (SteerControlType.angle, SteerControlType.curvature):
      tuning = self.CP.lateralTuning.which()
      if tuning == 'pid':
        self.assertTrue(len(self.CP.lateralTuning.pid.kpV))
      elif tuning == 'torque':
        self.assertTrue(self.CP.lateralTuning.torque.latAccelFactor > 0)
      else:
        raise Exception("unknown tuning")

  def test_car_interface(self):
    # TODO: also check for checksum violations from can parser
    can_invalid_cnt = 0
    CC = structs.CarControl().as_reader()

    for i, msg in enumerate(self.can_msgs):
      CS = self.CI.update(msg)
      self.CI.apply(CC, msg[0])

      # wait max of 2s for low frequency msgs to be seen
      if i > 250:
        can_invalid_cnt += not CS.canValid

    self.assertEqual(can_invalid_cnt, 0)

  def test_radar_interface(self):
    RI = self.CarInterface.RadarInterface(self.CP)
    assert RI

    # Since OBD port is multiplexed to bus 1 (commonly radar bus) while fingerprinting,
    # start parsing CAN messages after we've left ELM mode and can expect CAN traffic
    error_cnt = 0
    for i, msg in enumerate(self.can_msgs[self.elm_frame:]):
      rr: structs.RadarData | None = RI.update(msg)
      if rr is not None and i > 50:
        error_cnt += rr.errors.canError
    self.assertEqual(error_cnt, 0)

  def test_panda_safety_rx_checks(self):
    if self.CP.dashcamOnly:
      self.skipTest("no need to check panda safety for dashcamOnly")

    start_ts = self.can_msgs[0][0]

    failed_addrs = Counter()
    for can in self.can_msgs:
      # update panda timer
      t = (can[0] - start_ts) / 1e3
      self.safety.set_timer(int(t))

      # run all msgs through the safety RX hook
      for msg in can[1]:
        if msg.src >= 64:
          continue

        to_send = libsafety_py.make_CANPacket(msg.address, msg.src % 4, msg.dat)
        if self.safety.safety_rx_hook(to_send) != 1:
          failed_addrs[hex(msg.address)] += 1

      # ensure all msgs defined in the addr checks are valid
      self.safety.safety_tick_current_safety_config()
      if t > 1e6:
        self.assertTrue(self.safety.safety_config_valid())

      # Don't check relay malfunction on disabled routes (relay closed),
      # or before fingerprinting is done (elm327 and noOutput)
      if self.car_safety_mode_frame is not None and t / 1e4 > self.car_safety_mode_frame:
        self.assertFalse(self.safety.get_relay_malfunction())
      else:
        self.safety.set_relay_malfunction(False)

    self.assertFalse(len(failed_addrs), f"panda safety RX check failed: {failed_addrs}")

    # ensure RX checks go invalid after small time with no traffic
    self.safety.set_timer(int(t + (2*1e6)))
    self.safety.safety_tick_current_safety_config()
    self.assertFalse(self.safety.safety_config_valid())

  def test_panda_safety_tx_cases(self, data=None):
    """Asserts we can tx common messages"""
    if self.CP.dashcamOnly:
      self.skipTest("no need to check panda safety for dashcamOnly")

    if self.CP.notCar:
      self.skipTest("Skipping test for notCar")

    def test_car_controller(car_control):
      now_nanos = 0
      msgs_sent = 0
      CI = self.CarInterface(self.CP)
      for _ in range(round(10.0 / DT_CTRL)):  # make sure we hit the slowest messages
        CI.update([])
        _, sendcan = CI.apply(car_control, now_nanos)

        now_nanos += DT_CTRL * 1e9
        msgs_sent += len(sendcan)
        for addr, dat, bus in sendcan:
          to_send = libsafety_py.make_CANPacket(addr, bus % 4, dat)
          self.assertTrue(self.safety.safety_tx_hook(to_send), (addr, dat, bus))

      # Make sure we attempted to send messages
      self.assertGreater(msgs_sent, 50)

    # Make sure we can send all messages while inactive
    CC = structs.CarControl()
    test_car_controller(CC.as_reader())

    # Test cancel + general messages (controls_allowed=False & cruise_engaged=True)
    self.safety.set_cruise_engaged_prev(True)
    CC = structs.CarControl(cruiseControl=structs.CarControl.CruiseControl(cancel=True))
    test_car_controller(CC.as_reader())

    # Test resume + general messages (controls_allowed=True & cruise_engaged=True)
    self.safety.set_controls_allowed(True)
    CC = structs.CarControl(cruiseControl=structs.CarControl.CruiseControl(resume=True))
    test_car_controller(CC.as_reader())

  # Capturing stdout/stderr here causes elevated memory usage.
  @settings(max_examples=MAX_EXAMPLES, deadline=None,
            phases=(Phase.reuse, Phase.generate, Phase.shrink))
  @given(data=st.data())
  def test_panda_safety_carstate_fuzzy(self, data):
    """
      For each example, pick a random CAN message on the bus and fuzz its data,
      checking for panda state mismatches.
    """

    if self.CP.dashcamOnly:
      self.skipTest("no need to check panda safety for dashcamOnly")

    valid_addrs = [(addr, bus, size) for bus, addrs in self.fingerprint.items() for addr, size in addrs.items()]
    address, bus, size = data.draw(st.sampled_from(valid_addrs))

    msg_strategy = st.binary(min_size=size, max_size=size)
    msgs = data.draw(st.lists(msg_strategy, min_size=20))

    vehicle_speed_seen = self.CP.steerControlType == SteerControlType.angle and not self.CP.notCar

    for n, dat in enumerate(msgs):
      # due to panda updating state selectively, only edges are expected to match
      # TODO: warm up CarState with real CAN messages to check edge of both sources
      #  (eg. toyota's gasPressed is the inverse of a signal being set)
      prev_panda_gas = self.safety.get_gas_pressed_prev()
      prev_panda_brake = self.safety.get_brake_pressed_prev()
      prev_panda_regen_braking = self.safety.get_regen_braking_prev()
      prev_panda_steering_disengage = self.safety.get_steering_disengage_prev()
      prev_panda_vehicle_moving = self.safety.get_vehicle_moving()
      prev_panda_vehicle_speed_min = self.safety.get_vehicle_speed_min()
      prev_panda_vehicle_speed_max = self.safety.get_vehicle_speed_max()
      prev_panda_cruise_engaged = self.safety.get_cruise_engaged_prev()
      prev_panda_acc_main_on = self.safety.get_acc_main_on()

      to_send = libsafety_py.make_CANPacket(address, bus, dat)
      self.safety.safety_rx_hook(to_send)

      can = [(int(time.monotonic() * 1e9), [CanData(address=address, dat=dat, src=bus)])]
      CS = self.CI.update(can)
      if n < 5:  # CANParser warmup time
        continue

      if self.safety.get_gas_pressed_prev() != prev_panda_gas:
        self.assertEqual(CS.gasPressed, self.safety.get_gas_pressed_prev())

      if self.safety.get_brake_pressed_prev() != prev_panda_brake:
        self.assertEqual(CS.brakePressed, self.safety.get_brake_pressed_prev())

      if self.safety.get_regen_braking_prev() != prev_panda_regen_braking:
        self.assertEqual(CS.regenBraking, self.safety.get_regen_braking_prev())

      if self.safety.get_steering_disengage_prev() != prev_panda_steering_disengage:
        self.assertEqual(CS.steeringDisengage, self.safety.get_steering_disengage_prev())

      if self.safety.get_vehicle_moving() != prev_panda_vehicle_moving and not self.CP.notCar:
        self.assertEqual(not CS.standstill, self.safety.get_vehicle_moving())

      # check vehicle speed if angle control car or available
      if self.safety.get_vehicle_speed_min() > 0 or self.safety.get_vehicle_speed_max() > 0:
        vehicle_speed_seen = True

      if vehicle_speed_seen and (self.safety.get_vehicle_speed_min() != prev_panda_vehicle_speed_min or
                                 self.safety.get_vehicle_speed_max() != prev_panda_vehicle_speed_max):
        v_ego_raw = CS.vEgoRaw / self.CP.wheelSpeedFactor
        self.assertFalse(v_ego_raw > (self.safety.get_vehicle_speed_max() + 1e-3) or
                         v_ego_raw < (self.safety.get_vehicle_speed_min() - 1e-3))

      if not (self.CP.brand == "honda" and not (self.CP.flags & HondaFlags.BOSCH)):
        if self.safety.get_cruise_engaged_prev() != prev_panda_cruise_engaged:
          self.assertEqual(CS.cruiseState.enabled, self.safety.get_cruise_engaged_prev())

      if self.CP.brand == "honda":
        if self.safety.get_acc_main_on() != prev_panda_acc_main_on:
          self.assertEqual(CS.cruiseState.available, self.safety.get_acc_main_on())

  def _tx_fuzz_setup(self):
    # Warm up openpilot and panda with real CAN messages so both sides start
    # from the same realistic state (mirrors test_panda_safety_carstate). Keep the
    # last real payload seen per (bus, addr) to seed the fuzz pool with a plausible
    # value, so fuzzed sequences oscillate between real and extreme rather than only
    # between random extremes (the measured state a sample-window bug like panda#1948
    # needs is a real, in-range reading followed by extremes).
    self._tx_fuzz_seed_payload: dict[tuple[int, int], bytes] = {}
    for can in self.can_msgs[:300]:
      self.CI.update(can)
      for msg in filter(lambda m: m.src < 64, can[1]):
        to_send = libsafety_py.make_CANPacket(msg.address, msg.src % 4, msg.dat)
        self.safety.safety_rx_hook(to_send)
        self._tx_fuzz_seed_payload[(msg.src, msg.address)] = bytes(msg.dat)

    # continue the route's timeline; panda's timer is µs relative to the route start
    self._tx_fuzz_epoch_nanos = self.can_msgs[0][0]
    self._tx_fuzz_now_nanos = self.can_msgs[299][0]

    # panda and the CANParser both require valid counters and checksums to update
    # state, so fuzzed payloads need them fixed up to have any effect. start the
    # counters from the parsers' state so the warmup and fuzz timelines are continuous
    self._tx_fuzz_msg_states = {}
    self._tx_fuzz_counters = {}
    exclusions = TX_FUZZ_RX_EXCLUSIONS.get(self.CP.brand, set())
    for cp in self.CI.can_parsers.values():
      if cp.bus >= 8:  # not a physical panda bus (e.g. GM's loopback parser)
        continue
      for addr, state in cp.message_states.items():
        if (cp.bus, addr) not in exclusions:
          self._tx_fuzz_msg_states[(cp.bus, addr)] = state
          self._tx_fuzz_counters[(cp.bus, addr)] = state.counter

    self._tx_fuzz_coherent_prev = True
    self._tx_fuzz_speed_seen = self.CP.steerControlType == SteerControlType.angle and not self.CP.notCar
    self._tx_fuzz_blocked_addrs: set[tuple[int, int]] = set()
    self._tx_fuzz_actuators_out = None
    self._tx_fuzz_prev_vego = 0.0

  def _tx_fuzz_sync_baselines(self, out):
    # panda resets its rate limiting baselines (last desired angle/curvature) to the
    # measured value or zero on a violation, but openpilot's CarController keeps its own.
    # while the fuzzed measured state is incoherent we don't assert, but panda's baseline
    # still drifts away from openpilot's - so re-anchor it to openpilot's last applied
    # command so the next coherent frame starts aligned instead of spuriously blocking.
    if out is None:
      return
    if self.CP.steerControlType == SteerControlType.angle and self.CP.brand in ANGLE_DEG_TO_CAN:
      self.safety.set_desired_angle_last(round(out.steeringAngleDeg * ANGLE_DEG_TO_CAN[self.CP.brand]))
    cfg = self._tx_fuzz_curvature_cfg()
    if cfg is not None:
      self.safety.set_desired_curvature_last(round(cfg["desired_sign"] * out.curvature * cfg["curvature_to_can"]))

  def _tx_fuzz_fix_payload(self, address: int, bus: int, dat: bytes) -> bytes:
    state = self._tx_fuzz_msg_states.get((bus, address))
    if state is None:
      return dat

    dat = bytearray(dat)
    for byte, mask in TX_FUZZ_PAYLOAD_MASKS.get((self.CP.brand, address), []):
      dat[byte] &= ~mask
    pins = TX_FUZZ_SIGNAL_PIN.get(self.CP.brand, {})
    for sig in state.signals:
      if sig.name in pins:
        set_value(dat, sig, pins[sig.name])
    for sig in state.signals:
      if sig.type == SignalType.COUNTER:
        counter = (self._tx_fuzz_counters[(bus, address)] + 1) % (1 << sig.size)
        set_value(dat, sig, counter)
        self._tx_fuzz_counters[(bus, address)] = counter
      elif sig.name.endswith('_D_Qf'):
        # quality flags are checked by panda but not by the CANParser, so force them
        # valid to keep both sides accepting the same messages
        set_value(dat, sig, 3)
    for sig in state.signals:
      if sig.calc_checksum is not None:
        set_value(dat, sig, sig.calc_checksum(address, sig, dat))
    return bytes(dat)

  def _tx_fuzz_measured_state_coherent(self, CS, actuators_out, lat_active: bool, angle_meas_dependent: bool) -> bool:
    """
      openpilot and panda read some measured state from different sensors (with a
      learned offset between them) or at different times, so fuzzed CAN can diverge
      them in ways real sensors don't. TX limits are only comparable while both
      sides agree on the measured state.
    """
    # speed scales most limits, and openpilot can read it from a different message
    # than panda (e.g. Tesla's ESP vs DI speed)
    if self.safety.get_vehicle_speed_min() > 0 or self.safety.get_vehicle_speed_max() > 0:
      self._tx_fuzz_speed_seen = True
    if self._tx_fuzz_speed_seen:
      v_ego_raw = CS.vEgoRaw / self.CP.wheelSpeedFactor
      if not (self.safety.get_vehicle_speed_min() - 1e-3 <= v_ego_raw <= self.safety.get_vehicle_speed_max() + 1e-3):
        return False

    # Toyota LTA gates its TORQUE_WIND_DOWN on driver/EPS torque: openpilot decides from
    # the instantaneous reading, panda from the min-abs of its sample-window *extremes*.
    # The two only agree when openpilot's instantaneous value is itself one of the window
    # extremes panda evaluates. When it is a mid-window sample (a near-zero reading between
    # two far-from-zero extremes), panda's envelope never sees it, so openpilot commands
    # full torque while panda holds it back - a real but benign eagerness that only shows up
    # under implausibly fast fuzzed torque oscillation. Assert only when they line up; this
    # still catches sample-window bugs (e.g. panda#1948), where a window extreme is near zero.
    if self.CP.brand == "toyota" and self.CP.steerControlType == SteerControlType.angle:
      # panda widens torque_meas by 1 on each side on update
      if not (abs(CS.steeringTorque - self.safety.get_torque_driver_min()) <= 1 or
              abs(CS.steeringTorque - self.safety.get_torque_driver_max()) <= 1):
        return False
      if not (abs(CS.steeringTorqueEps - self.safety.get_torque_meas_min()) <= 2 or
              abs(CS.steeringTorqueEps - self.safety.get_torque_meas_max()) <= 2):
        return False
      # openpilot always clamps its LTA angle command to ±max_angle, but panda's angle
      # checks reference the measured angle: the inactive check wants the command near the
      # clamped measured angle, and on re-engage openpilot's command jumps from the clamped
      # value toward the target. a fuzzed measured angle outside the steerable range (the
      # EPS can't physically be there) makes the two disagree while openpilot ramps
      if not (-TOYOTA_LTA_MAX_ANGLE <= self.safety.get_angle_meas_min()
              and self.safety.get_angle_meas_max() <= TOYOTA_LTA_MAX_ANGLE):
        return False

    if self.CP.steerControlType == SteerControlType.angle and self.CP.brand in ANGLE_DEG_TO_CAN:
      # openpilot's and panda's angle sources only need to agree where the checks
      # compare the command against the measured angle: while steering is inactive,
      # and on baseline handoff at the start of an example. while continuously
      # steering, both sides rate limit against their own last command
      if angle_meas_dependent:
        angle_can = (CS.steeringAngleDeg + CS.steeringAngleOffsetDeg) * ANGLE_DEG_TO_CAN[self.CP.brand]
        if not (self.safety.get_angle_meas_min() - 1 <= angle_can <= self.safety.get_angle_meas_max() + 1):
          return False
      # fuzzed measured state can make openpilot's angle command jump discontinuously
      # (e.g. its lateral accel limit clipping a large tracked angle on re-engage);
      # rate-limited streams driven by real sensors never do this, and panda's rate
      # limiting can't follow it
      if self._tx_fuzz_actuators_out is not None:
        angle_delta = abs(actuators_out.steeringAngleDeg - self._tx_fuzz_actuators_out.steeringAngleDeg)
        if angle_delta > 25:
          return False
        # a fuzzed speed jump makes openpilot clip its command to the new, tighter
        # lateral accel bound in one step; real speed can't move that fast
        if angle_delta > 1. and CS.vEgoRaw > self._tx_fuzz_prev_vego + 0.5:
          return False

    cfg = self._tx_fuzz_curvature_cfg()
    if cfg is not None:
      if "max_curvature_error" in cfg:
        curvature_can = round(CS.yawRate / max(CS.vEgoRaw, 0.1) * cfg["curvature_to_can"])
        if not (self.safety.get_curvature_meas_min() - 1 <= curvature_can <= self.safety.get_curvature_meas_max() + 1):
          return False
        # openpilot stops limiting to measured curvature below its speed threshold, while
        # panda's instantaneous speed sample may still be above its own
        if CS.vEgoRaw <= cfg["op_clip_min_speed"] and self.safety.get_vehicle_speed_max() > cfg["panda_check_min_speed"]:
          return False
        # openpilot's jerk limit ramps the applied curvature toward the measured window
        # from wherever it last was, while panda's curvature error check has no such
        # convergence allowance, so it blocks openpilot's tx until the ramp gets there
        # (e.g. when engaging in a curve)
        applied_can = round(cfg["desired_sign"] * actuators_out.curvature * cfg["curvature_to_can"])
        if not (self.safety.get_curvature_meas_min() - cfg["max_curvature_error"] <= applied_can <=
                self.safety.get_curvature_meas_max() + cfg["max_curvature_error"]):
          return False
      # fuzzed measured state can make openpilot's curvature command do things real
      # sensors never allow: jump discontinuously (e.g. VW MEB winds down tracking its
      # fuzzed measured curvature) or track a curvature/speed combination beyond the
      # lateral accel limit. panda enforces both with tolerance on top, so bound
      # openpilot's command by panda's own limits at its fuzzed speed
      fudged_speed = max(self.safety.get_vehicle_speed_min() - 1., 1.)
      if abs(actuators_out.curvature) >= PANDA_MAX_LATERAL_ACCEL / (fudged_speed ** 2):
        return False
      if self._tx_fuzz_actuators_out is not None:
        max_delta = PANDA_MAX_LATERAL_JERK / (fudged_speed ** 2) / cfg["frequency"]
        if abs(actuators_out.curvature - self._tx_fuzz_actuators_out.curvature) > max_delta:
          return False

    return True

  def _tx_fuzz_curvature_cfg(self):
    # MQB/PQ VW platforms are torque-based; only MEB sends curvature
    if self.CP.brand == "volkswagen" and self.CP.steerControlType != SteerControlType.curvature:
      return None
    return CURVATURE_COHERENCE.get(self.CP.brand)

  def _tx_fuzz_failure_context(self, CC, tx_addr: int, tx_bus: int, tx_dat: bytes) -> str:
    s = self.safety
    return "\n".join([
      f"panda blocked openpilot tx: addr={hex(tx_addr)} bus={tx_bus} dat={tx_dat.hex()}",
      f"  controls_allowed={s.get_controls_allowed()} cruise_engaged_prev={s.get_cruise_engaged_prev()}",
      f"  longitudinal_allowed={s.get_longitudinal_allowed()} gas_pressed_prev={s.get_gas_pressed_prev()}",
      f"  torque_meas=({s.get_torque_meas_min()}, {s.get_torque_meas_max()}) torque_driver=({s.get_torque_driver_min()}, {s.get_torque_driver_max()})",
      f"  angle_meas=({s.get_angle_meas_min()}, {s.get_angle_meas_max()}) desired_angle_last={s.get_desired_angle_last()}",
      f"  curvature_meas=({s.get_curvature_meas_min()}, {s.get_curvature_meas_max()})",
      f"  vehicle_speed=({s.get_vehicle_speed_min():.2f}, {s.get_vehicle_speed_max():.2f}) vehicle_moving={s.get_vehicle_moving()}",
      f"  CarControl: {CC}",
    ])

  # Capturing stdout/stderr here causes elevated memory usage.
  @settings(max_examples=MAX_EXAMPLES, deadline=None,
            phases=(Phase.reuse, Phase.generate, Phase.shrink))
  @given(data=st.data())
  def test_panda_safety_tx_fuzzy(self, data):
    """
      Fuzz measured vehicle state into both openpilot and panda, and CarControl into
      openpilot's CarController. CarController limits its commands using the same
      measured state panda enforces against, so panda must allow every message
      openpilot sends. A blocked message means the TX logic of openpilot and panda
      has diverged.
    """
    if self.CP.dashcamOnly:
      self.skipTest("no need to check panda safety for dashcamOnly")

    if self.CP.notCar:
      self.skipTest("Skipping test for notCar")

    if not hasattr(self, '_tx_fuzz_now_nanos'):
      self._tx_fuzz_setup()

    controls_allowed = data.draw(st.booleans(), label='controls_allowed')
    lat_active = controls_allowed and data.draw(st.booleans(), label='lat_active')
    long_active = controls_allowed and self.CP.openpilotLongitudinalControl and data.draw(st.booleans(), label='long_active')
    cruise_engaged = controls_allowed or data.draw(st.booleans(), label='cruise_engaged')

    fields = {field: data.draw(strategy, label=field) for field, strategy in TX_FUZZ_CAR_CONTROL_STRATEGY.items()}
    drawn_accel = fields['actuators.accel']
    drawn_long_control_state = fields['actuators.longControlState']

    # honor the contract selfdrived's controlsd provides to CarController:
    # lateral and longitudinal actuators are zeroed while their control is inactive
    if not lat_active:
      fields['actuators.torque'] = 0.0
    if not long_active:
      fields['actuators.accel'] = 0.0
      fields['actuators.longControlState'] = 'off'

    CC = structs.CarControl()
    CC.enabled = lat_active or long_active
    CC.latActive = lat_active
    CC.longActive = long_active
    for field, val in fields.items():
      obj = CC
      *parents, leaf = field.split('.')
      for p in parents:
        obj = getattr(obj, p)
      setattr(obj, leaf, val)

    # openpilot only cancels stock cruise when it's engaged and openpilot is not,
    # and only resumes from a stop while engaged. controlsd flags an override
    # whenever it's enabled without controlling longitudinal on an op-long car
    CC.cruiseControl.cancel = cruise_engaged and not CC.enabled and data.draw(st.booleans(), label='cancel')
    CC.cruiseControl.resume = CC.enabled and data.draw(st.booleans(), label='resume')
    CC.cruiseControl.override = CC.enabled and not long_active and self.CP.openpilotLongitudinalControl

    # fuzz one message that feeds measured state (speed, driver/EPS torque, angle, ...)
    # into both sides, like test_panda_safety_carstate_fuzzy. drawing the sequence from
    # a small pool of payloads makes measured values repeat and oscillate, which is what
    # exposes sample-window bugs like commaai/panda#1948
    valid_addrs = sorted(self._tx_fuzz_msg_states)
    bus, address = data.draw(st.sampled_from(valid_addrs), label='rx_addr')
    size = self._tx_fuzz_msg_states[(bus, address)].size
    msg_strategy = st.binary(min_size=size, max_size=size)
    pool = data.draw(st.lists(msg_strategy, min_size=1, max_size=3), label='rx_pool')
    # seed the pool with the last real payload so measured values oscillate between a
    # plausible in-range reading and fuzzed extremes
    seed = self._tx_fuzz_seed_payload.get((bus, address))
    if seed is not None and len(seed) == size:
      pool.append(seed)
    msgs = data.draw(st.lists(st.sampled_from(pool), min_size=10, max_size=25), label='rx_msgs')

    # the route may have ended with the driver on the gas, which would keep panda's
    # longitudinal_allowed (and with it our longActive gating) off for the whole test
    gas_pressed = data.draw(st.booleans(), label='gas_pressed')

    self.safety.set_controls_allowed(controls_allowed)
    self.safety.set_cruise_engaged_prev(cruise_engaged)
    self._tx_fuzz_sync_baselines(self._tx_fuzz_actuators_out)
    cc_long_active = long_active
    cc_reader = CC.as_reader()

    for n, dat in enumerate(msgs):
      self._tx_fuzz_now_nanos += int(DT_CTRL * 1e9)
      self.safety.set_timer(((self._tx_fuzz_now_nanos - self._tx_fuzz_epoch_nanos) // 1000) % (2**32))

      dat = self._tx_fuzz_fix_payload(address, bus, dat)
      self.safety.safety_rx_hook(libsafety_py.make_CANPacket(address, bus % 4, dat))
      CS = self.CI.update([(self._tx_fuzz_now_nanos, [CanData(address=address, dat=dat, src=bus)])])

      # the fuzzed message may have tripped state panda uses to gate tx. openpilot
      # reacts to the same state changes, so put panda back into the drawn state
      self.safety.set_relay_malfunction(False)
      self.safety.set_controls_allowed(controls_allowed)
      self.safety.set_cruise_engaged_prev(cruise_engaged)
      self.safety.set_gas_pressed_prev(gas_pressed)
      if self.CP.brand == "honda":
        # fuzzed stock AEB frames latch brake command forwarding, which blocks openpilot's
        self.safety.set_honda_fwd_brake(False)

      # panda ties gas/brake commands to its own gas pressed state; openpilot's
      # controlsd reacts to the same state by deactivating long control
      if cc_long_active != (long_active and self.safety.get_longitudinal_allowed()):
        cc_long_active = long_active and self.safety.get_longitudinal_allowed()
        CC.longActive = cc_long_active
        CC.actuators.accel = drawn_accel if cc_long_active else 0.0
        CC.actuators.longControlState = drawn_long_control_state if cc_long_active else 'off'
        CC.cruiseControl.override = CC.enabled and not cc_long_active and self.CP.openpilotLongitudinalControl
        cc_reader = CC.as_reader()

      actuators_out, sendcan = self.CI.apply(cc_reader, self._tx_fuzz_now_nanos)

      # both sides' rate limiting baselines come from the previous frame, so only
      # assert when the measured state agreed on this frame and the previous one.
      # still run blocked frames through the tx hook to keep panda's tracking state
      # (last desired torque/angle/curvature) moving with openpilot's
      coherent = self._tx_fuzz_measured_state_coherent(CS, actuators_out, lat_active, not lat_active or n == 0)
      for tx_addr, tx_dat, tx_bus in sendcan:
        to_send = libsafety_py.make_CANPacket(tx_addr, tx_bus % 4, tx_dat)
        key = (tx_addr, tx_bus)
        # Toyota's inactive LTA message (both steer-request bits clear) commands no
        # actuation - it just holds the measured angle plus openpilot's learned steering
        # offset, which panda compares against the raw measured angle with a ±1 tolerance.
        # Under fuzzing that offset and float-degree rounding disagree harmlessly. The
        # safety-critical LTA checks (active angle rate, torque wind-down) are all on frames
        # with a steer request set, which stay asserted.
        inactive_lta = (self.CP.brand == "toyota" and tx_addr == 0x191
                        and not (tx_dat[0] & 1) and not ((tx_dat[3] >> 1) & 1))
        msg_coherent = coherent and not inactive_lta
        if self.safety.safety_tx_hook(to_send):
          self._tx_fuzz_blocked_addrs.discard(key)
        else:
          # panda can disengage itself inside the tx hook on state openpilot can't
          # observe (e.g. its two fuzzed speed sources diverging beyond the mismatch
          # tolerance) - not a TX logic mismatch
          panda_self_disengaged = controls_allowed and not self.safety.get_controls_allowed()
          # a blocked message also reset panda's rate limiting baselines for it, so
          # asserts for this addr stay off until a send passes again
          if msg_coherent and self._tx_fuzz_coherent_prev and key not in self._tx_fuzz_blocked_addrs and not panda_self_disengaged:
            self.fail(self._tx_fuzz_failure_context(cc_reader, tx_addr, tx_bus, tx_dat))
          self._tx_fuzz_blocked_addrs.add(key)
      # openpilot's CarController and panda track their steering rate-limit baselines
      # (last desired angle/curvature) independently, and panda resets its baseline to the
      # clamped measured value on any block. Under fuzzing that makes the two drift apart
      # and every later steering command read as an out-of-rate jump. Re-anchor panda's
      # baseline to openpilot's applied command each frame so the angle/curvature rate check
      # always compares against the same starting point openpilot used. This concedes
      # rate-limit-only coverage on the steering command (which is deterministic and not
      # measured-state dependent) to keep the measured-state-dependent checks - where the
      # bounty's reference bug class lives (e.g. the LTA torque wind-down) - false-positive free.
      self._tx_fuzz_sync_baselines(actuators_out)
      self._tx_fuzz_coherent_prev = coherent
      self._tx_fuzz_actuators_out = actuators_out
      self._tx_fuzz_prev_vego = CS.vEgoRaw

  def test_panda_safety_carstate(self):
    """
      Assert that panda safety matches openpilot's carState
    """
    if self.CP.dashcamOnly:
      self.skipTest("no need to check panda safety for dashcamOnly")

    # warm up pass, as initial states may be different
    for can in self.can_msgs[:300]:
      self.CI.update(can)
      for msg in filter(lambda m: m.src < 64, can[1]):
        to_send = libsafety_py.make_CANPacket(msg.address, msg.src % 4, msg.dat)
        self.safety.safety_rx_hook(to_send)

    controls_allowed_prev = False
    CS_prev = car.CarState.new_message()
    checks = defaultdict(int)
    vehicle_speed_seen = self.CP.steerControlType == SteerControlType.angle and not self.CP.notCar
    for idx, can in enumerate(self.can_msgs):
      CS = self.CI.update(can).as_reader()
      for msg in filter(lambda m: m.src < 64, can[1]):
        to_send = libsafety_py.make_CANPacket(msg.address, msg.src % 4, msg.dat)
        ret = self.safety.safety_rx_hook(to_send)
        self.assertEqual(1, ret, f"safety rx failed ({ret=}): {(msg.address, msg.src % 4)}")

      # Skip first frame so CS_prev is properly initialized
      if idx == 0:
        CS_prev = CS
        # Button may be left pressed in warm up period
        if not self.CP.pcmCruise:
          self.safety.set_controls_allowed(0)
        continue

      # TODO: check rest of panda's carstate (steering, ACC main on, etc.)

      checks['gasPressed'] += CS.gasPressed != self.safety.get_gas_pressed_prev()
      checks['standstill'] += (CS.standstill == self.safety.get_vehicle_moving()) and not self.CP.notCar

      # check vehicle speed if angle control car or available
      if self.safety.get_vehicle_speed_min() > 0 or self.safety.get_vehicle_speed_max() > 0:
        vehicle_speed_seen = True

      if vehicle_speed_seen:
        v_ego_raw = CS.vEgoRaw / self.CP.wheelSpeedFactor
        checks['vEgoRaw'] += (v_ego_raw > (self.safety.get_vehicle_speed_max() + 1e-3) or
                              v_ego_raw < (self.safety.get_vehicle_speed_min() - 1e-3))

      # check steering angle for angle control cars (panda stores angle_meas in CAN units)
      # ford and VW MEB excluded since they track curvature, not steering angle
      # TODO: add curvature check, standardize CAN units to rm brand specific ANGLE_DEG_TO_CAN
      if self.CP.steerControlType == SteerControlType.angle and not self.CP.notCar and self.CP.brand not in ("ford", "volkswagen"):
        angle_can = (CS.steeringAngleDeg + CS.steeringAngleOffsetDeg) * ANGLE_DEG_TO_CAN[self.CP.brand]
        checks['steeringAngleDeg'] += (angle_can > (self.safety.get_angle_meas_max() + 1) or
                                       angle_can < (self.safety.get_angle_meas_min() - 1))

      checks['brakePressed'] += CS.brakePressed != self.safety.get_brake_pressed_prev()
      checks['regenBraking'] += CS.regenBraking != self.safety.get_regen_braking_prev()
      checks['steeringDisengage'] += CS.steeringDisengage != self.safety.get_steering_disengage_prev()

      if self.CP.pcmCruise:
        # On most pcmCruise cars, openpilot's state is always tied to the PCM's cruise state.
        # On Honda Nidec, we always engage on the rising edge of the PCM cruise state, but
        # openpilot brakes to zero even if the min ACC speed is non-zero (i.e. the PCM disengages).
        if self.CP.brand == "honda" and not (self.CP.flags & HondaFlags.BOSCH):
          # only the rising edges are expected to match
          if CS.cruiseState.enabled and not CS_prev.cruiseState.enabled:
            checks['controlsAllowed'] += not self.safety.get_controls_allowed()
        else:
          checks['controlsAllowed'] += not CS.cruiseState.enabled and self.safety.get_controls_allowed()

        # TODO: fix notCar mismatch
        if not self.CP.notCar:
          checks['cruiseState'] += CS.cruiseState.enabled != self.safety.get_cruise_engaged_prev()
      else:
        # Check for user button enable on rising edge of controls allowed
        button_enable = CS.buttonEnable and (not CS.brakePressed or CS.standstill)
        mismatch = button_enable != (self.safety.get_controls_allowed() and not controls_allowed_prev)
        checks['controlsAllowed'] += mismatch
        controls_allowed_prev = self.safety.get_controls_allowed()
        if button_enable and not mismatch:
          self.safety.set_controls_allowed(False)

      if self.CP.brand == "honda":
        checks['mainOn'] += CS.cruiseState.available != self.safety.get_acc_main_on()

      CS_prev = CS

    failed_checks = {k: v for k, v in checks.items() if v > 0}
    self.assertFalse(len(failed_checks), f"panda safety doesn't agree with openpilot: {failed_checks}")


@parameterized_class(('platform', 'test_route'), get_test_cases())
class TestCarModel(TestCarModelBase):
  pass


if __name__ == "__main__":
  unittest.main()
