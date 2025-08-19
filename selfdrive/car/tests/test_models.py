import time
import os
import pytest
import random
import unittest # noqa: TID251
from collections import defaultdict, Counter
import hypothesis.strategies as st
from hypothesis import Phase, given, settings
from parameterized import parameterized_class

from opendbc.car import DT_CTRL, gen_empty_fingerprint, structs
from opendbc.car.can_definitions import CanData
from opendbc.car.car_helpers import FRAME_FINGERPRINT, interfaces
from opendbc.car.fingerprints import MIGRATION
from opendbc.car.honda.values import CAR as HONDA, HondaFlags
from opendbc.car.structs import car
from opendbc.car.tests.routes import non_tested_cars, routes, CarTestRoute
from opendbc.car.values import Platform, PLATFORMS
from opendbc.safety.tests.libsafety import libsafety_py
from openpilot.common.basedir import BASEDIR
from openpilot.selfdrive.pandad import can_capnp_to_list
from openpilot.selfdrive.test.helpers import read_segment_list
from openpilot.system.hardware.hw import DEFAULT_DOWNLOAD_CACHE_ROOT
from openpilot.tools.lib.logreader import LogReader, LogsUnavailable, openpilotci_source, internal_source, comma_api_source
from openpilot.tools.lib.route import SegmentName

SafetyModel = car.CarParams.SafetyModel
SteerControlType = structs.CarParams.SteerControlType

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


@pytest.mark.slow
@pytest.mark.shared_download_cache
class TestCarModelBase(unittest.TestCase):
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
        sources = [internal_source] if len(INTERNAL_SEG_LIST) else [openpilotci_source, comma_api_source]
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

    if self.CP.steerControlType != SteerControlType.angle:
      tuning = self.CP.lateralTuning.which()
      if tuning == 'pid':
        self.assertTrue(len(self.CP.lateralTuning.pid.kpV))
      elif tuning == 'torque':
        self.assertTrue(self.CP.lateralTuning.torque.kf > 0)
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
      if self.openpilot_enabled and t / 1e4 > self.car_safety_mode_frame:
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

  # Skip stdout/stderr capture with pytest, causes elevated memory usage
  @pytest.mark.nocapture
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
        # TODO: remove this exception once this mismatch is resolved
        brake_pressed = CS.brakePressed
        if CS.brakePressed and not self.safety.get_brake_pressed_prev():
          if self.CP.carFingerprint in (HONDA.HONDA_PILOT, HONDA.HONDA_RIDGELINE) and CS.brake > 0.05:
            brake_pressed = False

        self.assertEqual(brake_pressed, self.safety.get_brake_pressed_prev())

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

      # TODO: remove this exception once this mismatch is resolved
      brake_pressed = CS.brakePressed
      if CS.brakePressed and not self.safety.get_brake_pressed_prev():
        if self.CP.carFingerprint in (HONDA.HONDA_PILOT, HONDA.HONDA_RIDGELINE) and CS.brake > 0.05:
          brake_pressed = False
      checks['brakePressed'] += brake_pressed != self.safety.get_brake_pressed_prev()
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
@pytest.mark.xdist_group_class_property('test_route')
class TestCarModel(TestCarModelBase):
  pass


if __name__ == "__main__":
  unittest.main()
