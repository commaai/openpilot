#!/usr/bin/env python3
import capnp
import os
import importlib
import time
import pytest
import random
import unittest
from collections import defaultdict, Counter
from typing import List, Optional, Tuple
from parameterized import parameterized_class
import hypothesis.strategies as st
from hypothesis import HealthCheck, Phase, assume, given, settings, seed
from pympler.tracker import SummaryTracker
import gc

from cereal import messaging, log, car
from openpilot.common.basedir import BASEDIR
from openpilot.common.params import Params
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.car import CanBusBase
from openpilot.selfdrive.car.fingerprints import all_known_cars
from openpilot.selfdrive.car.car_helpers import FRAME_FINGERPRINT, interfaces
from openpilot.selfdrive.car.toyota.values import TSS2_CAR
from openpilot.selfdrive.car.honda.values import CAR as HONDA, HONDA_BOSCH
from openpilot.selfdrive.car.tests.routes import non_tested_cars, routes, CarTestRoute
from openpilot.selfdrive.controls.controlsd import Controls
from openpilot.selfdrive.test.openpilotci import get_url
from openpilot.tools.lib.logreader import LogReader
from openpilot.tools.lib.route import Route, SegmentName, RouteName

from panda.tests.libpanda import libpanda_py

EventName = car.CarEvent.EventName
PandaType = log.PandaState.PandaType
SafetyModel = car.CarParams.SafetyModel

NUM_JOBS = int(os.environ.get("NUM_JOBS", "1"))
JOB_ID = int(os.environ.get("JOB_ID", "0"))
INTERNAL_SEG_LIST = os.environ.get("INTERNAL_SEG_LIST", "")
INTERNAL_SEG_CNT = int(os.environ.get("INTERNAL_SEG_CNT", "0"))


def get_test_cases() -> List[Tuple[str, Optional[CarTestRoute]]]:
  # build list of test cases
  test_cases = []
  if not len(INTERNAL_SEG_LIST):
    routes_by_car = defaultdict(set)
    for r in routes:
      routes_by_car[r.car_model].add(r)

    for i, c in enumerate(sorted(all_known_cars())):
      if i % NUM_JOBS == JOB_ID:
        test_cases.extend(sorted((c.value, r) for r in routes_by_car.get(c, (None,))))

  else:
    with open(os.path.join(BASEDIR, INTERNAL_SEG_LIST), "r") as f:
      seg_list = f.read().splitlines()

    cnt = INTERNAL_SEG_CNT or len(seg_list)
    seg_list_iter = iter(seg_list[:cnt])

    for platform in seg_list_iter:
      platform = platform[2:]  # get rid of comment
      segment_name = SegmentName(next(seg_list_iter))
      test_cases.append((platform, CarTestRoute(segment_name.route_name.canonical_name, platform,
                                                segment=segment_name.segment_num)))
  return test_cases


@pytest.mark.slow
class TestCarModelBase(unittest.TestCase):
  car_model: Optional[str] = None
  test_route: Optional[CarTestRoute] = None
  ci: bool = True

  can_msgs: List[capnp.lib.capnp._DynamicStructReader]
  elm_frame: Optional[int]
  car_safety_mode_frame: Optional[int]

  @classmethod
  def setUpClass(cls):
    if cls.__name__ == 'TestCarModel' or cls.__name__.endswith('Base'):
      raise unittest.SkipTest

    if 'FILTER' in os.environ:
      if not cls.car_model.startswith(tuple(os.environ.get('FILTER').split(','))):
        raise unittest.SkipTest

    if cls.test_route is None:
      if cls.car_model in non_tested_cars:
        print(f"Skipping tests for {cls.car_model}: missing route")
        raise unittest.SkipTest
      raise Exception(f"missing test route for {cls.car_model}")

    test_segs = (2, 1, 0)
    if cls.test_route.segment is not None:
      test_segs = (cls.test_route.segment,)

    for seg in test_segs:
      try:
        if len(INTERNAL_SEG_LIST):
          route_name = RouteName(cls.test_route.route)
          lr = LogReader(f"cd:/{route_name.dongle_id}/{route_name.time_str}/{seg}/rlog.bz2")
        elif cls.ci:
          lr = LogReader(get_url(cls.test_route.route, seg))
        else:
          lr = LogReader(Route(cls.test_route.route).log_paths()[seg])
      except Exception:
        continue

      car_fw = []
      can_msgs = []
      cls.elm_frame = None
      cls.car_safety_mode_frame = None
      cls.fingerprint = defaultdict(dict)
      experimental_long = False
      for msg in lr:
        if msg.which() == "can":
          can_msgs.append(msg)
          if len(can_msgs) <= FRAME_FINGERPRINT:
            for m in msg.can:
              if m.src < 64:
                cls.fingerprint[m.src][m.address] = len(m.dat)

        elif msg.which() == "carParams":
          car_fw = msg.carParams.carFw
          if msg.carParams.openpilotLongitudinalControl:
            experimental_long = True
          if cls.car_model is None and not cls.ci:
            cls.car_model = msg.carParams.carFingerprint

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

      if len(can_msgs) > int(50 / DT_CTRL):
        break
    else:
      raise Exception(f"Route: {repr(cls.test_route.route)} with segments: {test_segs} not found or no CAN msgs found. Is it uploaded?")

    # if relay is expected to be open in the route
    cls.openpilot_enabled = cls.car_safety_mode_frame is not None

    cls.can_msgs = sorted(can_msgs, key=lambda msg: msg.logMonoTime)

    cls.CarInterface, cls.CarController, cls.CarState = interfaces[cls.car_model]
    cls.CP = cls.CarInterface.get_params(cls.car_model, cls.fingerprint, car_fw, experimental_long, docs=False)
    assert cls.CP
    assert cls.CP.carFingerprint == cls.car_model

    cls.car_state_dict = {'panda': {'gas_pressed': False}, 'CS': {'gasPressed': False}}
    cls.init_gas_pressed = False

  @classmethod
  def tearDownClass(cls):
    del cls.can_msgs
    gc.collect()

  def setUp(self):
    # print('SETUP HEREHEREHEREHEREHEREHEREHEREHEREHEREHEREHEREHEREHEREHEREHEREHEREHERE')
    self.CI = self.CarInterface(self.CP.copy(), self.CarController, self.CarState)
    assert self.CI

    Params().put_bool("OpenpilotEnabledToggle", self.openpilot_enabled)

    # TODO: check safetyModel is in release panda build
    self.safety = libpanda_py.libpanda

    cfg = self.CP.safetyConfigs[-1]
    set_status = self.safety.set_safety_hooks(cfg.safetyModel.raw, cfg.safetyParam)
    self.assertEqual(0, set_status, f"failed to set safetyModel {cfg}")
    self.safety.init_tests()

    # self.tracker = SummaryTracker()
    # for _ in range(5):
    #   self.tracker.print_diff()

  # def tearDown(self):
  #   self.tracker.print_diff()
  #   # for _type, num_objects, total_size in self.tracker.diff():
  #   #   print(_type, num_objects, total_size)
  #   #   # with self.subTest(_type=_type):
  #   #   #   self.assertLess(total_size / 1024, 10, f'Object {_type} ({num_objects=}) grew larger than 10 kB while uploading file')

  # def test_honda_buttons(self):
  #   if self.CP.carFingerprint in HONDA_BOSCH:
  #     return
  #
  #   # self.assertTrue(0x17C in self.fingerprint[1])
  #   self.assertTrue(0x1BE in self.fingerprint[0])
  #   # for msg in self.can_msgs:
  #   #   for can in msg.can:
  #   #     self.assertTrue(can.address != 0x1a6)

  @settings(max_examples=600, deadline=None,
            phases=(Phase.reuse, Phase.generate, ),
            suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow, HealthCheck.large_base_example],
            )
  @given(data=st.data())
  @seed(1)  # for reproduction
  def test_panda_safety_carstate_fuzzy(self, data):
    # raise unittest.SkipTest
    # return
    # TODO: how much of test_panda_safety_carstate can we re-use?
    """
      Assert that panda safety matches openpilot's carState by fuzzing the CAN data
    """
    if self.CP.dashcamOnly:
      self.skipTest("no need to check panda safety for dashcamOnly")
    state_has_changed = lambda prev_state, new_state: prev_state != new_state
    # cfg = self.CP.safetyConfigs[-1]
    # set_status = self.safety.set_safety_hooks(cfg.safetyModel.raw, cfg.safetyParam)
    # self.assertEqual(0, set_status, f"failed to set safetyModel {cfg}")
    # self.safety.init_tests()


    # bus = 0  # random.randint(0, 3)
    bus_offset = CanBusBase(None, fingerprint=self.fingerprint).offset
    bus = bus_offset
    # address = data.draw(st.sampled_from([i for i in self.fingerprint[0] if i < 0x700]))  # random.randint(0x200, 0x300)
    address = data.draw(st.sampled_from([i for i in self.fingerprint[bus]]))  # random.randint(0x200, 0x300)
    # addresses = [i for i in self.fingerprint[bus_offset]]
    # weighted_address_strategy = st.sampled_from(sorted(addresses, key=lambda x: random.choices(addresses, weights=[(1 / (i + 1)) for i in range(len(addresses))])[0]))
    # address = data.draw(weighted_address_strategy)
    # if address not in self.fingerprint[bus_offset]:
    #   raise unittest.SkipTest
    size = self.fingerprint[bus][address]
    print(address, size)
    # if self.CP.carFingerprint in TSS2_CAR:
    #   raise unittest.SkipTest
    # print(self.fingerprint)

    # address = data.draw(st.integers(0x201, 0x226))

    # ORIG:
    # msg_strategy = st.tuples(st.integers(min_value=0, max_value=0), st.integers(min_value=0x100, max_value=0x400), st.binary(min_size=8, max_size=8))
    # msg_strategy = st.tuples(st.integers(min_value=0xaa, max_value=0xaa), st.binary(min_size=8, max_size=8))

    msg_strategy = st.binary(min_size=size, max_size=size)
    msgs = data.draw(st.lists(msg_strategy, min_size=20))
    # time.sleep(8)
    # print(len(msgs))

    prev_panda_gas = self.safety.get_gas_pressed_prev()
    prev_panda_brake = self.safety.get_brake_pressed_prev()
    prev_panda_regen_braking = self.safety.get_regen_braking_prev()
    prev_panda_vehicle_moving = self.safety.get_vehicle_moving()
    prev_panda_cruise_engaged = self.safety.get_cruise_engaged_prev()
    prev_panda_acc_main_on = self.safety.get_acc_main_on()

    start_gas = self.safety.get_gas_pressed_prev()
    start_gas_int_detected = self.safety.get_gas_interceptor_detected()

    # for bus, address, dat in msgs:
    # since all toyotas can detect fake interceptor, but we want to test PCM gas too
    for dat in msgs:
      # set interceptor detected so we don't accidentally trigger gas_pressed with other message
      self.safety.set_gas_interceptor_detected(self.CP.enableGasInterceptor)
      # if not self.CP.enableGasInterceptor:
      #   self.safety.set_gas_interceptor_detected(False)
      print()

      # for i in range(100):
      to_send = libpanda_py.make_CANPacket(address, bus, dat)
      did_rx = self.safety.safety_rx_hook(to_send)

      can = messaging.new_message('can', 1)
      can.can = [log.CanData(address=address, dat=dat, src=bus)]
      # del can.can[0]
      # del can.can
      print('rxing', dict(address=address, dat=dat, src=bus))
      # continue

      CC = car.CarControl.new_message()
      CS = self.CI.update(CC, (can.to_bytes(),))
      # del can
      # del CC
      # continue

      # test multiple CAN packets as well as multiple messages per CAN packet
      # for (_dat1, _dat2) in ((dat1, dat2), (dat3, dat4)):
      #   can = messaging.new_message('can', 2)
      #   can.can = [log.CanData(address=address, dat=_dat1, src=bus), log.CanData(address=address, dat=_dat2, src=bus)]
      #   print('rxing', dict(address=address, _dat1=_dat1, _dat2=_dat2, src=bus))
      #
      #   CC = car.CarControl.new_message()
      #   CS = self.CI.update(CC, (can.to_bytes(),))

      if self.safety.get_gas_pressed_prev():
        self.init_gas_pressed = True

      # due to panda updating state selectively, per message, we can only compare on a change

      # if self.safety.get_gas_interceptor_detected():# and state_has_changed(start_gas, self.safety.get_gas_pressed_prev()):
      # print('ret.gas', CS.gas, 'safety gas', self.safety.get_gas_interceptor_prev())
      # print('both', CS.gasPressed, self.safety.get_gas_pressed_prev(), 'int')
      if self.safety.get_gas_pressed_prev() != prev_panda_gas:
        print()
        print('ret.gas', CS.gas, 'safety gas', self.safety.get_gas_interceptor_prev())
        print('both', CS.gasPressed, self.safety.get_gas_pressed_prev(), 'int')
        print('get_gas_interceptor_detected!')
        print('can.can', can.can)
        # self.assertEqual(CS.gasPressed, self.safety.get_gas_interceptor_prev())
        self.assertEqual(CS.gasPressed, self.safety.get_gas_pressed_prev())
        # self.assertEqual(CS.gas, self.safety.get_gas_interceptor_prev())
        # self.assertFalse(True)

      # TODO: don't fully skip
      if self.CP.carFingerprint not in (HONDA.PILOT, HONDA.RIDGELINE):
        print('both', CS.brakePressed, 'safety brake', self.safety.get_brake_pressed_prev())
        if self.safety.get_brake_pressed_prev() != prev_panda_brake:
          # print('brake change!')
          # print('both', CS.brakePressed, self.safety.get_brake_pressed_prev())
          self.assertEqual(CS.brakePressed, self.safety.get_brake_pressed_prev())

      if self.safety.get_regen_braking_prev() != prev_panda_regen_braking:
        print('regen change!')
        print('both', CS.regenBraking, self.safety.get_regen_braking_prev())
        self.assertEqual(CS.regenBraking, self.safety.get_regen_braking_prev())

      # print('both', not CS.standstill, 'safety moving', self.safety.get_vehicle_moving())
      if self.safety.get_vehicle_moving() != prev_panda_vehicle_moving:
        self.assertEqual(not CS.standstill, self.safety.get_vehicle_moving())

      if not (self.CP.carName == "honda" and self.CP.carFingerprint not in HONDA_BOSCH):
        if self.safety.get_cruise_engaged_prev() != prev_panda_cruise_engaged:
          self.assertEqual(CS.cruiseState.enabled, self.safety.get_cruise_engaged_prev())

      if self.CP.carName == "honda":
        if self.safety.get_acc_main_on() != prev_panda_acc_main_on:
          self.assertEqual(CS.cruiseState.available, self.safety.get_acc_main_on())

      prev_panda_gas = self.safety.get_gas_pressed_prev()
      prev_panda_brake = self.safety.get_brake_pressed_prev()
      prev_panda_regen_braking = self.safety.get_regen_braking_prev()
      prev_panda_vehicle_moving = self.safety.get_vehicle_moving()
      prev_panda_cruise_engaged = self.safety.get_cruise_engaged_prev()
      prev_panda_acc_main_on = self.safety.get_acc_main_on()
      # if self.safety.get_gas_pressed_prev() and self.safety.get_cruise_engaged_prev():
      #   self.assertFalse(True)
      # self.assertFalse(self.safety.get_cruise_engaged_prev())

      # print('gas_pressed', CS.gasPressed, self.safety.get_gas_pressed_prev())
      # print('wheel_speeds', CS.wheelSpeeds)
      # print('standstill', CS.standstill, not self.safety.get_vehicle_moving())

      # print('did_rx', did_rx)
      # if did_rx:
      #   self.assertFalse(True, 'finally did rx: {}, {}'.format(i, dat))
      # self.assertTrue(CS.standstill, (not CS.standstill, self.safety.get_vehicle_moving(), CS.vEgoRaw, CS.wheelSpeeds))


      # self.assertEqual(CS.gasPressed, self.safety.get_gas_pressed_prev())
      # self.assertEqual(not CS.standstill, self.safety.get_vehicle_moving())
      # self.assertEqual(CS.brakePressed, self.safety.get_brake_pressed_prev())
      # self.assertEqual(CS.regenBraking, self.safety.get_regen_braking_prev())
      #
      # if self.CP.pcmCruise:
      #   self.assertEqual(CS.cruiseState.enabled, self.safety.get_cruise_engaged_prev())
      #
      # if self.CP.carName == "honda":
      #   self.assertEqual(CS.cruiseState.available, self.safety.get_acc_main_on())


    # if self.safety.get_gas_interceptor_detected():
    #   print('get_gas_interceptor_detected!')
    #   # self.assertEqual(CS.gasPressed, self.safety.get_gas_interceptor_prev())
    #   self.assertEqual(CS.gasPressed, self.safety.get_gas_pressed_prev())
    #   # self.assertFalse(True)

    # return
    # self.car_state_dict['panda'] = {'gas_pressed': self.safety.get_gas_pressed_prev()}
    # self.car_state_dict['CS'] = {'gasPressed': CS.gasPressed}

    # print(self.safety.get_gas_pressed_prev(), self.safety.get_brake_pressed_prev(), self.safety.get_vehicle_moving(), self.safety.get_cruise_engaged_prev())
    # assume(state_has_changed(False, self.safety.get_gas_pressed_prev()))
    # assume(state_has_changed(start_gas, self.safety.get_gas_pressed_prev()))  # this just goes on forever EDIT: actually no it doesn't
    # assume(state_has_changed(start_gas_int_detected, self.safety.get_gas_interceptor_detected()))
    # assume(state_has_changed(False, self.safety.get_brake_pressed_prev()))
    # assume(state_has_changed(False, self.safety.get_vehicle_moving()))
    # assume(state_has_changed(False, self.safety.get_cruise_engaged_prev()))

    # print(msgs)
    # print('\nresults', self.safety.get_gas_pressed_prev(), self.safety.get_vehicle_moving(), self.safety.get_brake_pressed_prev(), self.safety.get_regen_braking_prev(), self.safety.get_cruise_engaged_prev(), self.safety.get_acc_main_on())
    del msgs

  # def test_panda_safety_carstate(self):
  #   """
  #     Assert that panda safety matches openpilot's carState
  #   """
  #   if self.CP.dashcamOnly:
  #     self.skipTest("no need to check panda safety for dashcamOnly")
  #
  #   CC = car.CarControl.new_message()
  #
  #   # warm up pass, as initial states may be different
  #   for can in self.can_msgs[:300]:
  #     self.CI.update(CC, (can.as_builder().to_bytes(), ))
  #     for msg in filter(lambda m: m.src in range(64), can.can):
  #       to_send = libpanda_py.make_CANPacket(msg.address, msg.src % 4, msg.dat)
  #       self.safety.safety_rx_hook(to_send)
  #
  #   controls_allowed_prev = False
  #   CS_prev = car.CarState.new_message()
  #   checks = defaultdict(lambda: 0)
  #   controlsd = Controls(CI=self.CI)
  #   controlsd.initialized = True
  #   for idx, can in enumerate(self.can_msgs):
  #     CS = self.CI.update(CC, (can.as_builder().to_bytes(), ))
  #     for msg in filter(lambda m: m.src in range(64), can.can):
  #       to_send = libpanda_py.make_CANPacket(msg.address, msg.src % 4, msg.dat)
  #       ret = self.safety.safety_rx_hook(to_send)
  #       self.assertEqual(1, ret, f"safety rx failed ({ret=}): {to_send}")
  #
  #     # Skip first frame so CS_prev is properly initialized
  #     if idx == 0:
  #       CS_prev = CS
  #       # Button may be left pressed in warm up period
  #       if not self.CP.pcmCruise:
  #         self.safety.set_controls_allowed(0)
  #       continue
  #
  #     # TODO: check rest of panda's carstate (steering, ACC main on, etc.)
  #
  #     checks['gasPressed'] += CS.gasPressed != self.safety.get_gas_pressed_prev()
  #     checks['standstill'] += CS.standstill == self.safety.get_vehicle_moving()
  #
  #     # TODO: remove this exception once this mismatch is resolved
  #     brake_pressed = CS.brakePressed
  #     if CS.brakePressed and not self.safety.get_brake_pressed_prev():
  #       if self.CP.carFingerprint in (HONDA.PILOT, HONDA.RIDGELINE) and CS.brake > 0.05:
  #         brake_pressed = False
  #     checks['brakePressed'] += brake_pressed != self.safety.get_brake_pressed_prev()
  #     checks['regenBraking'] += CS.regenBraking != self.safety.get_regen_braking_prev()
  #
  #     if self.CP.pcmCruise:
  #       # On most pcmCruise cars, openpilot's state is always tied to the PCM's cruise state.
  #       # On Honda Nidec, we always engage on the rising edge of the PCM cruise state, but
  #       # openpilot brakes to zero even if the min ACC speed is non-zero (i.e. the PCM disengages).
  #       if self.CP.carName == "honda" and self.CP.carFingerprint not in HONDA_BOSCH:
  #         # only the rising edges are expected to match
  #         if CS.cruiseState.enabled and not CS_prev.cruiseState.enabled:
  #           checks['controlsAllowed'] += not self.safety.get_controls_allowed()
  #       else:
  #         checks['controlsAllowed'] += not CS.cruiseState.enabled and self.safety.get_controls_allowed()
  #
  #       # TODO: fix notCar mismatch
  #       if not self.CP.notCar:
  #         checks['cruiseState'] += CS.cruiseState.enabled != self.safety.get_cruise_engaged_prev()
  #     else:
  #       # Check for enable events on rising edge of controls allowed
  #       controlsd.update_events(CS)
  #       controlsd.CS_prev = CS
  #       button_enable = (any(evt.enable for evt in CS.events) and
  #                        not any(evt == EventName.pedalPressed for evt in controlsd.events.names))
  #       mismatch = button_enable != (self.safety.get_controls_allowed() and not controls_allowed_prev)
  #       checks['controlsAllowed'] += mismatch
  #       controls_allowed_prev = self.safety.get_controls_allowed()
  #       if button_enable and not mismatch:
  #         self.safety.set_controls_allowed(False)
  #
  #     if self.CP.carName == "honda":
  #       checks['mainOn'] += CS.cruiseState.available != self.safety.get_acc_main_on()
  #
  #     CS_prev = CS
  #
  #   failed_checks = {k: v for k, v in checks.items() if v > 0}
  #   self.assertFalse(len(failed_checks), f"panda safety doesn't agree with openpilot: {failed_checks}")


@parameterized_class(('car_model', 'test_route'), get_test_cases())
@pytest.mark.xdist_group_class_property('car_model')
class TestCarModel(TestCarModelBase):
  pass


if __name__ == "__main__":
  unittest.main()
