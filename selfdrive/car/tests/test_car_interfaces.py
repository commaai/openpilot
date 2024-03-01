#!/usr/bin/env python3
import os
import math
import unittest
import hypothesis.strategies as st
from hypothesis import Phase, given, settings
import importlib
from parameterized import parameterized

from cereal import car, messaging
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.car import gen_empty_fingerprint
from openpilot.selfdrive.car.car_helpers import interfaces
from openpilot.selfdrive.car.fingerprints import all_known_cars
from openpilot.selfdrive.car.fw_versions import FW_VERSIONS
from openpilot.selfdrive.car.interfaces import get_interface_attr
from openpilot.selfdrive.controls.lib.latcontrol_angle import LatControlAngle
from openpilot.selfdrive.controls.lib.latcontrol_pid import LatControlPID
from openpilot.selfdrive.controls.lib.latcontrol_torque import LatControlTorque
from openpilot.selfdrive.controls.lib.longcontrol import LongControl
from openpilot.selfdrive.test.fuzzy_generation import DrawType, FuzzyGenerator

ALL_ECUS = list({ecu for ecus in FW_VERSIONS.values() for ecu in ecus.keys()})

MAX_EXAMPLES = int(os.environ.get('MAX_EXAMPLES', '40'))


def get_fuzzy_car_interface_args(draw: DrawType) -> dict:
  # Fuzzy CAN fingerprints and FW versions to test more states of the CarInterface
  fingerprint_strategy = st.fixed_dictionaries({key: st.dictionaries(st.integers(min_value=0, max_value=0x800),
                                                                     st.integers(min_value=0, max_value=64)) for key in
                                                gen_empty_fingerprint()})

  # only pick from possible ecus to reduce search space
  car_fw_strategy = st.lists(st.sampled_from(ALL_ECUS))

  params_strategy = st.fixed_dictionaries({
    'fingerprints': fingerprint_strategy,
    'car_fw': car_fw_strategy,
    'experimental_long': st.booleans(),
  })

  params: dict = draw(params_strategy)
  params['car_fw'] = [car.CarParams.CarFw(ecu=fw[0], address=fw[1], subAddress=fw[2] or 0) for fw in params['car_fw']]
  return params


class TestCarInterfaces(unittest.TestCase):
  # FIXME: Due to the lists used in carParams, Phase.target is very slow and will cause
  #  many generated examples to overrun when max_examples > ~20, don't use it
  @parameterized.expand([(car,) for car in sorted(all_known_cars())])
  @settings(max_examples=MAX_EXAMPLES, deadline=None,
            phases=(Phase.reuse, Phase.generate, Phase.shrink))
  @given(data=st.data())
  def test_car_interfaces(self, car_name, data):
    CarInterface, CarController, CarState = interfaces[car_name]

    args = get_fuzzy_car_interface_args(data.draw)

    car_params = CarInterface.get_params(car_name, args['fingerprints'], args['car_fw'],
                                         experimental_long=args['experimental_long'], docs=False)
    car_interface = CarInterface(car_params, CarController, CarState)
    assert car_params
    assert car_interface

    self.assertGreater(car_params.mass, 1)
    self.assertGreater(car_params.wheelbase, 0)
    # centerToFront is center of gravity to front wheels, assert a reasonable range
    self.assertTrue(car_params.wheelbase * 0.3 < car_params.centerToFront < car_params.wheelbase * 0.7)
    self.assertGreater(car_params.maxLateralAccel, 0)

    # Longitudinal sanity checks
    self.assertEqual(len(car_params.longitudinalTuning.kpV), len(car_params.longitudinalTuning.kpBP))
    self.assertEqual(len(car_params.longitudinalTuning.kiV), len(car_params.longitudinalTuning.kiBP))
    self.assertEqual(len(car_params.longitudinalTuning.deadzoneV), len(car_params.longitudinalTuning.deadzoneBP))

    # If we're using the interceptor for gasPressed, we should be commanding gas with it
    if car_params.enableGasInterceptor:
      self.assertTrue(car_params.openpilotLongitudinalControl)

    # Lateral sanity checks
    if car_params.steerControlType != car.CarParams.SteerControlType.angle:
      tune = car_params.lateralTuning
      if tune.which() == 'pid':
        self.assertTrue(not math.isnan(tune.pid.kf) and tune.pid.kf > 0)
        self.assertTrue(len(tune.pid.kpV) > 0 and len(tune.pid.kpV) == len(tune.pid.kpBP))
        self.assertTrue(len(tune.pid.kiV) > 0 and len(tune.pid.kiV) == len(tune.pid.kiBP))

      elif tune.which() == 'torque':
        self.assertTrue(not math.isnan(tune.torque.kf) and tune.torque.kf > 0)
        self.assertTrue(not math.isnan(tune.torque.friction) and tune.torque.friction > 0)

    cc_msg = FuzzyGenerator.get_random_msg(data.draw, car.CarControl, real_floats=True)
    # Run car interface
    now_nanos = 0
    CC = car.CarControl.new_message(**cc_msg)
    for _ in range(10):
      car_interface.update(CC, [])
      car_interface.apply(CC, now_nanos)
      car_interface.apply(CC, now_nanos)
      now_nanos += DT_CTRL * 1e9  # 10 ms

    CC = car.CarControl.new_message(**cc_msg)
    CC.enabled = True
    for _ in range(10):
      car_interface.update(CC, [])
      car_interface.apply(CC, now_nanos)
      car_interface.apply(CC, now_nanos)
      now_nanos += DT_CTRL * 1e9  # 10ms

    # Test controller initialization
    # TODO: wait until card refactor is merged to run controller a few times,
    #  hypothesis also slows down significantly with just one more message draw
    LongControl(car_params)
    if car_params.steerControlType == car.CarParams.SteerControlType.angle:
      LatControlAngle(car_params, car_interface)
    elif car_params.lateralTuning.which() == 'pid':
      LatControlPID(car_params, car_interface)
    elif car_params.lateralTuning.which() == 'torque':
      LatControlTorque(car_params, car_interface)

    # Test radar interface
    RadarInterface = importlib.import_module(f'selfdrive.car.{car_params.carName}.radar_interface').RadarInterface
    radar_interface = RadarInterface(car_params)
    assert radar_interface

    # Run radar interface once
    radar_interface.update([])
    if not car_params.radarUnavailable and radar_interface.rcp is not None and \
       hasattr(radar_interface, '_update') and hasattr(radar_interface, 'trigger_msg'):
      radar_interface._update([radar_interface.trigger_msg])

    # Test radar fault
    if not car_params.radarUnavailable and radar_interface.rcp is not None:
      cans = [messaging.new_message('can', 1).to_bytes() for _ in range(5)]
      rr = radar_interface.update(cans)
      self.assertTrue(rr is None or len(rr.errors) > 0)

  def test_interface_attrs(self):
    """Asserts basic behavior of interface attribute getter"""
    num_brands = len(get_interface_attr('CAR'))
    self.assertGreaterEqual(num_brands, 13)

    # Should return value for all brands when not combining, even if attribute doesn't exist
    ret = get_interface_attr('FAKE_ATTR')
    self.assertEqual(len(ret), num_brands)

    # Make sure we can combine dicts
    ret = get_interface_attr('DBC', combine_brands=True)
    self.assertGreaterEqual(len(ret), 160)

    # We don't support combining non-dicts
    ret = get_interface_attr('CAR', combine_brands=True)
    self.assertEqual(len(ret), 0)

    # If brand has None value, it shouldn't return when ignore_none=True is specified
    none_brands = {b for b, v in get_interface_attr('FINGERPRINTS').items() if v is None}
    self.assertGreaterEqual(len(none_brands), 1)

    ret = get_interface_attr('FINGERPRINTS', ignore_none=True)
    none_brands_in_ret = none_brands.intersection(ret)
    self.assertEqual(len(none_brands_in_ret), 0, f'Brands with None values in ignore_none=True result: {none_brands_in_ret}')


if __name__ == "__main__":
  unittest.main()
