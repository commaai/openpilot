import math
import os

import hypothesis.strategies as st
from hypothesis import Phase, given, settings
from parameterized import parameterized

from cereal import car
from opendbc.car import DT_CTRL, gen_empty_fingerprint, structs
from opendbc.car.car_helpers import interfaces
from opendbc.car.fw_versions import FW_QUERY_CONFIGS, FW_VERSIONS
from opendbc.car.mock.values import CAR as MOCK
from opendbc.car.structs import CarParams
from opendbc.car.values import PLATFORMS
from openpilot.selfdrive.controls.lib.latcontrol_angle import LatControlAngle
from openpilot.selfdrive.controls.lib.latcontrol_pid import LatControlPID
from openpilot.selfdrive.controls.lib.latcontrol_torque import LatControlTorque
from openpilot.selfdrive.controls.lib.longcontrol import LongControl
from openpilot.selfdrive.test.fuzzy_generation import FuzzyGenerator

ALL_ECUS = {ecu for ecus in FW_VERSIONS.values() for ecu in ecus.keys()}
ALL_ECUS |= {ecu for config in FW_QUERY_CONFIGS.values() for ecu in config.extra_ecus}

ALL_REQUESTS = {tuple(r.request) for config in FW_QUERY_CONFIGS.values() for r in config.requests}

MAX_EXAMPLES = int(os.environ.get('MAX_EXAMPLES', '60'))

EMPTY_FINGERPRINT = gen_empty_fingerprint().keys()

TRIPLE_FP_STRATEGY = st.lists(
  st.tuples(st.sampled_from(tuple(EMPTY_FINGERPRINT)), st.integers(0, 0x800), st.integers(0, 64)),
)


def _build_fp(triples):
  fp = {i: {} for i in EMPTY_FINGERPRINT}
  for i, address, length in triples:
    fp[i][address] = length
  return fp


FINGERPRINT_STRATEGY = st.builds(_build_fp, TRIPLE_FP_STRATEGY)

REQUEST_STRATEGY = st.sampled_from(sorted(ALL_REQUESTS))


@st.composite
def car_fw_obj_list(draw):
  entries = draw(
    st.lists(
      st.sampled_from(sorted(ALL_ECUS)),
    )
  )
  return [structs.CarParams.CarFw(ecu=e[0], address=e[1], subAddress=e[2] or 0, request=draw(REQUEST_STRATEGY)) for e in entries]


PARAMS_STRATEGY = st.fixed_dictionaries(
  {
    'fingerprints': FINGERPRINT_STRATEGY,
    'car_fw': car_fw_obj_list(),
    'alpha_long': st.booleans(),
  }
)


class TestCarInterfaces:
  # FIXME: Due to the lists used in carParams, Phase.target is very slow and will cause
  #  many generated examples to overrun when max_examples > ~20, don't use it
  @parameterized.expand([(car,) for car in sorted(PLATFORMS)] + [MOCK.MOCK])
  @settings(max_examples=MAX_EXAMPLES, deadline=None,
            phases=(Phase.reuse, Phase.generate, Phase.shrink))
  @given(data=st.data())
  def test_car_interfaces(self, car_name, data):
    CarInterface = interfaces[car_name]

    car_interface_args = data.draw(PARAMS_STRATEGY)

    car_params = CarInterface.get_params(car_name, car_interface_args['fingerprints'], car_interface_args['car_fw'],
                                         alpha_long=car_interface_args['alpha_long'], docs=False)
    car_params = car_params.as_reader()
    car_interface = CarInterface(car_params)
    assert car_params
    assert car_interface

    assert car_params.mass > 1
    assert car_params.wheelbase > 0
    # centerToFront is center of gravity to front wheels, assert a reasonable range
    assert car_params.wheelbase * 0.3 < car_params.centerToFront < car_params.wheelbase * 0.7
    assert car_params.maxLateralAccel > 0

    # Longitudinal sanity checks
    assert len(car_params.longitudinalTuning.kpV) == len(car_params.longitudinalTuning.kpBP)
    assert len(car_params.longitudinalTuning.kiV) == len(car_params.longitudinalTuning.kiBP)

    # Lateral sanity checks
    if car_params.steerControlType != CarParams.SteerControlType.angle:
      tune = car_params.lateralTuning
      if tune.which() == 'pid':
        if car_name != MOCK.MOCK:
          assert not math.isnan(tune.pid.kf) and tune.pid.kf > 0
          assert len(tune.pid.kpV) > 0 and len(tune.pid.kpV) == len(tune.pid.kpBP)
          assert len(tune.pid.kiV) > 0 and len(tune.pid.kiV) == len(tune.pid.kiBP)

      elif tune.which() == 'torque':
        assert not math.isnan(tune.torque.kf) and tune.torque.kf > 0
        assert not math.isnan(tune.torque.friction) and tune.torque.friction > 0

    cc_msg = FuzzyGenerator.get_random_msg(data.draw, car.CarControl, real_floats=True)
    # Run car interface
    now_nanos = 0
    CC = car.CarControl.new_message(**cc_msg)
    CC = CC.as_reader()
    for _ in range(10):
      car_interface.update([])
      car_interface.apply(CC, now_nanos)
      now_nanos += DT_CTRL * 1e9  # 10 ms

    CC = car.CarControl.new_message(**cc_msg)
    CC.enabled = True
    CC.latActive = True
    CC.longActive = True
    CC = CC.as_reader()
    for _ in range(10):
      car_interface.update([])
      car_interface.apply(CC, now_nanos)
      now_nanos += DT_CTRL * 1e9  # 10ms

    # Test controller initialization
    # TODO: wait until card refactor is merged to run controller a few times,
    #  hypothesis also slows down significantly with just one more message draw
    LongControl(car_params)
    if car_params.steerControlType == CarParams.SteerControlType.angle:
      LatControlAngle(car_params, car_interface)
    elif car_params.lateralTuning.which() == 'pid':
      LatControlPID(car_params, car_interface)
    elif car_params.lateralTuning.which() == 'torque':
      LatControlTorque(car_params, car_interface)
