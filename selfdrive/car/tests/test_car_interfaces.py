
from hypothesis import given, settings, Phase, strategies as st
from parameterized import parameterized
from cereal import car
from opendbc.car import DT_CTRL
from opendbc.car.car_helpers import interfaces
from opendbc.car.structs import CarParams
from opendbc.car.tests.test_car_interfaces import get_fuzzy_car_interface_args
from opendbc.car.mock.values import CAR as MOCK
from opendbc.car.values import PLATFORMS
from openpilot.selfdrive.controls.lib.latcontrol_angle import LatControlAngle
from openpilot.selfdrive.controls.lib.latcontrol_pid import LatControlPID
from openpilot.selfdrive.controls.lib.latcontrol_torque import LatControlTorque
from openpilot.selfdrive.controls.lib.longcontrol import LongControl
from openpilot.selfdrive.test.fuzzy_generation import FuzzyGenerator

MAX_EXAMPLES = 50

class TestCarInterfaces:
  # Faster fuzzing by limiting max_examples and skipping slow Phases
  @parameterized.expand([(car,) for car in sorted(PLATFORMS)] + [MOCK.MOCK])
  @settings(
    max_examples=MAX_EXAMPLES,
    deadline=None,
    phases=(Phase.reuse, Phase.generate),  # skip shrink phase
    suppress_health_check=[],
  )
  @given(data=st.data())
  def test_car_interfaces(self, car_name, data):
    CarInterface = interfaces[car_name]

    # --- Get params and construct interface ---
    args = get_fuzzy_car_interface_args(data.draw)
    car_params = CarInterface.get_params(
      car_name, args['fingerprints'], False, args['car_fw'],
      alpha_long=args['alpha_long'], docs=False
    ).as_reader()
    CI = CarInterface(car_params)
    assert car_params.mass > 1
    assert car_params.wheelbase > 0
    assert car_params.wheelbase * 0.3 < car_params.centerToFront < car_params.wheelbase * 0.7
    assert car_params.maxLateralAccel > 0
    assert len(car_params.longitudinalTuning.kpV) == len(car_params.longitudinalTuning.kpBP)
    assert len(car_params.longitudinalTuning.kiV) == len(car_params.longitudinalTuning.kiBP)

    if car_params.steerControlType != CarParams.SteerControlType.angle:
      tune = car_params.lateralTuning
      if tune.which() == 'pid' and car_name != MOCK.MOCK:
        assert tune.pid.kf > 0
        assert len(tune.pid.kpV) > 0 and len(tune.pid.kpV) == len(tune.pid.kpBP)
        assert len(tune.pid.kiV) > 0 and len(tune.pid.kiV) == len(tune.pid.kiBP)
      elif tune.which() == 'torque':
        assert tune.torque.kf > 0
        assert tune.torque.friction > 0

    # --- Control message fuzz ---
    CC = car.CarControl.new_message(**FuzzyGenerator.get_random_msg(
      data.draw, car.CarControl, real_floats=True))
    CC.enabled = CC.latActive = CC.longActive = data.draw(st.booleans())
    CC = CC.as_reader()

    # --- Run minimal control loop ---
    now_nanos = 0
    for _ in range(10):
      CI.update([])
      CI.apply(CC, now_nanos)
      now_nanos += DT_CTRL * 1e9

    # --- Controller sanity check only ---
    LongControl(car_params)
    if car_params.steerControlType == CarParams.SteerControlType.angle:
      LatControlAngle(car_params, CI)
    elif car_params.lateralTuning.which() == 'pid':
      LatControlPID(car_params, CI)
    elif car_params.lateralTuning.which() == 'torque':
      LatControlTorque(car_params, CI)
