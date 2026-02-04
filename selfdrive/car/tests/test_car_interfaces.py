import os
import hypothesis.strategies as st
from hypothesis import Phase, given, settings
from parameterized import parameterized

from cereal import car
from opendbc.car import DT_CTRL
from opendbc.car.structs import CarParams
from opendbc.car.tests.test_car_interfaces import get_fuzzy_car_interface
from opendbc.car.mock.values import CAR as MOCK
from opendbc.car.values import PLATFORMS
from openpilot.selfdrive.controls.lib.latcontrol_angle import LatControlAngle
from openpilot.selfdrive.controls.lib.latcontrol_pid import LatControlPID
from openpilot.selfdrive.controls.lib.latcontrol_torque import LatControlTorque
from openpilot.selfdrive.controls.lib.longcontrol import LongControl
from openpilot.selfdrive.test.fuzzy_generation import FuzzyGenerator

MAX_EXAMPLES = int(os.environ.get('MAX_EXAMPLES', '60'))
DT_CTRL_NANOS = int(DT_CTRL * 1e9)  # 10ms in nanoseconds


class TestCarInterfaces:
  # FIXME: Due to the lists used in carParams, Phase.target is very slow and will cause
  #  many generated examples to overrun when max_examples > ~20, don't use it
  @parameterized.expand([(car,) for car in sorted(PLATFORMS)] + [MOCK.MOCK])
  @settings(max_examples=MAX_EXAMPLES, deadline=None,
            phases=(Phase.reuse, Phase.generate, Phase.shrink))
  @given(data=st.data())
  def test_car_interfaces(self, car_name, data):
    car_interface = get_fuzzy_car_interface(car_name, data.draw)
    car_params = car_interface.CP.as_reader()

    cc_msg = FuzzyGenerator.get_random_msg(data.draw, car.CarControl, real_floats=True)

    # Pre-create CC messages to avoid repeated creation in loops
    CC_fuzzy = car.CarControl.new_message(**cc_msg).as_reader()
    cc_msg['enabled'] = True
    cc_msg['latActive'] = True
    cc_msg['longActive'] = True
    CC_enabled = car.CarControl.new_message(**cc_msg).as_reader()

    # Run car interface with fuzzy input
    now_nanos = 0
    for _ in range(10):
      car_interface.update([])
      car_interface.apply(CC_fuzzy, now_nanos)
      now_nanos += DT_CTRL_NANOS

    # Run with enabled=True
    for _ in range(10):
      car_interface.update([])
      car_interface.apply(CC_enabled, now_nanos)
      now_nanos += DT_CTRL_NANOS

    # Test controller initialization
    # TODO: wait until card refactor is merged to run controller a few times,
    #  hypothesis also slows down significantly with just one more message draw
    LongControl(car_params)
    if car_params.steerControlType == CarParams.SteerControlType.angle:
      LatControlAngle(car_params, car_interface, DT_CTRL)
    elif car_params.lateralTuning.which() == 'pid':
      LatControlPID(car_params, car_interface, DT_CTRL)
    elif car_params.lateralTuning.which() == 'torque':
      LatControlTorque(car_params, car_interface, DT_CTRL)
