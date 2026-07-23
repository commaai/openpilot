import os
from openpilot.common.test import OpenpilotTestCase
from openpilot.common.parameterized import parameterized
from openpilot.common.fuzzy import capnp_random_dict, fuzzy_test

from opendbc.car.structs import car
from opendbc.car import DT_CTRL
from opendbc.car.car_helpers import interfaces
from opendbc.car.fingerprints import FW_VERSIONS
from opendbc.car.fw_versions import FW_QUERY_CONFIGS
from opendbc.car.structs import CarParams
from opendbc.car.mock.values import CAR as MOCK
from opendbc.car.values import PLATFORMS
from openpilot.selfdrive.controls.lib.latcontrol_angle import LatControlAngle
from openpilot.selfdrive.controls.lib.latcontrol_pid import LatControlPID
from openpilot.selfdrive.controls.lib.latcontrol_torque import LatControlTorque
from openpilot.selfdrive.controls.lib.longcontrol import LongControl

MAX_EXAMPLES = int(os.environ.get('MAX_EXAMPLES', '60'))

ALL_ECUS = tuple(sorted({ecu for ecus in FW_VERSIONS.values() for ecu in ecus} |
                        {ecu for config in FW_QUERY_CONFIGS.values() for ecu in config.extra_ecus}))
ALL_REQUESTS = tuple(sorted({tuple(request.request) for config in FW_QUERY_CONFIGS.values() for request in config.requests}))
DLC_TO_LEN = (0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20, 24, 32, 48, 64)


class TestCarInterfaces(OpenpilotTestCase):
  @parameterized.expand([(car,) for car in sorted(PLATFORMS)] + [MOCK.MOCK])
  @fuzzy_test(max_examples=MAX_EXAMPLES)
  def test_car_interfaces(self, car_name, fuzzy):
    fingerprint = dict(fuzzy.list(lambda: (fuzzy.integer(0, 0x800), fuzzy.choice(DLC_TO_LEN))))
    fingerprints = dict.fromkeys(range(7), fingerprint)

    def generate_car_fw():
      ecu, address, sub_address = fuzzy.choice(ALL_ECUS)
      return CarParams.CarFw(ecu=ecu, address=address, subAddress=sub_address or 0, request=fuzzy.choice(ALL_REQUESTS))

    CarInterface = interfaces[car_name]
    car_params = CarInterface.get_params(car_name, fingerprints, fuzzy.list(generate_car_fw),
                                         alpha_long=fuzzy.boolean(), is_release=False, docs=False)
    car_interface = CarInterface(car_params)
    car_params = car_interface.CP.as_reader()

    cc_msg = capnp_random_dict(fuzzy, car.CarControl.schema, real_floats=True)
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
    # TODO: wait until card refactor is merged to run controller a few times
    LongControl(car_params)
    if car_params.steerControlType == CarParams.SteerControlType.angle:
      LatControlAngle(car_params, car_interface, DT_CTRL)
    elif car_params.lateralTuning.which() == 'pid':
      LatControlPID(car_params, car_interface, DT_CTRL)
    elif car_params.lateralTuning.which() == 'torque':
      LatControlTorque(car_params, car_interface, DT_CTRL)
