import os
import random
from openpilot.common.parameterized import parameterized

from cereal import car
from opendbc.car import DT_CTRL
from opendbc.car.fingerprints import FW_VERSIONS
from opendbc.car.fw_versions import FW_QUERY_CONFIGS
from opendbc.car.car_helpers import interfaces
from opendbc.car.structs import CarParams
from opendbc.car.mock.values import CAR as MOCK
from opendbc.car.values import PLATFORMS
from openpilot.selfdrive.controls.lib.latcontrol_angle import LatControlAngle
from openpilot.selfdrive.controls.lib.latcontrol_pid import LatControlPID
from openpilot.selfdrive.controls.lib.latcontrol_torque import LatControlTorque
from openpilot.selfdrive.controls.lib.longcontrol import LongControl
from openpilot.selfdrive.test.fuzzy_generation import FuzzyGenerator

MAX_EXAMPLES = int(os.environ.get('MAX_EXAMPLES', '60'))

# From panda/python/__init__.py
DLC_TO_LEN = [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20, 24, 32, 48, 64]

ALL_ECUS = sorted({ecu for ecus in FW_VERSIONS.values() for ecu in ecus.keys()} |
                  {ecu for config in FW_QUERY_CONFIGS.values() for ecu in config.extra_ecus})
ALL_REQUESTS = sorted({tuple(r.request) for config in FW_QUERY_CONFIGS.values() for r in config.requests})


def _random_fuzzy_car_interface(car_name: str):
  # Random CAN fingerprints and FW versions to exercise more states of the CarInterface.
  fingerprint = {addr: random.choice(DLC_TO_LEN) for addr in
                 random.sample(range(0x801), random.randint(0, 32))}
  fingerprints = dict.fromkeys(range(7), fingerprint)

  car_fw = [CarParams.CarFw(ecu=ecu[0], address=ecu[1], subAddress=ecu[2] or 0, request=req)
            for ecu, req in ((random.choice(ALL_ECUS), random.choice(ALL_REQUESTS))
                             for _ in range(random.randint(0, 10)))]

  alpha_long = random.random() < 0.5
  CarInterface = interfaces[car_name]
  car_params = CarInterface.get_params(car_name, fingerprints, car_fw,
                                       alpha_long=alpha_long, is_release=False, docs=False)
  return CarInterface(car_params)


class TestCarInterfaces:
  @parameterized.expand([(car,) for car in sorted(PLATFORMS)] + [MOCK.MOCK])
  def test_car_interfaces(self, car_name):
    for _ in range(MAX_EXAMPLES):
      car_interface = _random_fuzzy_car_interface(car_name)
      car_params = car_interface.CP.as_reader()

      cc_msg = FuzzyGenerator.get_random_msg(car.CarControl, real_floats=True)
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

      LongControl(car_params)
      if car_params.steerControlType == CarParams.SteerControlType.angle:
        LatControlAngle(car_params, car_interface, DT_CTRL)
      elif car_params.lateralTuning.which() == 'pid':
        LatControlPID(car_params, car_interface, DT_CTRL)
      elif car_params.lateralTuning.which() == 'torque':
        LatControlTorque(car_params, car_interface, DT_CTRL)
