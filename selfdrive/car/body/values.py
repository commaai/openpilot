from typing import Dict

from cereal import car
from selfdrive.car import dbc_dict
from selfdrive.car.docs_definitions import CarInfo, Harness
Ecu = car.CarParams.Ecu

SPEED_FROM_RPM = 0.008587

class CarControllerParams:
  ANGLE_DELTA_BP = [0., 5., 15.]
  ANGLE_DELTA_V = [5., .8, .15]     # windup limit
  ANGLE_DELTA_VU = [5., 3.5, 0.4]   # unwind limit
  LKAS_MAX_TORQUE = 1               # A value of 1 is easy to overpower
  STEER_THRESHOLD = 1.0

class CAR:
  BODY = "COMMA BODY"

CAR_INFO: Dict[str, CarInfo] = {
  CAR.BODY: CarInfo("comma body", package="All", harness=Harness.none),
}

FW_VERSIONS = {
  CAR.BODY: {
    (Ecu.engine, 0x720, None): [
      b'0.0.01',
      b'02/27/2022'
    ],
    (Ecu.debug, 0x721, None): [
      b'166bd860' # git hash of the firmware used
    ],
  },
}

DBC = {
  CAR.BODY: dbc_dict('comma_body', None),
}
