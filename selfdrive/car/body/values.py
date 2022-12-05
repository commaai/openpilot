from typing import Dict

from cereal import car
from selfdrive.car import dbc_dict
from selfdrive.car.docs_definitions import CarInfo
from selfdrive.car.fw_query_definitions import FwQueryConfig, Request, StdQueries

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
  CAR.BODY: CarInfo("comma body", package="All"),
}

FINGERPRINTS = {
  CAR.BODY: [{
    513: 8, 516: 8, 514: 3, 515: 4,
  }],
}

FW_QUERY_CONFIG = FwQueryConfig(
  requests=[
    Request(
      [StdQueries.TESTER_PRESENT_REQUEST, StdQueries.UDS_VERSION_REQUEST],
      [StdQueries.TESTER_PRESENT_RESPONSE, StdQueries.UDS_VERSION_RESPONSE],
      bus=0,
    ),
  ],
)

FW_VERSIONS = {
  CAR.BODY: {
    (Ecu.engine, 0x720, None): [
      b'0.0.01',
      b'02/27/2022',
      b'0.3.00a',
    ],
    # git hash of the firmware used
    (Ecu.debug, 0x721, None): [
      b'166bd860',
      b'dc780f85',
    ],
  },
}

DBC = {
  CAR.BODY: dbc_dict('comma_body', None),
}
