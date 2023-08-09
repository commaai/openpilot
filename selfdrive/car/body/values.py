from typing import Dict

from selfdrive.car import dbc_dict
from selfdrive.car.docs_definitions import CarInfo

SPEED_FROM_RPM = 0.008587


class CarControllerParams:
  ANGLE_DELTA_BP = [0., 5., 15.]
  ANGLE_DELTA_V = [5., .8, .15]     # windup limit
  ANGLE_DELTA_VU = [5., 3.5, 0.4]   # unwind limit
  LKAS_MAX_TORQUE = 1               # A value of 1 is easy to overpower
  STEER_THRESHOLD = 1.0

  def __init__(self, CP):
    pass


class CAR:
  BODY = "COMMA BODY"


CAR_INFO: Dict[str, CarInfo] = {
  CAR.BODY: CarInfo("comma body", package="All"),
}

DBC = {
  CAR.BODY: dbc_dict('comma_body', None),
}
