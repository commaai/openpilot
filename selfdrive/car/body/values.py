from selfdrive.car import dbc_dict
from cereal import car
Ecu = car.CarParams.Ecu


class CarControllerParams:
  ANGLE_DELTA_BP = [0., 5., 15.]
  ANGLE_DELTA_V = [5., .8, .15]     # windup limit
  ANGLE_DELTA_VU = [5., 3.5, 0.4]   # unwind limit
  LKAS_MAX_TORQUE = 1               # A value of 1 is easy to overpower
  STEER_THRESHOLD = 1.0

class CAR:
  BODY = "COMMA BODY"

FW_VERSIONS = {
  CAR.BODY: {
    (Ecu.engine, 0x720, None): [
      b'02/27/2022'
    ],
  },
}

DBC = {
  CAR.BODY: dbc_dict('comma_body', None),
}
