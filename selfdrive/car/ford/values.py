# flake8: noqa

from selfdrive.car import dbc_dict
from cereal import car
Ecu = car.CarParams.Ecu

MAX_ANGLE = 87.  # make sure we never command the extremes (0xfff) which cause latching fault

class CAR:
  FUSION = "FORD FUSION 2018"

DBC = {
  CAR.FUSION: dbc_dict('ford_fusion_2018_pt', 'ford_fusion_2018_adas'),
}
