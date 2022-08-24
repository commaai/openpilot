from collections import namedtuple
from typing import Dict, List, Union

from cereal import car
from selfdrive.car import dbc_dict
from selfdrive.car.docs_definitions import CarInfo

Ecu = car.CarParams.Ecu
TransmissionType = car.CarParams.TransmissionType
GearShifter = car.CarState.GearShifter

AngleRateLimit = namedtuple('AngleRateLimit', ['speed_points', 'max_angle_diff_points'])


class CarControllerParams:
  # Messages: Lane_Assist_Data1, LateralMotionControl
  LKAS_STEER_STEP = 5
  # Message: IPMA_Data
  LKAS_UI_STEP = 100
  # Message: ACCDATA_3
  ACC_UI_STEP = 5

  STEER_RATE_LIMIT_UP = AngleRateLimit(speed_points=[0., 5., 15.], max_angle_diff_points=[5., .8, .15])
  STEER_RATE_LIMIT_DOWN = AngleRateLimit(speed_points=[0., 5., 15.], max_angle_diff_points=[5., 3.5, 0.4])


class RADAR:
  DELPHI_ESR = 'ford_fusion_2018_adas'
  DELPHI_MRR = 'FORD_CADS'


class CANBUS:
  main = 0
  radar = 1
  camera = 2


class CAR:
  ESCAPE_MK4 = "FORD ESCAPE 4TH GEN"
  FOCUS_MK4 = "FORD FOCUS 4TH GEN"


CAR_INFO: Dict[str, Union[CarInfo, List[CarInfo]]] = {
  CAR.ESCAPE_MK4: CarInfo("Ford Escape", "NA"),
  CAR.FOCUS_MK4: CarInfo("Ford Focus", "NA"),
}


FW_VERSIONS = {
  CAR.ESCAPE_MK4: {
    (Ecu.eps, 0x730, None): [
      b'LX6C-14D003-AH\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.esp, 0x760, None): [
      b'LX6C-2D053-NS\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.fwdRadar, 0x764, None): [
      b'LB5T-14D049-AB\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.fwdCamera, 0x706, None): [
      b'LJ6T-14F397-AD\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.engine, 0x7E0, None): [
      b'LX6A-14C204-ESG\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
  },
  CAR.FOCUS_MK4: {
    (Ecu.eps, 0x730, None): [
      b'JX6C-14D003-AH\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.esp, 0x760, None): [
      b'JX61-2D053-CJ\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.fwdRadar, 0x764, None): [
      b'JX7T-14D049-AC\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.fwdCamera, 0x706, None): [
      b'JX7T-14F397-AH\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.engine, 0x7E0, None): [
      b'JX6A-14C204-BPL\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
  },
}


DBC = {
  CAR.ESCAPE_MK4: dbc_dict('ford_lincoln_base_pt', RADAR.DELPHI_MRR),
  CAR.FOCUS_MK4: dbc_dict('ford_lincoln_base_pt', RADAR.DELPHI_MRR),
}
