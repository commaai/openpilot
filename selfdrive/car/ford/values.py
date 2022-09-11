from collections import namedtuple
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Union

from cereal import car
from selfdrive.car import dbc_dict
from selfdrive.car.docs_definitions import CarInfo, Harness
from selfdrive.car.fw_query_definitions import FwQueryConfig, Request, StdQueries

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

  STEER_RATIO = 2.75
  STEER_DRIVER_ALLOWANCE = 0.8

  RATE_LIMIT_UP = AngleRateLimit(speed_points=[0., 5., 15.], max_angle_diff_points=[5., .8, .15])
  RATE_LIMIT_DOWN = AngleRateLimit(speed_points=[0., 5., 15.], max_angle_diff_points=[5., 3.5, 0.4])


class RADAR:
  DELPHI_ESR = 'ford_fusion_2018_adas'
  DELPHI_MRR = 'FORD_CADS'


class CANBUS:
  main = 0
  radar = 1
  camera = 2


class CAR:
  ESCAPE_MK4 = "FORD ESCAPE 4TH GEN"
  EXPLORER_MK6 = "FORD EXPLORER 6TH GEN"
  FOCUS_MK4 = "FORD FOCUS 4TH GEN"


@dataclass
class FordCarInfo(CarInfo):
  package: str = "Co-Pilot360 Assist+"
  harness: Enum = Harness.ford_q3


CAR_INFO: Dict[str, Union[CarInfo, List[CarInfo]]] = {
  CAR.ESCAPE_MK4: [
    FordCarInfo("Ford Escape 2020"),
    FordCarInfo("Ford Kuga EU", "Driver Assistance Pack"),
  ],
  CAR.EXPLORER_MK6: FordCarInfo("Ford Explorer 2020-21"),
  CAR.FOCUS_MK4: FordCarInfo("Ford Focus EU 2019", "Driver Assistance Pack"),
}

FW_QUERY_CONFIG = FwQueryConfig(
  requests=[
    Request(
      [StdQueries.TESTER_PRESENT_REQUEST, StdQueries.MANUFACTURER_SOFTWARE_VERSION_REQUEST],
      [StdQueries.TESTER_PRESENT_RESPONSE, StdQueries.MANUFACTURER_SOFTWARE_VERSION_RESPONSE],
      whitelist_ecus=[Ecu.engine],
    ),
    Request(
      [StdQueries.TESTER_PRESENT_REQUEST, StdQueries.MANUFACTURER_SOFTWARE_VERSION_REQUEST],
      [StdQueries.TESTER_PRESENT_RESPONSE, StdQueries.MANUFACTURER_SOFTWARE_VERSION_RESPONSE],
      bus=0,
      whitelist_ecus=[Ecu.eps, Ecu.abs, Ecu.fwdRadar, Ecu.fwdCamera, Ecu.shiftByWire],
    ),
  ],
  ecus={(0x730, None): Ecu.eps, (0x760, None): Ecu.abs, (0x764, None): Ecu.fwdRadar,
        (0x706, None): Ecu.fwdCamera, (0x7e0, None): Ecu.engine, (0x732, None): Ecu.shiftByWire},
)

FW_VERSIONS = {
  CAR.ESCAPE_MK4: {
    Ecu.eps: [
      b'LX6C-14D003-AH\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    Ecu.abs: [
      b'LX6C-2D053-NS\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    Ecu.fwdRadar: [
      b'LB5T-14D049-AB\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    Ecu.fwdCamera: [
      b'LJ6T-14F397-AD\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    Ecu.engine: [
      b'LX6A-14C204-ESG\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    Ecu.shiftByWire: [
    ],
  },
  CAR.EXPLORER_MK6: {
    Ecu.eps: [
      b'L1MC-14D003-AK\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
      b'L1MC-14D003-AL\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    Ecu.abs: [
      b'L1MC-2D053-BB\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
      b'L1MC-2D053-BF\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    Ecu.fwdRadar: [
      b'LB5T-14D049-AB\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    Ecu.fwdCamera: [
      b'LB5T-14F397-AE\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
      b'LB5T-14F397-AF\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    Ecu.engine: [
      b'LB5A-14C204-EAC\x00\x00\x00\x00\x00\x00\x00\x00\x00',
      b'MB5A-14C204-MD\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    Ecu.shiftByWire: [
      b'L1MP-14G395-AD\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
      b'L1MP-14G395-AE\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
  },
  CAR.FOCUS_MK4: {
    Ecu.eps: [
      b'JX6C-14D003-AH\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    Ecu.abs: [
      b'JX61-2D053-CJ\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    Ecu.fwdRadar: [
      b'JX7T-14D049-AC\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    Ecu.fwdCamera: [
      b'JX7T-14F397-AH\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    Ecu.engine: [
      b'JX6A-14C204-BPL\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    Ecu.shiftByWire: [
    ],
  },
}


DBC = {
  CAR.ESCAPE_MK4: dbc_dict('ford_lincoln_base_pt', RADAR.DELPHI_MRR),
  CAR.EXPLORER_MK6: dbc_dict('ford_lincoln_base_pt', RADAR.DELPHI_MRR),
  CAR.FOCUS_MK4: dbc_dict('ford_lincoln_base_pt', RADAR.DELPHI_MRR),
}
