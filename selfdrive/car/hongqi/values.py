from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Union

from cereal import car
from panda.python import uds
from selfdrive.car import dbc_dict
from selfdrive.car.docs_definitions import CarInfo, Harness
from selfdrive.car.fw_query_definitions import FwQueryConfig, Request, p16

Ecu = car.CarParams.Ecu
GearShifter = car.CarState.GearShifter


class CarControllerParams:
  LKAS_STEP = 2                  # LKAS message frequency 50Hz

  # TODO: placeholder values pending discovery of true EPS limits
  STEER_MAX = 300                # As-yet unknown fault boundary, guessing 300 / 3.0Nm for now
  STEER_DELTA_UP = 6             # 10 unit/sec observed from factory LKAS, fault boundary unknown
  STEER_DELTA_DOWN = 10          # 10 unit/sec observed from factory LKAS, fault boundary unknown
  STEER_DRIVER_ALLOWANCE = 50
  STEER_DRIVER_MULTIPLIER = 3    # weight driver torque heavily
  STEER_DRIVER_FACTOR = 1        # from dbc


class CANBUS:
  pt = 0
  cam = 2


class DBC_FILES:
  hongqi = "hongqi_hs5"


DBC = defaultdict(lambda: dbc_dict(DBC_FILES.hongqi, None))  # type: Dict[str, Dict[str, str]]


class CAR:
  HS5_G1 = "HONGQI HS5 1ST GEN"          # First generation FAW Hongqi HS5 SUV


@dataclass
class HongqiCarInfo(CarInfo):
  package: str = "Who Knows"  # FIXME
  harness: Enum = Harness.custom


CAR_INFO: Dict[str, Union[HongqiCarInfo, List[HongqiCarInfo]]] = {
  CAR.HS5_G1: HongqiCarInfo("Hongqi HS5 2020"),
}


HONGQI_VERSION_REQUEST = bytes([uds.SERVICE_TYPE.READ_DATA_BY_IDENTIFIER]) + \
  p16(uds.DATA_IDENTIFIER_TYPE.VEHICLE_MANUFACTURER_SPARE_PART_NUMBER)
HONGQI_VERSION_RESPONSE = bytes([uds.SERVICE_TYPE.READ_DATA_BY_IDENTIFIER + 0x40]) + \
  p16(uds.DATA_IDENTIFIER_TYPE.VEHICLE_MANUFACTURER_SPARE_PART_NUMBER)

FW_QUERY_CONFIG = FwQueryConfig(
  requests=[
    Request(
      [HONGQI_VERSION_REQUEST],
      [HONGQI_VERSION_RESPONSE],
    ),
  ],
)


FW_VERSIONS = {
  CAR.HS5_G1: {
    (Ecu.engine, 0x7e0, None): [
      b'3601015-DD21    ',
    ],
    (Ecu.transmission, 0x7e1, None): [
      b'3611015-DD01\x00\x00\x00\x00',
    ],
    (Ecu.eps, 0x7a2, None): [
      b'3418310-DD01\x00\x00\x00\x00',
    ],
    (Ecu.fwdCamera, 0x722, None): [
      b'3616215-DD03-B  ',
    ],
  },
}
