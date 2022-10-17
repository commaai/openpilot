from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Union

from cereal import car
from selfdrive.car import dbc_dict
from selfdrive.car.docs_definitions import CarInfo, Harness

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
