from dataclasses import dataclass, field
from enum import StrEnum
from typing import Dict, List, Union

from cereal import car
from openpilot.selfdrive.car import dbc_dict
from openpilot.selfdrive.car.docs_definitions import CarHarness, CarInfo, CarParts
from openpilot.selfdrive.car.fw_query_definitions import FwQueryConfig, Request, StdQueries

Ecu = car.CarParams.Ecu


# Steer torque limits

class CarControllerParams:
  STEER_MAX = 800                # theoretical max_steer 2047
  STEER_DELTA_UP = 10             # torque increase per refresh
  STEER_DELTA_DOWN = 25           # torque decrease per refresh
  STEER_DRIVER_ALLOWANCE = 15     # allowed driver torque before start limiting
  STEER_DRIVER_MULTIPLIER = 1     # weight driver torque
  STEER_DRIVER_FACTOR = 1         # from dbc
  STEER_ERROR_MAX = 350           # max delta between torque cmd and torque motor
  STEER_STEP = 1  # 100 Hz

  def __init__(self, CP):
    pass


class CAR(StrEnum):
  CX5 = "MAZDA CX-5"
  CX9 = "MAZDA CX-9"
  MAZDA3 = "MAZDA 3"
  MAZDA6 = "MAZDA 6"
  CX9_2021 = "MAZDA CX-9 2021"
  CX5_2022 = "MAZDA CX-5 2022"


@dataclass
class MazdaCarInfo(CarInfo):
  package: str = "All"
  car_parts: CarParts = field(default_factory=CarParts.common([CarHarness.mazda]))


CAR_INFO: Dict[str, Union[MazdaCarInfo, List[MazdaCarInfo]]] = {
  CAR.CX5: MazdaCarInfo("Mazda CX-5 2017-21"),
  CAR.CX9: MazdaCarInfo("Mazda CX-9 2016-20"),
  CAR.MAZDA3: MazdaCarInfo("Mazda 3 2017-18"),
  CAR.MAZDA6: MazdaCarInfo("Mazda 6 2017-20"),
  CAR.CX9_2021: MazdaCarInfo("Mazda CX-9 2021-23", video_link="https://youtu.be/dA3duO4a0O4"),
  CAR.CX5_2022: MazdaCarInfo("Mazda CX-5 2022-24"),
}


class LKAS_LIMITS:
  STEER_THRESHOLD = 15
  DISABLE_SPEED = 45    # kph
  ENABLE_SPEED = 52     # kph


class Buttons:
  NONE = 0
  SET_PLUS = 1
  SET_MINUS = 2
  RESUME = 3
  CANCEL = 4


FW_QUERY_CONFIG = FwQueryConfig(
  requests=[
    Request(
      [StdQueries.MANUFACTURER_SOFTWARE_VERSION_REQUEST],
      [StdQueries.MANUFACTURER_SOFTWARE_VERSION_RESPONSE],
    ),
    # Log responses on powertrain bus
    Request(
      [StdQueries.MANUFACTURER_SOFTWARE_VERSION_REQUEST],
      [StdQueries.MANUFACTURER_SOFTWARE_VERSION_RESPONSE],
      bus=0,
      logging=True,
    ),
  ],
)


DBC = {
  CAR.CX5: dbc_dict('mazda_2017', None),
  CAR.CX9: dbc_dict('mazda_2017', None),
  CAR.MAZDA3: dbc_dict('mazda_2017', None),
  CAR.MAZDA6: dbc_dict('mazda_2017', None),
  CAR.CX9_2021: dbc_dict('mazda_2017', None),
  CAR.CX5_2022: dbc_dict('mazda_2017', None),
}

# Gen 1 hardware: same CAN messages and same camera
GEN1 = {CAR.CX5, CAR.CX9, CAR.CX9_2021, CAR.MAZDA3, CAR.MAZDA6, CAR.CX5_2022}
