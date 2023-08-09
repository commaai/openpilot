from dataclasses import dataclass, field
from enum import Enum, IntFlag
from typing import Dict, List, Union

from cereal import car
from selfdrive.car import dbc_dict
from selfdrive.car.docs_definitions import CarFootnote, CarHarness, CarInfo, CarParts, Tool, Column

Ecu = car.CarParams.Ecu


class CarControllerParams:
  def __init__(self, CP):
    self.STEER_STEP = 2                # how often we update the steer cmd
    self.STEER_DELTA_UP = 50           # torque increase per refresh, 0.8s to max
    self.STEER_DELTA_DOWN = 70         # torque decrease per refresh
    self.STEER_DRIVER_ALLOWANCE = 60   # allowed driver torque before start limiting
    self.STEER_DRIVER_MULTIPLIER = 50  # weight driver torque heavily
    self.STEER_DRIVER_FACTOR = 1       # from dbc

    if CP.carFingerprint in GLOBAL_GEN2:
      self.STEER_MAX = 1000
      self.STEER_DELTA_UP = 40
      self.STEER_DELTA_DOWN = 40
    elif CP.carFingerprint == CAR.IMPREZA_2020:
      self.STEER_MAX = 1439
    else:
      self.STEER_MAX = 2047


class SubaruFlags(IntFlag):
  SEND_INFOTAINMENT = 1


class CanBus:
  main = 0
  alt = 1
  camera = 2


class CAR:
  # Global platform
  ASCENT = "SUBARU ASCENT LIMITED 2019"
  IMPREZA = "SUBARU IMPREZA LIMITED 2019"
  IMPREZA_2020 = "SUBARU IMPREZA SPORT 2020"
  FORESTER = "SUBARU FORESTER 2019"
  OUTBACK = "SUBARU OUTBACK 6TH GEN"
  LEGACY = "SUBARU LEGACY 7TH GEN"
  FORESTER_2022 = "SUBARU FORESTER 2022"
  OUTBACK_2023 = "SUBARU OUTBACK 7TH GEN"

  # Pre-global
  FORESTER_PREGLOBAL = "SUBARU FORESTER 2017 - 2018"
  LEGACY_PREGLOBAL = "SUBARU LEGACY 2015 - 2018"
  OUTBACK_PREGLOBAL = "SUBARU OUTBACK 2015 - 2017"
  OUTBACK_PREGLOBAL_2018 = "SUBARU OUTBACK 2018 - 2019"


class Footnote(Enum):
  GLOBAL = CarFootnote(
    "In the non-US market, openpilot requires the car to come equipped with EyeSight with Lane Keep Assistance.",
    Column.PACKAGE)


@dataclass
class SubaruCarInfo(CarInfo):
  package: str = "EyeSight Driver Assistance"
  car_parts: CarParts = field(default_factory=CarParts.common([CarHarness.subaru_a]))
  footnotes: List[Enum] = field(default_factory=lambda: [Footnote.GLOBAL])

  def init_make(self, CP: car.CarParams):
    self.car_parts.parts.extend([Tool.socket_8mm_deep, Tool.pry_tool])

CAR_INFO: Dict[str, Union[SubaruCarInfo, List[SubaruCarInfo]]] = {
  CAR.ASCENT: SubaruCarInfo("Subaru Ascent 2019-21", "All"),
  CAR.OUTBACK: SubaruCarInfo("Subaru Outback 2020-22", "All", car_parts=CarParts.common([CarHarness.subaru_b])),
  CAR.LEGACY: SubaruCarInfo("Subaru Legacy 2020-22", "All", car_parts=CarParts.common([CarHarness.subaru_b])),
  CAR.IMPREZA: [
    SubaruCarInfo("Subaru Impreza 2017-19"),
    SubaruCarInfo("Subaru Crosstrek 2018-19", video_link="https://youtu.be/Agww7oE1k-s?t=26"),
    SubaruCarInfo("Subaru XV 2018-19", video_link="https://youtu.be/Agww7oE1k-s?t=26"),
  ],
  CAR.IMPREZA_2020: [
    SubaruCarInfo("Subaru Impreza 2020-22"),
    SubaruCarInfo("Subaru Crosstrek 2020-23"),
    SubaruCarInfo("Subaru XV 2020-21"),
  ],
  CAR.FORESTER: SubaruCarInfo("Subaru Forester 2019-21", "All"),
  CAR.FORESTER_PREGLOBAL: SubaruCarInfo("Subaru Forester 2017-18"),
  CAR.LEGACY_PREGLOBAL: SubaruCarInfo("Subaru Legacy 2015-18"),
  CAR.OUTBACK_PREGLOBAL: SubaruCarInfo("Subaru Outback 2015-17"),
  CAR.OUTBACK_PREGLOBAL_2018: SubaruCarInfo("Subaru Outback 2018-19"),
  CAR.FORESTER_2022: SubaruCarInfo("Subaru Forester 2022", "All", car_parts=CarParts.common([CarHarness.subaru_c])),
  CAR.OUTBACK_2023: SubaruCarInfo("Subaru Outback 2023", "All", car_parts=CarParts.common([CarHarness.subaru_d])),
}

DBC = {
  CAR.ASCENT: dbc_dict('subaru_global_2017_generated', None),
  CAR.IMPREZA: dbc_dict('subaru_global_2017_generated', None),
  CAR.IMPREZA_2020: dbc_dict('subaru_global_2017_generated', None),
  CAR.FORESTER: dbc_dict('subaru_global_2017_generated', None),
  CAR.FORESTER_2022: dbc_dict('subaru_global_2017_generated', None),
  CAR.OUTBACK: dbc_dict('subaru_global_2017_generated', None),
  CAR.OUTBACK_2023: dbc_dict('subaru_global_2017_generated', None),
  CAR.LEGACY: dbc_dict('subaru_global_2017_generated', None),
  CAR.FORESTER_PREGLOBAL: dbc_dict('subaru_forester_2017_generated', None),
  CAR.LEGACY_PREGLOBAL: dbc_dict('subaru_outback_2015_generated', None),
  CAR.OUTBACK_PREGLOBAL: dbc_dict('subaru_outback_2015_generated', None),
  CAR.OUTBACK_PREGLOBAL_2018: dbc_dict('subaru_outback_2019_generated', None),
}

LKAS_ANGLE = {CAR.FORESTER_2022, CAR.OUTBACK_2023}
GLOBAL_GEN2 = {CAR.OUTBACK, CAR.LEGACY, CAR.OUTBACK_2023}
PREGLOBAL_CARS = {CAR.FORESTER_PREGLOBAL, CAR.LEGACY_PREGLOBAL, CAR.OUTBACK_PREGLOBAL, CAR.OUTBACK_PREGLOBAL_2018}
