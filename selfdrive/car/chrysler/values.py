from enum import IntFlag
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from selfdrive.car import dbc_dict
from selfdrive.car.docs_definitions import CarHarness, CarInfo, CarParts


class ChryslerFlags(IntFlag):
  HIGHER_MIN_STEERING_SPEED = 1


class CAR:
  # Chrysler
  PACIFICA_2017_HYBRID = "CHRYSLER PACIFICA HYBRID 2017"
  PACIFICA_2018_HYBRID = "CHRYSLER PACIFICA HYBRID 2018"
  PACIFICA_2019_HYBRID = "CHRYSLER PACIFICA HYBRID 2019"
  PACIFICA_2018 = "CHRYSLER PACIFICA 2018"
  PACIFICA_2020 = "CHRYSLER PACIFICA 2020"

  # Jeep
  JEEP_CHEROKEE = "JEEP GRAND CHEROKEE V6 2018"   # includes 2017 Trailhawk
  JEEP_CHEROKEE_2019 = "JEEP GRAND CHEROKEE 2019" # includes 2020 Trailhawk

  # Ram
  RAM_1500 = "RAM 1500 5TH GEN"
  RAM_HD = "RAM HD 5TH GEN"


class CarControllerParams:
  def __init__(self, CP):
    self.STEER_STEP = 2  # 50 Hz
    self.STEER_ERROR_MAX = 80
    if CP.carFingerprint in RAM_HD:
      self.STEER_DELTA_UP = 14
      self.STEER_DELTA_DOWN = 14
      self.STEER_MAX = 361  # higher than this faults the EPS
    elif CP.carFingerprint in RAM_DT:
      self.STEER_DELTA_UP = 6
      self.STEER_DELTA_DOWN = 6
      self.STEER_MAX = 261  # EPS allows more, up to 350?
    else:
      self.STEER_DELTA_UP = 3
      self.STEER_DELTA_DOWN = 3
      self.STEER_MAX = 261  # higher than this faults the EPS


STEER_THRESHOLD = 120

RAM_DT = {CAR.RAM_1500, }
RAM_HD = {CAR.RAM_HD, }
RAM_CARS = RAM_DT | RAM_HD


@dataclass
class ChryslerCarInfo(CarInfo):
  package: str = "Adaptive Cruise Control (ACC)"
  car_parts: CarParts = field(default_factory=CarParts.common([CarHarness.fca]))


CAR_INFO: Dict[str, Optional[Union[ChryslerCarInfo, List[ChryslerCarInfo]]]] = {
  CAR.PACIFICA_2017_HYBRID: ChryslerCarInfo("Chrysler Pacifica Hybrid 2017-18"),
  CAR.PACIFICA_2018_HYBRID: None,  # same platforms
  CAR.PACIFICA_2019_HYBRID: ChryslerCarInfo("Chrysler Pacifica Hybrid 2019-23"),
  CAR.PACIFICA_2018: ChryslerCarInfo("Chrysler Pacifica 2017-18"),
  CAR.PACIFICA_2020: [
    ChryslerCarInfo("Chrysler Pacifica 2019-20"),
    ChryslerCarInfo("Chrysler Pacifica 2021", package="All"),
  ],
  CAR.JEEP_CHEROKEE: ChryslerCarInfo("Jeep Grand Cherokee 2016-18", video_link="https://www.youtube.com/watch?v=eLR9o2JkuRk"),
  CAR.JEEP_CHEROKEE_2019: ChryslerCarInfo("Jeep Grand Cherokee 2019-21", video_link="https://www.youtube.com/watch?v=jBe4lWnRSu4"),
  CAR.RAM_1500: ChryslerCarInfo("Ram 1500 2019-23", car_parts=CarParts.common([CarHarness.ram])),
  CAR.RAM_HD: [
    ChryslerCarInfo("Ram 2500 2020-22", car_parts=CarParts.common([CarHarness.ram])),
    ChryslerCarInfo("Ram 3500 2019-22", car_parts=CarParts.common([CarHarness.ram])),
  ],
}

DBC = {
  CAR.PACIFICA_2017_HYBRID: dbc_dict('chrysler_pacifica_2017_hybrid_generated', 'chrysler_pacifica_2017_hybrid_private_fusion'),
  CAR.PACIFICA_2018: dbc_dict('chrysler_pacifica_2017_hybrid_generated', 'chrysler_pacifica_2017_hybrid_private_fusion'),
  CAR.PACIFICA_2020: dbc_dict('chrysler_pacifica_2017_hybrid_generated', 'chrysler_pacifica_2017_hybrid_private_fusion'),
  CAR.PACIFICA_2018_HYBRID: dbc_dict('chrysler_pacifica_2017_hybrid_generated', 'chrysler_pacifica_2017_hybrid_private_fusion'),
  CAR.PACIFICA_2019_HYBRID: dbc_dict('chrysler_pacifica_2017_hybrid_generated', 'chrysler_pacifica_2017_hybrid_private_fusion'),
  CAR.JEEP_CHEROKEE: dbc_dict('chrysler_pacifica_2017_hybrid_generated', 'chrysler_pacifica_2017_hybrid_private_fusion'),
  CAR.JEEP_CHEROKEE_2019: dbc_dict('chrysler_pacifica_2017_hybrid_generated', 'chrysler_pacifica_2017_hybrid_private_fusion'),
  CAR.RAM_1500: dbc_dict('chrysler_ram_dt_generated', None),
  CAR.RAM_HD: dbc_dict('chrysler_ram_hd_generated', None),
}
