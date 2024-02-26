from enum import IntFlag
from dataclasses import dataclass, field

from cereal import car
from panda.python import uds
from openpilot.selfdrive.car import DbcDict, PlatformConfig, Platforms, dbc_dict
from openpilot.selfdrive.car.docs_definitions import CarHarness, CarInfo, CarParts
from openpilot.selfdrive.car.fw_query_definitions import FwQueryConfig, Request, p16

Ecu = car.CarParams.Ecu


class ChryslerFlags(IntFlag):
  HIGHER_MIN_STEERING_SPEED = 1

@dataclass
class ChryslerCarInfo(CarInfo):
  package: str = "Adaptive Cruise Control (ACC)"
  car_parts: CarParts = field(default_factory=CarParts.common([CarHarness.fca]))


@dataclass
class ChryslerPlatformConfig(PlatformConfig):
  dbc_dict: DbcDict = field(default_factory=lambda: dbc_dict('chrysler_pacifica_2017_hybrid_generated', None))


class CAR(Platforms):
  # Chrysler
  PACIFICA_2017_HYBRID = ChryslerPlatformConfig(
    "CHRYSLER PACIFICA HYBRID 2017",
    ChryslerCarInfo("Chrysler Pacifica Hybrid 2017"),
  )
  PACIFICA_2018_HYBRID = ChryslerPlatformConfig(
    "CHRYSLER PACIFICA HYBRID 2018",
    ChryslerCarInfo("Chrysler Pacifica Hybrid 2018"),
  )
  PACIFICA_2019_HYBRID = ChryslerPlatformConfig(
    "CHRYSLER PACIFICA HYBRID 2019",
    ChryslerCarInfo("Chrysler Pacifica Hybrid 2019-23"),
  )
  PACIFICA_2018 = ChryslerPlatformConfig(
    "CHRYSLER PACIFICA 2018",
    ChryslerCarInfo("Chrysler Pacifica 2017-18"),
  )
  PACIFICA_2020 = ChryslerPlatformConfig(
    "CHRYSLER PACIFICA 2020",
    [
      ChryslerCarInfo("Chrysler Pacifica 2019-20"),
      ChryslerCarInfo("Chrysler Pacifica 2021-23", package="All"),
    ]
  )

  # Dodge
  DODGE_DURANGO = ChryslerPlatformConfig(
    "DODGE DURANGO 2021",
    ChryslerCarInfo("Dodge Durango 2020-21"),
  )

  # Jeep
  JEEP_GRAND_CHEROKEE = ChryslerPlatformConfig( # includes 2017 Trailhawk
    "JEEP GRAND CHEROKEE V6 2018",
    ChryslerCarInfo("Jeep Grand Cherokee 2016-18", video_link="https://www.youtube.com/watch?v=eLR9o2JkuRk"),
  )

  JEEP_GRAND_CHEROKEE_2019 = ChryslerPlatformConfig(  # includes 2020 Trailhawk
    "JEEP GRAND CHEROKEE 2019",
    ChryslerCarInfo("Jeep Grand Cherokee 2019-21", video_link="https://www.youtube.com/watch?v=jBe4lWnRSu4"),
  )

  # Ram
  RAM_1500 = ChryslerPlatformConfig(
    "RAM 1500 5TH GEN",
    ChryslerCarInfo("Ram 1500 2019-24", car_parts=CarParts.common([CarHarness.ram])),
    dbc_dict('chrysler_ram_dt_generated', None),
  )
  RAM_HD = ChryslerPlatformConfig(
    "RAM HD 5TH GEN",
    [
      ChryslerCarInfo("Ram 2500 2020-24", car_parts=CarParts.common([CarHarness.ram])),
      ChryslerCarInfo("Ram 3500 2019-22", car_parts=CarParts.common([CarHarness.ram])),
    ],
    dbc_dict('chrysler_ram_hd_generated', None),
  )


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


CHRYSLER_VERSION_REQUEST = bytes([uds.SERVICE_TYPE.READ_DATA_BY_IDENTIFIER]) + \
  p16(0xf132)
CHRYSLER_VERSION_RESPONSE = bytes([uds.SERVICE_TYPE.READ_DATA_BY_IDENTIFIER + 0x40]) + \
  p16(0xf132)

CHRYSLER_SOFTWARE_VERSION_REQUEST = bytes([uds.SERVICE_TYPE.READ_DATA_BY_IDENTIFIER]) + \
  p16(uds.DATA_IDENTIFIER_TYPE.SYSTEM_SUPPLIER_ECU_SOFTWARE_NUMBER)
CHRYSLER_SOFTWARE_VERSION_RESPONSE = bytes([uds.SERVICE_TYPE.READ_DATA_BY_IDENTIFIER + 0x40]) + \
  p16(uds.DATA_IDENTIFIER_TYPE.SYSTEM_SUPPLIER_ECU_SOFTWARE_NUMBER)

CHRYSLER_RX_OFFSET = -0x280

FW_QUERY_CONFIG = FwQueryConfig(
  requests=[
    Request(
      [CHRYSLER_VERSION_REQUEST],
      [CHRYSLER_VERSION_RESPONSE],
      whitelist_ecus=[Ecu.abs, Ecu.eps, Ecu.srs, Ecu.fwdRadar, Ecu.fwdCamera, Ecu.combinationMeter],
      rx_offset=CHRYSLER_RX_OFFSET,
      bus=0,
    ),
    Request(
      [CHRYSLER_VERSION_REQUEST],
      [CHRYSLER_VERSION_RESPONSE],
      whitelist_ecus=[Ecu.abs, Ecu.hybrid, Ecu.engine, Ecu.transmission],
      bus=0,
    ),
    Request(
      [CHRYSLER_SOFTWARE_VERSION_REQUEST],
      [CHRYSLER_SOFTWARE_VERSION_RESPONSE],
      whitelist_ecus=[Ecu.engine, Ecu.transmission],
      bus=0,
    ),
  ],
  extra_ecus=[
    (Ecu.abs, 0x7e4, None),  # alt address for abs on hybrids, NOTE: not on all hybrid platforms
  ],
)

CAR_INFO = CAR.create_carinfo_map()
DBC = CAR.create_dbc_map()
