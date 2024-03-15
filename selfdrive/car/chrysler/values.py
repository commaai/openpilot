from enum import IntFlag
from dataclasses import dataclass, field

from cereal import car
from panda.python import uds
from openpilot.selfdrive.car import CarSpecs, DbcDict, PlatformConfig, Platforms, dbc_dict
from openpilot.selfdrive.car.docs_definitions import CarHarness, CarDocs, CarParts
from openpilot.selfdrive.car.fw_query_definitions import FwQueryConfig, Request, p16

Ecu = car.CarParams.Ecu


class ChryslerFlags(IntFlag):
  # Detected flags
  HIGHER_MIN_STEERING_SPEED = 1

@dataclass
class ChryslerCarDocs(CarDocs):
  package: str = "Adaptive Cruise Control (ACC)"
  car_parts: CarParts = field(default_factory=CarParts.common([CarHarness.fca]))


@dataclass
class ChryslerPlatformConfig(PlatformConfig):
  dbc_dict: DbcDict = field(default_factory=lambda: dbc_dict('chrysler_pacifica_2017_hybrid_generated', 'chrysler_pacifica_2017_hybrid_private_fusion'))


@dataclass(frozen=True)
class ChryslerCarSpecs(CarSpecs):
  minSteerSpeed: float = 3.8  # m/s


class CAR(Platforms):
  # Chrysler
  PACIFICA_2017_HYBRID = ChryslerPlatformConfig(
    "CHRYSLER PACIFICA HYBRID 2017",
    [ChryslerCarDocs("Chrysler Pacifica Hybrid 2017")],
    ChryslerCarSpecs(mass=2242., wheelbase=3.089, steerRatio=16.2),
  )
  PACIFICA_2018_HYBRID = ChryslerPlatformConfig(
    "CHRYSLER PACIFICA HYBRID 2018",
    [ChryslerCarDocs("Chrysler Pacifica Hybrid 2018")],
    PACIFICA_2017_HYBRID.specs,
  )
  PACIFICA_2019_HYBRID = ChryslerPlatformConfig(
    "CHRYSLER PACIFICA HYBRID 2019",
    [ChryslerCarDocs("Chrysler Pacifica Hybrid 2019-23")],
    PACIFICA_2017_HYBRID.specs,
  )
  PACIFICA_2018 = ChryslerPlatformConfig(
    "CHRYSLER PACIFICA 2018",
    [ChryslerCarDocs("Chrysler Pacifica 2017-18")],
    PACIFICA_2017_HYBRID.specs,
  )
  PACIFICA_2020 = ChryslerPlatformConfig(
    "CHRYSLER PACIFICA 2020",
    [
      ChryslerCarDocs("Chrysler Pacifica 2019-20"),
      ChryslerCarDocs("Chrysler Pacifica 2021-23", package="All"),
    ],
    PACIFICA_2017_HYBRID.specs,
  )

  # Dodge
  DODGE_DURANGO = ChryslerPlatformConfig(
    "DODGE DURANGO 2021",
    [ChryslerCarDocs("Dodge Durango 2020-21")],
    PACIFICA_2017_HYBRID.specs,
  )

  # Jeep
  JEEP_GRAND_CHEROKEE = ChryslerPlatformConfig(  # includes 2017 Trailhawk
    "JEEP GRAND CHEROKEE V6 2018",
    [ChryslerCarDocs("Jeep Grand Cherokee 2016-18", video_link="https://www.youtube.com/watch?v=eLR9o2JkuRk")],
    ChryslerCarSpecs(mass=1778., wheelbase=2.71, steerRatio=16.7),
  )

  JEEP_GRAND_CHEROKEE_2019 = ChryslerPlatformConfig(  # includes 2020 Trailhawk
    "JEEP GRAND CHEROKEE 2019",
    [ChryslerCarDocs("Jeep Grand Cherokee 2019-21", video_link="https://www.youtube.com/watch?v=jBe4lWnRSu4")],
    JEEP_GRAND_CHEROKEE.specs,
  )

  # Ram
  RAM_1500 = ChryslerPlatformConfig(
    "RAM 1500 5TH GEN",
    [ChryslerCarDocs("Ram 1500 2019-24", car_parts=CarParts.common([CarHarness.ram]))],
    ChryslerCarSpecs(mass=2493., wheelbase=3.88, steerRatio=16.3, minSteerSpeed=14.5),
    dbc_dict('chrysler_ram_dt_generated', None),
  )
  RAM_HD = ChryslerPlatformConfig(
    "RAM HD 5TH GEN",
    [
      ChryslerCarDocs("Ram 2500 2020-24", car_parts=CarParts.common([CarHarness.ram])),
      ChryslerCarDocs("Ram 3500 2019-22", car_parts=CarParts.common([CarHarness.ram])),
    ],
    ChryslerCarSpecs(mass=3405., wheelbase=3.785, steerRatio=15.61, minSteerSpeed=16.),
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

DBC = CAR.create_dbc_map()
