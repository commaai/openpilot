from enum import IntFlag, StrEnum
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


class WMI(StrEnum):
  US_CAR = "1C3"
  US_MPV = "1C4"
  US_TRUCK = "1C6"
  CANADA_CAR = "2C3"
  CANADA_MPV = "2C4"
  CANADA_TRUCK = "2C6"
  MEXICO_CAR = "3C3"
  MEXICO_MPV = "3C4"
  MEXICO_TRUCK = "3C6"

class CAR(Platforms):
  # https://vpic.nhtsa.dot.gov/mid/home/displayfile/b5f541b7-8157-415d-889c-7f63628e9e66 vin stuff 2024 ()
  # a platform is fully determined by [5:6]+[8]+[10] (chassis_code+engine+year)
  # Chrysler
  CHRYSLER_PACIFICA_2017_HYBRID = ChryslerPlatformConfig( # ru
    [ChryslerCarDocs("Chrysler Pacifica Hybrid 2017")],
    ChryslerCarSpecs(mass=2242., wheelbase=3.089, steerRatio=16.2),
    wmis=[],
    chassi_codes=["C1"],
    year="H",
    engine=["7"]
  )
  CHRYSLER_PACIFICA_2018_HYBRID = ChryslerPlatformConfig( # ru
    [ChryslerCarDocs("Chrysler Pacifica Hybrid 2018")],
    CHRYSLER_PACIFICA_2017_HYBRID.specs,
    wmis=[],
    chassi_codes=["C1"],
    year="J",
    engine=["7"]
  )
  CHRYSLER_PACIFICA_2019_HYBRID = ChryslerPlatformConfig( # ru
    [ChryslerCarDocs("Chrysler Pacifica Hybrid 2019-24")],
    CHRYSLER_PACIFICA_2017_HYBRID.specs,
    wmis=[],
    chassi_codes=["C1", "C3"],
    year="K",
    engine=["7"]
  )
  CHRYSLER_PACIFICA_2018 = ChryslerPlatformConfig( # ru
    [ChryslerCarDocs("Chrysler Pacifica 2017-18")],
    CHRYSLER_PACIFICA_2017_HYBRID.specs,
    wmis=[],
    chassi_codes=["C1", "C3"],
    year="J",
    engine=["7"]
  )
  CHRYSLER_PACIFICA_2020 = ChryslerPlatformConfig( # ru
    [
      ChryslerCarDocs("Chrysler Pacifica 2019-20"),
      ChryslerCarDocs("Chrysler Pacifica 2021-23", package="All"),
    ],
    CHRYSLER_PACIFICA_2017_HYBRID.specs,
    wmis=[],
    year="L",
    chassi_codes=["C1", "C3"],
    engine=["G"]
  )

  # Dodge
  DODGE_DURANGO = ChryslerPlatformConfig( # wd
    [ChryslerCarDocs("Dodge Durango 2020-21")],
    CHRYSLER_PACIFICA_2017_HYBRID.specs,
    wmis=[],
    chassi_codes=["DH", "DJ"],
    year="A",
    engine=["G"]
  )

  # Jeep
  JEEP_GRAND_CHEROKEE = ChryslerPlatformConfig(  # includes 2017 Trailhawk # wk
    [ChryslerCarDocs("Jeep Grand Cherokee 2016-18", video_link="https://www.youtube.com/watch?v=eLR9o2JkuRk")],
    ChryslerCarSpecs(mass=1778., wheelbase=2.71, steerRatio=16.7),
    wmis=[],
    chassi_codes=["JE", "JF"],
    year="J",
    engine=["G"],
  )

  JEEP_GRAND_CHEROKEE_2019 = ChryslerPlatformConfig(  # includes 2020 Trailhawk # wk
    [ChryslerCarDocs("Jeep Grand Cherokee 2019-21", video_link="https://www.youtube.com/watch?v=jBe4lWnRSu4")],
    JEEP_GRAND_CHEROKEE.specs,
    wmis=[],
    chassi_codes=["JG", "JH"],
    year="K",
    engine=["G"]
  )

  # Ram
  RAM_1500_5TH_GEN = ChryslerPlatformConfig(
    [ChryslerCarDocs("Ram 1500 2019-24", car_parts=CarParts.common([CarHarness.ram]))],
    ChryslerCarSpecs(mass=2493., wheelbase=3.88, steerRatio=16.3, minSteerSpeed=14.5),
    dbc_dict('chrysler_ram_dt_generated', None),
    wmis=[],
    chassi_codes=["C1"],
    year="K",
    engine=["G"]
  )
  RAM_HD_5TH_GEN = ChryslerPlatformConfig(
    [
      ChryslerCarDocs("Ram 2500 2020-24", car_parts=CarParts.common([CarHarness.ram])),
      ChryslerCarDocs("Ram 3500 2019-22", car_parts=CarParts.common([CarHarness.ram])),
    ],
    ChryslerCarSpecs(mass=3405., wheelbase=3.785, steerRatio=15.61, minSteerSpeed=16.),
    dbc_dict('chrysler_ram_hd_generated', None),
    wmis=[],
    chassi_codes=["C1"],
    year="K",
    engine=["G"]
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

def match_fw_to_car_fuzzy(live_fw_versions, vin, offline_fw_versions) -> set[str]:
  candidates = set()

  # Compile all FW versions for each ECU
  all_ecu_versions: dict[EcuAddrSubAddr, set[str]] = defaultdict(set)
  for ecus in offline_fw_versions.values():
    for ecu, versions in ecus.items():
      all_ecu_versions[ecu] |= set(versions)

  # Check the WMI and chassis code to determine the platform
  wmi = vin[:3]
  chassis_code = vin[4:6] # use vin[5:6]
  engine = vin[7]
  year = vin[9]

  for platform in CAR:
    valid_ecus = set()
    for ecu in offline_fw_versions[platform]:
      addr = ecu[1:]
      if ecu[0] not in CHECK_FUZZY_ECUS:
        continue

      # Sanity check that live FW is in the superset of all FW, Volkswagen ECU part numbers are commonly shared
      found_versions = live_fw_versions.get(addr, [])
      expected_versions = all_ecu_versions[ecu]
      if not any(found_version in expected_versions for found_version in found_versions):
        break

      valid_ecus.add(ecu[0])

    if valid_ecus != CHECK_FUZZY_ECUS:
      continue

    if wmi in platform.config.wmis and chassis_code in platform.config.chassis_codes: # code in all_comb_possible_of_wmi+chassis_code+engine+year
      candidates.add(platform)

  return {str(c) for c in candidates}

CHECK_FUZZY_ECUS = {}



STEER_THRESHOLD = 120

RAM_DT = {CAR.RAM_1500_5TH_GEN, }
RAM_HD = {CAR.RAM_HD_5TH_GEN, }
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
