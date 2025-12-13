from dataclasses import dataclass, field
from enum import StrEnum, IntFlag

from opendbc.car import Bus, CarSpecs, DbcDict, PlatformConfig, Platforms, structs, uds
from opendbc.car.docs_definitions import CarHarness, CarDocs, CarParts
from opendbc.car.fw_query_definitions import FwQueryConfig, Request, StdQueries, p16
from opendbc.car.vin import Vin


class WMI(StrEnum):
  RIVIAN_TRUCK = "7FC"
  RIVIAN_MPV = "7PD"


class ModelLine(StrEnum):
  R1T = "T"  # R1T 4-door Pickup Truck
  R1S = "S"  # R1S 4-door MPV


class ModelYear(StrEnum):
  N_2022 = "N"
  P_2023 = "P"
  R_2024 = "R"
  S_2025 = "S"


@dataclass
class RivianCarDocs(CarDocs):
  package: str = "All"
  car_parts: CarParts = field(default_factory=CarParts.common([CarHarness.rivian]))
  setup_video: str = "https://youtu.be/uaISd1j7Z4U"


@dataclass
class RivianPlatformConfig(PlatformConfig):
  dbc_dict: DbcDict = field(default_factory=lambda: {Bus.pt: 'rivian_primary_actuator', Bus.radar: 'rivian_mando_front_radar_generated'})
  wmis: set[WMI] = field(default_factory=set)
  lines: set[ModelLine] = field(default_factory=set)
  years: set[ModelYear] = field(default_factory=set)


class CAR(Platforms):
  RIVIAN_R1_GEN1 = RivianPlatformConfig(
    # TODO: verify this
    [
      RivianCarDocs("Rivian R1S 2022-24"),
      RivianCarDocs("Rivian R1T 2022-24"),
    ],
    CarSpecs(mass=3206., wheelbase=3.08, steerRatio=15.2),
    wmis={WMI.RIVIAN_TRUCK, WMI.RIVIAN_MPV},
    lines={ModelLine.R1T, ModelLine.R1S},
    years={ModelYear.N_2022, ModelYear.P_2023, ModelYear.R_2024},
  )


def match_fw_to_car_fuzzy(live_fw_versions, vin, offline_fw_versions) -> set[str]:
  # Rivian VIN reference: https://www.rivianforums.com/forum/threads/rivian-vin-decoder.1546
  vin_obj = Vin(vin)
  line = vin_obj.vds[:1]
  year = vin_obj.vis[:1]

  candidates = set()
  for platform in CAR:
    if vin_obj.wmi in platform.config.wmis and line in platform.config.lines and year in platform.config.years:
      candidates.add(platform)

  return {str(c) for c in candidates}


RIVIAN_VERSION_REQUEST = bytes([uds.SERVICE_TYPE.READ_DATA_BY_IDENTIFIER]) + \
  p16(0xf1a0)
RIVIAN_VERSION_RESPONSE = bytes([uds.SERVICE_TYPE.READ_DATA_BY_IDENTIFIER + 0x40])

FW_QUERY_CONFIG = FwQueryConfig(
  requests=[
    Request(
      [StdQueries.TESTER_PRESENT_REQUEST, StdQueries.SUPPLIER_SOFTWARE_VERSION_REQUEST],
      [StdQueries.TESTER_PRESENT_RESPONSE, StdQueries.SUPPLIER_SOFTWARE_VERSION_RESPONSE],
      rx_offset=0x40,
      bus=0,
    ),
    Request(
      [StdQueries.TESTER_PRESENT_REQUEST, StdQueries.MANUFACTURER_ECU_HARDWARE_NUMBER_REQUEST],
      [StdQueries.TESTER_PRESENT_RESPONSE, StdQueries.MANUFACTURER_ECU_HARDWARE_NUMBER_RESPONSE],
      rx_offset=0x40,
      bus=0,
      logging=True,
    ),
    Request(
      [StdQueries.TESTER_PRESENT_REQUEST, RIVIAN_VERSION_REQUEST],
      [StdQueries.TESTER_PRESENT_RESPONSE, RIVIAN_VERSION_RESPONSE],
      rx_offset=0x40,
      bus=0,
      logging=True,
    ),
  ],
  match_fw_to_car_fuzzy=match_fw_to_car_fuzzy,
)

GEAR_MAP = {
  0: structs.CarState.GearShifter.unknown,
  1: structs.CarState.GearShifter.park,
  2: structs.CarState.GearShifter.reverse,
  3: structs.CarState.GearShifter.neutral,
  4: structs.CarState.GearShifter.drive,
}


class CarControllerParams:
  # The R1T 2023 and R1S 2023 we tested on achieves slightly more lateral acceleration going left vs. right
  # and lateral acceleration falls linearly as speed decreases from 38 mph to 20 mph. These values are set
  # conservatively to reach a maximum of 3.0 m/s^2 turning left at 80 mph

  # These refer to turning left:
  # 250 is ~2.8 m/s^2 above 17 m/s, then linearly ramps to ~1.6 m/s^2 from 17 m/s to 9 m/s
  # TODO: it is theorized older models have different steering racks and achieve down to half the
  #  lateral acceleration referenced here at all speeds. detect this and ship a torque increase for those models
  STEER_MAX = 250  # 350 is intended to maintain lateral accel, not increase it
  STEER_MAX_LOOKUP = [9, 17], [350, 250]
  STEER_STEP = 1
  STEER_DELTA_UP = 3  # torque increase per refresh
  STEER_DELTA_DOWN = 5  # torque decrease per refresh
  STEER_DRIVER_ALLOWANCE = 100  # allowed driver torque before start limiting
  STEER_DRIVER_MULTIPLIER = 2  # weight driver torque
  STEER_DRIVER_FACTOR = 100

  ACCEL_MIN = -3.5  # m/s^2
  ACCEL_MAX = 2.0  # m/s^2

  def __init__(self, CP):
    pass


class RivianSafetyFlags(IntFlag):
  LONG_CONTROL = 1


DBC = CAR.create_dbc_map()
