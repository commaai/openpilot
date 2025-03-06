from dataclasses import dataclass, field
from enum import StrEnum, IntFlag

from opendbc.car.structs import CarParams
from opendbc.car import Bus, structs
from opendbc.car import CarSpecs, PlatformConfig, Platforms
from opendbc.car.docs_definitions import CarHarness, CarDocs, CarParts, Device
from opendbc.car.fw_query_definitions import FwQueryConfig, Request, StdQueries

Ecu = CarParams.Ecu


class WMI(StrEnum):
  RIVIAN_TRUCK = "7FC"
  RIVIAN_MPV = "7PD"


@dataclass
class RivianCarDocs(CarDocs):
  package: str = "All"
  car_parts: CarParts = field(default_factory=CarParts([Device.threex_angled_mount, CarHarness.rivian]))


class CAR(Platforms):
  RIVIAN_R1_GEN1 = PlatformConfig(
    # TODO: verify this
    [
      RivianCarDocs("Rivian R1S 2022-24"),
      RivianCarDocs("Rivian R1T 2022-24"),
    ],
    CarSpecs(mass=3206., wheelbase=3.08, steerRatio=15.2),
    {Bus.pt: 'rivian_primary_actuator', Bus.radar: 'rivian_mando_front_radar_generated'}
  )


FW_QUERY_CONFIG = FwQueryConfig(
  requests=[
    Request(
      [StdQueries.TESTER_PRESENT_REQUEST, StdQueries.SUPPLIER_SOFTWARE_VERSION_REQUEST],
      [StdQueries.TESTER_PRESENT_RESPONSE, StdQueries.SUPPLIER_SOFTWARE_VERSION_RESPONSE],
      rx_offset=0x40,
      bus=0,
    )
  ]
)

GEAR_MAP = [
  structs.CarState.GearShifter.unknown,
  structs.CarState.GearShifter.park,
  structs.CarState.GearShifter.reverse,
  structs.CarState.GearShifter.neutral,
  structs.CarState.GearShifter.drive,
]


class CarControllerParams:
  # The Rivian R1T we tested on achieves slightly more lateral acceleration going left vs. right
  # and lateral acceleration rises as speed increases. This value is set conservatively to
  # reach a maximum of 2.5-3.0 m/s^2 turning left at 80 mph, but is less at lower speeds
  STEER_MAX = 250  # ~2.5 m/s^2
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
