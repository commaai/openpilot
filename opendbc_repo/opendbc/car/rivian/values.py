from dataclasses import dataclass, field
from enum import StrEnum

from opendbc.car.structs import CarParams
from opendbc.car import Bus, structs
from opendbc.car import CarSpecs, PlatformConfig, Platforms
from opendbc.car.docs_definitions import CarHarness, CarDocs, CarParts
from opendbc.car.fw_query_definitions import FwQueryConfig

Ecu = CarParams.Ecu


class WMI(StrEnum):
  RIVIAN_TRUCK = "7FC"
  RIVIAN_MPV = "7PD"


@dataclass
class RivianCarDocs(CarDocs):
  package: str = "All"
  car_parts: CarParts = field(default_factory=CarParts.common([CarHarness.rivian]))


class CAR(Platforms):
  RIVIAN_R1_GEN1 = PlatformConfig(
    # TODO: verify this
    [
      RivianCarDocs("Rivian R1S 2022-24"),
      RivianCarDocs("Rivian R1T 2022-24"),
    ],
    CarSpecs(mass=3206., wheelbase=3.08, steerRatio=15.2),
    {Bus.pt: 'rivian_can'}
  )


# TODO: Placeholder ↓
FW_QUERY_CONFIG = FwQueryConfig(
  requests=[
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
  STEER_MAX = 350
  STEER_STEP = 1
  STEER_DELTA_UP = 4  # torque increase per refresh
  STEER_DELTA_DOWN = 6  # torque decrease per refresh
  STEER_DRIVER_ALLOWANCE = 15  # allowed driver torque before start limiting
  STEER_DRIVER_MULTIPLIER = 1  # weight driver torque
  STEER_DRIVER_FACTOR = 1

  ACCEL_MIN = -3.48  # m/s^2
  ACCEL_MAX = 2.0  # m/s^2

  def __init__(self, CP):
    pass


DBC = CAR.create_dbc_map()
