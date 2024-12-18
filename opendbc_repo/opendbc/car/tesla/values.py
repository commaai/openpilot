from enum import IntFlag
from opendbc.car.structs import CarParams
from opendbc.car import Bus, structs
from opendbc.car import AngleRateLimit, CarSpecs, PlatformConfig, Platforms
from opendbc.car.docs_definitions import CarDocs
from opendbc.car.fw_query_definitions import FwQueryConfig, Request, StdQueries

Ecu = CarParams.Ecu


class CAR(Platforms):
  TESLA_MODEL_3 = PlatformConfig(
    [CarDocs("Tesla Model 3", "All")],
    CarSpecs(mass=1899., wheelbase=2.875, steerRatio=12.0),
    {Bus.party: 'tesla_model3_party'},
  )
  TESLA_MODEL_Y = PlatformConfig(
    [CarDocs("Tesla Model Y", "All")],
    CarSpecs(mass=2072., wheelbase=2.890, steerRatio=12.0),
    {Bus.party: 'tesla_model3_party'},
  )


FW_QUERY_CONFIG = FwQueryConfig(
  requests=[
    Request(
      [StdQueries.TESTER_PRESENT_REQUEST, StdQueries.SUPPLIER_SOFTWARE_VERSION_REQUEST],
      [StdQueries.TESTER_PRESENT_RESPONSE, StdQueries.SUPPLIER_SOFTWARE_VERSION_RESPONSE],
      whitelist_ecus=[Ecu.eps],
      rx_offset=0x08,
      bus=0,
    )
  ]
)

class CANBUS:
  party = 0
  vehicle = 1
  autopilot_party = 2


GEAR_MAP = {
  "DI_GEAR_INVALID": structs.CarState.GearShifter.unknown,
  "DI_GEAR_P": structs.CarState.GearShifter.park,
  "DI_GEAR_R": structs.CarState.GearShifter.reverse,
  "DI_GEAR_N": structs.CarState.GearShifter.neutral,
  "DI_GEAR_D": structs.CarState.GearShifter.drive,
  "DI_GEAR_SNA": structs.CarState.GearShifter.unknown,
}

class CarControllerParams:
  ANGLE_RATE_LIMIT_UP = AngleRateLimit(speed_bp=[0., 5., 15.], angle_v=[10., 1.6, .3])
  ANGLE_RATE_LIMIT_DOWN = AngleRateLimit(speed_bp=[0., 5., 15.], angle_v=[10., 7.0, 0.8])
  ACCEL_MIN = -3.48  # m/s^2
  ACCEL_MAX = 2.0    # m/s^2
  JERK_LIMIT_MAX = 5
  JERK_LIMIT_MIN = -5


class TeslaFlags(IntFlag):
  FLAG_TESLA_LONG_CONTROL = 1


DBC = CAR.create_dbc_map()
