from dataclasses import dataclass, field
from enum import Enum, IntFlag
from opendbc.car import Bus, CarSpecs, DbcDict, PlatformConfig, Platforms, AngleSteeringLimits
from opendbc.car.structs import CarParams, CarState
from opendbc.car.docs_definitions import CarDocs, CarFootnote, CarHarness, CarParts, Column
from opendbc.car.fw_query_definitions import FwQueryConfig, Request, StdQueries

Ecu = CarParams.Ecu


class Footnote(Enum):
  HW_TYPE = CarFootnote(
    "Some 2023 model years have HW4. To check which hardware type your vehicle has, look for " +
    "<b>Autopilot computer</b> under <b>Software -> Additional Vehicle Information</b> on your vehicle's touchscreen. </br></br>" +
    "See <a href=\"https://www.notateslaapp.com/news/2173/how-to-check-if-your-tesla-has-hardware-4-ai4-or-hardware-3\">this page</a> for more information.",
    Column.MODEL)

  SETUP = CarFootnote(
    "See more setup details for <a href=\"https://github.com/commaai/openpilot/wiki/tesla\" target=\"_blank\">Tesla</a>.",
    Column.MAKE, setup_note=True)


@dataclass
class TeslaCarDocsHW3(CarDocs):
  package: str = "All"
  car_parts: CarParts = field(default_factory=CarParts.common([CarHarness.tesla_a]))
  footnotes: list[Enum] = field(default_factory=lambda: [Footnote.HW_TYPE, Footnote.SETUP])


@dataclass
class TeslaCarDocsHW4(CarDocs):
  package: str = "All"
  car_parts: CarParts = field(default_factory=CarParts.common([CarHarness.tesla_b]))
  footnotes: list[Enum] = field(default_factory=lambda: [Footnote.HW_TYPE, Footnote.SETUP])


@dataclass
class TeslaPlatformConfig(PlatformConfig):
  dbc_dict: DbcDict = field(default_factory=lambda: {Bus.party: 'tesla_model3_party'})


class CAR(Platforms):
  TESLA_MODEL_3 = TeslaPlatformConfig(
    [
      # TODO: do we support 2017? It's HW3
      TeslaCarDocsHW3("Tesla Model 3 (with HW3) 2019-23"),
      TeslaCarDocsHW4("Tesla Model 3 (with HW4) 2024-25"),
    ],
    CarSpecs(mass=1899., wheelbase=2.875, steerRatio=12.0),
  )
  TESLA_MODEL_Y = TeslaPlatformConfig(
    [
      TeslaCarDocsHW3("Tesla Model Y (with HW3) 2020-23"),
      TeslaCarDocsHW4("Tesla Model Y (with HW4) 2024"),
     ],
    CarSpecs(mass=2072., wheelbase=2.890, steerRatio=12.0),
  )


FW_QUERY_CONFIG = FwQueryConfig(
  requests=[
    Request(
      [StdQueries.TESTER_PRESENT_REQUEST, StdQueries.SUPPLIER_SOFTWARE_VERSION_REQUEST],
      [StdQueries.TESTER_PRESENT_RESPONSE, StdQueries.SUPPLIER_SOFTWARE_VERSION_RESPONSE],
      bus=0,
    )
  ]
)


class CANBUS:
  party = 0
  vehicle = 1
  autopilot_party = 2


GEAR_MAP = {
  "DI_GEAR_INVALID": CarState.GearShifter.unknown,
  "DI_GEAR_P": CarState.GearShifter.park,
  "DI_GEAR_R": CarState.GearShifter.reverse,
  "DI_GEAR_N": CarState.GearShifter.neutral,
  "DI_GEAR_D": CarState.GearShifter.drive,
  "DI_GEAR_SNA": CarState.GearShifter.unknown,
}


class CarControllerParams:
  ANGLE_LIMITS: AngleSteeringLimits = AngleSteeringLimits(
    # EPAS faults above this angle
    360,  # deg
    # Angle rate limits are set using the Tesla Model Y VehicleModel such that they maximally meet ISO 11270
    # At 5 m/s, FSD has been seen hitting up to ~4 deg/frame with ~5 deg/frame at very low creeping speeds
    # At 30 m/s, FSD has been seen hitting mostly 0.1 deg/frame, sometimes 0.2 deg/frame, and rarely 0.3 deg/frame
    ([0., 5., 25.], [2.5, 1.5, 0.2]),
    ([0., 5., 25.], [5., 2.0, 0.3]),
  )

  STEER_STEP = 2  # Angle command is sent at 50 Hz
  ACCEL_MAX = 2.0    # m/s^2
  ACCEL_MIN = -3.48  # m/s^2
  JERK_LIMIT_MAX = 4.9  # m/s^3, ACC faults at 5.0
  JERK_LIMIT_MIN = -4.9  # m/s^3, ACC faults at 5.0


class TeslaSafetyFlags(IntFlag):
  LONG_CONTROL = 1


class TeslaFlags(IntFlag):
  LONG_CONTROL = 1


DBC = CAR.create_dbc_map()

STEER_THRESHOLD = 1
