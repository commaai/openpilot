from dataclasses import dataclass, field

from openpilot.selfdrive.car import dbc_dict, CarSpecs, DbcDict, PlatformConfig, Platforms
from openpilot.selfdrive.car.docs_definitions import CarHarness, CarDocs, CarParts
from openpilot.selfdrive.car.fw_query_definitions import FwQueryConfig, Request, StdQueries


class CarControllerParams:
  STEER_STEP = 1
  HUD_1_STEP = 50
  HUD_2_STEP = 25

  STEER_MAX = 300
  STEER_DRIVER_ALLOWANCE = 80
  STEER_DRIVER_MULTIPLIER = 3  # weight driver torque heavily
  STEER_DRIVER_FACTOR = 1  # from dbc
  STEER_DELTA_UP = 4
  STEER_DELTA_DOWN = 4

  def __init__(self, CP):
    pass


class CANBUS:
  pt = 0
  cam = 2


@dataclass
class FcaGiorgioPlatformConfig(PlatformConfig):
  dbc_dict: DbcDict = field(default_factory=lambda: dbc_dict('fca_giorgio', None))


@dataclass(frozen=True, kw_only=True)
class FcaGiorgioCarSpecs(CarSpecs):
  centerToFrontRatio: float = 0.45
  steerRatio: float = 14.2


@dataclass
class FcaGiorgioCarDocs(CarDocs):
  package: str = "Adaptive Cruise Control (ACC) & Lane Assist"
  car_parts: CarParts = field(default_factory=CarParts.common([CarHarness.vw_a]))


class CAR(Platforms):
  config: FcaGiorgioPlatformConfig

  ALFA_ROMEO_STELVIO_1ST_GEN = FcaGiorgioPlatformConfig(
    [FcaGiorgioCarDocs("Alfa Romeo Stelvio 2017-24")],
    FcaGiorgioCarSpecs(mass=1660, wheelbase=2.82),
  )


FW_QUERY_CONFIG = FwQueryConfig(
  requests=[
    # TODO: check data to ensure ABS does not skip ISO-TP frames on bus 0
    Request(
      [StdQueries.MANUFACTURER_SOFTWARE_VERSION_REQUEST],
      [StdQueries.MANUFACTURER_SOFTWARE_VERSION_RESPONSE],
      bus=0,
    ),
  ],
)


DBC = CAR.create_dbc_map()
