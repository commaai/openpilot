from dataclasses import dataclass, field

from openpilot.selfdrive.car import dbc_dict, CarSpecs, DbcDict, PlatformConfig, Platforms
from openpilot.selfdrive.car.docs_definitions import CarHarness, CarDocs, CarParts
from openpilot.selfdrive.car.fw_query_definitions import FwQueryConfig, Request, StdQueries


class CarControllerParams:
  def __init__(self, CP):
    pass


class CANBUS:
  pt = 0
  cam = 2


@dataclass
class GwmPlatformConfig(PlatformConfig):
  dbc_dict: DbcDict = field(default_factory=lambda: dbc_dict('gwm_haval_h6_mk3', None))


@dataclass(frozen=True, kw_only=True)
class GwmCarSpecs(CarSpecs):
  centerToFrontRatio: float = 0.45
  steerRatio: float = 15.6


@dataclass
class GwmCarDocs(CarDocs):
  package: str = "Adaptive Cruise Control (ACC) & Lane Assist"
  car_parts: CarParts = field(default_factory=CarParts.common([CarHarness.custom]))


class CAR(Platforms):
  config: GwmPlatformConfig

  GWM_HAVAL_H6_PHEV_3RD_GEN = GwmPlatformConfig(
    [GwmCarDocs("GWM Haval H6 hybrid plug-in 2020-24")],
    GwmCarSpecs(mass=2050, wheelbase=2.74),
  )


FW_QUERY_CONFIG = FwQueryConfig(
  requests=[
    # TODO:
    Request(
      [StdQueries.MANUFACTURER_SOFTWARE_VERSION_REQUEST],
      [StdQueries.MANUFACTURER_SOFTWARE_VERSION_RESPONSE],
      bus=0,
    ),
  ],
)


DBC = CAR.create_dbc_map()