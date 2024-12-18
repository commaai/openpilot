from dataclasses import dataclass, field

from panda import uds
from opendbc.car import AngleRateLimit, Bus, CarSpecs, DbcDict, PlatformConfig, Platforms
from opendbc.car.structs import CarParams
from opendbc.car.docs_definitions import CarDocs, CarHarness, CarParts
from opendbc.car.fw_query_definitions import FwQueryConfig, Request, StdQueries

Ecu = CarParams.Ecu


class CarControllerParams:
  ANGLE_RATE_LIMIT_UP = AngleRateLimit(speed_bp=[0., 5., 15.], angle_v=[5., .8, .15])
  ANGLE_RATE_LIMIT_DOWN = AngleRateLimit(speed_bp=[0., 5., 15.], angle_v=[5., 3.5, 0.4])
  LKAS_MAX_TORQUE = 1               # A value of 1 is easy to overpower
  STEER_THRESHOLD = 1.0

  # When output steering Angle not within range -1311 and 1310,
  #   CANPacker packs wrong angle output to be decoded by panda
  MAX_STEER_ANGLE = 1310

  def __init__(self, CP):
    pass


@dataclass
class NissanCarDocs(CarDocs):
  package: str = "ProPILOT Assist"
  car_parts: CarParts = field(default_factory=CarParts.common([CarHarness.nissan_a]))


@dataclass(frozen=True)
class NissanCarSpecs(CarSpecs):
  centerToFrontRatio: float = 0.44
  steerRatio: float = 17.


@dataclass
class NissanPlatformConfig(PlatformConfig):
  dbc_dict: DbcDict = field(default_factory=lambda: {Bus.pt: 'nissan_x_trail_2017_generated'})


class CAR(Platforms):
  NISSAN_XTRAIL = NissanPlatformConfig(
    [NissanCarDocs("Nissan X-Trail 2017")],
    NissanCarSpecs(mass=1610, wheelbase=2.705)
  )
  NISSAN_LEAF = NissanPlatformConfig(
    [NissanCarDocs("Nissan Leaf 2018-23", video_link="https://youtu.be/vaMbtAh_0cY")],
    NissanCarSpecs(mass=1610, wheelbase=2.705),
    {Bus.pt: 'nissan_leaf_2018_generated'},
  )
  # Leaf with ADAS ECU found behind instrument cluster instead of glovebox
  # Currently the only known difference between them is the inverted seatbelt signal.
  NISSAN_LEAF_IC = NISSAN_LEAF.override(car_docs=[])
  NISSAN_ROGUE = NissanPlatformConfig(
    [NissanCarDocs("Nissan Rogue 2018-20")],
    NissanCarSpecs(mass=1610, wheelbase=2.705)
  )
  NISSAN_ALTIMA = NissanPlatformConfig(
    [NissanCarDocs("Nissan Altima 2019-20", car_parts=CarParts.common([CarHarness.nissan_b]))],
    NissanCarSpecs(mass=1492, wheelbase=2.824)
  )


DBC = CAR.create_dbc_map()

# Default diagnostic session
NISSAN_DIAGNOSTIC_REQUEST_KWP = bytes([uds.SERVICE_TYPE.DIAGNOSTIC_SESSION_CONTROL, 0x81])
NISSAN_DIAGNOSTIC_RESPONSE_KWP = bytes([uds.SERVICE_TYPE.DIAGNOSTIC_SESSION_CONTROL + 0x40, 0x81])

# Manufacturer specific
NISSAN_DIAGNOSTIC_REQUEST_KWP_2 = bytes([uds.SERVICE_TYPE.DIAGNOSTIC_SESSION_CONTROL, 0xda])
NISSAN_DIAGNOSTIC_RESPONSE_KWP_2 = bytes([uds.SERVICE_TYPE.DIAGNOSTIC_SESSION_CONTROL + 0x40, 0xda])

NISSAN_VERSION_REQUEST_KWP = b'\x21\x83'
NISSAN_VERSION_RESPONSE_KWP = b'\x61\x83'

NISSAN_RX_OFFSET = 0x20

FW_QUERY_CONFIG = FwQueryConfig(
  requests=[request for bus, logging in ((0, False), (1, True)) for request in [
    Request(
      [NISSAN_DIAGNOSTIC_REQUEST_KWP, NISSAN_VERSION_REQUEST_KWP],
      [NISSAN_DIAGNOSTIC_RESPONSE_KWP, NISSAN_VERSION_RESPONSE_KWP],
      bus=bus,
      logging=logging,
    ),
    Request(
      [NISSAN_DIAGNOSTIC_REQUEST_KWP, NISSAN_VERSION_REQUEST_KWP],
      [NISSAN_DIAGNOSTIC_RESPONSE_KWP, NISSAN_VERSION_RESPONSE_KWP],
      rx_offset=NISSAN_RX_OFFSET,
      bus=bus,
      logging=logging,
    ),
    # Rogue's engine solely responds to this
    Request(
      [NISSAN_DIAGNOSTIC_REQUEST_KWP_2, NISSAN_VERSION_REQUEST_KWP],
      [NISSAN_DIAGNOSTIC_RESPONSE_KWP_2, NISSAN_VERSION_RESPONSE_KWP],
      bus=bus,
      logging=logging,
    ),
    Request(
      [StdQueries.MANUFACTURER_SOFTWARE_VERSION_REQUEST],
      [StdQueries.MANUFACTURER_SOFTWARE_VERSION_RESPONSE],
      rx_offset=NISSAN_RX_OFFSET,
      bus=bus,
      logging=logging,
    ),
  ]],
)
