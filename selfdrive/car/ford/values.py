import copy
from dataclasses import dataclass, field, replace
from enum import Enum, IntFlag

import panda.python.uds as uds
from cereal import car
from openpilot.selfdrive.car import AngleRateLimit, CarSpecs, dbc_dict, DbcDict, PlatformConfig, Platforms
from openpilot.selfdrive.car.docs_definitions import CarFootnote, CarHarness, CarDocs, CarParts, Column, \
                                                     Device
from openpilot.selfdrive.car.fw_query_definitions import FwQueryConfig, Request, StdQueries, p16

Ecu = car.CarParams.Ecu


class CarControllerParams:
  STEER_STEP = 5        # LateralMotionControl, 20Hz
  LKA_STEP = 3          # Lane_Assist_Data1, 33Hz
  ACC_CONTROL_STEP = 2  # ACCDATA, 50Hz
  LKAS_UI_STEP = 100    # IPMA_Data, 1Hz
  ACC_UI_STEP = 20      # ACCDATA_3, 5Hz
  BUTTONS_STEP = 5      # Steering_Data_FD1, 10Hz, but send twice as fast

  CURVATURE_MAX = 0.02  # Max curvature for steering command, m^-1
  STEER_DRIVER_ALLOWANCE = 1.0  # Driver intervention threshold, Nm

  # Curvature rate limits
  # The curvature signal is limited to 0.003 to 0.009 m^-1/sec by the EPS depending on speed and direction
  # Limit to ~2 m/s^3 up, ~3 m/s^3 down at 75 mph
  # Worst case, the low speed limits will allow 4.3 m/s^3 up, 4.9 m/s^3 down at 75 mph
  ANGLE_RATE_LIMIT_UP = AngleRateLimit(speed_bp=[5, 25], angle_v=[0.0002, 0.0001])
  ANGLE_RATE_LIMIT_DOWN = AngleRateLimit(speed_bp=[5, 25], angle_v=[0.000225, 0.00015])
  CURVATURE_ERROR = 0.002  # ~6 degrees at 10 m/s, ~10 degrees at 35 m/s

  ACCEL_MAX = 2.0               # m/s^2 max acceleration
  ACCEL_MIN = -3.5              # m/s^2 max deceleration
  MIN_GAS = -0.5
  INACTIVE_GAS = -5.0

  def __init__(self, CP):
    pass


class FordFlags(IntFlag):
  # Static flags
  CANFD = 1


class RADAR:
  DELPHI_ESR = 'ford_fusion_2018_adas'
  DELPHI_MRR = 'FORD_CADS'


class Footnote(Enum):
  FOCUS = CarFootnote(
    "Refers only to the Focus Mk4 (C519) available in Europe/China/Taiwan/Australasia, not the Focus Mk3 (C346) in " +
    "North and South America/Southeast Asia.",
    Column.MODEL,
  )


@dataclass
class FordCarDocs(CarDocs):
  package: str = "Co-Pilot360 Assist+"
  hybrid: bool = False
  plug_in_hybrid: bool = False

  def init_make(self, CP: car.CarParams):
    harness = CarHarness.ford_q4 if CP.flags & FordFlags.CANFD else CarHarness.ford_q3
    if CP.carFingerprint in (CAR.BRONCO_SPORT_MK1, CAR.MAVERICK_MK1, CAR.F_150_MK14, CAR.F_150_LIGHTNING_MK1):
      self.car_parts = CarParts([Device.threex_angled_mount, harness])
    else:
      self.car_parts = CarParts([Device.threex, harness])


@dataclass
class FordPlatformConfig(PlatformConfig):
  dbc_dict: DbcDict = field(default_factory=lambda: dbc_dict('ford_lincoln_base_pt', RADAR.DELPHI_MRR))

  def init(self):
    for car_docs in list(self.car_docs):
      if car_docs.hybrid:
        name = f"{car_docs.make} {car_docs.model} Hybrid {car_docs.years}"
        self.car_docs.append(replace(copy.deepcopy(car_docs), name=name))
      if car_docs.plug_in_hybrid:
        name = f"{car_docs.make} {car_docs.model} Plug-in Hybrid {car_docs.years}"
        self.car_docs.append(replace(copy.deepcopy(car_docs), name=name))


@dataclass
class FordCANFDPlatformConfig(FordPlatformConfig):
  dbc_dict: DbcDict = field(default_factory=lambda: dbc_dict('ford_lincoln_base_pt', None))

  def init(self):
    super().init()
    self.flags |= FordFlags.CANFD


class CAR(Platforms):
  BRONCO_SPORT_MK1 = FordPlatformConfig(
    "FORD BRONCO SPORT 1ST GEN",
    [FordCarDocs("Ford Bronco Sport 2021-23")],
    CarSpecs(mass=1625, wheelbase=2.67, steerRatio=17.7),
  )
  ESCAPE_MK4 = FordPlatformConfig(
    "FORD ESCAPE 4TH GEN",
    [
      FordCarDocs("Ford Escape 2020-22", hybrid=True, plug_in_hybrid=True),
      FordCarDocs("Ford Kuga 2020-22", "Adaptive Cruise Control with Lane Centering", hybrid=True, plug_in_hybrid=True),
    ],
    CarSpecs(mass=1750, wheelbase=2.71, steerRatio=16.7),
  )
  EXPLORER_MK6 = FordPlatformConfig(
    "FORD EXPLORER 6TH GEN",
    [
      FordCarDocs("Ford Explorer 2020-23", hybrid=True),  # Hybrid: Limited and Platinum only
      FordCarDocs("Lincoln Aviator 2020-23", "Co-Pilot360 Plus", plug_in_hybrid=True),  # Hybrid: Grand Touring only
    ],
    CarSpecs(mass=2050, wheelbase=3.025, steerRatio=16.8),
  )
  F_150_MK14 = FordCANFDPlatformConfig(
    "FORD F-150 14TH GEN",
    [FordCarDocs("Ford F-150 2022-23", "Co-Pilot360 Active 2.0", hybrid=True)],
    CarSpecs(mass=2000, wheelbase=3.69, steerRatio=17.0),
  )
  F_150_LIGHTNING_MK1 = FordCANFDPlatformConfig(
    "FORD F-150 LIGHTNING 1ST GEN",
    [FordCarDocs("Ford F-150 Lightning 2021-23", "Co-Pilot360 Active 2.0")],
    CarSpecs(mass=2948, wheelbase=3.70, steerRatio=16.9),
  )
  FOCUS_MK4 = FordPlatformConfig(
    "FORD FOCUS 4TH GEN",
    [FordCarDocs("Ford Focus 2018", "Adaptive Cruise Control with Lane Centering", footnotes=[Footnote.FOCUS], hybrid=True)],  # mHEV only
    CarSpecs(mass=1350, wheelbase=2.7, steerRatio=15.0),
  )
  MAVERICK_MK1 = FordPlatformConfig(
    "FORD MAVERICK 1ST GEN",
    [
      FordCarDocs("Ford Maverick 2022", "LARIAT Luxury", hybrid=True),
      FordCarDocs("Ford Maverick 2023-24", "Co-Pilot360 Assist", hybrid=True),
    ],
    CarSpecs(mass=1650, wheelbase=3.076, steerRatio=17.0),
  )
  MUSTANG_MACH_E_MK1 = FordCANFDPlatformConfig(
    "FORD MUSTANG MACH-E 1ST GEN",
    [FordCarDocs("Ford Mustang Mach-E 2021-23", "Co-Pilot360 Active 2.0")],
    CarSpecs(mass=2200, wheelbase=2.984, steerRatio=17.0),  # TODO: check steer ratio
  )


DATA_IDENTIFIER_FORD_ASBUILT = 0xDE00

ASBUILT_BLOCKS: list[tuple[int, list]] = [
  (1, [Ecu.debug, Ecu.fwdCamera, Ecu.eps]),
  (2, [Ecu.abs, Ecu.debug, Ecu.eps]),
  (3, [Ecu.abs, Ecu.debug, Ecu.eps]),
  (4, [Ecu.debug, Ecu.fwdCamera]),
  (5, [Ecu.debug]),
  (6, [Ecu.debug]),
  (7, [Ecu.debug]),
  (8, [Ecu.debug]),
  (9, [Ecu.debug]),
  (16, [Ecu.debug, Ecu.fwdCamera]),
  (18, [Ecu.fwdCamera]),
  (20, [Ecu.fwdCamera]),
  (21, [Ecu.fwdCamera]),
]


def ford_asbuilt_block_request(block_id: int):
  return bytes([uds.SERVICE_TYPE.READ_DATA_BY_IDENTIFIER]) + p16(DATA_IDENTIFIER_FORD_ASBUILT + block_id - 1)


def ford_asbuilt_block_response(block_id: int):
  return bytes([uds.SERVICE_TYPE.READ_DATA_BY_IDENTIFIER + 0x40]) + p16(DATA_IDENTIFIER_FORD_ASBUILT + block_id - 1)


FW_QUERY_CONFIG = FwQueryConfig(
  requests=[
    # CAN and CAN FD queries are combined.
    # FIXME: For CAN FD, ECUs respond with frames larger than 8 bytes on the powertrain bus
    Request(
      [StdQueries.TESTER_PRESENT_REQUEST, StdQueries.MANUFACTURER_SOFTWARE_VERSION_REQUEST],
      [StdQueries.TESTER_PRESENT_RESPONSE, StdQueries.MANUFACTURER_SOFTWARE_VERSION_RESPONSE],
      whitelist_ecus=[Ecu.abs, Ecu.debug, Ecu.engine, Ecu.eps, Ecu.fwdCamera, Ecu.fwdRadar, Ecu.shiftByWire],
      logging=True,
    ),
    Request(
      [StdQueries.TESTER_PRESENT_REQUEST, StdQueries.MANUFACTURER_SOFTWARE_VERSION_REQUEST],
      [StdQueries.TESTER_PRESENT_RESPONSE, StdQueries.MANUFACTURER_SOFTWARE_VERSION_RESPONSE],
      whitelist_ecus=[Ecu.abs, Ecu.debug, Ecu.engine, Ecu.eps, Ecu.fwdCamera, Ecu.fwdRadar, Ecu.shiftByWire],
      bus=0,
      auxiliary=True,
    ),
    *[Request(
      [StdQueries.TESTER_PRESENT_REQUEST, ford_asbuilt_block_request(block_id)],
      [StdQueries.TESTER_PRESENT_RESPONSE, ford_asbuilt_block_response(block_id)],
      whitelist_ecus=ecus,
      bus=0,
      logging=True,
    ) for block_id, ecus in ASBUILT_BLOCKS],
  ],
  extra_ecus=[
    (Ecu.engine, 0x7e0, None),        # Powertrain Control Module
                                      # Note: We are unlikely to get a response from behind the gateway
    (Ecu.shiftByWire, 0x732, None),   # Gear Shift Module
    (Ecu.debug, 0x7d0, None),         # Accessory Protocol Interface Module
  ],
)

DBC = CAR.create_dbc_map()
