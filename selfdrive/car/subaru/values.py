from dataclasses import dataclass, field
from enum import Enum, IntFlag

from cereal import car
from panda.python import uds
from openpilot.selfdrive.car import CarSpecs, DbcDict, PlatformConfig, Platforms, dbc_dict
from openpilot.selfdrive.car.docs_definitions import CarFootnote, CarHarness, CarDocs, CarParts, Tool, Column
from openpilot.selfdrive.car.fw_query_definitions import FwQueryConfig, Request, StdQueries, p16

Ecu = car.CarParams.Ecu


class CarControllerParams:
  def __init__(self, CP):
    self.STEER_STEP = 2                # how often we update the steer cmd
    self.STEER_DELTA_UP = 50           # torque increase per refresh, 0.8s to max
    self.STEER_DELTA_DOWN = 70         # torque decrease per refresh
    self.STEER_DRIVER_ALLOWANCE = 60   # allowed driver torque before start limiting
    self.STEER_DRIVER_MULTIPLIER = 50  # weight driver torque heavily
    self.STEER_DRIVER_FACTOR = 1       # from dbc

    if CP.flags & SubaruFlags.GLOBAL_GEN2:
      self.STEER_MAX = 1000
      self.STEER_DELTA_UP = 40
      self.STEER_DELTA_DOWN = 40
    elif CP.carFingerprint == CAR.SUBARU_IMPREZA_2020:
      self.STEER_MAX = 1439
    else:
      self.STEER_MAX = 2047

  THROTTLE_MIN = 808
  THROTTLE_MAX = 3400

  THROTTLE_INACTIVE     = 1818   # corresponds to zero acceleration
  THROTTLE_ENGINE_BRAKE = 808    # while braking, eyesight sets throttle to this, probably for engine braking

  BRAKE_MIN = 0
  BRAKE_MAX = 600                # about -3.5m/s2 from testing

  RPM_MIN = 0
  RPM_MAX = 3600

  RPM_INACTIVE = 600             # a good base rpm for zero acceleration

  THROTTLE_LOOKUP_BP = [0, 2]
  THROTTLE_LOOKUP_V = [THROTTLE_INACTIVE, THROTTLE_MAX]

  RPM_LOOKUP_BP = [0, 2]
  RPM_LOOKUP_V = [RPM_INACTIVE, RPM_MAX]

  BRAKE_LOOKUP_BP = [-3.5, 0]
  BRAKE_LOOKUP_V = [BRAKE_MAX, BRAKE_MIN]


class SubaruFlags(IntFlag):
  # Detected flags
  SEND_INFOTAINMENT = 1
  DISABLE_EYESIGHT = 2

  # Static flags
  GLOBAL_GEN2 = 4

  # Cars that temporarily fault when steering angle rate is greater than some threshold.
  # Appears to be all torque-based cars produced around 2019 - present
  STEER_RATE_LIMITED = 8
  PREGLOBAL = 16
  HYBRID = 32
  LKAS_ANGLE = 64


GLOBAL_ES_ADDR = 0x787
GEN2_ES_BUTTONS_DID = b'\x11\x30'


class CanBus:
  main = 0
  alt = 1
  camera = 2


class Footnote(Enum):
  GLOBAL = CarFootnote(
    "In the non-US market, openpilot requires the car to come equipped with EyeSight with Lane Keep Assistance.",
    Column.PACKAGE)
  EXP_LONG = CarFootnote(
    "Enabling longitudinal control (alpha) will disable all EyeSight functionality, including AEB, LDW, and RAB.",
    Column.LONGITUDINAL)


@dataclass
class SubaruCarDocs(CarDocs):
  package: str = "EyeSight Driver Assistance"
  car_parts: CarParts = field(default_factory=CarParts.common([CarHarness.subaru_a]))
  footnotes: list[Enum] = field(default_factory=lambda: [Footnote.GLOBAL])

  def init_make(self, CP: car.CarParams):
    self.car_parts.parts.extend([Tool.socket_8mm_deep, Tool.pry_tool])

    if CP.experimentalLongitudinalAvailable:
      self.footnotes.append(Footnote.EXP_LONG)


@dataclass
class SubaruPlatformConfig(PlatformConfig):
  dbc_dict: DbcDict = field(default_factory=lambda: dbc_dict('subaru_global_2017_generated', None))

  def init(self):
    if self.flags & SubaruFlags.HYBRID:
      self.dbc_dict = dbc_dict('subaru_global_2020_hybrid_generated', None)


@dataclass
class SubaruGen2PlatformConfig(SubaruPlatformConfig):
  def init(self):
    super().init()
    self.flags |= SubaruFlags.GLOBAL_GEN2
    if not (self.flags & SubaruFlags.LKAS_ANGLE):
      self.flags |= SubaruFlags.STEER_RATE_LIMITED


class CAR(Platforms):
  # Global platform
  SUBARU_ASCENT = SubaruPlatformConfig(
    [SubaruCarDocs("Subaru Ascent 2019-21", "All")],
    CarSpecs(mass=2031, wheelbase=2.89, steerRatio=13.5),
  )
  SUBARU_OUTBACK = SubaruGen2PlatformConfig(
    [SubaruCarDocs("Subaru Outback 2020-22", "All", car_parts=CarParts.common([CarHarness.subaru_b]))],
    CarSpecs(mass=1568, wheelbase=2.67, steerRatio=17),
  )
  SUBARU_LEGACY = SubaruGen2PlatformConfig(
    [SubaruCarDocs("Subaru Legacy 2020-22", "All", car_parts=CarParts.common([CarHarness.subaru_b]))],
    SUBARU_OUTBACK.specs,
  )
  SUBARU_IMPREZA = SubaruPlatformConfig(
    [
      SubaruCarDocs("Subaru Impreza 2017-19"),
      SubaruCarDocs("Subaru Crosstrek 2018-19", video_link="https://youtu.be/Agww7oE1k-s?t=26"),
      SubaruCarDocs("Subaru XV 2018-19", video_link="https://youtu.be/Agww7oE1k-s?t=26"),
    ],
    CarSpecs(mass=1568, wheelbase=2.67, steerRatio=15),
  )
  SUBARU_IMPREZA_2020 = SubaruPlatformConfig(
    [
      SubaruCarDocs("Subaru Impreza 2020-22"),
      SubaruCarDocs("Subaru Crosstrek 2020-23"),
      SubaruCarDocs("Subaru XV 2020-21"),
    ],
    CarSpecs(mass=1480, wheelbase=2.67, steerRatio=17),
    flags=SubaruFlags.STEER_RATE_LIMITED,
  )
  # TODO: is there an XV and Impreza too?
  SUBARU_CROSSTREK_HYBRID = SubaruPlatformConfig(
    [SubaruCarDocs("Subaru Crosstrek Hybrid 2020", car_parts=CarParts.common([CarHarness.subaru_b]))],
    CarSpecs(mass=1668, wheelbase=2.67, steerRatio=17),
    flags=SubaruFlags.HYBRID,
  )
  SUBARU_FORESTER = SubaruPlatformConfig(
    [SubaruCarDocs("Subaru Forester 2019-21", "All")],
    CarSpecs(mass=1568, wheelbase=2.67, steerRatio=17),
    flags=SubaruFlags.STEER_RATE_LIMITED,
  )
  SUBARU_FORESTER_HYBRID = SubaruPlatformConfig(
    [SubaruCarDocs("Subaru Forester Hybrid 2020")],
    SUBARU_FORESTER.specs,
    flags=SubaruFlags.HYBRID,
  )
  # Pre-global
  SUBARU_FORESTER_PREGLOBAL = SubaruPlatformConfig(
    [SubaruCarDocs("Subaru Forester 2017-18")],
    CarSpecs(mass=1568, wheelbase=2.67, steerRatio=20),
    dbc_dict('subaru_forester_2017_generated', None),
    flags=SubaruFlags.PREGLOBAL,
  )
  SUBARU_LEGACY_PREGLOBAL = SubaruPlatformConfig(
    [SubaruCarDocs("Subaru Legacy 2015-18")],
    CarSpecs(mass=1568, wheelbase=2.67, steerRatio=12.5),
    dbc_dict('subaru_outback_2015_generated', None),
    flags=SubaruFlags.PREGLOBAL,
  )
  SUBARU_OUTBACK_PREGLOBAL = SubaruPlatformConfig(
    [SubaruCarDocs("Subaru Outback 2015-17")],
    SUBARU_FORESTER_PREGLOBAL.specs,
    dbc_dict('subaru_outback_2015_generated', None),
    flags=SubaruFlags.PREGLOBAL,
  )
  SUBARU_OUTBACK_PREGLOBAL_2018 = SubaruPlatformConfig(
    [SubaruCarDocs("Subaru Outback 2018-19")],
    SUBARU_FORESTER_PREGLOBAL.specs,
    dbc_dict('subaru_outback_2019_generated', None),
    flags=SubaruFlags.PREGLOBAL,
  )
  # Angle LKAS
  SUBARU_FORESTER_2022 = SubaruPlatformConfig(
    [SubaruCarDocs("Subaru Forester 2022-24", "All", car_parts=CarParts.common([CarHarness.subaru_c]))],
    SUBARU_FORESTER.specs,
    flags=SubaruFlags.LKAS_ANGLE,
  )
  SUBARU_OUTBACK_2023 = SubaruGen2PlatformConfig(
    [SubaruCarDocs("Subaru Outback 2023", "All", car_parts=CarParts.common([CarHarness.subaru_d]))],
    SUBARU_OUTBACK.specs,
    flags=SubaruFlags.LKAS_ANGLE,
  )
  SUBARU_ASCENT_2023 = SubaruGen2PlatformConfig(
    [SubaruCarDocs("Subaru Ascent 2023", "All", car_parts=CarParts.common([CarHarness.subaru_d]))],
    SUBARU_ASCENT.specs,
    flags=SubaruFlags.LKAS_ANGLE,
  )


SUBARU_VERSION_REQUEST = bytes([uds.SERVICE_TYPE.READ_DATA_BY_IDENTIFIER]) + \
  p16(uds.DATA_IDENTIFIER_TYPE.APPLICATION_DATA_IDENTIFICATION)
SUBARU_VERSION_RESPONSE = bytes([uds.SERVICE_TYPE.READ_DATA_BY_IDENTIFIER + 0x40]) + \
  p16(uds.DATA_IDENTIFIER_TYPE.APPLICATION_DATA_IDENTIFICATION)

# The EyeSight ECU takes 10s to respond to SUBARU_VERSION_REQUEST properly,
# log this alternate manufacturer-specific query
SUBARU_ALT_VERSION_REQUEST = bytes([uds.SERVICE_TYPE.READ_DATA_BY_IDENTIFIER]) + \
  p16(0xf100)
SUBARU_ALT_VERSION_RESPONSE = bytes([uds.SERVICE_TYPE.READ_DATA_BY_IDENTIFIER + 0x40]) + \
  p16(0xf100)

FW_QUERY_CONFIG = FwQueryConfig(
  requests=[
    Request(
      [StdQueries.TESTER_PRESENT_REQUEST, SUBARU_VERSION_REQUEST],
      [StdQueries.TESTER_PRESENT_RESPONSE, SUBARU_VERSION_RESPONSE],
      whitelist_ecus=[Ecu.abs, Ecu.eps, Ecu.fwdCamera, Ecu.engine, Ecu.transmission],
      logging=True,
    ),
    # Non-OBD requests
    # Some Eyesight modules fail on TESTER_PRESENT_REQUEST
    # TODO: check if this resolves the fingerprinting issue for the 2023 Ascent and other new Subaru cars
    Request(
      [SUBARU_VERSION_REQUEST],
      [SUBARU_VERSION_RESPONSE],
      whitelist_ecus=[Ecu.fwdCamera],
      bus=0,
    ),
    Request(
      [SUBARU_ALT_VERSION_REQUEST],
      [SUBARU_ALT_VERSION_RESPONSE],
      whitelist_ecus=[Ecu.fwdCamera],
      bus=0,
      logging=True,
    ),
    Request(
      [StdQueries.DEFAULT_DIAGNOSTIC_REQUEST, StdQueries.TESTER_PRESENT_REQUEST, SUBARU_VERSION_REQUEST],
      [StdQueries.DEFAULT_DIAGNOSTIC_RESPONSE, StdQueries.TESTER_PRESENT_RESPONSE, SUBARU_VERSION_RESPONSE],
      whitelist_ecus=[Ecu.fwdCamera],
      bus=0,
      logging=True,
    ),
    Request(
      [StdQueries.TESTER_PRESENT_REQUEST, SUBARU_VERSION_REQUEST],
      [StdQueries.TESTER_PRESENT_RESPONSE, SUBARU_VERSION_RESPONSE],
      whitelist_ecus=[Ecu.abs, Ecu.eps, Ecu.fwdCamera, Ecu.engine, Ecu.transmission],
      bus=0,
    ),
    # GEN2 powertrain bus query
    Request(
      [StdQueries.TESTER_PRESENT_REQUEST, SUBARU_VERSION_REQUEST],
      [StdQueries.TESTER_PRESENT_RESPONSE, SUBARU_VERSION_RESPONSE],
      whitelist_ecus=[Ecu.abs, Ecu.eps, Ecu.fwdCamera, Ecu.engine, Ecu.transmission],
      bus=1,
      obd_multiplexing=False,
    ),
  ],
  # We don't get the EPS from non-OBD queries on GEN2 cars. Note that we still attempt to match when it exists
  non_essential_ecus={
    Ecu.eps: list(CAR.with_flags(SubaruFlags.GLOBAL_GEN2)),
  }
)

DBC = CAR.create_dbc_map()
