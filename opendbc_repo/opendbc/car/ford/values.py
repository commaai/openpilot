import copy
import re
from dataclasses import dataclass, field, replace
from enum import Enum, IntFlag

from panda import uds
from opendbc.car import AngleRateLimit, Bus, CarSpecs, DbcDict, PlatformConfig, Platforms
from opendbc.car.structs import CarParams
from opendbc.car.docs_definitions import CarFootnote, CarHarness, CarDocs, CarParts, Column, \
                                                     Device, SupportType
from opendbc.car.fw_query_definitions import FwQueryConfig, LiveFwVersions, OfflineFwVersions, Request, StdQueries, p16

Ecu = CarParams.Ecu


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
  # Max curvature is limited by the EPS to an equivalent of ~2.0 m/s^2 at all speeds,
  #  however max curvature rate linearly decreases as speed increases:
  #  ~0.009 m^-1/sec at 7 m/s, ~0.002 m^-1/sec at 35 m/s
  # Limit to ~2 m/s^3 up, ~3.3 m/s^3 down at 75 mph and match EPS limit at low speed
  ANGLE_RATE_LIMIT_UP = AngleRateLimit(speed_bp=[5, 25], angle_v=[0.00045, 0.0001])
  ANGLE_RATE_LIMIT_DOWN = AngleRateLimit(speed_bp=[5, 25], angle_v=[0.00045, 0.00015])
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

  def init_make(self, CP: CarParams):
    harness = CarHarness.ford_q4 if CP.flags & FordFlags.CANFD else CarHarness.ford_q3
    if CP.carFingerprint in (CAR.FORD_BRONCO_SPORT_MK1, CAR.FORD_MAVERICK_MK1, CAR.FORD_F_150_MK14, CAR.FORD_F_150_LIGHTNING_MK1):
      self.car_parts = CarParts([Device.threex_angled_mount, harness])
    else:
      self.car_parts = CarParts([Device.threex, harness])


@dataclass
class FordPlatformConfig(PlatformConfig):
  dbc_dict: DbcDict = field(default_factory=lambda: {
    Bus.pt: 'ford_lincoln_base_pt',
    Bus.radar: RADAR.DELPHI_MRR,
  })

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
  dbc_dict: DbcDict = field(default_factory=lambda: {
    Bus.pt: 'ford_lincoln_base_pt',
  })

  def init(self):
    super().init()
    self.flags |= FordFlags.CANFD


class CAR(Platforms):
  FORD_BRONCO_SPORT_MK1 = FordPlatformConfig(
    [FordCarDocs("Ford Bronco Sport 2021-24")],
    CarSpecs(mass=1625, wheelbase=2.67, steerRatio=17.7),
  )
  FORD_ESCAPE_MK4 = FordPlatformConfig(
    [
      FordCarDocs("Ford Escape 2020-22", hybrid=True, plug_in_hybrid=True),
      FordCarDocs("Ford Kuga 2020-22", "Adaptive Cruise Control with Lane Centering", hybrid=True, plug_in_hybrid=True),
    ],
    CarSpecs(mass=1750, wheelbase=2.71, steerRatio=16.7),
  )
  FORD_EXPLORER_MK6 = FordPlatformConfig(
    [
      FordCarDocs("Ford Explorer 2020-24", hybrid=True),  # Hybrid: Limited and Platinum only
      FordCarDocs("Lincoln Aviator 2020-24", "Co-Pilot360 Plus", plug_in_hybrid=True),  # Hybrid: Grand Touring only
    ],
    CarSpecs(mass=2050, wheelbase=3.025, steerRatio=16.8),
  )
  FORD_F_150_MK14 = FordCANFDPlatformConfig(
    [FordCarDocs("Ford F-150 2022-23", "Co-Pilot360 Active 2.0", hybrid=True, support_type=SupportType.REVIEW)],
    CarSpecs(mass=2000, wheelbase=3.69, steerRatio=17.0),
  )
  FORD_F_150_LIGHTNING_MK1 = FordCANFDPlatformConfig(
    [FordCarDocs("Ford F-150 Lightning 2021-23", "Co-Pilot360 Active 2.0", support_type=SupportType.REVIEW)],
    CarSpecs(mass=2948, wheelbase=3.70, steerRatio=16.9),
  )
  FORD_FOCUS_MK4 = FordPlatformConfig(
    [FordCarDocs("Ford Focus 2018", "Adaptive Cruise Control with Lane Centering", footnotes=[Footnote.FOCUS], hybrid=True)],  # mHEV only
    CarSpecs(mass=1350, wheelbase=2.7, steerRatio=15.0),
  )
  FORD_MAVERICK_MK1 = FordPlatformConfig(
    [
      FordCarDocs("Ford Maverick 2022", "LARIAT Luxury", hybrid=True),
      FordCarDocs("Ford Maverick 2023-24", "Co-Pilot360 Assist", hybrid=True),
    ],
    CarSpecs(mass=1650, wheelbase=3.076, steerRatio=17.0),
  )
  FORD_MUSTANG_MACH_E_MK1 = FordCANFDPlatformConfig(
    [FordCarDocs("Ford Mustang Mach-E 2021-23", "Co-Pilot360 Active 2.0", support_type=SupportType.REVIEW)],
    CarSpecs(mass=2200, wheelbase=2.984, steerRatio=17.0),  # TODO: check steer ratio
  )
  FORD_RANGER_MK2 = FordCANFDPlatformConfig(
    [FordCarDocs("Ford Ranger 2024", "Adaptive Cruise Control with Lane Centering", support_type=SupportType.REVIEW)],
    CarSpecs(mass=2000, wheelbase=3.27, steerRatio=17.0),
  )


# FW response contains a combined software and part number
# A-Z except no I, O or W
# e.g. NZ6A-14C204-AAA
#      1222-333333-444
# 1 = Model year hint (approximates model year/generation)
# 2 = Platform hint
# 3 = Part number
# 4 = Software version
FW_ALPHABET = b'A-HJ-NP-VX-Z'
FW_PATTERN = re.compile(b'^(?P<model_year_hint>[' + FW_ALPHABET + b'])' +
                        b'(?P<platform_hint>[0-9' + FW_ALPHABET + b']{3})-' +
                        b'(?P<part_number>[0-9' + FW_ALPHABET + b']{5,6})-' +
                        b'(?P<software_revision>[' + FW_ALPHABET + b']{2,})\x00*$')


def get_platform_codes(fw_versions: list[bytes] | set[bytes]) -> set[tuple[bytes, bytes]]:
  codes = set()
  for fw in fw_versions:
    match = FW_PATTERN.match(fw)
    if match is not None:
      codes.add((match.group('platform_hint'), match.group('model_year_hint')))

  return codes


def match_fw_to_car_fuzzy(live_fw_versions: LiveFwVersions, vin: str, offline_fw_versions: OfflineFwVersions) -> set[str]:
  candidates: set[str] = set()

  for candidate, fws in offline_fw_versions.items():
    # Keep track of ECUs which pass all checks (platform hint, within model year hint range)
    valid_found_ecus = set()
    valid_expected_ecus = {ecu[1:] for ecu in fws if ecu[0] in PLATFORM_CODE_ECUS}
    for ecu, expected_versions in fws.items():
      addr = ecu[1:]
      # Only check ECUs expected to have platform codes
      if ecu[0] not in PLATFORM_CODE_ECUS:
        continue

      # Expected platform codes & model year hints
      codes = get_platform_codes(expected_versions)
      expected_platform_codes = {code for code, _ in codes}
      expected_model_year_hints = {model_year_hint for _, model_year_hint in codes}

      # Found platform codes & model year hints
      codes = get_platform_codes(live_fw_versions.get(addr, set()))
      found_platform_codes = {code for code, _ in codes}
      found_model_year_hints = {model_year_hint for _, model_year_hint in codes}

      # Check platform code matches for any found versions
      if not any(found_platform_code in expected_platform_codes for found_platform_code in found_platform_codes):
        break

      # Check any model year hint within range in the database. Note that some models have more than one
      # platform code per ECU which we don't consider as separate ranges
      if not any(min(expected_model_year_hints) <= found_model_year_hint <= max(expected_model_year_hints) for
                 found_model_year_hint in found_model_year_hints):
        break

      valid_found_ecus.add(addr)

    # If all live ECUs pass all checks for candidate, add it as a match
    if valid_expected_ecus.issubset(valid_found_ecus):
      candidates.add(candidate)

  return candidates


# All of these ECUs must be present and are expected to have platform codes we can match
PLATFORM_CODE_ECUS = (Ecu.abs, Ecu.fwdCamera, Ecu.fwdRadar, Ecu.eps)

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
  # Custom fuzzy fingerprinting function using platform and model year hints
  match_fw_to_car_fuzzy=match_fw_to_car_fuzzy,
)

DBC = CAR.create_dbc_map()
