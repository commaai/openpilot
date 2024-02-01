import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, IntFlag, StrEnum
from typing import Dict, List, Set, Union

from cereal import car
from openpilot.common.conversions import Conversions as CV
from openpilot.selfdrive.car import AngleRateLimit, dbc_dict
from openpilot.selfdrive.car.docs_definitions import CarFootnote, CarInfo, Column, CarParts, CarHarness
from openpilot.selfdrive.car.fw_query_definitions import FwQueryConfig, Request, StdQueries

Ecu = car.CarParams.Ecu
MIN_ACC_SPEED = 19. * CV.MPH_TO_MS
PEDAL_TRANSITION = 10. * CV.MPH_TO_MS


class CarControllerParams:
  ACCEL_MAX = 1.5  # m/s2, lower than allowed 2.0 m/s2 for tuning reasons
  ACCEL_MIN = -3.5  # m/s2

  STEER_STEP = 1
  STEER_MAX = 1500
  STEER_ERROR_MAX = 350     # max delta between torque cmd and torque motor

  # Lane Tracing Assist (LTA) control limits
  # Assuming a steering ratio of 13.7:
  # Limit to ~2.0 m/s^3 up (7.5 deg/s), ~3.5 m/s^3 down (13 deg/s) at 75 mph
  # Worst case, the low speed limits will allow ~4.0 m/s^3 up (15 deg/s) and ~4.9 m/s^3 down (18 deg/s) at 75 mph,
  # however the EPS has its own internal limits at all speeds which are less than that:
  # Observed internal torque rate limit on TSS 2.5 Camry and RAV4 is ~1500 units/sec up and down when using LTA
  ANGLE_RATE_LIMIT_UP = AngleRateLimit(speed_bp=[5, 25], angle_v=[0.3, 0.15])
  ANGLE_RATE_LIMIT_DOWN = AngleRateLimit(speed_bp=[5, 25], angle_v=[0.36, 0.26])

  def __init__(self, CP):
    if CP.lateralTuning.which == 'torque':
      self.STEER_DELTA_UP = 15       # 1.0s time to peak torque
      self.STEER_DELTA_DOWN = 25     # always lower than 45 otherwise the Rav4 faults (Prius seems ok with 50)
    else:
      self.STEER_DELTA_UP = 10       # 1.5s time to peak torque
      self.STEER_DELTA_DOWN = 25     # always lower than 45 otherwise the Rav4 faults (Prius seems ok with 50)


class ToyotaFlags(IntFlag):
  HYBRID = 1
  SMART_DSU = 2
  DISABLE_RADAR = 4


class CAR(StrEnum):
  # Toyota
  ALPHARD_TSS2 = "TOYOTA ALPHARD 2020"
  AVALON = "TOYOTA AVALON 2016"
  AVALON_2019 = "TOYOTA AVALON 2019"
  AVALON_TSS2 = "TOYOTA AVALON 2022"  # TSS 2.5
  CAMRY = "TOYOTA CAMRY 2018"
  CAMRY_TSS2 = "TOYOTA CAMRY 2021"  # TSS 2.5
  CHR = "TOYOTA C-HR 2018"
  CHR_TSS2 = "TOYOTA C-HR 2021"
  COROLLA = "TOYOTA COROLLA 2017"
  # LSS2 Lexus UX Hybrid is same as a TSS2 Corolla Hybrid
  COROLLA_TSS2 = "TOYOTA COROLLA TSS2 2019"
  HIGHLANDER = "TOYOTA HIGHLANDER 2017"
  HIGHLANDER_TSS2 = "TOYOTA HIGHLANDER 2020"
  PRIUS = "TOYOTA PRIUS 2017"
  PRIUS_V = "TOYOTA PRIUS v 2017"
  PRIUS_TSS2 = "TOYOTA PRIUS TSS2 2021"
  RAV4 = "TOYOTA RAV4 2017"
  RAV4H = "TOYOTA RAV4 HYBRID 2017"
  RAV4_TSS2 = "TOYOTA RAV4 2019"
  RAV4_TSS2_2022 = "TOYOTA RAV4 2022"
  RAV4_TSS2_2023 = "TOYOTA RAV4 2023"
  MIRAI = "TOYOTA MIRAI 2021"  # TSS 2.5
  SIENNA = "TOYOTA SIENNA 2018"

  # Lexus
  LEXUS_CTH = "LEXUS CT HYBRID 2018"
  LEXUS_ES = "LEXUS ES 2018"
  LEXUS_ES_TSS2 = "LEXUS ES 2019"
  LEXUS_IS = "LEXUS IS 2018"
  LEXUS_IS_TSS2 = "LEXUS IS 2023"
  LEXUS_NX = "LEXUS NX 2018"
  LEXUS_NX_TSS2 = "LEXUS NX 2020"
  LEXUS_RC = "LEXUS RC 2020"
  LEXUS_RX = "LEXUS RX 2016"
  LEXUS_RX_TSS2 = "LEXUS RX 2020"
  LEXUS_GS_F = "LEXUS GS F 2016"


class Footnote(Enum):
  CAMRY = CarFootnote(
    "openpilot operates above 28mph for Camry 4CYL L, 4CYL LE and 4CYL SE which don't have Full-Speed Range Dynamic Radar Cruise Control.",
    Column.FSR_LONGITUDINAL)


@dataclass
class ToyotaCarInfo(CarInfo):
  package: str = "All"
  car_parts: CarParts = field(default_factory=CarParts.common([CarHarness.toyota_a]))


CAR_INFO: Dict[str, Union[ToyotaCarInfo, List[ToyotaCarInfo]]] = {
  # Toyota
  CAR.ALPHARD_TSS2: [
    ToyotaCarInfo("Toyota Alphard 2019-20"),
    ToyotaCarInfo("Toyota Alphard Hybrid 2021"),
  ],
  CAR.AVALON: [
    ToyotaCarInfo("Toyota Avalon 2016", "Toyota Safety Sense P"),
    ToyotaCarInfo("Toyota Avalon 2017-18"),
  ],
  CAR.AVALON_2019: [
    ToyotaCarInfo("Toyota Avalon 2019-21"),
    ToyotaCarInfo("Toyota Avalon Hybrid 2019-21"),
  ],
  CAR.AVALON_TSS2: [
    ToyotaCarInfo("Toyota Avalon 2022"),
    ToyotaCarInfo("Toyota Avalon Hybrid 2022"),
  ],
  CAR.CAMRY: [
    ToyotaCarInfo("Toyota Camry 2018-20", video_link="https://www.youtube.com/watch?v=fkcjviZY9CM", footnotes=[Footnote.CAMRY]),
    ToyotaCarInfo("Toyota Camry Hybrid 2018-20", video_link="https://www.youtube.com/watch?v=Q2DYY0AWKgk"),
  ],
  CAR.CAMRY_TSS2: [
    ToyotaCarInfo("Toyota Camry 2021-24", footnotes=[Footnote.CAMRY]),
    ToyotaCarInfo("Toyota Camry Hybrid 2021-24"),
  ],
  CAR.CHR: [
    ToyotaCarInfo("Toyota C-HR 2017-20"),
    ToyotaCarInfo("Toyota C-HR Hybrid 2017-20"),
  ],
  CAR.CHR_TSS2: [
    ToyotaCarInfo("Toyota C-HR 2021"),
    ToyotaCarInfo("Toyota C-HR Hybrid 2021-22"),
  ],
  CAR.COROLLA: ToyotaCarInfo("Toyota Corolla 2017-19"),
  CAR.COROLLA_TSS2: [
    ToyotaCarInfo("Toyota Corolla 2020-22", video_link="https://www.youtube.com/watch?v=_66pXk0CBYA"),
    ToyotaCarInfo("Toyota Corolla Cross (Non-US only) 2020-23", min_enable_speed=7.5),
    ToyotaCarInfo("Toyota Corolla Hatchback 2019-22", video_link="https://www.youtube.com/watch?v=_66pXk0CBYA"),
    # Hybrid platforms
    ToyotaCarInfo("Toyota Corolla Hybrid 2020-22"),
    ToyotaCarInfo("Toyota Corolla Hybrid (Non-US only) 2020-23", min_enable_speed=7.5),
    ToyotaCarInfo("Toyota Corolla Cross Hybrid (Non-US only) 2020-22", min_enable_speed=7.5),
    ToyotaCarInfo("Lexus UX Hybrid 2019-23"),
  ],
  CAR.HIGHLANDER: [
    ToyotaCarInfo("Toyota Highlander 2017-19", video_link="https://www.youtube.com/watch?v=0wS0wXSLzoo"),
    ToyotaCarInfo("Toyota Highlander Hybrid 2017-19"),
  ],
  CAR.HIGHLANDER_TSS2: [
    ToyotaCarInfo("Toyota Highlander 2020-23"),
    ToyotaCarInfo("Toyota Highlander Hybrid 2020-23"),
  ],
  CAR.PRIUS: [
    ToyotaCarInfo("Toyota Prius 2016", "Toyota Safety Sense P", video_link="https://www.youtube.com/watch?v=8zopPJI8XQ0"),
    ToyotaCarInfo("Toyota Prius 2017-20", video_link="https://www.youtube.com/watch?v=8zopPJI8XQ0"),
    ToyotaCarInfo("Toyota Prius Prime 2017-20", video_link="https://www.youtube.com/watch?v=8zopPJI8XQ0"),
  ],
  CAR.PRIUS_V: ToyotaCarInfo("Toyota Prius v 2017", "Toyota Safety Sense P", min_enable_speed=MIN_ACC_SPEED),
  CAR.PRIUS_TSS2: [
    ToyotaCarInfo("Toyota Prius 2021-22", video_link="https://www.youtube.com/watch?v=J58TvCpUd4U"),
    ToyotaCarInfo("Toyota Prius Prime 2021-22", video_link="https://www.youtube.com/watch?v=J58TvCpUd4U"),
  ],
  CAR.RAV4: [
    ToyotaCarInfo("Toyota RAV4 2016", "Toyota Safety Sense P"),
    ToyotaCarInfo("Toyota RAV4 2017-18")
  ],
  CAR.RAV4H: [
    ToyotaCarInfo("Toyota RAV4 Hybrid 2016", "Toyota Safety Sense P", video_link="https://youtu.be/LhT5VzJVfNI?t=26"),
    ToyotaCarInfo("Toyota RAV4 Hybrid 2017-18", video_link="https://youtu.be/LhT5VzJVfNI?t=26")
  ],
  CAR.RAV4_TSS2: [
    ToyotaCarInfo("Toyota RAV4 2019-21", video_link="https://www.youtube.com/watch?v=wJxjDd42gGA"),
    ToyotaCarInfo("Toyota RAV4 Hybrid 2019-21"),
  ],
  CAR.RAV4_TSS2_2022: [
    ToyotaCarInfo("Toyota RAV4 2022"),
    ToyotaCarInfo("Toyota RAV4 Hybrid 2022", video_link="https://youtu.be/U0nH9cnrFB0"),
  ],
  CAR.RAV4_TSS2_2023: [
    ToyotaCarInfo("Toyota RAV4 2023-24"),
    ToyotaCarInfo("Toyota RAV4 Hybrid 2023-24"),
  ],
  CAR.MIRAI: ToyotaCarInfo("Toyota Mirai 2021"),
  CAR.SIENNA: ToyotaCarInfo("Toyota Sienna 2018-20", video_link="https://www.youtube.com/watch?v=q1UPOo4Sh68", min_enable_speed=MIN_ACC_SPEED),

  # Lexus
  CAR.LEXUS_CTH: ToyotaCarInfo("Lexus CT Hybrid 2017-18", "Lexus Safety System+"),
  CAR.LEXUS_ES: [
    ToyotaCarInfo("Lexus ES 2017-18"),
    ToyotaCarInfo("Lexus ES Hybrid 2017-18"),
  ],
  CAR.LEXUS_ES_TSS2: [
    ToyotaCarInfo("Lexus ES 2019-24"),
    ToyotaCarInfo("Lexus ES Hybrid 2019-24", video_link="https://youtu.be/BZ29osRVJeg?t=12"),
  ],
  CAR.LEXUS_IS: ToyotaCarInfo("Lexus IS 2017-19"),
  CAR.LEXUS_IS_TSS2: ToyotaCarInfo("Lexus IS 2022-23"),
  CAR.LEXUS_GS_F: ToyotaCarInfo("Lexus GS F 2016"),
  CAR.LEXUS_NX: [
    ToyotaCarInfo("Lexus NX 2018-19"),
    ToyotaCarInfo("Lexus NX Hybrid 2018-19"),
  ],
  CAR.LEXUS_NX_TSS2: [
    ToyotaCarInfo("Lexus NX 2020-21"),
    ToyotaCarInfo("Lexus NX Hybrid 2020-21"),
  ],
  CAR.LEXUS_RC: ToyotaCarInfo("Lexus RC 2018-20"),
  CAR.LEXUS_RX: [
    ToyotaCarInfo("Lexus RX 2016", "Lexus Safety System+"),
    ToyotaCarInfo("Lexus RX 2017-19"),
    # Hybrid platforms
    ToyotaCarInfo("Lexus RX Hybrid 2016", "Lexus Safety System+"),
    ToyotaCarInfo("Lexus RX Hybrid 2017-19"),
  ],
  CAR.LEXUS_RX_TSS2: [
    ToyotaCarInfo("Lexus RX 2020-22"),
    ToyotaCarInfo("Lexus RX Hybrid 2020-22"),
  ],
}

# (addr, cars, bus, 1/freq*100, vl)
STATIC_DSU_MSGS = [
  (0x128, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RX, CAR.LEXUS_NX, CAR.RAV4, CAR.COROLLA, CAR.AVALON), 1,   3, b'\xf4\x01\x90\x83\x00\x37'),
  (0x128, (CAR.HIGHLANDER, CAR.SIENNA, CAR.LEXUS_CTH, CAR.LEXUS_ES), 1,   3, b'\x03\x00\x20\x00\x00\x52'),
  (0x141, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RX, CAR.LEXUS_NX, CAR.RAV4, CAR.COROLLA, CAR.HIGHLANDER, CAR.AVALON,
           CAR.SIENNA, CAR.LEXUS_CTH, CAR.LEXUS_ES, CAR.PRIUS_V), 1,   2, b'\x00\x00\x00\x46'),
  (0x160, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RX, CAR.LEXUS_NX, CAR.RAV4, CAR.COROLLA, CAR.HIGHLANDER, CAR.AVALON,
           CAR.SIENNA, CAR.LEXUS_CTH, CAR.LEXUS_ES, CAR.PRIUS_V), 1,   7, b'\x00\x00\x08\x12\x01\x31\x9c\x51'),
  (0x161, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RX, CAR.LEXUS_NX, CAR.RAV4, CAR.COROLLA, CAR.AVALON, CAR.PRIUS_V),
                                                                                               1,   7, b'\x00\x1e\x00\x00\x00\x80\x07'),
  (0X161, (CAR.HIGHLANDER, CAR.SIENNA, CAR.LEXUS_CTH, CAR.LEXUS_ES), 1,  7, b'\x00\x1e\x00\xd4\x00\x00\x5b'),
  (0x283, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RX, CAR.LEXUS_NX, CAR.RAV4, CAR.COROLLA, CAR.HIGHLANDER, CAR.AVALON,
           CAR.SIENNA, CAR.LEXUS_CTH, CAR.LEXUS_ES, CAR.PRIUS_V), 0,   3, b'\x00\x00\x00\x00\x00\x00\x8c'),
  (0x2E6, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RX), 0,   3, b'\xff\xf8\x00\x08\x7f\xe0\x00\x4e'),
  (0x2E7, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RX), 0,   3, b'\xa8\x9c\x31\x9c\x00\x00\x00\x02'),
  (0x33E, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RX), 0,  20, b'\x0f\xff\x26\x40\x00\x1f\x00'),
  (0x344, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RX, CAR.LEXUS_NX, CAR.RAV4, CAR.COROLLA, CAR.HIGHLANDER, CAR.AVALON,
           CAR.SIENNA, CAR.LEXUS_CTH, CAR.LEXUS_ES, CAR.PRIUS_V), 0,   5, b'\x00\x00\x01\x00\x00\x00\x00\x50'),
  (0x365, (CAR.PRIUS, CAR.LEXUS_NX, CAR.HIGHLANDER), 0,  20, b'\x00\x00\x00\x80\x03\x00\x08'),
  (0x365, (CAR.RAV4, CAR.RAV4H, CAR.COROLLA, CAR.AVALON, CAR.SIENNA, CAR.LEXUS_CTH, CAR.LEXUS_ES, CAR.LEXUS_RX,
           CAR.PRIUS_V), 0,  20, b'\x00\x00\x00\x80\xfc\x00\x08'),
  (0x366, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RX, CAR.LEXUS_NX, CAR.HIGHLANDER), 0,  20, b'\x00\x00\x4d\x82\x40\x02\x00'),
  (0x366, (CAR.RAV4, CAR.COROLLA, CAR.AVALON, CAR.SIENNA, CAR.LEXUS_CTH, CAR.LEXUS_ES, CAR.PRIUS_V),
          0,  20, b'\x00\x72\x07\xff\x09\xfe\x00'),
  (0x470, (CAR.PRIUS, CAR.LEXUS_RX), 1, 100, b'\x00\x00\x02\x7a'),
  (0x470, (CAR.HIGHLANDER, CAR.RAV4H, CAR.SIENNA, CAR.LEXUS_CTH, CAR.LEXUS_ES, CAR.PRIUS_V), 1,  100, b'\x00\x00\x01\x79'),
  (0x4CB, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RX, CAR.LEXUS_NX, CAR.RAV4, CAR.COROLLA, CAR.HIGHLANDER, CAR.AVALON,
           CAR.SIENNA, CAR.LEXUS_CTH, CAR.LEXUS_ES, CAR.PRIUS_V), 0, 100, b'\x0c\x00\x00\x00\x00\x00\x00\x00'),
]


def get_platform_codes(fw_versions: List[bytes]) -> Dict[bytes, Set[bytes]]:
  # Returns sub versions in a dict so comparisons can be made within part-platform-major_version combos
  codes = defaultdict(set)  # Optional[part]-platform-major_version: set of sub_version
  for fw in fw_versions:
    # FW versions returned from UDS queries can return multiple fields/chunks of data (different ECU calibrations, different data?)
    #  and are prefixed with a byte that describes how many chunks of data there are.
    # But FW returned from KWP requires querying of each sub-data id and does not have a length prefix.

    length_code = 1
    length_code_match = FW_LEN_CODE.search(fw)
    if length_code_match is not None:
      length_code = length_code_match.group()[0]
      fw = fw[1:]

    # fw length should be multiple of 16 bytes (per chunk, even if no length code), skip parsing if unexpected length
    if length_code * FW_CHUNK_LEN != len(fw):
      continue

    chunks = [fw[FW_CHUNK_LEN * i:FW_CHUNK_LEN * i + FW_CHUNK_LEN].strip(b'\x00 ') for i in range(length_code)]

    # only first is considered for now since second is commonly shared (TODO: understand that)
    first_chunk = chunks[0]
    if len(first_chunk) == 8:
      # TODO: no part number, but some short chunks have it in subsequent chunks
      fw_match = SHORT_FW_PATTERN.search(first_chunk)
      if fw_match is not None:
        platform, major_version, sub_version = fw_match.groups()
        codes[b'-'.join((platform, major_version))].add(sub_version)

    elif len(first_chunk) == 10:
      fw_match = MEDIUM_FW_PATTERN.search(first_chunk)
      if fw_match is not None:
        part, platform, major_version, sub_version = fw_match.groups()
        codes[b'-'.join((part, platform, major_version))].add(sub_version)

    elif len(first_chunk) == 12:
      fw_match = LONG_FW_PATTERN.search(first_chunk)
      if fw_match is not None:
        part, platform, major_version, sub_version = fw_match.groups()
        codes[b'-'.join((part, platform, major_version))].add(sub_version)

  return dict(codes)


def match_fw_to_car_fuzzy(live_fw_versions, offline_fw_versions) -> Set[str]:
  candidates = set()

  for candidate, fws in offline_fw_versions.items():
    # Keep track of ECUs which pass all checks (platform codes, within sub-version range)
    valid_found_ecus = set()
    valid_expected_ecus = {ecu[1:] for ecu in fws if ecu[0] in PLATFORM_CODE_ECUS}
    for ecu, expected_versions in fws.items():
      addr = ecu[1:]
      # Only check ECUs expected to have platform codes
      if ecu[0] not in PLATFORM_CODE_ECUS:
        continue

      # Expected platform codes & versions
      expected_platform_codes = get_platform_codes(expected_versions)

      # Found platform codes & versions
      found_platform_codes = get_platform_codes(live_fw_versions.get(addr, set()))

      # Check part number + platform code + major version matches for any found versions
      # Platform codes and major versions change for different physical parts, generation, API, etc.
      # Sub-versions are incremented for minor recalls, do not need to be checked.
      if not any(found_platform_code in expected_platform_codes for found_platform_code in found_platform_codes):
        break

      valid_found_ecus.add(addr)

    # If all live ECUs pass all checks for candidate, add it as a match
    if valid_expected_ecus.issubset(valid_found_ecus):
      candidates.add(candidate)

  return {str(c) for c in (candidates - FUZZY_EXCLUDED_PLATFORMS)}


# Regex patterns for parsing more general platform-specific identifiers from FW versions.
# - Part number: Toyota part number (usually last character needs to be ignored to find a match).
#    Each ECU address has just one part number.
# - Platform: usually multiple codes per an openpilot platform, however this is the least variable and
#    is usually shared across ECUs and model years signifying this describes something about the specific platform.
#    This describes more generational changes (TSS-P vs TSS2), or manufacture region.
# - Major version: second least variable part of the FW version. Seen splitting cars by model year/API such as
#    RAV4 2022/2023 and Avalon. Used to differentiate cars where API has changed slightly, but is not a generational change.
#    It is important to note that these aren't always consecutive, for example:
#    Avalon 2016-18's fwdCamera has these major versions: 01, 03 while 2019 has: 02
# - Sub version: exclusive to major version, but shared with other cars. Should only be used for further filtering.
#    Seen bumped in TSB FW updates, and describes other minor differences.
SHORT_FW_PATTERN = re.compile(b'[A-Z0-9](?P<platform>[A-Z0-9]{2})(?P<major_version>[A-Z0-9]{2})(?P<sub_version>[A-Z0-9]{3})')
MEDIUM_FW_PATTERN = re.compile(b'(?P<part>[A-Z0-9]{5})(?P<platform>[A-Z0-9]{2})(?P<major_version>[A-Z0-9]{1})(?P<sub_version>[A-Z0-9]{2})')
LONG_FW_PATTERN = re.compile(b'(?P<part>[A-Z0-9]{5})(?P<platform>[A-Z0-9]{2})(?P<major_version>[A-Z0-9]{2})(?P<sub_version>[A-Z0-9]{3})')
FW_LEN_CODE = re.compile(b'^[\x01-\x03]')  # highest seen is 3 chunks, 16 bytes each
FW_CHUNK_LEN = 16

# List of ECUs that are most unique across openpilot platforms
# - fwdCamera: describes actual features related to ADAS. For example, on the Avalon it describes
#    when TSS-P became standard, whether the car supports stop and go, and whether it's TSS2.
#    On the RAV4, it describes the move to the radar doing ACC, and the use of LTA for lane keeping.
#    Note that the platform codes & major versions do not describe features in plain text, only with
#    matching against other seen FW versions in the database they can describe features.
# - fwdRadar: sanity check against fwdCamera, commonly shares a platform code.
#    For example the RAV4 2022's new radar architecture is shown for both with platform code.
# - abs: differentiates hybrid/ICE on most cars (Corolla TSS2 is an exception, not used due to hybrid platform combination)
# - eps: describes lateral API changes for the EPS, such as using LTA for lane keeping and rejecting LKA messages
PLATFORM_CODE_ECUS = (Ecu.fwdCamera, Ecu.fwdRadar, Ecu.eps)

# These platforms have at least one platform code for all ECUs shared with another platform.
FUZZY_EXCLUDED_PLATFORMS: set[CAR] = set()

# Some ECUs that use KWP2000 have their FW versions on non-standard data identifiers.
# Toyota diagnostic software first gets the supported data ids, then queries them one by one.
# For example, sends: 0x1a8800, receives: 0x1a8800010203, queries: 0x1a8801, 0x1a8802, 0x1a8803
TOYOTA_VERSION_REQUEST_KWP = b'\x1a\x88\x01'
TOYOTA_VERSION_RESPONSE_KWP = b'\x5a\x88\x01'

FW_QUERY_CONFIG = FwQueryConfig(
  # TODO: look at data to whitelist new ECUs effectively
  requests=[
    Request(
      [StdQueries.SHORT_TESTER_PRESENT_REQUEST, TOYOTA_VERSION_REQUEST_KWP],
      [StdQueries.SHORT_TESTER_PRESENT_RESPONSE, TOYOTA_VERSION_RESPONSE_KWP],
      whitelist_ecus=[Ecu.fwdCamera, Ecu.fwdRadar, Ecu.dsu, Ecu.abs, Ecu.eps, Ecu.epb, Ecu.telematics,
                      Ecu.srs, Ecu.combinationMeter, Ecu.transmission, Ecu.gateway, Ecu.hvac],
      bus=0,
    ),
    Request(
      [StdQueries.SHORT_TESTER_PRESENT_REQUEST, StdQueries.OBD_VERSION_REQUEST],
      [StdQueries.SHORT_TESTER_PRESENT_RESPONSE, StdQueries.OBD_VERSION_RESPONSE],
      whitelist_ecus=[Ecu.engine, Ecu.epb, Ecu.telematics, Ecu.hybrid, Ecu.srs, Ecu.combinationMeter, Ecu.transmission,
                      Ecu.gateway, Ecu.hvac],
      bus=0,
    ),
    Request(
      [StdQueries.TESTER_PRESENT_REQUEST, StdQueries.DEFAULT_DIAGNOSTIC_REQUEST, StdQueries.EXTENDED_DIAGNOSTIC_REQUEST, StdQueries.UDS_VERSION_REQUEST],
      [StdQueries.TESTER_PRESENT_RESPONSE, StdQueries.DEFAULT_DIAGNOSTIC_RESPONSE, StdQueries.EXTENDED_DIAGNOSTIC_RESPONSE, StdQueries.UDS_VERSION_RESPONSE],
      whitelist_ecus=[Ecu.engine, Ecu.fwdRadar, Ecu.fwdCamera, Ecu.abs, Ecu.eps, Ecu.epb, Ecu.telematics,
                      Ecu.hybrid, Ecu.srs, Ecu.combinationMeter, Ecu.transmission, Ecu.gateway, Ecu.hvac],
      bus=0,
    ),
  ],
  non_essential_ecus={
    # FIXME: On some models, abs can sometimes be missing
    Ecu.abs: [CAR.RAV4, CAR.COROLLA, CAR.HIGHLANDER, CAR.SIENNA, CAR.LEXUS_IS, CAR.ALPHARD_TSS2],
    # On some models, the engine can show on two different addresses
    Ecu.engine: [CAR.HIGHLANDER, CAR.CAMRY, CAR.COROLLA_TSS2, CAR.CHR, CAR.CHR_TSS2, CAR.LEXUS_IS,
                 CAR.LEXUS_RC, CAR.LEXUS_NX, CAR.LEXUS_NX_TSS2, CAR.LEXUS_RX, CAR.LEXUS_RX_TSS2],
  },
  extra_ecus=[
    # All known ECUs on a late-model Toyota vehicle not queried here:
    # Responds to UDS:
    # - HV Battery (0x713, 0x747)
    # - Motor Generator (0x716, 0x724)
    # - 2nd ABS "Brake/EPB" (0x730)
    # Responds to KWP (0x1a8801):
    # - Steering Angle Sensor (0x7b3)
    # - EPS/EMPS (0x7a0, 0x7a1)
    # Responds to KWP (0x1a8881):
    # - Body Control Module ((0x750, 0x40))

    # Hybrid control computer can be on 0x7e2 (KWP) or 0x7d2 (UDS) depending on platform
    (Ecu.hybrid, 0x7e2, None),  # Hybrid Control Assembly & Computer
    # TODO: if these duplicate ECUs always exist together, remove one
    (Ecu.srs, 0x780, None),     # SRS Airbag
    (Ecu.srs, 0x784, None),     # SRS Airbag 2
    # Likely only exists on cars where EPB isn't standard (e.g. Camry, Avalon (/Hybrid))
    # On some cars, EPB is controlled by the ABS module
    (Ecu.epb, 0x750, 0x2c),     # Electronic Parking Brake
    # This isn't accessible on all cars
    (Ecu.gateway, 0x750, 0x5f),
    # On some cars, this only responds to b'\x1a\x88\x81', which is reflected by the b'\x1a\x88\x00' query
    (Ecu.telematics, 0x750, 0xc7),
    # Transmission is combined with engine on some platforms, such as TSS-P RAV4
    (Ecu.transmission, 0x701, None),
    # A few platforms have a tester present response on this address, add to log
    (Ecu.transmission, 0x7e1, None),
    # On some cars, this only responds to b'\x1a\x88\x80'
    (Ecu.combinationMeter, 0x7c0, None),
    (Ecu.hvac, 0x7c4, None),
  ],
  match_fw_to_car_fuzzy=match_fw_to_car_fuzzy,
)


STEER_THRESHOLD = 100

DBC = {
  CAR.RAV4H: dbc_dict('toyota_tnga_k_pt_generated', 'toyota_adas'),
  CAR.RAV4: dbc_dict('toyota_new_mc_pt_generated', 'toyota_adas'),
  CAR.PRIUS: dbc_dict('toyota_nodsu_pt_generated', 'toyota_adas'),
  CAR.PRIUS_V: dbc_dict('toyota_new_mc_pt_generated', 'toyota_adas'),
  CAR.COROLLA: dbc_dict('toyota_new_mc_pt_generated', 'toyota_adas'),
  CAR.LEXUS_RC: dbc_dict('toyota_tnga_k_pt_generated', 'toyota_adas'),
  CAR.LEXUS_RX: dbc_dict('toyota_tnga_k_pt_generated', 'toyota_adas'),
  CAR.LEXUS_RX_TSS2: dbc_dict('toyota_nodsu_pt_generated', 'toyota_tss2_adas'),
  CAR.CHR: dbc_dict('toyota_nodsu_pt_generated', 'toyota_adas'),
  CAR.CHR_TSS2: dbc_dict('toyota_nodsu_pt_generated', None),
  CAR.CAMRY: dbc_dict('toyota_nodsu_pt_generated', 'toyota_adas'),
  CAR.CAMRY_TSS2: dbc_dict('toyota_nodsu_pt_generated', 'toyota_tss2_adas'),
  CAR.HIGHLANDER: dbc_dict('toyota_tnga_k_pt_generated', 'toyota_adas'),
  CAR.HIGHLANDER_TSS2: dbc_dict('toyota_nodsu_pt_generated', 'toyota_tss2_adas'),
  CAR.AVALON: dbc_dict('toyota_tnga_k_pt_generated', 'toyota_adas'),
  CAR.AVALON_2019: dbc_dict('toyota_nodsu_pt_generated', 'toyota_adas'),
  CAR.AVALON_TSS2: dbc_dict('toyota_nodsu_pt_generated', 'toyota_tss2_adas'),
  CAR.RAV4_TSS2: dbc_dict('toyota_nodsu_pt_generated', 'toyota_tss2_adas'),
  CAR.RAV4_TSS2_2022: dbc_dict('toyota_nodsu_pt_generated', None),
  CAR.RAV4_TSS2_2023: dbc_dict('toyota_nodsu_pt_generated', None),
  CAR.COROLLA_TSS2: dbc_dict('toyota_nodsu_pt_generated', 'toyota_tss2_adas'),
  CAR.LEXUS_ES: dbc_dict('toyota_new_mc_pt_generated', 'toyota_adas'),
  CAR.LEXUS_ES_TSS2: dbc_dict('toyota_nodsu_pt_generated', 'toyota_tss2_adas'),
  CAR.SIENNA: dbc_dict('toyota_tnga_k_pt_generated', 'toyota_adas'),
  CAR.LEXUS_IS: dbc_dict('toyota_tnga_k_pt_generated', 'toyota_adas'),
  CAR.LEXUS_IS_TSS2: dbc_dict('toyota_nodsu_pt_generated', 'toyota_tss2_adas'),
  CAR.LEXUS_CTH: dbc_dict('toyota_new_mc_pt_generated', 'toyota_adas'),
  CAR.LEXUS_NX: dbc_dict('toyota_tnga_k_pt_generated', 'toyota_adas'),
  CAR.LEXUS_NX_TSS2: dbc_dict('toyota_nodsu_pt_generated', 'toyota_tss2_adas'),
  CAR.PRIUS_TSS2: dbc_dict('toyota_nodsu_pt_generated', 'toyota_tss2_adas'),
  CAR.MIRAI: dbc_dict('toyota_nodsu_pt_generated', 'toyota_tss2_adas'),
  CAR.ALPHARD_TSS2: dbc_dict('toyota_nodsu_pt_generated', 'toyota_tss2_adas'),
  CAR.LEXUS_GS_F: dbc_dict('toyota_new_mc_pt_generated', 'toyota_adas'),
}

# These cars have non-standard EPS torque scale factors. All others are 73
EPS_SCALE = defaultdict(lambda: 73, {CAR.PRIUS: 66, CAR.COROLLA: 88, CAR.LEXUS_IS: 77, CAR.LEXUS_RC: 77, CAR.LEXUS_CTH: 100, CAR.PRIUS_V: 100})

# Toyota/Lexus Safety Sense 2.0 and 2.5
TSS2_CAR = {CAR.RAV4_TSS2, CAR.RAV4_TSS2_2022, CAR.RAV4_TSS2_2023, CAR.COROLLA_TSS2, CAR.LEXUS_ES_TSS2,
            CAR.LEXUS_RX_TSS2, CAR.HIGHLANDER_TSS2, CAR.PRIUS_TSS2, CAR.CAMRY_TSS2, CAR.LEXUS_IS_TSS2,
            CAR.MIRAI, CAR.LEXUS_NX_TSS2, CAR.ALPHARD_TSS2, CAR.AVALON_TSS2, CAR.CHR_TSS2}

NO_DSU_CAR = TSS2_CAR | {CAR.CHR, CAR.CAMRY}

# the DSU uses the AEB message for longitudinal on these cars
UNSUPPORTED_DSU_CAR = {CAR.LEXUS_IS, CAR.LEXUS_RC, CAR.LEXUS_GS_F}

# these cars have a radar which sends ACC messages instead of the camera
RADAR_ACC_CAR = {CAR.RAV4_TSS2_2022, CAR.RAV4_TSS2_2023, CAR.CHR_TSS2}

# these cars use the Lane Tracing Assist (LTA) message for lateral control
ANGLE_CONTROL_CAR = {CAR.RAV4_TSS2_2023}

# no resume button press required
NO_STOP_TIMER_CAR = TSS2_CAR | {CAR.PRIUS_V, CAR.RAV4H, CAR.HIGHLANDER, CAR.SIENNA}
