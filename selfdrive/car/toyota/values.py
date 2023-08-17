from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, IntFlag
from typing import Dict, List, Union

from cereal import car
from common.conversions import Conversions as CV
from selfdrive.car import AngleRateLimit, dbc_dict
from selfdrive.car.docs_definitions import CarFootnote, CarInfo, Column, CarParts, CarHarness
from selfdrive.car.fw_query_definitions import FwQueryConfig, Request, StdQueries

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
  # however the EPS has its own internal limits at all speeds which are less than that
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


class CAR:
  # Toyota
  ALPHARD_TSS2 = "TOYOTA ALPHARD 2020"
  ALPHARDH_TSS2 = "TOYOTA ALPHARD HYBRID 2021"
  AVALON = "TOYOTA AVALON 2016"
  AVALON_2019 = "TOYOTA AVALON 2019"
  AVALONH_2019 = "TOYOTA AVALON HYBRID 2019"
  AVALON_TSS2 = "TOYOTA AVALON 2022"  # TSS 2.5
  AVALONH_TSS2 = "TOYOTA AVALON HYBRID 2022"
  CAMRY = "TOYOTA CAMRY 2018"
  CAMRYH = "TOYOTA CAMRY HYBRID 2018"
  CAMRY_TSS2 = "TOYOTA CAMRY 2021"  # TSS 2.5
  CAMRYH_TSS2 = "TOYOTA CAMRY HYBRID 2021"
  CHR = "TOYOTA C-HR 2018"
  CHR_TSS2 = "TOYOTA C-HR 2021"
  CHRH = "TOYOTA C-HR HYBRID 2018"
  CHRH_TSS2 = "TOYOTA C-HR HYBRID 2022"
  COROLLA = "TOYOTA COROLLA 2017"
  COROLLA_TSS2 = "TOYOTA COROLLA TSS2 2019"
  # LSS2 Lexus UX Hybrid is same as a TSS2 Corolla Hybrid
  COROLLAH_TSS2 = "TOYOTA COROLLA HYBRID TSS2 2019"
  HIGHLANDER = "TOYOTA HIGHLANDER 2017"
  HIGHLANDER_TSS2 = "TOYOTA HIGHLANDER 2020"
  HIGHLANDERH = "TOYOTA HIGHLANDER HYBRID 2018"
  HIGHLANDERH_TSS2 = "TOYOTA HIGHLANDER HYBRID 2020"
  PRIUS = "TOYOTA PRIUS 2017"
  PRIUS_V = "TOYOTA PRIUS v 2017"
  PRIUS_TSS2 = "TOYOTA PRIUS TSS2 2021"
  RAV4 = "TOYOTA RAV4 2017"
  RAV4H = "TOYOTA RAV4 HYBRID 2017"
  RAV4_TSS2 = "TOYOTA RAV4 2019"
  RAV4_TSS2_2022 = "TOYOTA RAV4 2022"
  RAV4_TSS2_2023 = "TOYOTA RAV4 2023"
  RAV4H_TSS2 = "TOYOTA RAV4 HYBRID 2019"
  RAV4H_TSS2_2022 = "TOYOTA RAV4 HYBRID 2022"
  RAV4H_TSS2_2023 = "TOYOTA RAV4 HYBRID 2023"
  MIRAI = "TOYOTA MIRAI 2021"  # TSS 2.5
  SIENNA = "TOYOTA SIENNA 2018"

  # Lexus
  LEXUS_CTH = "LEXUS CT HYBRID 2018"
  LEXUS_ES = "LEXUS ES 2018"
  LEXUS_ESH = "LEXUS ES HYBRID 2018"
  LEXUS_ES_TSS2 = "LEXUS ES 2019"
  LEXUS_ESH_TSS2 = "LEXUS ES HYBRID 2019"
  LEXUS_IS = "LEXUS IS 2018"
  LEXUS_IS_TSS2 = "LEXUS IS 2023"
  LEXUS_NX = "LEXUS NX 2018"
  LEXUS_NXH = "LEXUS NX HYBRID 2018"
  LEXUS_NX_TSS2 = "LEXUS NX 2020"
  LEXUS_NXH_TSS2 = "LEXUS NX HYBRID 2020"
  LEXUS_RC = "LEXUS RC 2020"
  LEXUS_RX = "LEXUS RX 2016"
  LEXUS_RXH = "LEXUS RX HYBRID 2017"
  LEXUS_RX_TSS2 = "LEXUS RX 2020"
  LEXUS_RXH_TSS2 = "LEXUS RX HYBRID 2020"


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
  CAR.ALPHARD_TSS2: ToyotaCarInfo("Toyota Alphard 2019-20"),
  CAR.ALPHARDH_TSS2: ToyotaCarInfo("Toyota Alphard Hybrid 2021"),
  CAR.AVALON: [
    ToyotaCarInfo("Toyota Avalon 2016", "Toyota Safety Sense P"),
    ToyotaCarInfo("Toyota Avalon 2017-18"),
  ],
  CAR.AVALON_2019: ToyotaCarInfo("Toyota Avalon 2019-21"),
  CAR.AVALONH_2019: ToyotaCarInfo("Toyota Avalon Hybrid 2019-21"),
  CAR.AVALON_TSS2: ToyotaCarInfo("Toyota Avalon 2022"),
  CAR.AVALONH_TSS2: ToyotaCarInfo("Toyota Avalon Hybrid 2022"),
  CAR.CAMRY: ToyotaCarInfo("Toyota Camry 2018-20", video_link="https://www.youtube.com/watch?v=fkcjviZY9CM", footnotes=[Footnote.CAMRY]),
  CAR.CAMRYH: ToyotaCarInfo("Toyota Camry Hybrid 2018-20", video_link="https://www.youtube.com/watch?v=Q2DYY0AWKgk"),
  CAR.CAMRY_TSS2: ToyotaCarInfo("Toyota Camry 2021-23", footnotes=[Footnote.CAMRY]),
  CAR.CAMRYH_TSS2: ToyotaCarInfo("Toyota Camry Hybrid 2021-23"),
  CAR.CHR: ToyotaCarInfo("Toyota C-HR 2017-20"),
  CAR.CHR_TSS2: ToyotaCarInfo("Toyota C-HR 2021"),
  CAR.CHRH: ToyotaCarInfo("Toyota C-HR Hybrid 2017-20"),
  CAR.CHRH_TSS2: ToyotaCarInfo("Toyota C-HR Hybrid 2021-22"),
  CAR.COROLLA: ToyotaCarInfo("Toyota Corolla 2017-19"),
  CAR.COROLLA_TSS2: [
    ToyotaCarInfo("Toyota Corolla 2020-22", video_link="https://www.youtube.com/watch?v=_66pXk0CBYA"),
    ToyotaCarInfo("Toyota Corolla Cross (Non-US only) 2020-23", min_enable_speed=7.5),
    ToyotaCarInfo("Toyota Corolla Hatchback 2019-22", video_link="https://www.youtube.com/watch?v=_66pXk0CBYA"),
  ],
  CAR.COROLLAH_TSS2: [
    ToyotaCarInfo("Toyota Corolla Hybrid 2020-22"),
    ToyotaCarInfo("Toyota Corolla Hybrid (Non-US only) 2020-23", min_enable_speed=7.5),
    ToyotaCarInfo("Toyota Corolla Cross Hybrid (Non-US only) 2020-22", min_enable_speed=7.5),
    ToyotaCarInfo("Lexus UX Hybrid 2019-23"),
  ],
  CAR.HIGHLANDER: ToyotaCarInfo("Toyota Highlander 2017-19", video_link="https://www.youtube.com/watch?v=0wS0wXSLzoo"),
  CAR.HIGHLANDER_TSS2: ToyotaCarInfo("Toyota Highlander 2020-23"),
  CAR.HIGHLANDERH: ToyotaCarInfo("Toyota Highlander Hybrid 2017-19"),
  CAR.HIGHLANDERH_TSS2: ToyotaCarInfo("Toyota Highlander Hybrid 2020-23"),
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
  CAR.RAV4_TSS2: ToyotaCarInfo("Toyota RAV4 2019-21", video_link="https://www.youtube.com/watch?v=wJxjDd42gGA"),
  CAR.RAV4_TSS2_2022: ToyotaCarInfo("Toyota RAV4 2022"),
  CAR.RAV4_TSS2_2023: ToyotaCarInfo("Toyota RAV4 2023"),
  CAR.RAV4H_TSS2: ToyotaCarInfo("Toyota RAV4 Hybrid 2019-21"),
  CAR.RAV4H_TSS2_2022: ToyotaCarInfo("Toyota RAV4 Hybrid 2022", video_link="https://youtu.be/U0nH9cnrFB0"),
  CAR.RAV4H_TSS2_2023: ToyotaCarInfo("Toyota RAV4 Hybrid 2023"),
  CAR.MIRAI: ToyotaCarInfo("Toyota Mirai 2021"),
  CAR.SIENNA: ToyotaCarInfo("Toyota Sienna 2018-20", video_link="https://www.youtube.com/watch?v=q1UPOo4Sh68", min_enable_speed=MIN_ACC_SPEED),

  # Lexus
  CAR.LEXUS_CTH: ToyotaCarInfo("Lexus CT Hybrid 2017-18", "Lexus Safety System+"),
  CAR.LEXUS_ES: ToyotaCarInfo("Lexus ES 2017-18"),
  CAR.LEXUS_ESH: ToyotaCarInfo("Lexus ES Hybrid 2017-18"),
  CAR.LEXUS_ES_TSS2: ToyotaCarInfo("Lexus ES 2019-22"),
  CAR.LEXUS_ESH_TSS2: ToyotaCarInfo("Lexus ES Hybrid 2019-23", video_link="https://youtu.be/BZ29osRVJeg?t=12"),
  CAR.LEXUS_IS: ToyotaCarInfo("Lexus IS 2017-19"),
  CAR.LEXUS_IS_TSS2: ToyotaCarInfo("Lexus IS 2023"),
  CAR.LEXUS_NX: ToyotaCarInfo("Lexus NX 2018-19"),
  CAR.LEXUS_NXH: ToyotaCarInfo("Lexus NX Hybrid 2018-19"),
  CAR.LEXUS_NX_TSS2: ToyotaCarInfo("Lexus NX 2020-21"),
  CAR.LEXUS_NXH_TSS2: ToyotaCarInfo("Lexus NX Hybrid 2020-21"),
  CAR.LEXUS_RC: ToyotaCarInfo("Lexus RC 2018-20"),
  CAR.LEXUS_RX: [
    ToyotaCarInfo("Lexus RX 2016", "Lexus Safety System+"),
    ToyotaCarInfo("Lexus RX 2017-19"),
  ],
  CAR.LEXUS_RXH: [
    ToyotaCarInfo("Lexus RX Hybrid 2016", "Lexus Safety System+"),
    ToyotaCarInfo("Lexus RX Hybrid 2017-19"),
  ],
  CAR.LEXUS_RX_TSS2: ToyotaCarInfo("Lexus RX 2020-22"),
  CAR.LEXUS_RXH_TSS2: ToyotaCarInfo("Lexus RX Hybrid 2020-22"),
}

# (addr, cars, bus, 1/freq*100, vl)
STATIC_DSU_MSGS = [
  (0x128, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RXH, CAR.LEXUS_NXH, CAR.LEXUS_NX, CAR.RAV4, CAR.COROLLA, CAR.AVALON), 1,   3, b'\xf4\x01\x90\x83\x00\x37'),
  (0x128, (CAR.HIGHLANDER, CAR.HIGHLANDERH, CAR.SIENNA, CAR.LEXUS_CTH, CAR.LEXUS_ES, CAR.LEXUS_ESH), 1,   3, b'\x03\x00\x20\x00\x00\x52'),
  (0x141, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RXH, CAR.LEXUS_NXH, CAR.LEXUS_NX, CAR.RAV4, CAR.COROLLA, CAR.HIGHLANDER, CAR.HIGHLANDERH, CAR.AVALON,
           CAR.SIENNA, CAR.LEXUS_CTH, CAR.LEXUS_ES, CAR.LEXUS_ESH, CAR.LEXUS_RX, CAR.PRIUS_V), 1,   2, b'\x00\x00\x00\x46'),
  (0x160, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RXH, CAR.LEXUS_NXH, CAR.LEXUS_NX, CAR.RAV4, CAR.COROLLA, CAR.HIGHLANDER, CAR.HIGHLANDERH, CAR.AVALON,
           CAR.SIENNA, CAR.LEXUS_CTH, CAR.LEXUS_ES, CAR.LEXUS_ESH, CAR.LEXUS_RX, CAR.PRIUS_V), 1,   7, b'\x00\x00\x08\x12\x01\x31\x9c\x51'),
  (0x161, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RXH, CAR.LEXUS_NXH, CAR.LEXUS_NX, CAR.RAV4, CAR.COROLLA, CAR.AVALON, CAR.LEXUS_RX, CAR.PRIUS_V, CAR.LEXUS_ES),
                                                                                               1,   7, b'\x00\x1e\x00\x00\x00\x80\x07'),
  (0X161, (CAR.HIGHLANDERH, CAR.HIGHLANDER, CAR.SIENNA, CAR.LEXUS_CTH, CAR.LEXUS_ESH), 1,  7, b'\x00\x1e\x00\xd4\x00\x00\x5b'),
  (0x283, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RXH, CAR.LEXUS_NXH, CAR.LEXUS_NX, CAR.RAV4, CAR.COROLLA, CAR.HIGHLANDER, CAR.HIGHLANDERH, CAR.AVALON,
           CAR.SIENNA, CAR.LEXUS_CTH, CAR.LEXUS_ES, CAR.LEXUS_ESH, CAR.LEXUS_RX, CAR.PRIUS_V), 0,   3, b'\x00\x00\x00\x00\x00\x00\x8c'),
  (0x2E6, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RXH), 0,   3, b'\xff\xf8\x00\x08\x7f\xe0\x00\x4e'),
  (0x2E7, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RXH), 0,   3, b'\xa8\x9c\x31\x9c\x00\x00\x00\x02'),
  (0x33E, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RXH), 0,  20, b'\x0f\xff\x26\x40\x00\x1f\x00'),
  (0x344, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RXH, CAR.LEXUS_NXH, CAR.LEXUS_NX, CAR.RAV4, CAR.COROLLA, CAR.HIGHLANDER, CAR.HIGHLANDERH, CAR.AVALON,
           CAR.SIENNA, CAR.LEXUS_CTH, CAR.LEXUS_ES, CAR.LEXUS_ESH, CAR.LEXUS_RX, CAR.PRIUS_V), 0,   5, b'\x00\x00\x01\x00\x00\x00\x00\x50'),
  (0x365, (CAR.PRIUS, CAR.LEXUS_RXH, CAR.LEXUS_NXH, CAR.LEXUS_NX, CAR.HIGHLANDERH), 0,  20, b'\x00\x00\x00\x80\x03\x00\x08'),
  (0x365, (CAR.RAV4, CAR.RAV4H, CAR.COROLLA, CAR.HIGHLANDER, CAR.AVALON, CAR.SIENNA, CAR.LEXUS_CTH, CAR.LEXUS_ES, CAR.LEXUS_ESH, CAR.LEXUS_RX,
           CAR.PRIUS_V), 0,  20, b'\x00\x00\x00\x80\xfc\x00\x08'),
  (0x366, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RXH, CAR.LEXUS_NXH, CAR.LEXUS_NX, CAR.HIGHLANDERH), 0,  20, b'\x00\x00\x4d\x82\x40\x02\x00'),
  (0x366, (CAR.RAV4, CAR.COROLLA, CAR.HIGHLANDER, CAR.AVALON, CAR.SIENNA, CAR.LEXUS_CTH, CAR.LEXUS_ESH, CAR.LEXUS_RX, CAR.PRIUS_V),
          0,  20, b'\x00\x72\x07\xff\x09\xfe\x00'),
  (0x366, (CAR.LEXUS_ES,), 0,  20, b'\x00\x95\x07\xfe\x08\x05\x00'),
  (0x470, (CAR.PRIUS, CAR.LEXUS_RXH), 1, 100, b'\x00\x00\x02\x7a'),
  (0x470, (CAR.HIGHLANDER, CAR.HIGHLANDERH, CAR.RAV4H, CAR.SIENNA, CAR.LEXUS_CTH, CAR.LEXUS_ES, CAR.LEXUS_ESH, CAR.PRIUS_V), 1,  100, b'\x00\x00\x01\x79'),
  (0x4CB, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RXH, CAR.LEXUS_NXH, CAR.LEXUS_NX, CAR.RAV4, CAR.COROLLA, CAR.HIGHLANDERH, CAR.HIGHLANDER, CAR.AVALON,
           CAR.SIENNA, CAR.LEXUS_CTH, CAR.LEXUS_ES, CAR.LEXUS_ESH, CAR.LEXUS_RX, CAR.PRIUS_V), 0, 100, b'\x0c\x00\x00\x00\x00\x00\x00\x00'),
]

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
                      Ecu.hybrid, Ecu.srs, Ecu.combinationMeter, Ecu.transmission, Ecu.gateway, Ecu.hvac],
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
    Ecu.abs: [CAR.RAV4, CAR.COROLLA, CAR.HIGHLANDER, CAR.SIENNA, CAR.LEXUS_IS],
    # On some models, the engine can show on two different addresses
    Ecu.engine: [CAR.CAMRY, CAR.COROLLA_TSS2, CAR.CHR, CAR.CHR_TSS2, CAR.LEXUS_IS, CAR.LEXUS_RC],
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
)

FW_VERSIONS = {
  CAR.AVALON: {
    (Ecu.abs, 0x7b0, None): [
      b'F152607060\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.dsu, 0x791, None): [
      b'881510701300\x00\x00\x00\x00',
      b'881510705100\x00\x00\x00\x00',
      b'881510705200\x00\x00\x00\x00',
    ],
    (Ecu.eps, 0x7a1, None): [
      b'8965B41051\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'\x0230721100\x00\x00\x00\x00\x00\x00\x00\x00A0C01000\x00\x00\x00\x00\x00\x00\x00\x00',
      b'\x0230721200\x00\x00\x00\x00\x00\x00\x00\x00A0C01000\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.fwdRadar, 0x750, 0xf): [
      b'8821F4702000\x00\x00\x00\x00',
      b'8821F4702100\x00\x00\x00\x00',
    ],
    (Ecu.fwdCamera, 0x750, 0x6d): [
      b'8646F0701100\x00\x00\x00\x00',
      b'8646F0703000\x00\x00\x00\x00',
    ],
  },
  CAR.AVALON_2019: {
    (Ecu.abs, 0x7b0, None): [
      b'F152607140\x00\x00\x00\x00\x00\x00',
      b'F152607171\x00\x00\x00\x00\x00\x00',
      b'F152607110\x00\x00\x00\x00\x00\x00',
      b'F152607180\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.dsu, 0x791, None): [
      b'881510703200\x00\x00\x00\x00',
    ],
    (Ecu.eps, 0x7a1, None): [
      b'8965B41080\x00\x00\x00\x00\x00\x00',
      b'8965B07010\x00\x00\x00\x00\x00\x00',
      b'8965B41090\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.engine, 0x700, None): [
      b'\x01896630725200\x00\x00\x00\x00',
      b'\x01896630725300\x00\x00\x00\x00',
      b'\x01896630735100\x00\x00\x00\x00',
      b'\x01896630738000\x00\x00\x00\x00',
    ],
    (Ecu.fwdRadar, 0x750, 0xf): [
      b'8821F4702300\x00\x00\x00\x00',
    ],
    (Ecu.fwdCamera, 0x750, 0x6d): [
      b'8646F0702100\x00\x00\x00\x00',
    ],
  },
  CAR.AVALONH_2019: {
    (Ecu.abs, 0x7b0, None): [
      b'F152641040\x00\x00\x00\x00\x00\x00',
      b'F152641061\x00\x00\x00\x00\x00\x00',
      b'F152641050\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.dsu, 0x791, None): [
      b'881510704200\x00\x00\x00\x00',
      b'881514107100\x00\x00\x00\x00',
    ],
    (Ecu.eps, 0x7a1, None): [
      b'8965B07010\x00\x00\x00\x00\x00\x00',
      b'8965B41090\x00\x00\x00\x00\x00\x00',
      b'8965B41070\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.engine, 0x700, None): [
      b'\x02896630724000\x00\x00\x00\x00897CF3302002\x00\x00\x00\x00',
      b'\x02896630737000\x00\x00\x00\x00897CF3305001\x00\x00\x00\x00',
      b'\x02896630728000\x00\x00\x00\x00897CF3302002\x00\x00\x00\x00',
    ],
    (Ecu.fwdRadar, 0x750, 0xf): [
      b'8821F4702300\x00\x00\x00\x00',
    ],
    (Ecu.fwdCamera, 0x750, 0x6d): [
      b'8646F0702100\x00\x00\x00\x00',
    ],
  },
  CAR.AVALON_TSS2: {
    (Ecu.abs, 0x7b0, None): [
      b'\x01F152607240\x00\x00\x00\x00\x00\x00',
      b'\x01F152607250\x00\x00\x00\x00\x00\x00',
      b'\x01F152607280\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.eps, 0x7a1, None): [
      b'8965B41110\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.engine, 0x700, None): [
      b'\x01896630742000\x00\x00\x00\x00',
      b'\x01896630743000\x00\x00\x00\x00',
    ],
    (Ecu.fwdRadar, 0x750, 0xf): [
      b'\x018821F6201200\x00\x00\x00\x00',
      b'\x018821F6201300\x00\x00\x00\x00',
    ],
    (Ecu.fwdCamera, 0x750, 0x6d): [
      b'\x028646F4104100\x00\x00\x00\x008646G5301200\x00\x00\x00\x00',
      b'\x028646F4104100\x00\x00\x00\x008646G3304000\x00\x00\x00\x00',
    ],
  },
  CAR.AVALONH_TSS2: {
    (Ecu.abs, 0x7b0, None): [
      b'F152641080\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.eps, 0x7a1, None): [
      b'8965B41110\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.engine, 0x700, None): [
      b'\x018966306Q6000\x00\x00\x00\x00',
    ],
    (Ecu.fwdRadar, 0x750, 0xf): [
      b'\x018821F6201200\x00\x00\x00\x00',
      b'\x018821F6201300\x00\x00\x00\x00',
    ],
    (Ecu.fwdCamera, 0x750, 0x6d): [
      b'\x028646F4104100\x00\x00\x00\x008646G5301200\x00\x00\x00\x00',
      b'\x028646F4104100\x00\x00\x00\x008646G3304000\x00\x00\x00\x00',
    ],
  },
  CAR.CAMRY: {
    (Ecu.engine, 0x700, None): [
      b'\x018966306L3100\x00\x00\x00\x00',
      b'\x018966306L4200\x00\x00\x00\x00',
      b'\x018966306L5200\x00\x00\x00\x00',
      b'\x018966306P8000\x00\x00\x00\x00',
      b'\x018966306Q3100\x00\x00\x00\x00',
      b'\x018966306Q4000\x00\x00\x00\x00',
      b'\x018966306Q4100\x00\x00\x00\x00',
      b'\x018966306Q4200\x00\x00\x00\x00',
      b'\x018966333Q9200\x00\x00\x00\x00',
      b'\x018966333P3100\x00\x00\x00\x00',
      b'\x018966333P3200\x00\x00\x00\x00',
      b'\x018966333P4200\x00\x00\x00\x00',
      b'\x018966333P4300\x00\x00\x00\x00',
      b'\x018966333P4400\x00\x00\x00\x00',
      b'\x018966333P4500\x00\x00\x00\x00',
      b'\x018966333P4700\x00\x00\x00\x00',
      b'\x018966333P4900\x00\x00\x00\x00',
      b'\x018966333Q6000\x00\x00\x00\x00',
      b'\x018966333Q6200\x00\x00\x00\x00',
      b'\x018966333Q6300\x00\x00\x00\x00',
      b'\x018966333Q6500\x00\x00\x00\x00',
      b'\x018966333W6000\x00\x00\x00\x00',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'\x02333P1100\x00\x00\x00\x00\x00\x00\x00\x00A0202000\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.dsu, 0x791, None): [
      b'8821F0601200    ',
      b'8821F0601300    ',
      b'8821F0602000    ',
      b'8821F0603300    ',
      b'8821F0604100    ',
      b'8821F0605200    ',
      b'8821F0607200    ',
      b'8821F0608000    ',
      b'8821F0608200    ',
      b'8821F0609100    ',
    ],
    (Ecu.abs, 0x7b0, None): [
      b'F152606210\x00\x00\x00\x00\x00\x00',
      b'F152606230\x00\x00\x00\x00\x00\x00',
      b'F152606270\x00\x00\x00\x00\x00\x00',
      b'F152606290\x00\x00\x00\x00\x00\x00',
      b'F152606410\x00\x00\x00\x00\x00\x00',
      b'F152633540\x00\x00\x00\x00\x00\x00',
      b'F152633A10\x00\x00\x00\x00\x00\x00',
      b'F152633A20\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.eps, 0x7a1, None): [
      b'8965B33540\x00\x00\x00\x00\x00\x00',
      b'8965B33542\x00\x00\x00\x00\x00\x00',
      b'8965B33580\x00\x00\x00\x00\x00\x00',
      b'8965B33581\x00\x00\x00\x00\x00\x00',
      b'8965B33621\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.fwdRadar, 0x750, 0xf): [  # Same as 0x791
      b'8821F0601200    ',
      b'8821F0601300    ',
      b'8821F0602000    ',
      b'8821F0603300    ',
      b'8821F0604100    ',
      b'8821F0605200    ',
      b'8821F0607200    ',
      b'8821F0608000    ',
      b'8821F0608200    ',
      b'8821F0609100    ',
    ],
    (Ecu.fwdCamera, 0x750, 0x6d): [
      b'8646F0601200    ',
      b'8646F0601300    ',
      b'8646F0601400    ',
      b'8646F0603400    ',
      b'8646F0604100    ',
      b'8646F0605000    ',
      b'8646F0606000    ',
      b'8646F0606100    ',
      b'8646F0607100    ',
    ],
  },
  CAR.CAMRYH: {
    (Ecu.engine, 0x700, None): [
      b'\x018966306Q6000\x00\x00\x00\x00',
      b'\x018966333N1100\x00\x00\x00\x00',
      b'\x018966333N4300\x00\x00\x00\x00',
      b'\x018966333X0000\x00\x00\x00\x00',
      b'\x018966333X4000\x00\x00\x00\x00',
      b'\x01896633T16000\x00\x00\x00\x00',
      b'\x028966306B2100\x00\x00\x00\x00897CF3302002\x00\x00\x00\x00',
      b'\x028966306B2300\x00\x00\x00\x00897CF3302002\x00\x00\x00\x00',
      b'\x028966306B2500\x00\x00\x00\x00897CF3302002\x00\x00\x00\x00',
      b'\x028966306N8100\x00\x00\x00\x00897CF3302002\x00\x00\x00\x00',
      b'\x028966306N8200\x00\x00\x00\x00897CF3302002\x00\x00\x00\x00',
      b'\x028966306N8300\x00\x00\x00\x00897CF3302002\x00\x00\x00\x00',
      b'\x028966306N8400\x00\x00\x00\x00897CF3302002\x00\x00\x00\x00',
      b'\x028966306R5000\x00\x00\x00\x00897CF3302002\x00\x00\x00\x00',
      b'\x028966306R5000\x00\x00\x00\x00897CF3305001\x00\x00\x00\x00',
      b'\x028966306R6000\x00\x00\x00\x00897CF3302002\x00\x00\x00\x00',
      b'\x028966306R6000\x00\x00\x00\x00897CF3305001\x00\x00\x00\x00',
      b'\x028966306S0000\x00\x00\x00\x00897CF3305001\x00\x00\x00\x00',
      b'\x028966306S0100\x00\x00\x00\x00897CF3305001\x00\x00\x00\x00',
      b'\x028966306S1100\x00\x00\x00\x00897CF3305001\x00\x00\x00\x00',
    ],
    (Ecu.abs, 0x7b0, None): [
      b'F152633214\x00\x00\x00\x00\x00\x00',
      b'F152633660\x00\x00\x00\x00\x00\x00',
      b'F152633712\x00\x00\x00\x00\x00\x00',
      b'F152633713\x00\x00\x00\x00\x00\x00',
      b'F152633B51\x00\x00\x00\x00\x00\x00',
      b'F152633B60\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.dsu, 0x791, None): [
      b'8821F0601200    ',
      b'8821F0601300    ',
      b'8821F0603400    ',
      b'8821F0604000    ',
      b'8821F0604100    ',
      b'8821F0604200    ',
      b'8821F0605200    ',
      b'8821F0606200    ',
      b'8821F0607200    ',
      b'8821F0608000    ',
      b'8821F0608200    ',
      b'8821F0609000    ',
      b'8821F0609100    ',
    ],
    (Ecu.eps, 0x7a1, None): [
      b'8965B33540\x00\x00\x00\x00\x00\x00',
      b'8965B33542\x00\x00\x00\x00\x00\x00',
      b'8965B33550\x00\x00\x00\x00\x00\x00',
      b'8965B33551\x00\x00\x00\x00\x00\x00',
      b'8965B33580\x00\x00\x00\x00\x00\x00',
      b'8965B33581\x00\x00\x00\x00\x00\x00',
      b'8965B33611\x00\x00\x00\x00\x00\x00',
      b'8965B33621\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.fwdRadar, 0x750, 0xf): [  # Same as 0x791
      b'8821F0601200    ',
      b'8821F0601300    ',
      b'8821F0603400    ',
      b'8821F0604000    ',
      b'8821F0604100    ',
      b'8821F0604200    ',
      b'8821F0605200    ',
      b'8821F0606200    ',
      b'8821F0607200    ',
      b'8821F0608000    ',
      b'8821F0608200    ',
      b'8821F0609000    ',
      b'8821F0609100    ',
    ],
    (Ecu.fwdCamera, 0x750, 0x6d): [
      b'8646F0601200    ',
      b'8646F0601300    ',
      b'8646F0601400    ',
      b'8646F0603400    ',
      b'8646F0603500    ',
      b'8646F0604100    ',
      b'8646F0605000    ',
      b'8646F0606000    ',
      b'8646F0606100    ',
      b'8646F0607000    ',
      b'8646F0607100    ',
    ],
  },
  CAR.CAMRY_TSS2: {
    (Ecu.eps, 0x7a1, None): [
      b'8965B33630\x00\x00\x00\x00\x00\x00',
      b'8965B33640\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.abs, 0x7b0, None): [
      b'\x01F152606370\x00\x00\x00\x00\x00\x00',
      b'\x01F152606390\x00\x00\x00\x00\x00\x00',
      b'\x01F152606400\x00\x00\x00\x00\x00\x00',
      b'\x01F152606431\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.engine, 0x700, None): [
      b'\x018966306Q5000\x00\x00\x00\x00',
      b'\x018966306Q9000\x00\x00\x00\x00',
      b'\x018966306R3000\x00\x00\x00\x00',
      b'\x018966306R8000\x00\x00\x00\x00',
      b'\x018966306T3100\x00\x00\x00\x00',
      b'\x018966306T3200\x00\x00\x00\x00',
      b'\x018966306T4000\x00\x00\x00\x00',
      b'\x018966306T4100\x00\x00\x00\x00',
    ],
    (Ecu.fwdRadar, 0x750, 0xf): [
      b'\x018821F6201200\x00\x00\x00\x00',
      b'\x018821F6201300\x00\x00\x00\x00',
    ],
    (Ecu.fwdCamera, 0x750, 0x6d): [
      b'\x028646F0602100\x00\x00\x00\x008646G5301200\x00\x00\x00\x00',
      b'\x028646F0602200\x00\x00\x00\x008646G5301200\x00\x00\x00\x00',
      b'\x028646F0602300\x00\x00\x00\x008646G3304000\x00\x00\x00\x00',
      b'\x028646F3305200\x00\x00\x00\x008646G5301200\x00\x00\x00\x00',
      b'\x028646F3305200\x00\x00\x00\x008646G3304000\x00\x00\x00\x00',
      b'\x028646F3305300\x00\x00\x00\x008646G5301200\x00\x00\x00\x00',
      b'\x028646F3305500\x00\x00\x00\x008646G3304000\x00\x00\x00\x00',
    ],
  },
  CAR.CAMRYH_TSS2: {
    (Ecu.eps, 0x7a1, None): [
      b'8965B33630\x00\x00\x00\x00\x00\x00',
      b'8965B33650\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.abs, 0x7b0, None): [
      b'F152633D00\x00\x00\x00\x00\x00\x00',
      b'F152633D60\x00\x00\x00\x00\x00\x00',
      b'F152633310\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.engine, 0x700, None): [
      b'\x018966306Q6000\x00\x00\x00\x00',
      b'\x018966306Q7000\x00\x00\x00\x00',
      b'\x018966306V1000\x00\x00\x00\x00',
      b'\x01896633T20000\x00\x00\x00\x00',
    ],
    (Ecu.fwdRadar, 0x750, 15): [
      b'\x018821F6201200\x00\x00\x00\x00',
      b'\x018821F6201300\x00\x00\x00\x00',
    ],
    (Ecu.fwdCamera, 0x750, 109): [
      b'\x028646F3305200\x00\x00\x00\x008646G5301200\x00\x00\x00\x00',
      b'\x028646F3305300\x00\x00\x00\x008646G5301200\x00\x00\x00\x00',
      b'\x028646F3305300\x00\x00\x00\x008646G3304000\x00\x00\x00\x00',
      b'\x028646F3305500\x00\x00\x00\x008646G3304000\x00\x00\x00\x00',
    ],
  },
  CAR.CHR: {
    (Ecu.engine, 0x700, None): [
      b'\x01896631021100\x00\x00\x00\x00',
      b'\x01896631017100\x00\x00\x00\x00',
      b'\x01896631017200\x00\x00\x00\x00',
      b'\x0189663F413100\x00\x00\x00\x00',
      b'\x0189663F414100\x00\x00\x00\x00',
    ],
    (Ecu.dsu, 0x791, None): [
      b'8821F0W01000    ',
      b'8821F0W01100    ',
      b'8821FF401600    ',
      b'8821FF404000    ',
      b'8821FF404100    ',
      b'8821FF405100    ',
      b'8821FF406000    ',
      b'8821FF407100    ',
    ],
    (Ecu.abs, 0x7b0, None): [
      b'F152610020\x00\x00\x00\x00\x00\x00',
      b'F152610153\x00\x00\x00\x00\x00\x00',
      b'F152610210\x00\x00\x00\x00\x00\x00',
      b'F1526F4034\x00\x00\x00\x00\x00\x00',
      b'F1526F4044\x00\x00\x00\x00\x00\x00',
      b'F1526F4073\x00\x00\x00\x00\x00\x00',
      b'F1526F4121\x00\x00\x00\x00\x00\x00',
      b'F1526F4122\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.eps, 0x7a1, None): [
      b'8965B10011\x00\x00\x00\x00\x00\x00',
      b'8965B10040\x00\x00\x00\x00\x00\x00',
      b'8965B10070\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'\x0331024000\x00\x00\x00\x00\x00\x00\x00\x00A0202000\x00\x00\x00\x00\x00\x00\x00\x00895231203202\x00\x00\x00\x00',
      b'\x0331024000\x00\x00\x00\x00\x00\x00\x00\x00A0202000\x00\x00\x00\x00\x00\x00\x00\x00895231203302\x00\x00\x00\x00',
      b'\x0331036000\x00\x00\x00\x00\x00\x00\x00\x00A0202000\x00\x00\x00\x00\x00\x00\x00\x00895231203302\x00\x00\x00\x00',
      b'\x033F401100\x00\x00\x00\x00\x00\x00\x00\x00A0202000\x00\x00\x00\x00\x00\x00\x00\x00895231203102\x00\x00\x00\x00',
      b'\x033F401200\x00\x00\x00\x00\x00\x00\x00\x00A0202000\x00\x00\x00\x00\x00\x00\x00\x00895231203202\x00\x00\x00\x00',
      b'\x033F424000\x00\x00\x00\x00\x00\x00\x00\x00A0202000\x00\x00\x00\x00\x00\x00\x00\x00895231203202\x00\x00\x00\x00',
      b'\x033F424000\x00\x00\x00\x00\x00\x00\x00\x00A0202000\x00\x00\x00\x00\x00\x00\x00\x00895231203302\x00\x00\x00\x00',
    ],
    (Ecu.fwdRadar, 0x750, 0xf): [
      b'8821F0W01000    ',
      b'8821FF401600    ',
      b'8821FF404000    ',
      b'8821FF404100    ',
      b'8821FF405100    ',
      b'8821FF406000    ',
      b'8821FF407100    ',
      b'8821F0W01100    ',
    ],
    (Ecu.fwdCamera, 0x750, 0x6d): [
      b'8646FF401700    ',
      b'8646FF401800    ',
      b'8646FF404000    ',
      b'8646FF406000    ',
      b'8646FF407000    ',
    ],
  },
  CAR.CHR_TSS2: {
    (Ecu.abs, 0x7b0, None): [
      b'F152610260\x00\x00\x00\x00\x00\x00',
      b'F1526F4270\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.eps, 0x7a1, None): [
      b'8965B10091\x00\x00\x00\x00\x00\x00',
      b'8965B10110\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.engine, 0x700, None): [
      b'\x0189663F459000\x00\x00\x00\x00',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'\x0331014000\x00\x00\x00\x00\x00\x00\x00\x00A0202000\x00\x00\x00\x00\x00\x00\x00\x00895231203402\x00\x00\x00\x00',
    ],
    (Ecu.fwdRadar, 0x750, 0xf): [
      b'\x018821FF410200\x00\x00\x00\x00',
      b'\x018821FF410300\x00\x00\x00\x00',
    ],
    (Ecu.fwdCamera, 0x750, 0x6d): [
      b'\x028646FF410200\x00\x00\x00\x008646GF408200\x00\x00\x00\x00',
      b'\x028646FF411100\x00\x00\x00\x008646GF409000\x00\x00\x00\x00',
    ],
  },
  CAR.CHRH: {
    (Ecu.engine, 0x700, None): [
      b'\x0289663F405100\x00\x00\x00\x008966A4703000\x00\x00\x00\x00',
      b'\x02896631013200\x00\x00\x00\x008966A4703000\x00\x00\x00\x00',
      b'\x0289663F405000\x00\x00\x00\x008966A4703000\x00\x00\x00\x00',
      b'\x0289663F418000\x00\x00\x00\x008966A4703000\x00\x00\x00\x00',
      b'\x0289663F423000\x00\x00\x00\x008966A4703000\x00\x00\x00\x00',
      b'\x0289663F431000\x00\x00\x00\x008966A4703000\x00\x00\x00\x00',
      b'\x0189663F438000\x00\x00\x00\x00',
    ],
    (Ecu.abs, 0x7b0, None): [
      b'F152610012\x00\x00\x00\x00\x00\x00',
      b'F152610013\x00\x00\x00\x00\x00\x00',
      b'F152610014\x00\x00\x00\x00\x00\x00',
      b'F152610040\x00\x00\x00\x00\x00\x00',
      b'F152610190\x00\x00\x00\x00\x00\x00',
      b'F152610200\x00\x00\x00\x00\x00\x00',
      b'F152610220\x00\x00\x00\x00\x00\x00',
      b'F152610230\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.dsu, 0x791, None): [
      b'8821F0W01000    ',
      b'8821FF402300    ',
      b'8821FF402400    ',
      b'8821FF404000    ',
      b'8821FF404100    ',
      b'8821FF405000    ',
      b'8821FF406000    ',
      b'8821FF407100    ',
    ],
    (Ecu.eps, 0x7a1, None): [
      b'8965B10011\x00\x00\x00\x00\x00\x00',
      b'8965B10020\x00\x00\x00\x00\x00\x00',
      b'8965B10040\x00\x00\x00\x00\x00\x00',
      b'8965B10050\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.fwdRadar, 0x750, 0xf): [
      b'8821F0W01000    ',
      b'8821FF402300    ',
      b'8821FF402400    ',
      b'8821FF404000    ',
      b'8821FF404100    ',
      b'8821FF405000    ',
      b'8821FF406000    ',
      b'8821FF407100    ',
    ],
    (Ecu.fwdCamera, 0x750, 0x6d): [
      b'8646FF401700    ',
      b'8646FF402100    ',
      b'8646FF404000    ',
      b'8646FF406000    ',
      b'8646FF407000    ',
      b'8646FF407100    ',
    ],
  },
  CAR.CHRH_TSS2: {
    (Ecu.eps, 0x7a1, None): [
      b'8965B10092\x00\x00\x00\x00\x00\x00',
      b'8965B10091\x00\x00\x00\x00\x00\x00',
      b'8965B10111\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.abs, 0x7b0, None): [
      b'F152610041\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.engine, 0x700, None): [
      b'\x0189663F438000\x00\x00\x00\x00',
      b'\x02896631025000\x00\x00\x00\x008966A4703000\x00\x00\x00\x00',
      b'\x0289663F453000\x00\x00\x00\x008966A4703000\x00\x00\x00\x00',
    ],
    (Ecu.fwdRadar, 0x750, 15): [
      b'\x018821FF410500\x00\x00\x00\x00',
      b'\x018821FF410300\x00\x00\x00\x00',
    ],
    (Ecu.fwdCamera, 0x750, 109): [
      b'\x028646FF413100\x00\x00\x00\x008646GF411100\x00\x00\x00\x00',
      b'\x028646FF411100\x00\x00\x00\x008646GF409000\x00\x00\x00\x00',
    ],
  },
  CAR.COROLLA: {
    (Ecu.engine, 0x7e0, None): [
      b'\x0230ZC2000\x00\x00\x00\x00\x00\x00\x00\x0050212000\x00\x00\x00\x00\x00\x00\x00\x00',
      b'\x0230ZC2100\x00\x00\x00\x00\x00\x00\x00\x0050212000\x00\x00\x00\x00\x00\x00\x00\x00',
      b'\x0230ZC2200\x00\x00\x00\x00\x00\x00\x00\x0050212000\x00\x00\x00\x00\x00\x00\x00\x00',
      b'\x0230ZC2300\x00\x00\x00\x00\x00\x00\x00\x0050212000\x00\x00\x00\x00\x00\x00\x00\x00',
      b'\x0230ZC3000\x00\x00\x00\x00\x00\x00\x00\x0050212000\x00\x00\x00\x00\x00\x00\x00\x00',
      b'\x0230ZC3100\x00\x00\x00\x00\x00\x00\x00\x0050212000\x00\x00\x00\x00\x00\x00\x00\x00',
      b'\x0230ZC3200\x00\x00\x00\x00\x00\x00\x00\x0050212000\x00\x00\x00\x00\x00\x00\x00\x00',
      b'\x0230ZC3300\x00\x00\x00\x00\x00\x00\x00\x0050212000\x00\x00\x00\x00\x00\x00\x00\x00',
      b'\x0330ZC1200\x00\x00\x00\x00\x00\x00\x00\x0050212000\x00\x00\x00\x00\x00\x00\x00\x00895231203202\x00\x00\x00\x00',
    ],
    (Ecu.dsu, 0x791, None): [
      b'881510201100\x00\x00\x00\x00',
      b'881510201200\x00\x00\x00\x00',
    ],
    (Ecu.abs, 0x7b0, None): [
      b'F152602190\x00\x00\x00\x00\x00\x00',
      b'F152602191\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.eps, 0x7a1, None): [
      b'8965B02181\x00\x00\x00\x00\x00\x00',
      b'8965B02191\x00\x00\x00\x00\x00\x00',
      b'8965B48150\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.fwdRadar, 0x750, 0xf): [
      b'8821F4702100\x00\x00\x00\x00',
      b'8821F4702300\x00\x00\x00\x00',
    ],
    (Ecu.fwdCamera, 0x750, 0x6d): [
      b'8646F0201101\x00\x00\x00\x00',
      b'8646F0201200\x00\x00\x00\x00',
    ],
  },
  CAR.COROLLA_TSS2: {
    (Ecu.engine, 0x700, None): [
      b'\x01896630A22000\x00\x00\x00\x00',
      b'\x01896630ZG2000\x00\x00\x00\x00',
      b'\x01896630ZG5000\x00\x00\x00\x00',
      b'\x01896630ZG5100\x00\x00\x00\x00',
      b'\x01896630ZG5200\x00\x00\x00\x00',
      b'\x01896630ZG5300\x00\x00\x00\x00',
      b'\x01896630ZP1000\x00\x00\x00\x00',
      b'\x01896630ZP2000\x00\x00\x00\x00',
      b'\x01896630ZQ5000\x00\x00\x00\x00',
      b'\x01896630ZU9000\x00\x00\x00\x00',
      b'\x01896630ZX4000\x00\x00\x00\x00',
      b'\x018966312L8000\x00\x00\x00\x00',
      b'\x018966312M0000\x00\x00\x00\x00',
      b'\x018966312M9000\x00\x00\x00\x00',
      b'\x018966312P9000\x00\x00\x00\x00',
      b'\x018966312P9100\x00\x00\x00\x00',
      b'\x018966312P9200\x00\x00\x00\x00',
      b'\x018966312P9300\x00\x00\x00\x00',
      b'\x018966312Q2300\x00\x00\x00\x00',
      b'\x018966312Q8000\x00\x00\x00\x00',
      b'\x018966312R0000\x00\x00\x00\x00',
      b'\x018966312R0100\x00\x00\x00\x00',
      b'\x018966312R0200\x00\x00\x00\x00',
      b'\x018966312R1000\x00\x00\x00\x00',
      b'\x018966312R1100\x00\x00\x00\x00',
      b'\x018966312R3100\x00\x00\x00\x00',
      b'\x018966312S5000\x00\x00\x00\x00',
      b'\x018966312S7000\x00\x00\x00\x00',
      b'\x018966312W3000\x00\x00\x00\x00',
      b'\x018966312W9000\x00\x00\x00\x00',
      b'\x01896637644000\x00\x00\x00\x00',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'\x0230A10000\x00\x00\x00\x00\x00\x00\x00\x00A0202000\x00\x00\x00\x00\x00\x00\x00\x00',
      b'\x0230A11000\x00\x00\x00\x00\x00\x00\x00\x00A0202000\x00\x00\x00\x00\x00\x00\x00\x00',
      b'\x0230ZN4000\x00\x00\x00\x00\x00\x00\x00\x00A0202000\x00\x00\x00\x00\x00\x00\x00\x00',
      b'\x03312K7000\x00\x00\x00\x00\x00\x00\x00\x00A0202000\x00\x00\x00\x00\x00\x00\x00\x00895231203402\x00\x00\x00\x00',
      b'\x03312M3000\x00\x00\x00\x00\x00\x00\x00\x00A0202000\x00\x00\x00\x00\x00\x00\x00\x00895231203402\x00\x00\x00\x00',
      b'\x03312N6000\x00\x00\x00\x00\x00\x00\x00\x00A0202000\x00\x00\x00\x00\x00\x00\x00\x00895231203202\x00\x00\x00\x00',
      b'\x03312N6000\x00\x00\x00\x00\x00\x00\x00\x00A0202000\x00\x00\x00\x00\x00\x00\x00\x00895231203302\x00\x00\x00\x00',
      b'\x03312N6000\x00\x00\x00\x00\x00\x00\x00\x00A0202000\x00\x00\x00\x00\x00\x00\x00\x00895231203402\x00\x00\x00\x00',
      b'\x03312N6100\x00\x00\x00\x00\x00\x00\x00\x00A0202000\x00\x00\x00\x00\x00\x00\x00\x00895231203302\x00\x00\x00\x00',
      b'\x03312N6100\x00\x00\x00\x00\x00\x00\x00\x00A0202000\x00\x00\x00\x00\x00\x00\x00\x00895231203402\x00\x00\x00\x00',
      b'\x03312N6200\x00\x00\x00\x00\x00\x00\x00\x00A0202000\x00\x00\x00\x00\x00\x00\x00\x00895231203202\x00\x00\x00\x00',
      b'\x03312N6200\x00\x00\x00\x00\x00\x00\x00\x00A0202000\x00\x00\x00\x00\x00\x00\x00\x00895231203302\x00\x00\x00\x00',
      b'\x03312N6200\x00\x00\x00\x00\x00\x00\x00\x00A0202000\x00\x00\x00\x00\x00\x00\x00\x00895231203402\x00\x00\x00\x00',
      b'\x02312K4000\x00\x00\x00\x00\x00\x00\x00\x00A0202000\x00\x00\x00\x00\x00\x00\x00\x00',
      b'\x02312U5000\x00\x00\x00\x00\x00\x00\x00\x00A0202000\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.eps, 0x7a1, None): [
      b'\x018965B12350\x00\x00\x00\x00\x00\x00',
      b'\x018965B12470\x00\x00\x00\x00\x00\x00',
      b'\x018965B12490\x00\x00\x00\x00\x00\x00',
      b'\x018965B12500\x00\x00\x00\x00\x00\x00',
      b'\x018965B12520\x00\x00\x00\x00\x00\x00',
      b'\x018965B12530\x00\x00\x00\x00\x00\x00',
      b'\x018965B1255000\x00\x00\x00\x00',
      b'8965B12361\x00\x00\x00\x00\x00\x00',
      b'8965B16011\x00\x00\x00\x00\x00\x00',
      b'8965B76012\x00\x00\x00\x00\x00\x00',
      b'\x018965B12510\x00\x00\x00\x00\x00\x00',
      b'\x018965B1256000\x00\x00\x00\x00',
    ],
    (Ecu.abs, 0x7b0, None): [
      b'\x01F152602280\x00\x00\x00\x00\x00\x00',
      b'\x01F152602560\x00\x00\x00\x00\x00\x00',
      b'\x01F152602590\x00\x00\x00\x00\x00\x00',
      b'\x01F152602650\x00\x00\x00\x00\x00\x00',
      b"\x01F15260A010\x00\x00\x00\x00\x00\x00",
      b'\x01F15260A050\x00\x00\x00\x00\x00\x00',
      b'\x01F152612641\x00\x00\x00\x00\x00\x00',
      b'\x01F152612651\x00\x00\x00\x00\x00\x00',
      b'\x01F152612B10\x00\x00\x00\x00\x00\x00',
      b'\x01F152612B51\x00\x00\x00\x00\x00\x00',
      b'\x01F152612B60\x00\x00\x00\x00\x00\x00',
      b'\x01F152612B61\x00\x00\x00\x00\x00\x00',
      b'\x01F152612B62\x00\x00\x00\x00\x00\x00',
      b'\x01F152612B70\x00\x00\x00\x00\x00\x00',
      b'\x01F152612B71\x00\x00\x00\x00\x00\x00',
      b'\x01F152612B81\x00\x00\x00\x00\x00\x00',
      b'\x01F152612B90\x00\x00\x00\x00\x00\x00',
      b'\x01F152612C00\x00\x00\x00\x00\x00\x00',
      b'\x01F152612862\x00\x00\x00\x00\x00\x00',
      b'\x01F152612B91\x00\x00\x00\x00\x00\x00',
      b'\x01F15260A070\x00\x00\x00\x00\x00\x00',
      b'\x01F152676250\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.fwdRadar, 0x750, 0xf): [
      b'\x018821F3301100\x00\x00\x00\x00',
      b'\x018821F3301200\x00\x00\x00\x00',
      b'\x018821F3301300\x00\x00\x00\x00',
      b'\x018821F3301400\x00\x00\x00\x00',
    ],
    (Ecu.fwdCamera, 0x750, 0x6d): [
      b'\x028646F12010D0\x00\x00\x00\x008646G26011A0\x00\x00\x00\x00',
      b'\x028646F1201100\x00\x00\x00\x008646G26011A0\x00\x00\x00\x00',
      b'\x028646F1201200\x00\x00\x00\x008646G26011A0\x00\x00\x00\x00',
      b'\x028646F1201300\x00\x00\x00\x008646G2601400\x00\x00\x00\x00',
      b'\x028646F1201400\x00\x00\x00\x008646G2601500\x00\x00\x00\x00',
      b'\x028646F1202000\x00\x00\x00\x008646G2601200\x00\x00\x00\x00',
      b'\x028646F1202100\x00\x00\x00\x008646G2601400\x00\x00\x00\x00',
      b'\x028646F1202200\x00\x00\x00\x008646G2601500\x00\x00\x00\x00',
      b'\x028646F1601100\x00\x00\x00\x008646G2601400\x00\x00\x00\x00',
      b'\x028646F1601300\x00\x00\x00\x008646G2601400\x00\x00\x00\x00',
    ],
  },
  CAR.COROLLAH_TSS2: {
    (Ecu.engine, 0x700, None): [
      b'\x01896630ZJ1000\x00\x00\x00\x00',
      b'\x01896630ZU8000\x00\x00\x00\x00',
      b'\x01896637621000\x00\x00\x00\x00',
      b'\x01896637623000\x00\x00\x00\x00',
      b'\x01896637624000\x00\x00\x00\x00',
      b'\x01896637626000\x00\x00\x00\x00',
      b'\x01896637639000\x00\x00\x00\x00',
      b'\x01896637648000\x00\x00\x00\x00',
      b'\x01896637643000\x00\x00\x00\x00',
      b'\x02896630A07000\x00\x00\x00\x008966A4703000\x00\x00\x00\x00',
      b'\x02896630A21000\x00\x00\x00\x008966A4703000\x00\x00\x00\x00',
      b'\x02896630ZJ5000\x00\x00\x00\x008966A4703000\x00\x00\x00\x00',
      b'\x02896630ZK8000\x00\x00\x00\x008966A4703000\x00\x00\x00\x00',
      b'\x02896630ZN8000\x00\x00\x00\x008966A4703000\x00\x00\x00\x00',
      b'\x02896630ZQ3000\x00\x00\x00\x008966A4703000\x00\x00\x00\x00',
      b'\x02896630ZR2000\x00\x00\x00\x008966A4703000\x00\x00\x00\x00',
      b'\x02896630ZT8000\x00\x00\x00\x008966A4703000\x00\x00\x00\x00',
      b'\x02896630ZT9000\x00\x00\x00\x008966A4703000\x00\x00\x00\x00',
      b'\x02896630ZZ0000\x00\x00\x00\x008966A4703000\x00\x00\x00\x00',
      b'\x028966312K6000\x00\x00\x00\x008966A4703000\x00\x00\x00\x00',
      b'\x028966312L0000\x00\x00\x00\x008966A4703000\x00\x00\x00\x00',
      b'\x028966312Q3000\x00\x00\x00\x008966A4703000\x00\x00\x00\x00',
      b'\x028966312Q3100\x00\x00\x00\x008966A4703000\x00\x00\x00\x00',
      b'\x028966312Q4000\x00\x00\x00\x008966A4703000\x00\x00\x00\x00',
      b'\x038966312L7000\x00\x00\x00\x008966A4703000\x00\x00\x00\x00897CF1205001\x00\x00\x00\x00',
      b'\x038966312N1000\x00\x00\x00\x008966A4703000\x00\x00\x00\x00897CF1203001\x00\x00\x00\x00',
      b'\x038966312T3000\x00\x00\x00\x008966A4703000\x00\x00\x00\x00897CF1205001\x00\x00\x00\x00',
    ],
    (Ecu.eps, 0x7a1, None): [
      b'8965B12361\x00\x00\x00\x00\x00\x00',
      b'8965B12451\x00\x00\x00\x00\x00\x00',
      b'8965B16011\x00\x00\x00\x00\x00\x00',
      b'8965B16101\x00\x00\x00\x00\x00\x00',
      b'8965B16170\x00\x00\x00\x00\x00\x00',
      b'8965B76012\x00\x00\x00\x00\x00\x00',
      b'8965B76050\x00\x00\x00\x00\x00\x00',
      b'8965B76091\x00\x00\x00\x00\x00\x00',
      b'\x018965B12350\x00\x00\x00\x00\x00\x00',
      b'\x018965B12470\x00\x00\x00\x00\x00\x00',
      b'\x018965B12490\x00\x00\x00\x00\x00\x00',
      b'\x018965B12500\x00\x00\x00\x00\x00\x00',
      b'\x018965B12510\x00\x00\x00\x00\x00\x00',
      b'\x018965B12520\x00\x00\x00\x00\x00\x00',
      b'\x018965B12530\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.abs, 0x7b0, None): [
      b'F152612590\x00\x00\x00\x00\x00\x00',
      b'F152612691\x00\x00\x00\x00\x00\x00',
      b'F152612692\x00\x00\x00\x00\x00\x00',
      b'F152612700\x00\x00\x00\x00\x00\x00',
      b'F152612710\x00\x00\x00\x00\x00\x00',
      b'F152612790\x00\x00\x00\x00\x00\x00',
      b'F152612800\x00\x00\x00\x00\x00\x00',
      b'F152612820\x00\x00\x00\x00\x00\x00',
      b'F152612840\x00\x00\x00\x00\x00\x00',
      b'F152612842\x00\x00\x00\x00\x00\x00',
      b'F152612890\x00\x00\x00\x00\x00\x00',
      b'F152612A00\x00\x00\x00\x00\x00\x00',
      b'F152612A10\x00\x00\x00\x00\x00\x00',
      b'F152612D00\x00\x00\x00\x00\x00\x00',
      b'F152616011\x00\x00\x00\x00\x00\x00',
      b'F152616060\x00\x00\x00\x00\x00\x00',
      b'F152616030\x00\x00\x00\x00\x00\x00',
      b'F152642540\x00\x00\x00\x00\x00\x00',
      b'F152676293\x00\x00\x00\x00\x00\x00',
      b'F152676303\x00\x00\x00\x00\x00\x00',
      b'F152676304\x00\x00\x00\x00\x00\x00',
      b'F152676371\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.fwdRadar, 0x750, 0xf): [
      b'\x018821F3301100\x00\x00\x00\x00',
      b'\x018821F3301200\x00\x00\x00\x00',
      b'\x018821F3301300\x00\x00\x00\x00',
      b'\x018821F3301400\x00\x00\x00\x00',
      b'\x018821F6201400\x00\x00\x00\x00',
    ],
    (Ecu.fwdCamera, 0x750, 0x6d): [
      b'\x028646F12010D0\x00\x00\x00\x008646G26011A0\x00\x00\x00\x00',
      b'\x028646F1201100\x00\x00\x00\x008646G26011A0\x00\x00\x00\x00',
      b'\x028646F1201300\x00\x00\x00\x008646G2601400\x00\x00\x00\x00',
      b'\x028646F1201400\x00\x00\x00\x008646G2601500\x00\x00\x00\x00',
      b'\x028646F1202000\x00\x00\x00\x008646G2601200\x00\x00\x00\x00',
      b'\x028646F1202100\x00\x00\x00\x008646G2601400\x00\x00\x00\x00',
      b'\x028646F1202200\x00\x00\x00\x008646G2601500\x00\x00\x00\x00',
      b'\x028646F1601100\x00\x00\x00\x008646G2601400\x00\x00\x00\x00',
      b'\x028646F1601200\x00\x00\x00\x008646G2601400\x00\x00\x00\x00',
      b"\x028646F1601300\x00\x00\x00\x008646G2601400\x00\x00\x00\x00",
      b'\x028646F4203400\x00\x00\x00\x008646G2601200\x00\x00\x00\x00',
      b'\x028646F76020C0\x00\x00\x00\x008646G26011A0\x00\x00\x00\x00',
      b'\x028646F7603100\x00\x00\x00\x008646G2601200\x00\x00\x00\x00',
      b'\x028646F7603200\x00\x00\x00\x008646G2601400\x00\x00\x00\x00',
      b'\x028646F7605100\x00\x00\x00\x008646G3304000\x00\x00\x00\x00',
    ],
  },
  CAR.HIGHLANDER: {
    (Ecu.engine, 0x700, None): [
      b'\x01896630E09000\x00\x00\x00\x00',
      b'\x01896630E43000\x00\x00\x00\x00',
      b'\x01896630E43100\x00\x00\x00\x00',
      b'\x01896630E43200\x00\x00\x00\x00',
      b'\x01896630E44200\x00\x00\x00\x00',
      b'\x01896630E45000\x00\x00\x00\x00',
      b'\x01896630E45100\x00\x00\x00\x00',
      b'\x01896630E45200\x00\x00\x00\x00',
      b'\x01896630E46000\x00\x00\x00\x00',
      b'\x01896630E46200\x00\x00\x00\x00',
      b'\x01896630E74000\x00\x00\x00\x00',
      b'\x01896630E75000\x00\x00\x00\x00',
      b'\x01896630E76000\x00\x00\x00\x00',
      b'\x01896630E77000\x00\x00\x00\x00',
      b'\x01896630E83000\x00\x00\x00\x00',
      b'\x01896630E84000\x00\x00\x00\x00',
      b'\x01896630E85000\x00\x00\x00\x00',
      b'\x01896630E86000\x00\x00\x00\x00',
      b'\x01896630E88000\x00\x00\x00\x00',
      b'\x01896630EA0000\x00\x00\x00\x00',
    ],
    (Ecu.eps, 0x7a1, None): [
      b'8965B48140\x00\x00\x00\x00\x00\x00',
      b'8965B48150\x00\x00\x00\x00\x00\x00',
      b'8965B48210\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.abs, 0x7b0, None): [b'F15260E011\x00\x00\x00\x00\x00\x00'],
    (Ecu.dsu, 0x791, None): [
      b'881510E01100\x00\x00\x00\x00',
      b'881510E01200\x00\x00\x00\x00',
    ],
    (Ecu.fwdRadar, 0x750, 0xf): [
      b'8821F4702100\x00\x00\x00\x00',
      b'8821F4702300\x00\x00\x00\x00',
    ],
    (Ecu.fwdCamera, 0x750, 0x6d): [
      b'8646F0E01200\x00\x00\x00\x00',
      b'8646F0E01300\x00\x00\x00\x00',
    ],
  },
  CAR.HIGHLANDERH: {
    (Ecu.eps, 0x7a1, None): [
      b'8965B48160\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.abs, 0x7b0, None): [
      b'F152648541\x00\x00\x00\x00\x00\x00',
      b'F152648542\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'\x0230E40000\x00\x00\x00\x00\x00\x00\x00\x00A4802000\x00\x00\x00\x00\x00\x00\x00\x00',
      b'\x0230E40100\x00\x00\x00\x00\x00\x00\x00\x00A4802000\x00\x00\x00\x00\x00\x00\x00\x00',
      b'\x0230EA2000\x00\x00\x00\x00\x00\x00\x00\x00A4802000\x00\x00\x00\x00\x00\x00\x00\x00',
      b'\x0230EA2100\x00\x00\x00\x00\x00\x00\x00\x00A4802000\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.fwdRadar, 0x750, 0xf): [
      b'8821F4702100\x00\x00\x00\x00',
      b'8821F4702300\x00\x00\x00\x00',
    ],
    (Ecu.fwdCamera, 0x750, 0x6d): [
      b'8646F0E01200\x00\x00\x00\x00',
      b'8646F0E01300\x00\x00\x00\x00',
    ],
  },
  CAR.HIGHLANDER_TSS2: {
    (Ecu.eps, 0x7a1, None): [
      b'8965B48241\x00\x00\x00\x00\x00\x00',
      b'8965B48310\x00\x00\x00\x00\x00\x00',
      b'8965B48320\x00\x00\x00\x00\x00\x00',
      b'8965B48400\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.abs, 0x7b0, None): [
      b'\x01F15260E051\x00\x00\x00\x00\x00\x00',
      b'\x01F15260E061\x00\x00\x00\x00\x00\x00',
      b'\x01F15260E110\x00\x00\x00\x00\x00\x00',
      b'\x01F15260E170\x00\x00\x00\x00\x00\x00',
      b'\x01F15260E05300\x00\x00\x00\x00',
    ],
    (Ecu.engine, 0x700, None): [
      b'\x01896630E62100\x00\x00\x00\x00',
      b'\x01896630E62200\x00\x00\x00\x00',
      b'\x01896630E64100\x00\x00\x00\x00',
      b'\x01896630E64200\x00\x00\x00\x00',
      b'\x01896630E64400\x00\x00\x00\x00',
      b'\x01896630EB1000\x00\x00\x00\x00',
      b'\x01896630EB1100\x00\x00\x00\x00',
      b'\x01896630EB1200\x00\x00\x00\x00',
      b'\x01896630EB2000\x00\x00\x00\x00',
      b'\x01896630EB2100\x00\x00\x00\x00',
      b'\x01896630EB2200\x00\x00\x00\x00',
      b'\x01896630EC4000\x00\x00\x00\x00',
      b'\x01896630ED9000\x00\x00\x00\x00',
      b'\x01896630ED9100\x00\x00\x00\x00',
      b'\x01896630EE1000\x00\x00\x00\x00',
      b'\x01896630EE1100\x00\x00\x00\x00',
      b'\x01896630EG3000\x00\x00\x00\x00',
      b'\x01896630EG5000\x00\x00\x00\x00',
    ],
    (Ecu.fwdRadar, 0x750, 0xf): [
      b'\x018821F3301400\x00\x00\x00\x00',
      b'\x018821F6201200\x00\x00\x00\x00',
      b'\x018821F6201300\x00\x00\x00\x00',
    ],
    (Ecu.fwdCamera, 0x750, 0x6d): [
      b'\x028646F0E02100\x00\x00\x00\x008646G2601200\x00\x00\x00\x00',
      b'\x028646F4803000\x00\x00\x00\x008646G5301200\x00\x00\x00\x00',
      b'\x028646F4803000\x00\x00\x00\x008646G3304000\x00\x00\x00\x00',
      b'\x028646F4803200\x00\x00\x00\x008646G3304000\x00\x00\x00\x00',
    ],
  },
  CAR.HIGHLANDERH_TSS2: {
    (Ecu.eps, 0x7a1, None): [
      b'8965B48241\x00\x00\x00\x00\x00\x00',
      b'8965B48310\x00\x00\x00\x00\x00\x00',
      b'8965B48400\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.abs, 0x7b0, None): [
      b'\x01F15264872300\x00\x00\x00\x00',
      b'\x01F15264872400\x00\x00\x00\x00',
      b'\x01F15264872500\x00\x00\x00\x00',
      b'\x01F15264872600\x00\x00\x00\x00',
      b'\x01F15264873500\x00\x00\x00\x00',
      b'\x01F152648C6300\x00\x00\x00\x00',
      b'\x01F152648J4000\x00\x00\x00\x00',
      b'\x01F152648J5000\x00\x00\x00\x00',
      b'\x01F152648J6000\x00\x00\x00\x00',
      b'\x01F15264872700\x00\x00\x00\x00',
    ],
    (Ecu.engine, 0x700, None): [
      b'\x01896630E67000\x00\x00\x00\x00',
      b'\x01896630EA1000\x00\x00\x00\x00',
      b'\x01896630EE4000\x00\x00\x00\x00',
      b'\x01896630EE4100\x00\x00\x00\x00',
      b'\x01896630EE5000\x00\x00\x00\x00',
      b'\x01896630EE6000\x00\x00\x00\x00',
      b'\x01896630EF8000\x00\x00\x00\x00',
      b'\x02896630E66000\x00\x00\x00\x00897CF4801001\x00\x00\x00\x00',
      b'\x02896630E66100\x00\x00\x00\x00897CF4801001\x00\x00\x00\x00',
      b'\x02896630EB3000\x00\x00\x00\x00897CF4801001\x00\x00\x00\x00',
      b'\x02896630EB3100\x00\x00\x00\x00897CF4801001\x00\x00\x00\x00',
    ],
    (Ecu.fwdRadar, 0x750, 0xf): [
      b'\x018821F3301400\x00\x00\x00\x00',
      b'\x018821F6201200\x00\x00\x00\x00',
      b'\x018821F6201300\x00\x00\x00\x00',
    ],
    (Ecu.fwdCamera, 0x750, 0x6d): [
      b'\x028646F0E02100\x00\x00\x00\x008646G2601200\x00\x00\x00\x00',
      b'\x028646F4803000\x00\x00\x00\x008646G5301200\x00\x00\x00\x00',
      b'\x028646F4803000\x00\x00\x00\x008646G3304000\x00\x00\x00\x00',
      b'\x028646F4803200\x00\x00\x00\x008646G3304000\x00\x00\x00\x00',
    ],
  },
  CAR.LEXUS_IS: {
    (Ecu.engine, 0x700, None): [
      b'\x018966353M7000\x00\x00\x00\x00',
      b'\x018966353M7100\x00\x00\x00\x00',
      b'\x018966353Q2000\x00\x00\x00\x00',
      b'\x018966353Q2300\x00\x00\x00\x00',
      b'\x018966353Q4000\x00\x00\x00\x00',
      b'\x018966353R1100\x00\x00\x00\x00',
      b'\x018966353R7100\x00\x00\x00\x00',
      b'\x018966353R8100\x00\x00\x00\x00',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'\x0232480000\x00\x00\x00\x00\x00\x00\x00\x00A4701000\x00\x00\x00\x00\x00\x00\x00\x00',
      b'\x02353P7000\x00\x00\x00\x00\x00\x00\x00\x00530J5000\x00\x00\x00\x00\x00\x00\x00\x00',
      b'\x02353P9000\x00\x00\x00\x00\x00\x00\x00\x00553C1000\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.abs, 0x7b0, None): [
      b'F152653300\x00\x00\x00\x00\x00\x00',
      b'F152653301\x00\x00\x00\x00\x00\x00',
      b'F152653310\x00\x00\x00\x00\x00\x00',
      b'F152653330\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.dsu, 0x791, None): [
      b'881515306200\x00\x00\x00\x00',
      b'881515306400\x00\x00\x00\x00',
      b'881515306500\x00\x00\x00\x00',
      b'881515307400\x00\x00\x00\x00',
    ],
    (Ecu.eps, 0x7a1, None): [
      b'8965B53270\x00\x00\x00\x00\x00\x00',
      b'8965B53271\x00\x00\x00\x00\x00\x00',
      b'8965B53280\x00\x00\x00\x00\x00\x00',
      b'8965B53281\x00\x00\x00\x00\x00\x00',
      b'8965B53311\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.fwdRadar, 0x750, 0xf): [
      b'8821F4702300\x00\x00\x00\x00',
      b'8821F4702100\x00\x00\x00\x00',
    ],
    (Ecu.fwdCamera, 0x750, 0x6d): [
      b'8646F5301101\x00\x00\x00\x00',
      b'8646F5301200\x00\x00\x00\x00',
      b'8646F5301300\x00\x00\x00\x00',
      b'8646F5301400\x00\x00\x00\x00',
    ],
  },
  CAR.LEXUS_IS_TSS2: {
    (Ecu.engine, 0x700, None): [
      b'\x018966353S1000\x00\x00\x00\x00',
    ],
    (Ecu.abs, 0x7b0, None): [
      b'\x01F15265342000\x00\x00\x00\x00',
    ],
    (Ecu.eps, 0x7a1, None): [
      b'8965B53450\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.fwdRadar, 0x750, 0xf): [
      b'\x018821F6201300\x00\x00\x00\x00',
    ],
    (Ecu.fwdCamera, 0x750, 0x6d): [
      b'\x028646F5303400\x00\x00\x00\x008646G3304000\x00\x00\x00\x00',
    ],
  },
  CAR.PRIUS: {
    (Ecu.engine, 0x700, None): [
      b'\x02896634761000\x00\x00\x00\x008966A4703000\x00\x00\x00\x00',
      b'\x02896634761100\x00\x00\x00\x008966A4703000\x00\x00\x00\x00',
      b'\x02896634761200\x00\x00\x00\x008966A4703000\x00\x00\x00\x00',
      b'\x02896634762000\x00\x00\x00\x008966A4703000\x00\x00\x00\x00',
      b'\x02896634763000\x00\x00\x00\x008966A4703000\x00\x00\x00\x00',
      b'\x02896634763100\x00\x00\x00\x008966A4703000\x00\x00\x00\x00',
      b'\x02896634765000\x00\x00\x00\x008966A4703000\x00\x00\x00\x00',
      b'\x02896634765100\x00\x00\x00\x008966A4703000\x00\x00\x00\x00',
      b'\x02896634769000\x00\x00\x00\x008966A4703000\x00\x00\x00\x00',
      b'\x02896634769100\x00\x00\x00\x008966A4703000\x00\x00\x00\x00',
      b'\x02896634769200\x00\x00\x00\x008966A4703000\x00\x00\x00\x00',
      b'\x02896634770000\x00\x00\x00\x008966A4703000\x00\x00\x00\x00',
      b'\x02896634774000\x00\x00\x00\x008966A4703000\x00\x00\x00\x00',
      b'\x02896634774100\x00\x00\x00\x008966A4703000\x00\x00\x00\x00',
      b'\x02896634774200\x00\x00\x00\x008966A4703000\x00\x00\x00\x00',
      b'\x02896634782000\x00\x00\x00\x008966A4703000\x00\x00\x00\x00',
      b'\x02896634784000\x00\x00\x00\x008966A4703000\x00\x00\x00\x00',
      b'\x028966347A0000\x00\x00\x00\x008966A4703000\x00\x00\x00\x00',
      b'\x028966347A5000\x00\x00\x00\x008966A4703000\x00\x00\x00\x00',
      b'\x028966347A8000\x00\x00\x00\x008966A4703000\x00\x00\x00\x00',
      b'\x028966347B0000\x00\x00\x00\x008966A4703000\x00\x00\x00\x00',
      b'\x03896634759100\x00\x00\x00\x008966A4703000\x00\x00\x00\x00897CF4701003\x00\x00\x00\x00',
      b'\x03896634759200\x00\x00\x00\x008966A4703000\x00\x00\x00\x00897CF4701003\x00\x00\x00\x00',
      b'\x03896634759200\x00\x00\x00\x008966A4703000\x00\x00\x00\x00897CF4701004\x00\x00\x00\x00',
      b'\x03896634759300\x00\x00\x00\x008966A4703000\x00\x00\x00\x00897CF4701004\x00\x00\x00\x00',
      b'\x03896634760000\x00\x00\x00\x008966A4703000\x00\x00\x00\x00897CF4701002\x00\x00\x00\x00',
      b'\x03896634760000\x00\x00\x00\x008966A4703000\x00\x00\x00\x00897CF4701003\x00\x00\x00\x00',
      b'\x03896634760000\x00\x00\x00\x008966A4703000\x00\x00\x00\x00897CF4701004\x00\x00\x00\x00',
      b'\x03896634760100\x00\x00\x00\x008966A4703000\x00\x00\x00\x00897CF4701003\x00\x00\x00\x00',
      b'\x03896634760200\x00\x00\x00\x008966A4703000\x00\x00\x00\x00897CF4701003\x00\x00\x00\x00',
      b'\x03896634760200\x00\x00\x00\x008966A4703000\x00\x00\x00\x00897CF4701004\x00\x00\x00\x00',
      b'\x03896634760300\x00\x00\x00\x008966A4703000\x00\x00\x00\x00897CF4701004\x00\x00\x00\x00',
      b'\x03896634768000\x00\x00\x00\x008966A4703000\x00\x00\x00\x00897CF4703001\x00\x00\x00\x00',
      b'\x03896634768000\x00\x00\x00\x008966A4703000\x00\x00\x00\x00897CF4703002\x00\x00\x00\x00',
      b'\x03896634768100\x00\x00\x00\x008966A4703000\x00\x00\x00\x00897CF4703002\x00\x00\x00\x00',
      b'\x03896634785000\x00\x00\x00\x008966A4703000\x00\x00\x00\x00897CF4705001\x00\x00\x00\x00',
      b'\x03896634785000\x00\x00\x00\x008966A4703000\x00\x00\x00\x00897CF4710001\x00\x00\x00\x00',
      b'\x03896634786000\x00\x00\x00\x008966A4703000\x00\x00\x00\x00897CF4705001\x00\x00\x00\x00',
      b'\x03896634786000\x00\x00\x00\x008966A4703000\x00\x00\x00\x00897CF4710001\x00\x00\x00\x00',
      b'\x03896634789000\x00\x00\x00\x008966A4703000\x00\x00\x00\x00897CF4703002\x00\x00\x00\x00',
      b'\x038966347A3000\x00\x00\x00\x008966A4703000\x00\x00\x00\x00897CF4701003\x00\x00\x00\x00',
      b'\x038966347A3000\x00\x00\x00\x008966A4703000\x00\x00\x00\x00897CF4707001\x00\x00\x00\x00',
      b'\x038966347B6000\x00\x00\x00\x008966A4703000\x00\x00\x00\x00897CF4710001\x00\x00\x00\x00',
      b'\x038966347B7000\x00\x00\x00\x008966A4703000\x00\x00\x00\x00897CF4710001\x00\x00\x00\x00',
    ],
    (Ecu.eps, 0x7a1, None): [
      b'8965B47021\x00\x00\x00\x00\x00\x00',
      b'8965B47022\x00\x00\x00\x00\x00\x00',
      b'8965B47023\x00\x00\x00\x00\x00\x00',
      b'8965B47050\x00\x00\x00\x00\x00\x00',
      b'8965B47060\x00\x00\x00\x00\x00\x00',  # This is the EPS with good angle sensor
    ],
    (Ecu.abs, 0x7b0, None): [
      b'F152647290\x00\x00\x00\x00\x00\x00',
      b'F152647300\x00\x00\x00\x00\x00\x00',
      b'F152647310\x00\x00\x00\x00\x00\x00',
      b'F152647414\x00\x00\x00\x00\x00\x00',
      b'F152647415\x00\x00\x00\x00\x00\x00',
      b'F152647416\x00\x00\x00\x00\x00\x00',
      b'F152647417\x00\x00\x00\x00\x00\x00',
      b'F152647470\x00\x00\x00\x00\x00\x00',
      b'F152647490\x00\x00\x00\x00\x00\x00',
      b'F152647682\x00\x00\x00\x00\x00\x00',
      b'F152647683\x00\x00\x00\x00\x00\x00',
      b'F152647684\x00\x00\x00\x00\x00\x00',
      b'F152647862\x00\x00\x00\x00\x00\x00',
      b'F152647863\x00\x00\x00\x00\x00\x00',
      b'F152647864\x00\x00\x00\x00\x00\x00',
      b'F152647865\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.dsu, 0x791, None): [
      b'881514702300\x00\x00\x00\x00',
      b'881514702400\x00\x00\x00\x00',
      b'881514703100\x00\x00\x00\x00',
      b'881514704100\x00\x00\x00\x00',
      b'881514706000\x00\x00\x00\x00',
      b'881514706100\x00\x00\x00\x00',
    ],
    (Ecu.fwdRadar, 0x750, 0xf): [
      b'8821F4702000\x00\x00\x00\x00',
      b'8821F4702100\x00\x00\x00\x00',
      b'8821F4702300\x00\x00\x00\x00',
    ],
    (Ecu.fwdCamera, 0x750, 0x6d): [
      b'8646F4701300\x00\x00\x00\x00',
      b'8646F4702001\x00\x00\x00\x00',
      b'8646F4702100\x00\x00\x00\x00',
      b'8646F4702200\x00\x00\x00\x00',
      b'8646F4705000\x00\x00\x00\x00',
      b'8646F4705200\x00\x00\x00\x00',
    ],
  },
  CAR.PRIUS_V: {
    (Ecu.abs, 0x7b0, None): [
      b'F152647280\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'\x0234781000\x00\x00\x00\x00\x00\x00\x00\x00A4701000\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.dsu, 0x791, None): [
      b'881514705100\x00\x00\x00\x00',
    ],
    (Ecu.fwdRadar, 0x750, 0xf): [
      b'8821F4702300\x00\x00\x00\x00',
    ],
    (Ecu.fwdCamera, 0x750, 0x6d): [
      b'8646F4703300\x00\x00\x00\x00',
    ],
  },
  CAR.RAV4: {
    (Ecu.engine, 0x7e0, None): [
      b'\x02342Q1000\x00\x00\x00\x00\x00\x00\x00\x0054212000\x00\x00\x00\x00\x00\x00\x00\x00',
      b'\x02342Q1100\x00\x00\x00\x00\x00\x00\x00\x0054212000\x00\x00\x00\x00\x00\x00\x00\x00',
      b'\x02342Q1200\x00\x00\x00\x00\x00\x00\x00\x0054212000\x00\x00\x00\x00\x00\x00\x00\x00',
      b'\x02342Q1300\x00\x00\x00\x00\x00\x00\x00\x0054212000\x00\x00\x00\x00\x00\x00\x00\x00',
      b'\x02342Q2000\x00\x00\x00\x00\x00\x00\x00\x0054213000\x00\x00\x00\x00\x00\x00\x00\x00',
      b'\x02342Q2100\x00\x00\x00\x00\x00\x00\x00\x0054213000\x00\x00\x00\x00\x00\x00\x00\x00',
      b'\x02342Q2200\x00\x00\x00\x00\x00\x00\x00\x0054213000\x00\x00\x00\x00\x00\x00\x00\x00',
      b'\x02342Q4000\x00\x00\x00\x00\x00\x00\x00\x0054215000\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.eps, 0x7a1, None): [
      b'8965B42063\x00\x00\x00\x00\x00\x00',
      b'8965B42073\x00\x00\x00\x00\x00\x00',
      b'8965B42082\x00\x00\x00\x00\x00\x00',
      b'8965B42083\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.abs, 0x7b0, None): [
      b'F15260R102\x00\x00\x00\x00\x00\x00',
      b'F15260R103\x00\x00\x00\x00\x00\x00',
      b'F152642493\x00\x00\x00\x00\x00\x00',
      b'F152642492\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.dsu, 0x791, None): [
      b'881514201200\x00\x00\x00\x00',
      b'881514201300\x00\x00\x00\x00',
      b'881514201400\x00\x00\x00\x00',
    ],
    (Ecu.fwdRadar, 0x750, 0xf): [
      b'8821F4702000\x00\x00\x00\x00',
      b'8821F4702100\x00\x00\x00\x00',
      b'8821F4702300\x00\x00\x00\x00',
    ],
    (Ecu.fwdCamera, 0x750, 0x6d): [
      b'8646F4201100\x00\x00\x00\x00',
      b'8646F4201200\x00\x00\x00\x00',
      b'8646F4202001\x00\x00\x00\x00',
      b'8646F4202100\x00\x00\x00\x00',
      b'8646F4204000\x00\x00\x00\x00',
    ],
  },
  CAR.RAV4H: {
    (Ecu.engine, 0x7e0, None): [
      b'\x02342N9000\x00\x00\x00\x00\x00\x00\x00\x00A4701000\x00\x00\x00\x00\x00\x00\x00\x00',
      b'\x02342N9100\x00\x00\x00\x00\x00\x00\x00\x00A4701000\x00\x00\x00\x00\x00\x00\x00\x00',
      b'\x02342P0000\x00\x00\x00\x00\x00\x00\x00\x00A4701000\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.eps, 0x7a1, None): [
      b'8965B42102\x00\x00\x00\x00\x00\x00',
      b'8965B42103\x00\x00\x00\x00\x00\x00',
      b'8965B42112\x00\x00\x00\x00\x00\x00',
      b'8965B42162\x00\x00\x00\x00\x00\x00',
      b'8965B42163\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.abs, 0x7b0, None): [
      b'F152642090\x00\x00\x00\x00\x00\x00',
      b'F152642110\x00\x00\x00\x00\x00\x00',
      b'F152642120\x00\x00\x00\x00\x00\x00',
      b'F152642400\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.dsu, 0x791, None): [
      b'881514202200\x00\x00\x00\x00',
      b'881514202300\x00\x00\x00\x00',
      b'881514202400\x00\x00\x00\x00',
    ],
    (Ecu.fwdRadar, 0x750, 0xf): [
      b'8821F4702000\x00\x00\x00\x00',
      b'8821F4702100\x00\x00\x00\x00',
      b'8821F4702300\x00\x00\x00\x00',
    ],
    (Ecu.fwdCamera, 0x750, 0x6d): [
      b'8646F4201100\x00\x00\x00\x00',
      b'8646F4201200\x00\x00\x00\x00',
      b'8646F4202001\x00\x00\x00\x00',
      b'8646F4202100\x00\x00\x00\x00',
      b'8646F4204000\x00\x00\x00\x00',
    ],
  },
  CAR.RAV4_TSS2: {
    (Ecu.engine, 0x700, None): [
      b'\x01896630R58000\x00\x00\x00\x00',
      b'\x01896630R58100\x00\x00\x00\x00',
      b'\x018966342E2000\x00\x00\x00\x00',
      b'\x018966342M8000\x00\x00\x00\x00',
      b'\x018966342S9000\x00\x00\x00\x00',
      b'\x018966342T1000\x00\x00\x00\x00',
      b'\x018966342T6000\x00\x00\x00\x00',
      b'\x018966342T9000\x00\x00\x00\x00',
      b'\x018966342U4000\x00\x00\x00\x00',
      b'\x018966342U4100\x00\x00\x00\x00',
      b'\x018966342U5100\x00\x00\x00\x00',
      b'\x018966342V0000\x00\x00\x00\x00',
      b'\x018966342V3000\x00\x00\x00\x00',
      b'\x018966342V3100\x00\x00\x00\x00',
      b'\x018966342V3200\x00\x00\x00\x00',
      b'\x01896634A05000\x00\x00\x00\x00',
      b'\x01896634A19000\x00\x00\x00\x00',
      b'\x01896634A19100\x00\x00\x00\x00',
      b'\x01896634A20000\x00\x00\x00\x00',
      b'\x01896634A20100\x00\x00\x00\x00',
      b'\x01896634A22000\x00\x00\x00\x00',
      b'\x01896634A22100\x00\x00\x00\x00',
      b'\x01896634A30000\x00\x00\x00\x00',
      b'\x01896634A44000\x00\x00\x00\x00',
      b'\x01896634A45000\x00\x00\x00\x00',
      b'\x01896634A46000\x00\x00\x00\x00',
      b'\x028966342M7000\x00\x00\x00\x00897CF1201001\x00\x00\x00\x00',
      b'\x028966342T0000\x00\x00\x00\x00897CF1201001\x00\x00\x00\x00',
      b'\x028966342V1000\x00\x00\x00\x00897CF1202001\x00\x00\x00\x00',
      b'\x028966342Y8000\x00\x00\x00\x00897CF1201001\x00\x00\x00\x00',
      b'\x02896634A18000\x00\x00\x00\x00897CF1201001\x00\x00\x00\x00',
      b'\x02896634A18100\x00\x00\x00\x00897CF1201001\x00\x00\x00\x00',
      b'\x02896634A43000\x00\x00\x00\x00897CF4201001\x00\x00\x00\x00',
      b'\x02896634A47000\x00\x00\x00\x00897CF4201001\x00\x00\x00\x00',
      b'\x028966342Z8000\x00\x00\x00\x00897CF1201001\x00\x00\x00\x00',
    ],
    (Ecu.abs, 0x7b0, None): [
      b'\x01F15260R210\x00\x00\x00\x00\x00\x00',
      b'\x01F15260R220\x00\x00\x00\x00\x00\x00',
      b'\x01F15260R290\x00\x00\x00\x00\x00\x00',
      b'\x01F15260R300\x00\x00\x00\x00\x00\x00',
      b'\x01F15260R302\x00\x00\x00\x00\x00\x00',
      b'\x01F152642551\x00\x00\x00\x00\x00\x00',
      b'\x01F152642561\x00\x00\x00\x00\x00\x00',
      b'\x01F152642601\x00\x00\x00\x00\x00\x00',
      b'\x01F152642700\x00\x00\x00\x00\x00\x00',
      b'\x01F152642701\x00\x00\x00\x00\x00\x00',
      b'\x01F152642710\x00\x00\x00\x00\x00\x00',
      b'\x01F152642711\x00\x00\x00\x00\x00\x00',
      b'\x01F152642750\x00\x00\x00\x00\x00\x00',
      b'\x01F152642751\x00\x00\x00\x00\x00\x00',
      b'\x01F15260R292\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.eps, 0x7a1, None): [
      b'8965B42170\x00\x00\x00\x00\x00\x00',
      b'8965B42171\x00\x00\x00\x00\x00\x00',
      b'8965B42180\x00\x00\x00\x00\x00\x00',
      b'8965B42181\x00\x00\x00\x00\x00\x00',
      b'\x028965B0R01200\x00\x00\x00\x008965B0R02200\x00\x00\x00\x00',
      b'\x028965B0R01300\x00\x00\x00\x008965B0R02300\x00\x00\x00\x00',
      b'\x028965B0R01400\x00\x00\x00\x008965B0R02400\x00\x00\x00\x00',
    ],
    (Ecu.fwdRadar, 0x750, 0xf): [
      b'\x018821F3301100\x00\x00\x00\x00',
      b'\x018821F3301200\x00\x00\x00\x00',
      b'\x018821F3301300\x00\x00\x00\x00',
      b'\x018821F3301400\x00\x00\x00\x00',
    ],
    (Ecu.fwdCamera, 0x750, 0x6d): [
      b'\x028646F4203200\x00\x00\x00\x008646G26011A0\x00\x00\x00\x00',
      b'\x028646F4203300\x00\x00\x00\x008646G26011A0\x00\x00\x00\x00',
      b'\x028646F4203400\x00\x00\x00\x008646G2601200\x00\x00\x00\x00',
      b'\x028646F4203500\x00\x00\x00\x008646G2601200\x00\x00\x00\x00',
      b'\x028646F4203700\x00\x00\x00\x008646G2601400\x00\x00\x00\x00',
      b'\x028646F4203800\x00\x00\x00\x008646G2601500\x00\x00\x00\x00',
    ],
  },
  CAR.RAV4_TSS2_2022: {
    (Ecu.abs, 0x7b0, None): [
      b'\x01F15260R350\x00\x00\x00\x00\x00\x00',
      b'\x01F15260R361\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.eps, 0x7a1, None): [
      b'\x028965B0R01500\x00\x00\x00\x008965B0R02500\x00\x00\x00\x00',
    ],
    (Ecu.engine, 0x700, None): [
      b'\x01896634AA0000\x00\x00\x00\x00',
      b'\x01896634AA0100\x00\x00\x00\x00',
      b'\x01896634AA1000\x00\x00\x00\x00',
      b'\x01896634A88000\x00\x00\x00\x00',
      b'\x01896634A89000\x00\x00\x00\x00',
      b'\x01896634A89100\x00\x00\x00\x00',
    ],
    (Ecu.fwdRadar, 0x750, 0xf): [
      b'\x018821F0R01100\x00\x00\x00\x00',
    ],
    (Ecu.fwdCamera, 0x750, 0x6d): [
      b'\x028646F0R02100\x00\x00\x00\x008646G0R01100\x00\x00\x00\x00',
    ],
  },
  CAR.RAV4_TSS2_2023: {
    (Ecu.abs, 0x7b0, None): [
      b'\x01F15260R450\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.eps, 0x7a1, None): [
      b'\x028965B0R11000\x00\x00\x00\x008965B0R12000\x00\x00\x00\x00',
    ],
    (Ecu.engine, 0x700, None): [
      b'\x01896634A88100\x00\x00\x00\x00',
      b'\x01896634AJ2000\x00\x00\x00\x00',
    ],
    (Ecu.fwdRadar, 0x750, 0xf): [
      b'\x018821F0R03100\x00\x00\x00\x00',
    ],
    (Ecu.fwdCamera, 0x750, 0x6d): [
      b'\x028646F0R05100\x00\x00\x00\x008646G0R02100\x00\x00\x00\x00',
    ],
  },
  CAR.RAV4H_TSS2: {
    (Ecu.engine, 0x700, None): [
      b'\x01896634A15000\x00\x00\x00\x00',
      b'\x018966342M5000\x00\x00\x00\x00',
      b'\x018966342W8000\x00\x00\x00\x00',
      b'\x018966342X5000\x00\x00\x00\x00',
      b'\x018966342X6000\x00\x00\x00\x00',
      b'\x01896634A25000\x00\x00\x00\x00',
      b'\x018966342W5000\x00\x00\x00\x00',
      b'\x018966342W7000\x00\x00\x00\x00',
      b'\x028966342W4001\x00\x00\x00\x00897CF1203001\x00\x00\x00\x00',
      b'\x02896634A13000\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
      b'\x02896634A13001\x00\x00\x00\x00897CF4801001\x00\x00\x00\x00',
      b'\x02896634A13101\x00\x00\x00\x00897CF4801001\x00\x00\x00\x00',
      b'\x02896634A14001\x00\x00\x00\x00897CF1203001\x00\x00\x00\x00',
      b'\x02896634A23000\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
      b'\x02896634A23001\x00\x00\x00\x00897CF1203001\x00\x00\x00\x00',
      b'\x02896634A23101\x00\x00\x00\x00897CF1203001\x00\x00\x00\x00',
      b'\x02896634A14001\x00\x00\x00\x00897CF4801001\x00\x00\x00\x00',
      b'\x02896634A14101\x00\x00\x00\x00897CF4801001\x00\x00\x00\x00',
    ],
    (Ecu.abs, 0x7b0, None): [
      b'F152642291\x00\x00\x00\x00\x00\x00',
      b'F152642290\x00\x00\x00\x00\x00\x00',
      b'F152642322\x00\x00\x00\x00\x00\x00',
      b'F152642330\x00\x00\x00\x00\x00\x00',
      b'F152642331\x00\x00\x00\x00\x00\x00',
      b'F152642531\x00\x00\x00\x00\x00\x00',
      b'F152642532\x00\x00\x00\x00\x00\x00',
      b'F152642520\x00\x00\x00\x00\x00\x00',
      b'F152642521\x00\x00\x00\x00\x00\x00',
      b'F152642540\x00\x00\x00\x00\x00\x00',
      b'F152642541\x00\x00\x00\x00\x00\x00',
      b'F152642542\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.eps, 0x7a1, None): [
      b'8965B42170\x00\x00\x00\x00\x00\x00',
      b'8965B42171\x00\x00\x00\x00\x00\x00',
      b'8965B42180\x00\x00\x00\x00\x00\x00',
      b'8965B42181\x00\x00\x00\x00\x00\x00',
      b'\x028965B0R01200\x00\x00\x00\x008965B0R02200\x00\x00\x00\x00',
      b'\x028965B0R01300\x00\x00\x00\x008965B0R02300\x00\x00\x00\x00',
      b'\x028965B0R01400\x00\x00\x00\x008965B0R02400\x00\x00\x00\x00',
    ],
    (Ecu.fwdRadar, 0x750, 0xf): [
      b'\x018821F3301100\x00\x00\x00\x00',
      b'\x018821F3301200\x00\x00\x00\x00',
      b'\x018821F3301300\x00\x00\x00\x00',
      b'\x018821F3301400\x00\x00\x00\x00',
    ],
    (Ecu.fwdCamera, 0x750, 0x6d): [
      b'\x028646F4203200\x00\x00\x00\x008646G26011A0\x00\x00\x00\x00',
      b'\x028646F4203300\x00\x00\x00\x008646G26011A0\x00\x00\x00\x00',
      b'\x028646F4203400\x00\x00\x00\x008646G2601200\x00\x00\x00\x00',
      b'\x028646F4203500\x00\x00\x00\x008646G2601200\x00\x00\x00\x00',
      b'\x028646F4203700\x00\x00\x00\x008646G2601400\x00\x00\x00\x00',
      b'\x028646F4203800\x00\x00\x00\x008646G2601500\x00\x00\x00\x00',
    ],
  },
  CAR.RAV4H_TSS2_2022: {
    (Ecu.abs, 0x7b0, None): [
      b'\x01F15264283100\x00\x00\x00\x00',
      b'\x01F15264286200\x00\x00\x00\x00',
      b'\x01F15264286100\x00\x00\x00\x00',
      b'\x01F15264283200\x00\x00\x00\x00',
    ],
    (Ecu.eps, 0x7a1, None): [
      b'\x028965B0R01500\x00\x00\x00\x008965B0R02500\x00\x00\x00\x00',
      b'8965B42182\x00\x00\x00\x00\x00\x00',
      b'8965B42172\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.engine, 0x700, None): [
      b'\x01896634A02001\x00\x00\x00\x00',
      b'\x01896634A03000\x00\x00\x00\x00',
      b'\x01896634A08000\x00\x00\x00\x00',
      b'\x01896634A61000\x00\x00\x00\x00',
      b'\x01896634A62000\x00\x00\x00\x00',
      b'\x01896634A62100\x00\x00\x00\x00',
      b'\x01896634A63000\x00\x00\x00\x00',
    ],
    (Ecu.fwdRadar, 0x750, 0xf): [
      b'\x018821F0R01100\x00\x00\x00\x00',
    ],
    (Ecu.fwdCamera, 0x750, 0x6d): [
      b'\x028646F0R02100\x00\x00\x00\x008646G0R01100\x00\x00\x00\x00',
    ],
  },
  CAR.RAV4H_TSS2_2023: {
    (Ecu.abs, 0x7b0, None): [
      b'\x01F15264283200\x00\x00\x00\x00',
      b'\x01F15264283300\x00\x00\x00\x00',
    ],
    (Ecu.eps, 0x7a1, None): [
      b'\x028965B0R11000\x00\x00\x00\x008965B0R12000\x00\x00\x00\x00',
      b'8965B42371\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.engine, 0x700, None): [
      b'\x01896634AE1001\x00\x00\x00\x00',
      b'\x01896634AF0000\x00\x00\x00\x00',
    ],
    (Ecu.hybrid, 0x7d2, None): [
      b'\x02899830R41000\x00\x00\x00\x00899850R20000\x00\x00\x00\x00',
      b'\x028998342C0000\x00\x00\x00\x00899854224000\x00\x00\x00\x00',
      b'\x02899830R39000\x00\x00\x00\x00899850R20000\x00\x00\x00\x00',
    ],
    (Ecu.fwdRadar, 0x750, 0xf): [
      b'\x018821F0R03100\x00\x00\x00\x00',
    ],
    (Ecu.fwdCamera, 0x750, 0x6d): [
      b'\x028646F0R05100\x00\x00\x00\x008646G0R02100\x00\x00\x00\x00',
    ],
  },
  CAR.SIENNA: {
    (Ecu.engine, 0x700, None): [
      b'\x01896630832100\x00\x00\x00\x00',
      b'\x01896630832200\x00\x00\x00\x00',
      b'\x01896630838000\x00\x00\x00\x00',
      b'\x01896630838100\x00\x00\x00\x00',
      b'\x01896630842000\x00\x00\x00\x00',
      b'\x01896630843000\x00\x00\x00\x00',
      b'\x01896630851000\x00\x00\x00\x00',
      b'\x01896630851100\x00\x00\x00\x00',
      b'\x01896630851200\x00\x00\x00\x00',
      b'\x01896630852000\x00\x00\x00\x00',
      b'\x01896630852100\x00\x00\x00\x00',
      b'\x01896630859000\x00\x00\x00\x00',
      b'\x01896630860000\x00\x00\x00\x00',
    ],
    (Ecu.eps, 0x7a1, None): [
      b'8965B45070\x00\x00\x00\x00\x00\x00',
      b'8965B45080\x00\x00\x00\x00\x00\x00',
      b'8965B45082\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.abs, 0x7b0, None): [
      b'F152608130\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.dsu, 0x791, None): [
      b'881510801100\x00\x00\x00\x00',
    ],
    (Ecu.fwdRadar, 0x750, 0xf): [
      b'8821F4702100\x00\x00\x00\x00',
      b'8821F4702200\x00\x00\x00\x00',
      b'8821F4702300\x00\x00\x00\x00',
    ],
    (Ecu.fwdCamera, 0x750, 0x6d): [
      b'8646F0801100\x00\x00\x00\x00',
    ],
  },
  CAR.LEXUS_CTH: {
    (Ecu.dsu, 0x791, None): [
      b'881517601100\x00\x00\x00\x00',
    ],
    (Ecu.abs, 0x7b0, None): [
      b'F152676144\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'\x0237635000\x00\x00\x00\x00\x00\x00\x00\x00A4701000\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.fwdRadar, 0x750, 0xf): [
      b'8821F4702300\x00\x00\x00\x00',
    ],
    (Ecu.fwdCamera, 0x750, 0x6d): [
      b'8646F7601100\x00\x00\x00\x00',
    ],
  },
  CAR.LEXUS_ES_TSS2: {
    (Ecu.engine, 0x700, None): [
      b'\x018966306U6000\x00\x00\x00\x00',
      b'\x01896630EC9100\x00\x00\x00\x00',
      b'\x018966333T5000\x00\x00\x00\x00',
      b'\x018966333T5100\x00\x00\x00\x00',
      b'\x018966333X6000\x00\x00\x00\x00',
      b'\x01896633T07000\x00\x00\x00\x00',
    ],
    (Ecu.abs, 0x7b0, None): [
      b'\x01F152606281\x00\x00\x00\x00\x00\x00',
      b'\x01F152606340\x00\x00\x00\x00\x00\x00',
      b'\x01F152606461\x00\x00\x00\x00\x00\x00',
      b'\x01F15260E031\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.eps, 0x7a1, None): [
      b'8965B33252\x00\x00\x00\x00\x00\x00',
      b'8965B33590\x00\x00\x00\x00\x00\x00',
      b'8965B33690\x00\x00\x00\x00\x00\x00',
      b'8965B33721\x00\x00\x00\x00\x00\x00',
      b'8965B48271\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.fwdRadar, 0x750, 0xf): [
      b'\x018821F3301100\x00\x00\x00\x00',
      b'\x018821F3301200\x00\x00\x00\x00',
      b'\x018821F3301400\x00\x00\x00\x00',
      b'\x018821F6201300\x00\x00\x00\x00',
    ],
    (Ecu.fwdCamera, 0x750, 0x6d): [
      b'\x028646F33030D0\x00\x00\x00\x008646G26011A0\x00\x00\x00\x00',
      b'\x028646F3303200\x00\x00\x00\x008646G26011A0\x00\x00\x00\x00',
      b'\x028646F3304100\x00\x00\x00\x008646G2601200\x00\x00\x00\x00',
      b'\x028646F3304300\x00\x00\x00\x008646G2601500\x00\x00\x00\x00',
      b'\x028646F3309100\x00\x00\x00\x008646G3304000\x00\x00\x00\x00',
      b'\x028646F4810200\x00\x00\x00\x008646G2601400\x00\x00\x00\x00',
    ],
  },
  CAR.LEXUS_ESH_TSS2: {
    (Ecu.engine, 0x700, None): [
      b'\x028966333S8000\x00\x00\x00\x00897CF3302002\x00\x00\x00\x00',
      b'\x028966333S8000\x00\x00\x00\x00897CF3305001\x00\x00\x00\x00',
      b'\x028966333T0100\x00\x00\x00\x00897CF3305001\x00\x00\x00\x00',
      b'\x028966333V4000\x00\x00\x00\x00897CF3305001\x00\x00\x00\x00',
      b'\x028966333W1000\x00\x00\x00\x00897CF3305001\x00\x00\x00\x00',
      b'\x02896633T09000\x00\x00\x00\x00897CF3307001\x00\x00\x00\x00',
      b'\x01896633T38000\x00\x00\x00\x00',
      b'\x01896633T58000\x00\x00\x00\x00',
    ],
    (Ecu.abs, 0x7b0, None): [
      b'F152633423\x00\x00\x00\x00\x00\x00',
      b'F152633680\x00\x00\x00\x00\x00\x00',
      b'F152633681\x00\x00\x00\x00\x00\x00',
      b'F152633F50\x00\x00\x00\x00\x00\x00',
      b'F152633F51\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.eps, 0x7a1, None): [
      b'8965B33252\x00\x00\x00\x00\x00\x00',
      b'8965B33590\x00\x00\x00\x00\x00\x00',
      b'8965B33690\x00\x00\x00\x00\x00\x00',
      b'8965B33721\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.fwdRadar, 0x750, 0xf): [
      b'\x018821F3301100\x00\x00\x00\x00',
      b'\x018821F3301200\x00\x00\x00\x00',
      b'\x018821F3301300\x00\x00\x00\x00',
      b'\x018821F3301400\x00\x00\x00\x00',
      b'\x018821F6201300\x00\x00\x00\x00',
    ],
    (Ecu.fwdCamera, 0x750, 0x6d): [
      b'\x028646F0610000\x00\x00\x00\x008646G3304000\x00\x00\x00\x00',
      b'\x028646F33030D0\x00\x00\x00\x008646G26011A0\x00\x00\x00\x00',
      b'\x028646F3303100\x00\x00\x00\x008646G26011A0\x00\x00\x00\x00',
      b'\x028646F3303200\x00\x00\x00\x008646G26011A0\x00\x00\x00\x00',
      b'\x028646F3304100\x00\x00\x00\x008646G2601200\x00\x00\x00\x00',
      b'\x028646F3304200\x00\x00\x00\x008646G2601400\x00\x00\x00\x00',
      b'\x028646F3304300\x00\x00\x00\x008646G2601500\x00\x00\x00\x00',
      b'\x028646F3309100\x00\x00\x00\x008646G3304000\x00\x00\x00\x00',
    ],
  },
  CAR.LEXUS_ES: {
    (Ecu.engine, 0x7e0, None): [
      b'\x02333R0000\x00\x00\x00\x00\x00\x00\x00\x00A0C01000\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.abs, 0x7b0, None): [
      b'F152606202\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.dsu, 0x791, None): [
      b'881513309500\x00\x00\x00\x00',
    ],
    (Ecu.eps, 0x7a1, None): [
      b'8965B33502\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.fwdRadar, 0x750, 0xf): [
      b'8821F4701200\x00\x00\x00\x00',
    ],
    (Ecu.fwdCamera, 0x750, 0x6d): [
      b'8646F3302200\x00\x00\x00\x00',
    ],
  },
  CAR.LEXUS_ESH: {
      (Ecu.engine, 0x7e0, None): [
        b'\x02333M4200\x00\x00\x00\x00\x00\x00\x00\x00A4701000\x00\x00\x00\x00\x00\x00\x00\x00',
      ],
      (Ecu.abs, 0x7b0, None): [
        b'F152633171\x00\x00\x00\x00\x00\x00',
      ],
      (Ecu.dsu, 0x791, None): [
        b'881513310400\x00\x00\x00\x00',
      ],
      (Ecu.eps, 0x7a1, None): [
        b'8965B33512\x00\x00\x00\x00\x00\x00',
      ],
      (Ecu.fwdRadar, 0x750, 0xf): [
        b'8821F4701100\x00\x00\x00\x00',
        b'8821F4701300\x00\x00\x00\x00',
      ],
      (Ecu.fwdCamera, 0x750, 0x6d): [
        b'8646F3302001\x00\x00\x00\x00',
        b'8646F3302200\x00\x00\x00\x00',
      ],
  },
  CAR.LEXUS_NX: {
    (Ecu.engine, 0x700, None): [
      b'\x01896637850000\x00\x00\x00\x00',
      b'\x01896637851000\x00\x00\x00\x00',
      b'\x01896637852000\x00\x00\x00\x00',
      b'\x01896637854000\x00\x00\x00\x00',
      b'\x01896637878000\x00\x00\x00\x00',
    ],
    (Ecu.abs, 0x7b0, None): [
      b'F152678130\x00\x00\x00\x00\x00\x00',
      b'F152678140\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.dsu, 0x791, None): [
      b'881517803100\x00\x00\x00\x00',
      b'881517803300\x00\x00\x00\x00',
    ],
    (Ecu.eps, 0x7a1, None): [
      b'8965B78060\x00\x00\x00\x00\x00\x00',
      b'8965B78080\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.fwdRadar, 0x750, 0xf): [
      b'8821F4702100\x00\x00\x00\x00',
      b'8821F4702300\x00\x00\x00\x00',
    ],
    (Ecu.fwdCamera, 0x750, 0x6d): [
      b'8646F7801100\x00\x00\x00\x00',
      b'8646F7801300\x00\x00\x00\x00',
    ],
  },
  CAR.LEXUS_NX_TSS2: {
    (Ecu.engine, 0x700, None): [
      b'\x018966378B2100\x00\x00\x00\x00',
      b'\x018966378B3000\x00\x00\x00\x00',
      b'\x018966378B4100\x00\x00\x00\x00',
      b'\x018966378G2000\x00\x00\x00\x00',
      b'\x018966378G3000\x00\x00\x00\x00',
      b'\x018966378B2000\x00\x00\x00\x00',
    ],
    (Ecu.abs, 0x7b0, None): [
      b'\x01F152678221\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.eps, 0x7a1, None): [
      b'8965B78120\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.fwdRadar, 0x750, 0xf): [
      b"\x018821F3301400\x00\x00\x00\x00",
      b'\x018821F3301200\x00\x00\x00\x00',
      b'\x018821F3301300\x00\x00\x00\x00',
    ],
    (Ecu.fwdCamera, 0x750, 0x6d): [
      b'\x028646F78030A0\x00\x00\x00\x008646G2601200\x00\x00\x00\x00',
      b'\x028646F7803100\x00\x00\x00\x008646G2601400\x00\x00\x00\x00',
    ],
  },
  CAR.LEXUS_NXH_TSS2: {
    (Ecu.engine, 0x7e0, None): [
      b'\x0237887000\x00\x00\x00\x00\x00\x00\x00\x00A4701000\x00\x00\x00\x00\x00\x00\x00\x00',
      b'\x02378A0000\x00\x00\x00\x00\x00\x00\x00\x00A4701000\x00\x00\x00\x00\x00\x00\x00\x00',
      b'\x02378F4000\x00\x00\x00\x00\x00\x00\x00\x00A4701000\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.abs, 0x7b0, None): [
      b'F152678210\x00\x00\x00\x00\x00\x00',
      b'F152678211\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.eps, 0x7a1, None): [
      b'8965B78120\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.fwdRadar, 0x750, 0xf): [
      b'\x018821F3301400\x00\x00\x00\x00',
      b'\x018821F3301300\x00\x00\x00\x00',
    ],
    (Ecu.fwdCamera, 0x750, 0x6d): [
      b'\x028646F78030A0\x00\x00\x00\x008646G2601200\x00\x00\x00\x00',
      b'\x028646F7803100\x00\x00\x00\x008646G2601400\x00\x00\x00\x00',
    ],
  },
  CAR.LEXUS_NXH: {
    (Ecu.engine, 0x7e0, None): [
      b'\x0237841000\x00\x00\x00\x00\x00\x00\x00\x00A4701000\x00\x00\x00\x00\x00\x00\x00\x00',
      b'\x0237842000\x00\x00\x00\x00\x00\x00\x00\x00A4701000\x00\x00\x00\x00\x00\x00\x00\x00',
      b'\x0237880000\x00\x00\x00\x00\x00\x00\x00\x00A4701000\x00\x00\x00\x00\x00\x00\x00\x00',
      b'\x0237882000\x00\x00\x00\x00\x00\x00\x00\x00A4701000\x00\x00\x00\x00\x00\x00\x00\x00',
      b'\x0237886000\x00\x00\x00\x00\x00\x00\x00\x00A4701000\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.abs, 0x7b0, None): [
      b'F152678160\x00\x00\x00\x00\x00\x00',
      b'F152678170\x00\x00\x00\x00\x00\x00',
      b'F152678171\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.dsu, 0x791, None): [
      b'881517804300\x00\x00\x00\x00',
      b'881517804100\x00\x00\x00\x00',
    ],
    (Ecu.eps, 0x7a1, None): [
      b'8965B78060\x00\x00\x00\x00\x00\x00',
      b'8965B78080\x00\x00\x00\x00\x00\x00',
      b'8965B78100\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.fwdRadar, 0x750, 0xf): [
      b'8821F4702300\x00\x00\x00\x00',
      b'8821F4702100\x00\x00\x00\x00',
    ],
    (Ecu.fwdCamera, 0x750, 0x6d): [
      b'8646F7801300\x00\x00\x00\x00',
      b'8646F7801100\x00\x00\x00\x00',
    ],
  },
  CAR.LEXUS_RC: {
    (Ecu.engine, 0x700, None): [
      b'\x01896632478200\x00\x00\x00\x00',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'\x0232484000\x00\x00\x00\x00\x00\x00\x00\x0052422000\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.abs, 0x7b0, None): [
      b'F152624150\x00\x00\x00\x00\x00\x00',
      b'F152624221\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.dsu, 0x791, None): [
      b'881512407000\x00\x00\x00\x00',
      b'881512409100\x00\x00\x00\x00',
    ],
    (Ecu.eps, 0x7a1, None): [
      b'8965B24081\x00\x00\x00\x00\x00\x00',
      b'8965B24320\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.fwdRadar, 0x750, 0xf): [
      b'8821F4702300\x00\x00\x00\x00',
    ],
    (Ecu.fwdCamera, 0x750, 0x6d): [
      b'8646F2401200\x00\x00\x00\x00',
      b'8646F2402200\x00\x00\x00\x00',
    ],
  },
  CAR.LEXUS_RX: {
    (Ecu.engine, 0x700, None): [
      b'\x01896630E36200\x00\x00\x00\x00',
      b'\x01896630E36300\x00\x00\x00\x00',
      b'\x01896630E37200\x00\x00\x00\x00',
      b'\x01896630E37300\x00\x00\x00\x00',
      b'\x01896630E41000\x00\x00\x00\x00',
      b'\x01896630E41100\x00\x00\x00\x00',
      b'\x01896630E41200\x00\x00\x00\x00',
      b'\x01896630E41500\x00\x00\x00\x00',
      b'\x01896630EA3100\x00\x00\x00\x00',
      b'\x01896630EA3400\x00\x00\x00\x00',
      b'\x01896630EA4100\x00\x00\x00\x00',
      b'\x01896630EA4300\x00\x00\x00\x00',
      b'\x01896630EA4400\x00\x00\x00\x00',
      b'\x01896630EA6300\x00\x00\x00\x00',
      b'\x018966348R1300\x00\x00\x00\x00',
      b'\x018966348R8500\x00\x00\x00\x00',
      b'\x018966348W1300\x00\x00\x00\x00',
    ],
    (Ecu.abs, 0x7b0, None): [
      b'F152648472\x00\x00\x00\x00\x00\x00',
      b'F152648473\x00\x00\x00\x00\x00\x00',
      b'F152648492\x00\x00\x00\x00\x00\x00',
      b'F152648493\x00\x00\x00\x00\x00\x00',
      b'F152648474\x00\x00\x00\x00\x00\x00',
      b'F152648630\x00\x00\x00\x00\x00\x00',
      b'F152648494\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.dsu, 0x791, None): [
      b'881514810300\x00\x00\x00\x00',
      b'881514810500\x00\x00\x00\x00',
      b'881514810700\x00\x00\x00\x00',
    ],
    (Ecu.eps, 0x7a1, None): [
      b'8965B0E011\x00\x00\x00\x00\x00\x00',
      b'8965B0E012\x00\x00\x00\x00\x00\x00',
      b'8965B48102\x00\x00\x00\x00\x00\x00',
      b'8965B48111\x00\x00\x00\x00\x00\x00',
      b'8965B48112\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.fwdRadar, 0x750, 0xf): [
      b'8821F4701000\x00\x00\x00\x00',
      b'8821F4701100\x00\x00\x00\x00',
      b'8821F4701200\x00\x00\x00\x00',
      b'8821F4701300\x00\x00\x00\x00',
    ],
    (Ecu.fwdCamera, 0x750, 0x6d): [
      b'8646F4801100\x00\x00\x00\x00',
      b'8646F4801200\x00\x00\x00\x00',
      b'8646F4802001\x00\x00\x00\x00',
      b'8646F4802100\x00\x00\x00\x00',
      b'8646F4802200\x00\x00\x00\x00',
      b'8646F4809000\x00\x00\x00\x00',
    ],
  },
  CAR.LEXUS_RXH: {
    (Ecu.engine, 0x7e0, None): [
      b'\x02348J7000\x00\x00\x00\x00\x00\x00\x00\x00A4802000\x00\x00\x00\x00\x00\x00\x00\x00',
      b'\x02348N0000\x00\x00\x00\x00\x00\x00\x00\x00A4802000\x00\x00\x00\x00\x00\x00\x00\x00',
      b'\x02348Q4000\x00\x00\x00\x00\x00\x00\x00\x00A4802000\x00\x00\x00\x00\x00\x00\x00\x00',
      b'\x02348Q4100\x00\x00\x00\x00\x00\x00\x00\x00A4802000\x00\x00\x00\x00\x00\x00\x00\x00',
      b'\x02348T1100\x00\x00\x00\x00\x00\x00\x00\x00A4802000\x00\x00\x00\x00\x00\x00\x00\x00',
      b'\x02348T3000\x00\x00\x00\x00\x00\x00\x00\x00A4802000\x00\x00\x00\x00\x00\x00\x00\x00',
      b'\x02348V6000\x00\x00\x00\x00\x00\x00\x00\x00A4802000\x00\x00\x00\x00\x00\x00\x00\x00',
      b'\x02348Z3000\x00\x00\x00\x00\x00\x00\x00\x00A4802000\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.abs, 0x7b0, None): [
      b'F152648361\x00\x00\x00\x00\x00\x00',
      b'F152648501\x00\x00\x00\x00\x00\x00',
      b'F152648502\x00\x00\x00\x00\x00\x00',
      b'F152648504\x00\x00\x00\x00\x00\x00',
      b'F152648740\x00\x00\x00\x00\x00\x00',
      b'F152648A30\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.dsu, 0x791, None): [
      b'881514811300\x00\x00\x00\x00',
      b'881514811500\x00\x00\x00\x00',
      b'881514811700\x00\x00\x00\x00',
    ],
    (Ecu.eps, 0x7a1, None): [
      b'8965B0E011\x00\x00\x00\x00\x00\x00',
      b'8965B0E012\x00\x00\x00\x00\x00\x00',
      b'8965B48111\x00\x00\x00\x00\x00\x00',
      b'8965B48112\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.fwdRadar, 0x750, 0xf): [
      b'8821F4701000\x00\x00\x00\x00',
      b'8821F4701100\x00\x00\x00\x00',
      b'8821F4701200\x00\x00\x00\x00',
      b'8821F4701300\x00\x00\x00\x00',
    ],
    (Ecu.fwdCamera, 0x750, 0x6d): [
      b'8646F4801200\x00\x00\x00\x00',
      b'8646F4802001\x00\x00\x00\x00',
      b'8646F4802100\x00\x00\x00\x00',
      b'8646F4802200\x00\x00\x00\x00',
      b'8646F4809000\x00\x00\x00\x00',
    ],
  },
  CAR.LEXUS_RX_TSS2: {
    (Ecu.engine, 0x700, None): [
      b'\x01896630EA9000\x00\x00\x00\x00',
      b'\x01896630EB0000\x00\x00\x00\x00',
      b'\x01896630EC9000\x00\x00\x00\x00',
      b'\x01896630ED0000\x00\x00\x00\x00',
      b'\x01896630ED0100\x00\x00\x00\x00',
      b'\x01896630ED6000\x00\x00\x00\x00',
      b'\x018966348T8000\x00\x00\x00\x00',
      b'\x018966348W5100\x00\x00\x00\x00',
      b'\x018966348W9000\x00\x00\x00\x00',
      b'\x01896634D12000\x00\x00\x00\x00',
      b'\x01896634D12100\x00\x00\x00\x00',
      b'\x01896634D43000\x00\x00\x00\x00',
      b'\x01896634D44000\x00\x00\x00\x00',
      b'\x018966348X0000\x00\x00\x00\x00',
      b'\x01896630ED5000\x00\x00\x00\x00',
    ],
    (Ecu.abs, 0x7b0, None): [
      b'\x01F15260E031\x00\x00\x00\x00\x00\x00',
      b'\x01F15260E041\x00\x00\x00\x00\x00\x00',
      b'\x01F152648781\x00\x00\x00\x00\x00\x00',
      b'\x01F152648801\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.eps, 0x7a1, None): [
      b'8965B48261\x00\x00\x00\x00\x00\x00',
      b'8965B48271\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.fwdRadar, 0x750, 0xf): [
      b'\x018821F3301100\x00\x00\x00\x00',
      b'\x018821F3301300\x00\x00\x00\x00',
      b'\x018821F3301400\x00\x00\x00\x00',
    ],
    (Ecu.fwdCamera, 0x750, 0x6d): [
      b'\x028646F4810100\x00\x00\x00\x008646G2601200\x00\x00\x00\x00',
      b'\x028646F4810200\x00\x00\x00\x008646G2601400\x00\x00\x00\x00',
      b'\x028646F4810300\x00\x00\x00\x008646G2601400\x00\x00\x00\x00',
      b'\x028646F4810400\x00\x00\x00\x008646G2601400\x00\x00\x00\x00',
    ],
  },
  CAR.LEXUS_RXH_TSS2: {
    (Ecu.engine, 0x7e0, None): [
      b'\x02348X4000\x00\x00\x00\x00\x00\x00\x00\x00A4802000\x00\x00\x00\x00\x00\x00\x00\x00',
      b'\x02348X5000\x00\x00\x00\x00\x00\x00\x00\x00A4802000\x00\x00\x00\x00\x00\x00\x00\x00',
      b'\x02348X8000\x00\x00\x00\x00\x00\x00\x00\x00A4802000\x00\x00\x00\x00\x00\x00\x00\x00',
      b'\x02348Y3000\x00\x00\x00\x00\x00\x00\x00\x00A4802000\x00\x00\x00\x00\x00\x00\x00\x00',
      b'\x0234D14000\x00\x00\x00\x00\x00\x00\x00\x00A4802000\x00\x00\x00\x00\x00\x00\x00\x00',
      b'\x0234D16000\x00\x00\x00\x00\x00\x00\x00\x00A4802000\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.abs, 0x7b0, None): [
      b'F152648831\x00\x00\x00\x00\x00\x00',
      b'F152648891\x00\x00\x00\x00\x00\x00',
      b'F152648D00\x00\x00\x00\x00\x00\x00',
      b'F152648D60\x00\x00\x00\x00\x00\x00',
      b'F152648811\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.eps, 0x7a1, None): [
      b'8965B48261\x00\x00\x00\x00\x00\x00',
      b'8965B48271\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.fwdRadar, 0x750, 0xf): [
      b'\x018821F3301400\x00\x00\x00\x00',
    ],
    (Ecu.fwdCamera, 0x750, 0x6d): [
      b'\x028646F4810100\x00\x00\x00\x008646G2601200\x00\x00\x00\x00',
      b'\x028646F4810200\x00\x00\x00\x008646G2601400\x00\x00\x00\x00',
      b'\x028646F4810300\x00\x00\x00\x008646G2601400\x00\x00\x00\x00',
    ],
  },
  CAR.PRIUS_TSS2: {
    (Ecu.engine, 0x700, None): [
      b'\x028966347B1000\x00\x00\x00\x008966A4703000\x00\x00\x00\x00',
      b'\x028966347C4000\x00\x00\x00\x008966A4703000\x00\x00\x00\x00',
      b'\x028966347C6000\x00\x00\x00\x008966A4703000\x00\x00\x00\x00',
      b'\x028966347C7000\x00\x00\x00\x008966A4703000\x00\x00\x00\x00',
      b'\x028966347C8000\x00\x00\x00\x008966A4703000\x00\x00\x00\x00',
      b'\x038966347C0000\x00\x00\x00\x008966A4703000\x00\x00\x00\x00897CF4710101\x00\x00\x00\x00',
      b'\x038966347C1000\x00\x00\x00\x008966A4703000\x00\x00\x00\x00897CF4710101\x00\x00\x00\x00',
      b'\x038966347C5000\x00\x00\x00\x008966A4703000\x00\x00\x00\x00897CF4707101\x00\x00\x00\x00',
      b'\x038966347C5100\x00\x00\x00\x008966A4703000\x00\x00\x00\x00897CF4707101\x00\x00\x00\x00',
    ],
    (Ecu.abs, 0x7b0, None): [
      b'F152647500\x00\x00\x00\x00\x00\x00',
      b'F152647510\x00\x00\x00\x00\x00\x00',
      b'F152647520\x00\x00\x00\x00\x00\x00',
      b'F152647521\x00\x00\x00\x00\x00\x00',
      b'F152647531\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.eps, 0x7a1, None): [
      b'8965B47070\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.fwdRadar, 0x750, 0xf): [
      b'\x018821F3301400\x00\x00\x00\x00',
    ],
    (Ecu.fwdCamera, 0x750, 0x6d): [
      b'\x028646F4707000\x00\x00\x00\x008646G2601400\x00\x00\x00\x00',
      b'\x028646F4710000\x00\x00\x00\x008646G2601500\x00\x00\x00\x00',
      b'\x028646F4712000\x00\x00\x00\x008646G2601500\x00\x00\x00\x00',
    ],
  },
  CAR.MIRAI: {
    (Ecu.abs, 0x7D1, None): [b'\x01898A36203000\x00\x00\x00\x00',],
    (Ecu.abs, 0x7B0, None): [  # a second ABS ECU
      b'\x01F15266203200\x00\x00\x00\x00',
      b'\x01F15266203500\x00\x00\x00\x00',
    ],
    (Ecu.eps, 0x7A1, None): [b'\x028965B6204100\x00\x00\x00\x008965B6203100\x00\x00\x00\x00',],
    (Ecu.fwdRadar, 0x750, 0xf): [b'\x018821F6201200\x00\x00\x00\x00',],
    (Ecu.fwdCamera, 0x750, 0x6d): [b'\x028646F6201400\x00\x00\x00\x008646G5301200\x00\x00\x00\x00',],
  },
  CAR.ALPHARD_TSS2: {
    (Ecu.engine, 0x7e0, None): [
      b'\x0235870000\x00\x00\x00\x00\x00\x00\x00\x00A0202000\x00\x00\x00\x00\x00\x00\x00\x00',
      b'\x0235883000\x00\x00\x00\x00\x00\x00\x00\x00A0202000\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.eps, 0x7a1, None): [
      b'8965B58040\x00\x00\x00\x00\x00\x00',
      b'8965B58052\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.fwdRadar, 0x750, 0xf): [
      b'\x018821F3301200\x00\x00\x00\x00',
      b'\x018821F3301400\x00\x00\x00\x00',
    ],
    (Ecu.fwdCamera, 0x750, 0x6d): [
      b'\x028646F58010C0\x00\x00\x00\x008646G26011A0\x00\x00\x00\x00',
      b'\x028646F5803200\x00\x00\x00\x008646G2601400\x00\x00\x00\x00',
    ],
  },
  CAR.ALPHARDH_TSS2: {
    (Ecu.engine, 0x7e0, None): [
      b'\x0235879000\x00\x00\x00\x00\x00\x00\x00\x00A4701000\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.eps, 0x7a1, None): [
      b'8965B58040\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.abs, 0x7b0, None): [
      b'F152658341\x00\x00\x00\x00\x00\x00'
    ],
    (Ecu.fwdRadar, 0x750, 0xf): [
      b'\x018821F3301400\x00\x00\x00\x00',
    ],
    (Ecu.fwdCamera, 0x750, 0x6d): [
      b'\x028646FV201000\x00\x00\x00\x008646G2601400\x00\x00\x00\x00',
    ],
  },
}

STEER_THRESHOLD = 100

DBC = {
  CAR.RAV4H: dbc_dict('toyota_tnga_k_pt_generated', 'toyota_adas'),
  CAR.RAV4: dbc_dict('toyota_new_mc_pt_generated', 'toyota_adas'),
  CAR.PRIUS: dbc_dict('toyota_nodsu_pt_generated', 'toyota_adas'),
  CAR.PRIUS_V: dbc_dict('toyota_new_mc_pt_generated', 'toyota_adas'),
  CAR.COROLLA: dbc_dict('toyota_new_mc_pt_generated', 'toyota_adas'),
  CAR.LEXUS_RC: dbc_dict('toyota_tnga_k_pt_generated', 'toyota_adas'),
  CAR.LEXUS_RX: dbc_dict('toyota_tnga_k_pt_generated', 'toyota_adas'),
  CAR.LEXUS_RXH: dbc_dict('toyota_tnga_k_pt_generated', 'toyota_adas'),
  CAR.LEXUS_RX_TSS2: dbc_dict('toyota_nodsu_pt_generated', 'toyota_tss2_adas'),
  CAR.LEXUS_RXH_TSS2: dbc_dict('toyota_nodsu_pt_generated', 'toyota_tss2_adas'),
  CAR.CHR: dbc_dict('toyota_nodsu_pt_generated', 'toyota_adas'),
  CAR.CHR_TSS2: dbc_dict('toyota_nodsu_pt_generated', None),
  CAR.CHRH: dbc_dict('toyota_nodsu_pt_generated', 'toyota_adas'),
  CAR.CHRH_TSS2: dbc_dict('toyota_nodsu_pt_generated', None),
  CAR.CAMRY: dbc_dict('toyota_nodsu_pt_generated', 'toyota_adas'),
  CAR.CAMRYH: dbc_dict('toyota_nodsu_pt_generated', 'toyota_adas'),
  CAR.CAMRY_TSS2: dbc_dict('toyota_nodsu_pt_generated', 'toyota_tss2_adas'),
  CAR.CAMRYH_TSS2: dbc_dict('toyota_nodsu_pt_generated', 'toyota_tss2_adas'),
  CAR.HIGHLANDER: dbc_dict('toyota_tnga_k_pt_generated', 'toyota_adas'),
  CAR.HIGHLANDER_TSS2: dbc_dict('toyota_nodsu_pt_generated', 'toyota_tss2_adas'),
  CAR.HIGHLANDERH: dbc_dict('toyota_tnga_k_pt_generated', 'toyota_adas'),
  CAR.HIGHLANDERH_TSS2: dbc_dict('toyota_nodsu_pt_generated', 'toyota_tss2_adas'),
  CAR.AVALON: dbc_dict('toyota_tnga_k_pt_generated', 'toyota_adas'),
  CAR.AVALON_2019: dbc_dict('toyota_nodsu_pt_generated', 'toyota_adas'),
  CAR.AVALONH_2019: dbc_dict('toyota_nodsu_pt_generated', 'toyota_adas'),
  CAR.AVALON_TSS2: dbc_dict('toyota_nodsu_pt_generated', 'toyota_tss2_adas'),
  CAR.AVALONH_TSS2: dbc_dict('toyota_nodsu_pt_generated', 'toyota_tss2_adas'),
  CAR.RAV4_TSS2: dbc_dict('toyota_nodsu_pt_generated', 'toyota_tss2_adas'),
  CAR.RAV4_TSS2_2022: dbc_dict('toyota_nodsu_pt_generated', None),
  CAR.RAV4_TSS2_2023: dbc_dict('toyota_nodsu_pt_generated', None),
  CAR.COROLLA_TSS2: dbc_dict('toyota_nodsu_pt_generated', 'toyota_tss2_adas'),
  CAR.COROLLAH_TSS2: dbc_dict('toyota_nodsu_pt_generated', 'toyota_tss2_adas'),
  CAR.LEXUS_ES: dbc_dict('toyota_new_mc_pt_generated', 'toyota_adas'),
  CAR.LEXUS_ES_TSS2: dbc_dict('toyota_nodsu_pt_generated', 'toyota_tss2_adas'),
  CAR.LEXUS_ESH_TSS2: dbc_dict('toyota_nodsu_pt_generated', 'toyota_tss2_adas'),
  CAR.LEXUS_ESH: dbc_dict('toyota_new_mc_pt_generated', 'toyota_adas'),
  CAR.SIENNA: dbc_dict('toyota_tnga_k_pt_generated', 'toyota_adas'),
  CAR.LEXUS_IS: dbc_dict('toyota_tnga_k_pt_generated', 'toyota_adas'),
  CAR.LEXUS_IS_TSS2: dbc_dict('toyota_nodsu_pt_generated', 'toyota_tss2_adas'),
  CAR.LEXUS_CTH: dbc_dict('toyota_new_mc_pt_generated', 'toyota_adas'),
  CAR.RAV4H_TSS2: dbc_dict('toyota_nodsu_pt_generated', 'toyota_tss2_adas'),
  CAR.RAV4H_TSS2_2022: dbc_dict('toyota_nodsu_pt_generated', None),
  CAR.RAV4H_TSS2_2023: dbc_dict('toyota_nodsu_pt_generated', None),
  CAR.LEXUS_NXH: dbc_dict('toyota_tnga_k_pt_generated', 'toyota_adas'),
  CAR.LEXUS_NX: dbc_dict('toyota_tnga_k_pt_generated', 'toyota_adas'),
  CAR.LEXUS_NX_TSS2: dbc_dict('toyota_nodsu_pt_generated', 'toyota_tss2_adas'),
  CAR.LEXUS_NXH_TSS2: dbc_dict('toyota_nodsu_pt_generated', 'toyota_tss2_adas'),
  CAR.PRIUS_TSS2: dbc_dict('toyota_nodsu_pt_generated', 'toyota_tss2_adas'),
  CAR.MIRAI: dbc_dict('toyota_nodsu_pt_generated', 'toyota_tss2_adas'),
  CAR.ALPHARD_TSS2: dbc_dict('toyota_nodsu_pt_generated', 'toyota_tss2_adas'),
  CAR.ALPHARDH_TSS2: dbc_dict('toyota_nodsu_pt_generated', 'toyota_tss2_adas'),
}

# These cars have non-standard EPS torque scale factors. All others are 73
EPS_SCALE = defaultdict(lambda: 73, {CAR.PRIUS: 66, CAR.COROLLA: 88, CAR.LEXUS_IS: 77, CAR.LEXUS_RC: 77, CAR.LEXUS_CTH: 100, CAR.PRIUS_V: 100})

# Toyota/Lexus Safety Sense 2.0 and 2.5
TSS2_CAR = {CAR.RAV4_TSS2, CAR.RAV4_TSS2_2022, CAR.RAV4_TSS2_2023, CAR.COROLLA_TSS2, CAR.COROLLAH_TSS2, CAR.LEXUS_ES_TSS2, CAR.LEXUS_ESH_TSS2,
            CAR.RAV4H_TSS2, CAR.RAV4H_TSS2_2022, CAR.RAV4H_TSS2_2023, CAR.LEXUS_RX_TSS2, CAR.LEXUS_RXH_TSS2, CAR.HIGHLANDER_TSS2,
            CAR.HIGHLANDERH_TSS2, CAR.PRIUS_TSS2, CAR.CAMRY_TSS2, CAR.CAMRYH_TSS2, CAR.LEXUS_IS_TSS2, CAR.MIRAI, CAR.LEXUS_NX_TSS2,
            CAR.LEXUS_NXH_TSS2, CAR.ALPHARD_TSS2, CAR.AVALON_TSS2, CAR.AVALONH_TSS2, CAR.ALPHARDH_TSS2, CAR.CHR_TSS2, CAR.CHRH_TSS2}

NO_DSU_CAR = TSS2_CAR | {CAR.CHR, CAR.CHRH, CAR.CAMRY, CAR.CAMRYH}

# the DSU uses the AEB message for longitudinal on these cars
UNSUPPORTED_DSU_CAR = {CAR.LEXUS_IS, CAR.LEXUS_RC}

# these cars have a radar which sends ACC messages instead of the camera
RADAR_ACC_CAR = {CAR.RAV4H_TSS2_2022, CAR.RAV4_TSS2_2022, CAR.RAV4H_TSS2_2023, CAR.RAV4_TSS2_2023, CAR.CHR_TSS2, CAR.CHRH_TSS2}

# these cars use the Lane Tracing Assist (LTA) message for lateral control
ANGLE_CONTROL_CAR = {CAR.RAV4H_TSS2_2023, CAR.RAV4_TSS2_2023}

EV_HYBRID_CAR = {CAR.AVALONH_2019, CAR.AVALONH_TSS2, CAR.CAMRYH, CAR.CAMRYH_TSS2, CAR.CHRH, CAR.CHRH_TSS2, CAR.COROLLAH_TSS2,
                 CAR.HIGHLANDERH, CAR.HIGHLANDERH_TSS2, CAR.PRIUS, CAR.PRIUS_V, CAR.RAV4H, CAR.RAV4H_TSS2, CAR.RAV4H_TSS2_2022,
                 CAR.RAV4H_TSS2_2023, CAR.LEXUS_CTH, CAR.MIRAI, CAR.LEXUS_ESH, CAR.LEXUS_ESH_TSS2, CAR.LEXUS_NXH, CAR.LEXUS_RXH,
                 CAR.LEXUS_RXH_TSS2, CAR.LEXUS_NXH_TSS2, CAR.PRIUS_TSS2, CAR.ALPHARDH_TSS2}

# no resume button press required
NO_STOP_TIMER_CAR = TSS2_CAR | {CAR.PRIUS_V, CAR.RAV4H, CAR.HIGHLANDERH, CAR.HIGHLANDER, CAR.SIENNA, CAR.LEXUS_ESH}
