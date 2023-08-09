from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, IntFlag
from typing import Dict, List, Union

from common.conversions import Conversions as CV
from selfdrive.car import AngleRateLimit, dbc_dict
from selfdrive.car.docs_definitions import CarFootnote, CarInfo, Column, CarParts, CarHarness

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
