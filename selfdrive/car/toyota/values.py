import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, IntFlag

from cereal import car
from openpilot.common.conversions import Conversions as CV
from openpilot.selfdrive.car import CarSpecs, PlatformConfig, Platforms
from openpilot.selfdrive.car import AngleRateLimit, dbc_dict
from openpilot.selfdrive.car.docs_definitions import CarFootnote, CarDocs, Column, CarParts, CarHarness
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
  # Detected flags
  HYBRID = 1
  SMART_DSU = 2
  DISABLE_RADAR = 4
  RADAR_CAN_FILTER = 1024

  # Static flags
  TSS2 = 8
  NO_DSU = 16
  UNSUPPORTED_DSU = 32
  RADAR_ACC = 64
  # these cars use the Lane Tracing Assist (LTA) message for lateral control
  ANGLE_CONTROL = 128
  NO_STOP_TIMER = 256
  # these cars are speculated to allow stop and go when the DSU is unplugged or disabled with sDSU
  SNG_WITHOUT_DSU = 512


class Footnote(Enum):
  CAMRY = CarFootnote(
    "openpilot operates above 28mph for Camry 4CYL L, 4CYL LE and 4CYL SE which don't have Full-Speed Range Dynamic Radar Cruise Control.",
    Column.FSR_LONGITUDINAL)


@dataclass
class ToyotaCarDocs(CarDocs):
  package: str = "All"
  car_parts: CarParts = field(default_factory=CarParts.common([CarHarness.toyota_a]))


@dataclass
class ToyotaTSS2PlatformConfig(PlatformConfig):
  dbc_dict: dict = field(default_factory=lambda: dbc_dict('toyota_nodsu_pt_generated', 'toyota_tss2_adas'))

  def init(self):
    self.flags |= ToyotaFlags.TSS2 | ToyotaFlags.NO_STOP_TIMER | ToyotaFlags.NO_DSU

    if self.flags & ToyotaFlags.RADAR_ACC:
      self.dbc_dict = dbc_dict('toyota_nodsu_pt_generated', None)


class CAR(Platforms):
  # Toyota
  TOYOTA_ALPHARD_TSS2 = ToyotaTSS2PlatformConfig(
    [
      ToyotaCarDocs("Toyota Alphard 2019-20"),
      ToyotaCarDocs("Toyota Alphard Hybrid 2021"),
    ],
    CarSpecs(mass=4305. * CV.LB_TO_KG, wheelbase=3.0, steerRatio=14.2, tireStiffnessFactor=0.444),
  )
  TOYOTA_AVALON = PlatformConfig(
    [
      ToyotaCarDocs("Toyota Avalon 2016", "Toyota Safety Sense P"),
      ToyotaCarDocs("Toyota Avalon 2017-18"),
    ],
    CarSpecs(mass=3505. * CV.LB_TO_KG, wheelbase=2.82, steerRatio=14.8, tireStiffnessFactor=0.7983),
    dbc_dict('toyota_tnga_k_pt_generated', 'toyota_adas'),
  )
  TOYOTA_AVALON_2019 = PlatformConfig(
    [
      ToyotaCarDocs("Toyota Avalon 2019-21"),
      ToyotaCarDocs("Toyota Avalon Hybrid 2019-21"),
    ],
    TOYOTA_AVALON.specs,
    dbc_dict('toyota_nodsu_pt_generated', 'toyota_adas'),
  )
  TOYOTA_AVALON_TSS2 = ToyotaTSS2PlatformConfig( # TSS 2.5
    [
      ToyotaCarDocs("Toyota Avalon 2022"),
      ToyotaCarDocs("Toyota Avalon Hybrid 2022"),
    ],
    TOYOTA_AVALON.specs,
  )
  TOYOTA_CAMRY = PlatformConfig(
    [
      ToyotaCarDocs("Toyota Camry 2018-20", video_link="https://www.youtube.com/watch?v=fkcjviZY9CM", footnotes=[Footnote.CAMRY]),
      ToyotaCarDocs("Toyota Camry Hybrid 2018-20", video_link="https://www.youtube.com/watch?v=Q2DYY0AWKgk"),
    ],
    CarSpecs(mass=3400. * CV.LB_TO_KG, wheelbase=2.82448, steerRatio=13.7, tireStiffnessFactor=0.7933),
    dbc_dict('toyota_nodsu_pt_generated', 'toyota_adas'),
    flags=ToyotaFlags.NO_DSU,
  )
  TOYOTA_CAMRY_TSS2 = ToyotaTSS2PlatformConfig( # TSS 2.5
    [
      ToyotaCarDocs("Toyota Camry 2021-24", footnotes=[Footnote.CAMRY]),
      ToyotaCarDocs("Toyota Camry Hybrid 2021-24"),
    ],
    TOYOTA_CAMRY.specs,
  )
  TOYOTA_CHR = PlatformConfig(
    [
      ToyotaCarDocs("Toyota C-HR 2017-20"),
      ToyotaCarDocs("Toyota C-HR Hybrid 2017-20"),
    ],
    CarSpecs(mass=3300. * CV.LB_TO_KG, wheelbase=2.63906, steerRatio=13.6, tireStiffnessFactor=0.7933),
    dbc_dict('toyota_nodsu_pt_generated', 'toyota_adas'),
    flags=ToyotaFlags.NO_DSU,
  )
  TOYOTA_CHR_TSS2 = ToyotaTSS2PlatformConfig(
    [
      ToyotaCarDocs("Toyota C-HR 2021"),
      ToyotaCarDocs("Toyota C-HR Hybrid 2021-22"),
    ],
    TOYOTA_CHR.specs,
    flags=ToyotaFlags.RADAR_ACC,
  )
  TOYOTA_COROLLA = PlatformConfig(
    [ToyotaCarDocs("Toyota Corolla 2017-19")],
    CarSpecs(mass=2860. * CV.LB_TO_KG, wheelbase=2.7, steerRatio=18.27, tireStiffnessFactor=0.444),
    dbc_dict('toyota_new_mc_pt_generated', 'toyota_adas'),
  )
  # LSS2 Lexus UX Hybrid is same as a TSS2 Corolla Hybrid
  TOYOTA_COROLLA_TSS2 = ToyotaTSS2PlatformConfig(
    [
      ToyotaCarDocs("Toyota Corolla 2020-22", video_link="https://www.youtube.com/watch?v=_66pXk0CBYA"),
      ToyotaCarDocs("Toyota Corolla Cross (Non-US only) 2020-23", min_enable_speed=7.5),
      ToyotaCarDocs("Toyota Corolla Hatchback 2019-22", video_link="https://www.youtube.com/watch?v=_66pXk0CBYA"),
      # Hybrid platforms
      ToyotaCarDocs("Toyota Corolla Hybrid 2020-22"),
      ToyotaCarDocs("Toyota Corolla Hybrid (Non-US only) 2020-23", min_enable_speed=7.5),
      ToyotaCarDocs("Toyota Corolla Cross Hybrid (Non-US only) 2020-22", min_enable_speed=7.5),
      ToyotaCarDocs("Lexus UX Hybrid 2019-23"),
    ],
    CarSpecs(mass=3060. * CV.LB_TO_KG, wheelbase=2.67, steerRatio=13.9, tireStiffnessFactor=0.444),
  )
  TOYOTA_HIGHLANDER = PlatformConfig(
    [
      ToyotaCarDocs("Toyota Highlander 2017-19", video_link="https://www.youtube.com/watch?v=0wS0wXSLzoo"),
      ToyotaCarDocs("Toyota Highlander Hybrid 2017-19"),
    ],
    CarSpecs(mass=4516. * CV.LB_TO_KG, wheelbase=2.8194, steerRatio=16.0, tireStiffnessFactor=0.8),
    dbc_dict('toyota_tnga_k_pt_generated', 'toyota_adas'),
    flags=ToyotaFlags.NO_STOP_TIMER | ToyotaFlags.SNG_WITHOUT_DSU,
  )
  TOYOTA_HIGHLANDER_TSS2 = ToyotaTSS2PlatformConfig(
    [
      ToyotaCarDocs("Toyota Highlander 2020-23"),
      ToyotaCarDocs("Toyota Highlander Hybrid 2020-23"),
    ],
    TOYOTA_HIGHLANDER.specs,
  )
  TOYOTA_PRIUS = PlatformConfig(
    [
      ToyotaCarDocs("Toyota Prius 2016", "Toyota Safety Sense P", video_link="https://www.youtube.com/watch?v=8zopPJI8XQ0"),
      ToyotaCarDocs("Toyota Prius 2017-20", video_link="https://www.youtube.com/watch?v=8zopPJI8XQ0"),
      ToyotaCarDocs("Toyota Prius Prime 2017-20", video_link="https://www.youtube.com/watch?v=8zopPJI8XQ0"),
    ],
    CarSpecs(mass=3045. * CV.LB_TO_KG, wheelbase=2.7, steerRatio=15.74, tireStiffnessFactor=0.6371),
    dbc_dict('toyota_nodsu_pt_generated', 'toyota_adas'),
  )
  TOYOTA_PRIUS_V = PlatformConfig(
    [ToyotaCarDocs("Toyota Prius v 2017", "Toyota Safety Sense P", min_enable_speed=MIN_ACC_SPEED)],
    CarSpecs(mass=3340. * CV.LB_TO_KG, wheelbase=2.78, steerRatio=17.4, tireStiffnessFactor=0.5533),
    dbc_dict('toyota_new_mc_pt_generated', 'toyota_adas'),
    flags=ToyotaFlags.NO_STOP_TIMER | ToyotaFlags.SNG_WITHOUT_DSU,
  )
  TOYOTA_PRIUS_TSS2 = ToyotaTSS2PlatformConfig(
    [
      ToyotaCarDocs("Toyota Prius 2021-22", video_link="https://www.youtube.com/watch?v=J58TvCpUd4U"),
      ToyotaCarDocs("Toyota Prius Prime 2021-22", video_link="https://www.youtube.com/watch?v=J58TvCpUd4U"),
    ],
    CarSpecs(mass=3115. * CV.LB_TO_KG, wheelbase=2.70002, steerRatio=13.4, tireStiffnessFactor=0.6371),
  )
  TOYOTA_RAV4 = PlatformConfig(
    [
      ToyotaCarDocs("Toyota RAV4 2016", "Toyota Safety Sense P"),
      ToyotaCarDocs("Toyota RAV4 2017-18")
    ],
    CarSpecs(mass=3650. * CV.LB_TO_KG, wheelbase=2.65, steerRatio=16.88, tireStiffnessFactor=0.5533),
    dbc_dict('toyota_new_mc_pt_generated', 'toyota_adas'),
  )
  TOYOTA_RAV4H = PlatformConfig(
    [
      ToyotaCarDocs("Toyota RAV4 Hybrid 2016", "Toyota Safety Sense P", video_link="https://youtu.be/LhT5VzJVfNI?t=26"),
      ToyotaCarDocs("Toyota RAV4 Hybrid 2017-18", video_link="https://youtu.be/LhT5VzJVfNI?t=26")
    ],
    TOYOTA_RAV4.specs,
    dbc_dict('toyota_tnga_k_pt_generated', 'toyota_adas'),
    # Note that the ICE RAV4 does not respect positive acceleration commands under 19 mph
    flags=ToyotaFlags.NO_STOP_TIMER | ToyotaFlags.SNG_WITHOUT_DSU,
  )
  TOYOTA_RAV4_TSS2 = ToyotaTSS2PlatformConfig(
    [
      ToyotaCarDocs("Toyota RAV4 2019-21", video_link="https://www.youtube.com/watch?v=wJxjDd42gGA"),
      ToyotaCarDocs("Toyota RAV4 Hybrid 2019-21"),
    ],
    CarSpecs(mass=3585. * CV.LB_TO_KG, wheelbase=2.68986, steerRatio=14.3, tireStiffnessFactor=0.7933),
  )
  TOYOTA_RAV4_TSS2_2022 = ToyotaTSS2PlatformConfig(
    [
      ToyotaCarDocs("Toyota RAV4 2022"),
      ToyotaCarDocs("Toyota RAV4 Hybrid 2022", video_link="https://youtu.be/U0nH9cnrFB0"),
    ],
    TOYOTA_RAV4_TSS2.specs,
    flags=ToyotaFlags.RADAR_ACC,
  )
  TOYOTA_RAV4_TSS2_2023 = ToyotaTSS2PlatformConfig(
    [
      ToyotaCarDocs("Toyota RAV4 2023-24"),
      ToyotaCarDocs("Toyota RAV4 Hybrid 2023-24"),
    ],
    TOYOTA_RAV4_TSS2.specs,
    flags=ToyotaFlags.RADAR_ACC | ToyotaFlags.ANGLE_CONTROL,
  )
  TOYOTA_MIRAI = ToyotaTSS2PlatformConfig( # TSS 2.5
    [ToyotaCarDocs("Toyota Mirai 2021")],
    CarSpecs(mass=4300. * CV.LB_TO_KG, wheelbase=2.91, steerRatio=14.8, tireStiffnessFactor=0.8),
  )
  TOYOTA_SIENNA = PlatformConfig(
    [ToyotaCarDocs("Toyota Sienna 2018-20", video_link="https://www.youtube.com/watch?v=q1UPOo4Sh68", min_enable_speed=MIN_ACC_SPEED)],
    CarSpecs(mass=4590. * CV.LB_TO_KG, wheelbase=3.03, steerRatio=15.5, tireStiffnessFactor=0.444),
    dbc_dict('toyota_tnga_k_pt_generated', 'toyota_adas'),
    flags=ToyotaFlags.NO_STOP_TIMER,
  )

  # Lexus
  LEXUS_CTH = PlatformConfig(
    [ToyotaCarDocs("Lexus CT Hybrid 2017-18", "Lexus Safety System+")],
    CarSpecs(mass=3108. * CV.LB_TO_KG, wheelbase=2.6, steerRatio=18.6, tireStiffnessFactor=0.517),
    dbc_dict('toyota_new_mc_pt_generated', 'toyota_adas'),
  )
  LEXUS_ES = PlatformConfig(
    [
      ToyotaCarDocs("Lexus ES 2017-18"),
      ToyotaCarDocs("Lexus ES Hybrid 2017-18"),
    ],
    CarSpecs(mass=3677. * CV.LB_TO_KG, wheelbase=2.8702, steerRatio=16.0, tireStiffnessFactor=0.444),
    dbc_dict('toyota_new_mc_pt_generated', 'toyota_adas'),
  )
  LEXUS_ES_TSS2 = ToyotaTSS2PlatformConfig(
    [
      ToyotaCarDocs("Lexus ES 2019-24"),
      ToyotaCarDocs("Lexus ES Hybrid 2019-24", video_link="https://youtu.be/BZ29osRVJeg?t=12"),
    ],
    LEXUS_ES.specs,
  )
  LEXUS_IS = PlatformConfig(
    [ToyotaCarDocs("Lexus IS 2017-19")],
    CarSpecs(mass=3736.8 * CV.LB_TO_KG, wheelbase=2.79908, steerRatio=13.3, tireStiffnessFactor=0.444),
    dbc_dict('toyota_tnga_k_pt_generated', 'toyota_adas'),
    flags=ToyotaFlags.UNSUPPORTED_DSU,
  )
  LEXUS_IS_TSS2 = ToyotaTSS2PlatformConfig(
    [ToyotaCarDocs("Lexus IS 2022-23")],
    LEXUS_IS.specs,
  )
  LEXUS_NX = PlatformConfig(
    [
      ToyotaCarDocs("Lexus NX 2018-19"),
      ToyotaCarDocs("Lexus NX Hybrid 2018-19"),
    ],
    CarSpecs(mass=4070. * CV.LB_TO_KG, wheelbase=2.66, steerRatio=14.7, tireStiffnessFactor=0.444),
    dbc_dict('toyota_tnga_k_pt_generated', 'toyota_adas'),
  )
  LEXUS_NX_TSS2 = ToyotaTSS2PlatformConfig(
    [
      ToyotaCarDocs("Lexus NX 2020-21"),
      ToyotaCarDocs("Lexus NX Hybrid 2020-21"),
    ],
    LEXUS_NX.specs,
  )
  LEXUS_LC_TSS2 = ToyotaTSS2PlatformConfig(
    [ToyotaCarDocs("Lexus LC 2024")],
    CarSpecs(mass=4500. * CV.LB_TO_KG, wheelbase=2.87, steerRatio=13.0, tireStiffnessFactor=0.444),
  )
  LEXUS_RC = PlatformConfig(
    [ToyotaCarDocs("Lexus RC 2018-20")],
    LEXUS_IS.specs,
    dbc_dict('toyota_tnga_k_pt_generated', 'toyota_adas'),
    flags=ToyotaFlags.UNSUPPORTED_DSU,
  )
  LEXUS_RX = PlatformConfig(
    [
      ToyotaCarDocs("Lexus RX 2016", "Lexus Safety System+"),
      ToyotaCarDocs("Lexus RX 2017-19"),
      # Hybrid platforms
      ToyotaCarDocs("Lexus RX Hybrid 2016", "Lexus Safety System+"),
      ToyotaCarDocs("Lexus RX Hybrid 2017-19"),
    ],
    CarSpecs(mass=4481. * CV.LB_TO_KG, wheelbase=2.79, steerRatio=16., tireStiffnessFactor=0.5533),
    dbc_dict('toyota_tnga_k_pt_generated', 'toyota_adas'),
  )
  LEXUS_RX_TSS2 = ToyotaTSS2PlatformConfig(
    [
      ToyotaCarDocs("Lexus RX 2020-22"),
      ToyotaCarDocs("Lexus RX Hybrid 2020-22"),
    ],
    LEXUS_RX.specs,
  )
  LEXUS_GS_F = PlatformConfig(
    [ToyotaCarDocs("Lexus GS F 2016")],
    CarSpecs(mass=4034. * CV.LB_TO_KG, wheelbase=2.84988, steerRatio=13.3, tireStiffnessFactor=0.444),
    dbc_dict('toyota_new_mc_pt_generated', 'toyota_adas'),
    flags=ToyotaFlags.UNSUPPORTED_DSU,
  )


# (addr, cars, bus, 1/freq*100, vl)
STATIC_DSU_MSGS = [
  (0x128, (CAR.TOYOTA_PRIUS, CAR.TOYOTA_RAV4H, CAR.LEXUS_RX, CAR.LEXUS_NX, CAR.TOYOTA_RAV4, CAR.TOYOTA_COROLLA, CAR.TOYOTA_AVALON), \
                                                                                                                      1, 3, b'\xf4\x01\x90\x83\x00\x37'),
  (0x128, (CAR.TOYOTA_HIGHLANDER, CAR.TOYOTA_SIENNA, CAR.LEXUS_CTH, CAR.LEXUS_ES), 1,   3, b'\x03\x00\x20\x00\x00\x52'),
  (0x141, (CAR.TOYOTA_PRIUS, CAR.TOYOTA_RAV4H, CAR.LEXUS_RX, CAR.LEXUS_NX, CAR.TOYOTA_RAV4, CAR.TOYOTA_COROLLA, CAR.TOYOTA_HIGHLANDER, CAR.TOYOTA_AVALON,
           CAR.TOYOTA_SIENNA, CAR.LEXUS_CTH, CAR.LEXUS_ES, CAR.TOYOTA_PRIUS_V), 1,   2, b'\x00\x00\x00\x46'),
  (0x160, (CAR.TOYOTA_PRIUS, CAR.TOYOTA_RAV4H, CAR.LEXUS_RX, CAR.LEXUS_NX, CAR.TOYOTA_RAV4, CAR.TOYOTA_COROLLA, CAR.TOYOTA_HIGHLANDER, CAR.TOYOTA_AVALON,
           CAR.TOYOTA_SIENNA, CAR.LEXUS_CTH, CAR.LEXUS_ES, CAR.TOYOTA_PRIUS_V), 1,   7, b'\x00\x00\x08\x12\x01\x31\x9c\x51'),
  (0x161, (CAR.TOYOTA_PRIUS, CAR.TOYOTA_RAV4H, CAR.LEXUS_RX, CAR.LEXUS_NX, CAR.TOYOTA_RAV4, CAR.TOYOTA_COROLLA, CAR.TOYOTA_AVALON, CAR.TOYOTA_PRIUS_V),
                                                                                               1,   7, b'\x00\x1e\x00\x00\x00\x80\x07'),
  (0X161, (CAR.TOYOTA_HIGHLANDER, CAR.TOYOTA_SIENNA, CAR.LEXUS_CTH, CAR.LEXUS_ES), 1,  7, b'\x00\x1e\x00\xd4\x00\x00\x5b'),
  (0x283, (CAR.TOYOTA_PRIUS, CAR.TOYOTA_RAV4H, CAR.LEXUS_RX, CAR.LEXUS_NX, CAR.TOYOTA_RAV4, CAR.TOYOTA_COROLLA, CAR.TOYOTA_HIGHLANDER, CAR.TOYOTA_AVALON,
           CAR.TOYOTA_SIENNA, CAR.LEXUS_CTH, CAR.LEXUS_ES, CAR.TOYOTA_PRIUS_V), 0,   3, b'\x00\x00\x00\x00\x00\x00\x8c'),
  (0x2E6, (CAR.TOYOTA_PRIUS, CAR.TOYOTA_RAV4H, CAR.LEXUS_RX), 0,   3, b'\xff\xf8\x00\x08\x7f\xe0\x00\x4e'),
  (0x2E7, (CAR.TOYOTA_PRIUS, CAR.TOYOTA_RAV4H, CAR.LEXUS_RX), 0,   3, b'\xa8\x9c\x31\x9c\x00\x00\x00\x02'),
  (0x33E, (CAR.TOYOTA_PRIUS, CAR.TOYOTA_RAV4H, CAR.LEXUS_RX), 0,  20, b'\x0f\xff\x26\x40\x00\x1f\x00'),
  (0x344, (CAR.TOYOTA_PRIUS, CAR.TOYOTA_RAV4H, CAR.LEXUS_RX, CAR.LEXUS_NX, CAR.TOYOTA_RAV4, CAR.TOYOTA_COROLLA, CAR.TOYOTA_HIGHLANDER, CAR.TOYOTA_AVALON,
           CAR.TOYOTA_SIENNA, CAR.LEXUS_CTH, CAR.LEXUS_ES, CAR.TOYOTA_PRIUS_V), 0,   5, b'\x00\x00\x01\x00\x00\x00\x00\x50'),
  (0x365, (CAR.TOYOTA_PRIUS, CAR.LEXUS_NX, CAR.TOYOTA_HIGHLANDER), 0,  20, b'\x00\x00\x00\x80\x03\x00\x08'),
  (0x365, (CAR.TOYOTA_RAV4, CAR.TOYOTA_RAV4H, CAR.TOYOTA_COROLLA, CAR.TOYOTA_AVALON, CAR.TOYOTA_SIENNA, CAR.LEXUS_CTH, CAR.LEXUS_ES, CAR.LEXUS_RX,
           CAR.TOYOTA_PRIUS_V), 0,  20, b'\x00\x00\x00\x80\xfc\x00\x08'),
  (0x366, (CAR.TOYOTA_PRIUS, CAR.TOYOTA_RAV4H, CAR.LEXUS_RX, CAR.LEXUS_NX, CAR.TOYOTA_HIGHLANDER), 0,  20, b'\x00\x00\x4d\x82\x40\x02\x00'),
  (0x366, (CAR.TOYOTA_RAV4, CAR.TOYOTA_COROLLA, CAR.TOYOTA_AVALON, CAR.TOYOTA_SIENNA, CAR.LEXUS_CTH, CAR.LEXUS_ES, CAR.TOYOTA_PRIUS_V),
          0,  20, b'\x00\x72\x07\xff\x09\xfe\x00'),
  (0x470, (CAR.TOYOTA_PRIUS, CAR.LEXUS_RX), 1, 100, b'\x00\x00\x02\x7a'),
  (0x470, (CAR.TOYOTA_HIGHLANDER, CAR.TOYOTA_RAV4H, CAR.TOYOTA_SIENNA, CAR.LEXUS_CTH, CAR.LEXUS_ES, CAR.TOYOTA_PRIUS_V), 1,  100, b'\x00\x00\x01\x79'),
  (0x4CB, (CAR.TOYOTA_PRIUS, CAR.TOYOTA_RAV4H, CAR.LEXUS_RX, CAR.LEXUS_NX, CAR.TOYOTA_RAV4, CAR.TOYOTA_COROLLA, CAR.TOYOTA_HIGHLANDER, CAR.TOYOTA_AVALON,
           CAR.TOYOTA_SIENNA, CAR.LEXUS_CTH, CAR.LEXUS_ES, CAR.TOYOTA_PRIUS_V), 0, 100, b'\x0c\x00\x00\x00\x00\x00\x00\x00'),
]


def get_platform_codes(fw_versions: list[bytes]) -> dict[bytes, set[bytes]]:
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


def match_fw_to_car_fuzzy(live_fw_versions, offline_fw_versions) -> set[str]:
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
    Ecu.abs: [CAR.TOYOTA_RAV4, CAR.TOYOTA_COROLLA, CAR.TOYOTA_HIGHLANDER, CAR.TOYOTA_SIENNA, CAR.LEXUS_IS, CAR.TOYOTA_ALPHARD_TSS2],
    # On some models, the engine can show on two different addresses
    Ecu.engine: [CAR.TOYOTA_HIGHLANDER, CAR.TOYOTA_CAMRY, CAR.TOYOTA_COROLLA_TSS2, CAR.TOYOTA_CHR, CAR.TOYOTA_CHR_TSS2, CAR.LEXUS_IS,
                 CAR.LEXUS_IS_TSS2, CAR.LEXUS_RC, CAR.LEXUS_NX, CAR.LEXUS_NX_TSS2, CAR.LEXUS_RX, CAR.LEXUS_RX_TSS2],
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

# These cars have non-standard EPS torque scale factors. All others are 73
EPS_SCALE = defaultdict(lambda: 73,
                        {CAR.TOYOTA_PRIUS: 66, CAR.TOYOTA_COROLLA: 88, CAR.LEXUS_IS: 77, CAR.LEXUS_RC: 77, CAR.LEXUS_CTH: 100, CAR.TOYOTA_PRIUS_V: 100})

# Toyota/Lexus Safety Sense 2.0 and 2.5
TSS2_CAR = CAR.with_flags(ToyotaFlags.TSS2)

NO_DSU_CAR = CAR.with_flags(ToyotaFlags.NO_DSU)

# the DSU uses the AEB message for longitudinal on these cars
UNSUPPORTED_DSU_CAR = CAR.with_flags(ToyotaFlags.UNSUPPORTED_DSU)

# these cars have a radar which sends ACC messages instead of the camera
RADAR_ACC_CAR = CAR.with_flags(ToyotaFlags.RADAR_ACC)

ANGLE_CONTROL_CAR = CAR.with_flags(ToyotaFlags.ANGLE_CONTROL)

# no resume button press required
NO_STOP_TIMER_CAR = CAR.with_flags(ToyotaFlags.NO_STOP_TIMER)

DBC = CAR.create_dbc_map()
