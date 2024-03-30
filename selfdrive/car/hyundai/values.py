import re
from dataclasses import dataclass, field
from enum import Enum, IntFlag

from cereal import car
from panda.python import uds
from openpilot.common.conversions import Conversions as CV
from openpilot.selfdrive.car import CarSpecs, DbcDict, PlatformConfig, Platforms, dbc_dict
from openpilot.selfdrive.car.docs_definitions import CarFootnote, CarHarness, CarDocs, CarParts, Column
from openpilot.selfdrive.car.fw_query_definitions import FwQueryConfig, Request, p16

Ecu = car.CarParams.Ecu


class CarControllerParams:
  ACCEL_MIN = -3.5 # m/s
  ACCEL_MAX = 2.0 # m/s

  def __init__(self, CP):
    self.STEER_DELTA_UP = 3
    self.STEER_DELTA_DOWN = 7
    self.STEER_DRIVER_ALLOWANCE = 50
    self.STEER_DRIVER_MULTIPLIER = 2
    self.STEER_DRIVER_FACTOR = 1
    self.STEER_THRESHOLD = 150
    self.STEER_STEP = 1  # 100 Hz

    if CP.carFingerprint in CANFD_CAR:
      self.STEER_MAX = 270
      self.STEER_DRIVER_ALLOWANCE = 250
      self.STEER_DRIVER_MULTIPLIER = 2
      self.STEER_THRESHOLD = 250
      self.STEER_DELTA_UP = 2
      self.STEER_DELTA_DOWN = 3

    # To determine the limit for your car, find the maximum value that the stock LKAS will request.
    # If the max stock LKAS request is <384, add your car to this list.
    elif CP.carFingerprint in (CAR.GENESIS_G80, CAR.GENESIS_G90, CAR.HYUNDAI_ELANTRA, CAR.HYUNDAI_ELANTRA_GT_I30, CAR.HYUNDAI_IONIQ,
                               CAR.HYUNDAI_IONIQ_EV_LTD, CAR.HYUNDAI_SANTA_FE_PHEV_2022, CAR.HYUNDAI_SONATA_LF, CAR.KIA_FORTE, CAR.KIA_NIRO_PHEV,
                               CAR.KIA_OPTIMA_H, CAR.KIA_OPTIMA_H_G4_FL, CAR.KIA_SORENTO):
      self.STEER_MAX = 255

    # these cars have significantly more torque than most HKG; limit to 70% of max
    elif CP.flags & HyundaiFlags.ALT_LIMITS:
      self.STEER_MAX = 270
      self.STEER_DELTA_UP = 2
      self.STEER_DELTA_DOWN = 3

    # Default for most HKG
    else:
      self.STEER_MAX = 384


class HyundaiFlags(IntFlag):
  # Dynamic Flags
  CANFD_HDA2 = 1
  CANFD_ALT_BUTTONS = 2
  CANFD_ALT_GEARS = 2 ** 2
  CANFD_CAMERA_SCC = 2 ** 3

  ALT_LIMITS = 2 ** 4
  ENABLE_BLINKERS = 2 ** 5
  CANFD_ALT_GEARS_2 = 2 ** 6
  SEND_LFA = 2 ** 7
  USE_FCA = 2 ** 8
  CANFD_HDA2_ALT_STEERING = 2 ** 9

  # these cars use a different gas signal
  HYBRID = 2 ** 10
  EV = 2 ** 11

  # Static flags

  # If 0x500 is present on bus 1 it probably has a Mando radar outputting radar points.
  # If no points are outputted by default it might be possible to turn it on using  selfdrive/debug/hyundai_enable_radar_points.py
  MANDO_RADAR = 2 ** 12
  CANFD = 2 ** 13

  # The radar does SCC on these cars when HDA I, rather than the camera
  RADAR_SCC = 2 ** 14
  CAMERA_SCC = 2 ** 15
  CHECKSUM_CRC8 = 2 ** 16
  CHECKSUM_6B = 2 ** 17

  # these cars require a special panda safety mode due to missing counters and checksums in the messages
  LEGACY = 2 ** 18

  # these cars have not been verified to work with longitudinal yet - radar disable, sending correct messages, etc.
  UNSUPPORTED_LONGITUDINAL = 2 ** 19

  CANFD_NO_RADAR_DISABLE = 2 ** 20

  CLUSTER_GEARS = 2 ** 21
  TCU_GEARS = 2 ** 22

  MIN_STEER_32_MPH = 2 ** 23


class Footnote(Enum):
  CANFD = CarFootnote(
    "Requires a <a href=\"https://comma.ai/shop/can-fd-panda-kit\" target=\"_blank\">CAN FD panda kit</a> if not using " +
    "comma 3X for this <a href=\"https://en.wikipedia.org/wiki/CAN_FD\" target=\"_blank\">CAN FD car</a>.",
    Column.MODEL, shop_footnote=False)


@dataclass
class HyundaiCarDocs(CarDocs):
  package: str = "Smart Cruise Control (SCC)"

  def init_make(self, CP: car.CarParams):
    if CP.flags & HyundaiFlags.CANFD:
      self.footnotes.insert(0, Footnote.CANFD)


@dataclass
class HyundaiPlatformConfig(PlatformConfig):
  dbc_dict: DbcDict = field(default_factory=lambda: dbc_dict("hyundai_kia_generic", None))

  def init(self):
    if self.flags & HyundaiFlags.MANDO_RADAR:
      self.dbc_dict = dbc_dict('hyundai_kia_generic', 'hyundai_kia_mando_front_radar_generated')

    if self.flags & HyundaiFlags.MIN_STEER_32_MPH:
      self.specs = self.specs.override(minSteerSpeed=32 * CV.MPH_TO_MS)


@dataclass
class HyundaiCanFDPlatformConfig(PlatformConfig):
  dbc_dict: DbcDict = field(default_factory=lambda: dbc_dict("hyundai_canfd", None))

  def init(self):
    self.flags |= HyundaiFlags.CANFD


class CAR(Platforms):
  # Hyundai
  HYUNDAI_AZERA_6TH_GEN = HyundaiPlatformConfig(
    [HyundaiCarDocs("Hyundai Azera 2022", "All", car_parts=CarParts.common([CarHarness.hyundai_k]))],
    CarSpecs(mass=1600, wheelbase=2.885, steerRatio=14.5),
  )
  HYUNDAI_AZERA_HEV_6TH_GEN = HyundaiPlatformConfig(
    [
      HyundaiCarDocs("Hyundai Azera Hybrid 2019", "All", car_parts=CarParts.common([CarHarness.hyundai_c])),
      HyundaiCarDocs("Hyundai Azera Hybrid 2020", "All", car_parts=CarParts.common([CarHarness.hyundai_k])),
    ],
    CarSpecs(mass=1675, wheelbase=2.885, steerRatio=14.5),
    flags=HyundaiFlags.HYBRID,
  )
  HYUNDAI_ELANTRA = HyundaiPlatformConfig(
    [
      # TODO: 2017-18 could be Hyundai G
      HyundaiCarDocs("Hyundai Elantra 2017-18", min_enable_speed=19 * CV.MPH_TO_MS, car_parts=CarParts.common([CarHarness.hyundai_b])),
      HyundaiCarDocs("Hyundai Elantra 2019", min_enable_speed=19 * CV.MPH_TO_MS, car_parts=CarParts.common([CarHarness.hyundai_g])),
    ],
    # steerRatio: 14 is Stock | Settled Params Learner values are steerRatio: 15.401566348670535, stiffnessFactor settled on 1.0081302973865127
    CarSpecs(mass=1275, wheelbase=2.7, steerRatio=15.4, tireStiffnessFactor=0.385),
    flags=HyundaiFlags.LEGACY | HyundaiFlags.CLUSTER_GEARS | HyundaiFlags.MIN_STEER_32_MPH,
  )
  HYUNDAI_ELANTRA_GT_I30 = HyundaiPlatformConfig(
    [
      HyundaiCarDocs("Hyundai Elantra GT 2017-19", car_parts=CarParts.common([CarHarness.hyundai_e])),
      HyundaiCarDocs("Hyundai i30 2017-19", car_parts=CarParts.common([CarHarness.hyundai_e])),
    ],
    HYUNDAI_ELANTRA.specs,
    flags=HyundaiFlags.LEGACY | HyundaiFlags.CLUSTER_GEARS | HyundaiFlags.MIN_STEER_32_MPH,
  )
  HYUNDAI_ELANTRA_2021 = HyundaiPlatformConfig(
    [HyundaiCarDocs("Hyundai Elantra 2021-23", video_link="https://youtu.be/_EdYQtV52-c", car_parts=CarParts.common([CarHarness.hyundai_k]))],
    CarSpecs(mass=2800 * CV.LB_TO_KG, wheelbase=2.72, steerRatio=12.9, tireStiffnessFactor=0.65),
    flags=HyundaiFlags.CHECKSUM_CRC8,
  )
  HYUNDAI_ELANTRA_HEV_2021 = HyundaiPlatformConfig(
    [HyundaiCarDocs("Hyundai Elantra Hybrid 2021-23", video_link="https://youtu.be/_EdYQtV52-c",
                    car_parts=CarParts.common([CarHarness.hyundai_k]))],
    CarSpecs(mass=3017 * CV.LB_TO_KG, wheelbase=2.72, steerRatio=12.9, tireStiffnessFactor=0.65),
    flags=HyundaiFlags.CHECKSUM_CRC8 | HyundaiFlags.HYBRID,
  )
  HYUNDAI_GENESIS = HyundaiPlatformConfig(
    [
      # TODO: check 2015 packages
      HyundaiCarDocs("Hyundai Genesis 2015-16", min_enable_speed=19 * CV.MPH_TO_MS, car_parts=CarParts.common([CarHarness.hyundai_j])),
      HyundaiCarDocs("Genesis G80 2017", "All", min_enable_speed=19 * CV.MPH_TO_MS, car_parts=CarParts.common([CarHarness.hyundai_j])),
    ],
    CarSpecs(mass=2060, wheelbase=3.01, steerRatio=16.5, minSteerSpeed=60 * CV.KPH_TO_MS),
    flags=HyundaiFlags.CHECKSUM_6B | HyundaiFlags.LEGACY,
  )
  HYUNDAI_IONIQ = HyundaiPlatformConfig(
    [HyundaiCarDocs("Hyundai Ioniq Hybrid 2017-19", car_parts=CarParts.common([CarHarness.hyundai_c]))],
    CarSpecs(mass=1490, wheelbase=2.7, steerRatio=13.73, tireStiffnessFactor=0.385),
    flags=HyundaiFlags.HYBRID | HyundaiFlags.MIN_STEER_32_MPH,
  )
  HYUNDAI_IONIQ_HEV_2022 = HyundaiPlatformConfig(
    [HyundaiCarDocs("Hyundai Ioniq Hybrid 2020-22", car_parts=CarParts.common([CarHarness.hyundai_h]))],  # TODO: confirm 2020-21 harness,
    CarSpecs(mass=1490, wheelbase=2.7, steerRatio=13.73, tireStiffnessFactor=0.385),
    flags=HyundaiFlags.HYBRID | HyundaiFlags.LEGACY,
  )
  HYUNDAI_IONIQ_EV_LTD = HyundaiPlatformConfig(
    [HyundaiCarDocs("Hyundai Ioniq Electric 2019", car_parts=CarParts.common([CarHarness.hyundai_c]))],
    CarSpecs(mass=1490, wheelbase=2.7, steerRatio=13.73, tireStiffnessFactor=0.385),
    flags=HyundaiFlags.MANDO_RADAR | HyundaiFlags.EV | HyundaiFlags.LEGACY | HyundaiFlags.MIN_STEER_32_MPH,
  )
  HYUNDAI_IONIQ_EV_2020 = HyundaiPlatformConfig(
    [HyundaiCarDocs("Hyundai Ioniq Electric 2020", "All", car_parts=CarParts.common([CarHarness.hyundai_h]))],
    CarSpecs(mass=1490, wheelbase=2.7, steerRatio=13.73, tireStiffnessFactor=0.385),
    flags=HyundaiFlags.EV,
  )
  HYUNDAI_IONIQ_PHEV_2019 = HyundaiPlatformConfig(
    [HyundaiCarDocs("Hyundai Ioniq Plug-in Hybrid 2019", car_parts=CarParts.common([CarHarness.hyundai_c]))],
    CarSpecs(mass=1490, wheelbase=2.7, steerRatio=13.73, tireStiffnessFactor=0.385),
    flags=HyundaiFlags.HYBRID | HyundaiFlags.MIN_STEER_32_MPH,
  )
  HYUNDAI_IONIQ_PHEV = HyundaiPlatformConfig(
    [HyundaiCarDocs("Hyundai Ioniq Plug-in Hybrid 2020-22", "All", car_parts=CarParts.common([CarHarness.hyundai_h]))],
    CarSpecs(mass=1490, wheelbase=2.7, steerRatio=13.73, tireStiffnessFactor=0.385),
    flags=HyundaiFlags.HYBRID,
  )
  HYUNDAI_KONA = HyundaiPlatformConfig(
    [HyundaiCarDocs("Hyundai Kona 2020", car_parts=CarParts.common([CarHarness.hyundai_b]))],
    CarSpecs(mass=1275, wheelbase=2.6, steerRatio=13.42, tireStiffnessFactor=0.385),
    flags=HyundaiFlags.CLUSTER_GEARS,
  )
  HYUNDAI_KONA_EV = HyundaiPlatformConfig(
    [HyundaiCarDocs("Hyundai Kona Electric 2018-21", car_parts=CarParts.common([CarHarness.hyundai_g]))],
    CarSpecs(mass=1685, wheelbase=2.6, steerRatio=13.42, tireStiffnessFactor=0.385),
    flags=HyundaiFlags.EV,
  )
  HYUNDAI_KONA_EV_2022 = HyundaiPlatformConfig(
    [HyundaiCarDocs("Hyundai Kona Electric 2022-23", car_parts=CarParts.common([CarHarness.hyundai_o]))],
    CarSpecs(mass=1743, wheelbase=2.6, steerRatio=13.42, tireStiffnessFactor=0.385),
    flags=HyundaiFlags.CAMERA_SCC | HyundaiFlags.EV,
  )
  HYUNDAI_KONA_EV_2ND_GEN = HyundaiCanFDPlatformConfig(
    [HyundaiCarDocs("Hyundai Kona Electric (with HDA II, Korea only) 2023", video_link="https://www.youtube.com/watch?v=U2fOCmcQ8hw",
                    car_parts=CarParts.common([CarHarness.hyundai_r]))],
    CarSpecs(mass=1740, wheelbase=2.66, steerRatio=13.6, tireStiffnessFactor=0.385),
    flags=HyundaiFlags.EV | HyundaiFlags.CANFD_NO_RADAR_DISABLE,
  )
  HYUNDAI_KONA_HEV = HyundaiPlatformConfig(
    [HyundaiCarDocs("Hyundai Kona Hybrid 2020", car_parts=CarParts.common([CarHarness.hyundai_i]))],  # TODO: check packages,
    CarSpecs(mass=1425, wheelbase=2.6, steerRatio=13.42, tireStiffnessFactor=0.385),
    flags=HyundaiFlags.HYBRID,
  )
  HYUNDAI_SANTA_FE = HyundaiPlatformConfig(
    [HyundaiCarDocs("Hyundai Santa Fe 2019-20", "All", video_link="https://youtu.be/bjDR0YjM__s",
                    car_parts=CarParts.common([CarHarness.hyundai_d]))],
    CarSpecs(mass=3982 * CV.LB_TO_KG, wheelbase=2.766, steerRatio=16.55, tireStiffnessFactor=0.82),
    flags=HyundaiFlags.MANDO_RADAR | HyundaiFlags.CHECKSUM_CRC8,
  )
  HYUNDAI_SANTA_FE_2022 = HyundaiPlatformConfig(
    [HyundaiCarDocs("Hyundai Santa Fe 2021-23", "All", video_link="https://youtu.be/VnHzSTygTS4",
                    car_parts=CarParts.common([CarHarness.hyundai_l]))],
    HYUNDAI_SANTA_FE.specs,
    flags=HyundaiFlags.CHECKSUM_CRC8,
  )
  HYUNDAI_SANTA_FE_HEV_2022 = HyundaiPlatformConfig(
    [HyundaiCarDocs("Hyundai Santa Fe Hybrid 2022-23", "All", car_parts=CarParts.common([CarHarness.hyundai_l]))],
    HYUNDAI_SANTA_FE.specs,
    flags=HyundaiFlags.CHECKSUM_CRC8 | HyundaiFlags.HYBRID,
  )
  HYUNDAI_SANTA_FE_PHEV_2022 = HyundaiPlatformConfig(
    [HyundaiCarDocs("Hyundai Santa Fe Plug-in Hybrid 2022-23", "All", car_parts=CarParts.common([CarHarness.hyundai_l]))],
    HYUNDAI_SANTA_FE.specs,
    flags=HyundaiFlags.CHECKSUM_CRC8 | HyundaiFlags.HYBRID,
  )
  HYUNDAI_SONATA = HyundaiPlatformConfig(
    [HyundaiCarDocs("Hyundai Sonata 2020-23", "All", video_link="https://www.youtube.com/watch?v=ix63r9kE3Fw",
                   car_parts=CarParts.common([CarHarness.hyundai_a]))],
    CarSpecs(mass=1513, wheelbase=2.84, steerRatio=13.27 * 1.15, tireStiffnessFactor=0.65),  # 15% higher at the center seems reasonable
    flags=HyundaiFlags.MANDO_RADAR | HyundaiFlags.CHECKSUM_CRC8,
  )
  HYUNDAI_SONATA_LF = HyundaiPlatformConfig(
    [HyundaiCarDocs("Hyundai Sonata 2018-19", car_parts=CarParts.common([CarHarness.hyundai_e]))],
    CarSpecs(mass=1536, wheelbase=2.804, steerRatio=13.27 * 1.15),  # 15% higher at the center seems reasonable

    flags=HyundaiFlags.UNSUPPORTED_LONGITUDINAL | HyundaiFlags.TCU_GEARS,
  )
  HYUNDAI_STARIA_4TH_GEN = HyundaiCanFDPlatformConfig(
    [HyundaiCarDocs("Hyundai Staria 2023", "All", car_parts=CarParts.common([CarHarness.hyundai_k]))],
    CarSpecs(mass=2205, wheelbase=3.273, steerRatio=11.94),  # https://www.hyundai.com/content/dam/hyundai/au/en/models/staria-load/premium-pip-update-2023/spec-sheet/STARIA_Load_Spec-Table_March_2023_v3.1.pdf
  )
  HYUNDAI_TUCSON = HyundaiPlatformConfig(
    [
      HyundaiCarDocs("Hyundai Tucson 2021", min_enable_speed=19 * CV.MPH_TO_MS, car_parts=CarParts.common([CarHarness.hyundai_l])),
      HyundaiCarDocs("Hyundai Tucson Diesel 2019", car_parts=CarParts.common([CarHarness.hyundai_l])),
    ],
    CarSpecs(mass=3520 * CV.LB_TO_KG, wheelbase=2.67, steerRatio=16.1, tireStiffnessFactor=0.385),
    flags=HyundaiFlags.TCU_GEARS,
  )
  HYUNDAI_PALISADE = HyundaiPlatformConfig(
    [
      HyundaiCarDocs("Hyundai Palisade 2020-22", "All", video_link="https://youtu.be/TAnDqjF4fDY?t=456", car_parts=CarParts.common([CarHarness.hyundai_h])),
      HyundaiCarDocs("Kia Telluride 2020-22", "All", car_parts=CarParts.common([CarHarness.hyundai_h])),
    ],
    CarSpecs(mass=1999, wheelbase=2.9, steerRatio=15.6 * 1.15, tireStiffnessFactor=0.63),
    flags=HyundaiFlags.MANDO_RADAR | HyundaiFlags.CHECKSUM_CRC8,
  )
  HYUNDAI_VELOSTER = HyundaiPlatformConfig(
    [HyundaiCarDocs("Hyundai Veloster 2019-20", min_enable_speed=5. * CV.MPH_TO_MS, car_parts=CarParts.common([CarHarness.hyundai_e]))],
    CarSpecs(mass=2917 * CV.LB_TO_KG, wheelbase=2.8, steerRatio=13.75 * 1.15, tireStiffnessFactor=0.5),
    flags=HyundaiFlags.LEGACY | HyundaiFlags.TCU_GEARS,
  )
  HYUNDAI_SONATA_HYBRID = HyundaiPlatformConfig(
    [HyundaiCarDocs("Hyundai Sonata Hybrid 2020-23", "All", car_parts=CarParts.common([CarHarness.hyundai_a]))],
    HYUNDAI_SONATA.specs,
    flags=HyundaiFlags.MANDO_RADAR | HyundaiFlags.CHECKSUM_CRC8 | HyundaiFlags.HYBRID,
  )
  HYUNDAI_IONIQ_5 = HyundaiCanFDPlatformConfig(
    [
      HyundaiCarDocs("Hyundai Ioniq 5 (Southeast Asia only) 2022-23", "All", car_parts=CarParts.common([CarHarness.hyundai_q])),
      HyundaiCarDocs("Hyundai Ioniq 5 (without HDA II) 2022-23", "Highway Driving Assist", car_parts=CarParts.common([CarHarness.hyundai_k])),
      HyundaiCarDocs("Hyundai Ioniq 5 (with HDA II) 2022-23", "Highway Driving Assist II", car_parts=CarParts.common([CarHarness.hyundai_q])),
    ],
    CarSpecs(mass=1948, wheelbase=2.97, steerRatio=14.26, tireStiffnessFactor=0.65),
    flags=HyundaiFlags.EV,
  )
  HYUNDAI_IONIQ_6 = HyundaiCanFDPlatformConfig(
    [HyundaiCarDocs("Hyundai Ioniq 6 (with HDA II) 2023", "Highway Driving Assist II", car_parts=CarParts.common([CarHarness.hyundai_p]))],
    HYUNDAI_IONIQ_5.specs,
    flags=HyundaiFlags.EV | HyundaiFlags.CANFD_NO_RADAR_DISABLE,
  )
  HYUNDAI_TUCSON_4TH_GEN = HyundaiCanFDPlatformConfig(
    [
      HyundaiCarDocs("Hyundai Tucson 2022", car_parts=CarParts.common([CarHarness.hyundai_n])),
      HyundaiCarDocs("Hyundai Tucson 2023", "All", car_parts=CarParts.common([CarHarness.hyundai_n])),
      HyundaiCarDocs("Hyundai Tucson Hybrid 2022-24", "All", car_parts=CarParts.common([CarHarness.hyundai_n])),
    ],
    CarSpecs(mass=1630, wheelbase=2.756, steerRatio=13.7, tireStiffnessFactor=0.385),
  )
  HYUNDAI_SANTA_CRUZ_1ST_GEN = HyundaiCanFDPlatformConfig(
    [HyundaiCarDocs("Hyundai Santa Cruz 2022-24", car_parts=CarParts.common([CarHarness.hyundai_n]))],
    # weight from Limited trim - the only supported trim, steering ratio according to Hyundai News https://www.hyundainews.com/assets/documents/original/48035-2022SantaCruzProductGuideSpecsv2081521.pdf
    CarSpecs(mass=1870, wheelbase=3, steerRatio=14.2),
  )
  HYUNDAI_CUSTIN_1ST_GEN = HyundaiPlatformConfig(
    [HyundaiCarDocs("Hyundai Custin 2023", "All", car_parts=CarParts.common([CarHarness.hyundai_k]))],
    CarSpecs(mass=1690, wheelbase=3.055, steerRatio=17),  # mass: from https://www.hyundai-motor.com.tw/clicktobuy/custin#spec_0, steerRatio: from learner
    flags=HyundaiFlags.CHECKSUM_CRC8,
  )

  # Kia
  KIA_FORTE = HyundaiPlatformConfig(
    [
      HyundaiCarDocs("Kia Forte 2019-21", car_parts=CarParts.common([CarHarness.hyundai_g])),
      HyundaiCarDocs("Kia Forte 2023", car_parts=CarParts.common([CarHarness.hyundai_e])),
    ],
    CarSpecs(mass=2878 * CV.LB_TO_KG, wheelbase=2.8, steerRatio=13.75, tireStiffnessFactor=0.5)
  )
  KIA_K5_2021 = HyundaiPlatformConfig(
    [HyundaiCarDocs("Kia K5 2021-24", car_parts=CarParts.common([CarHarness.hyundai_a]))],
    CarSpecs(mass=3381 * CV.LB_TO_KG, wheelbase=2.85, steerRatio=13.27, tireStiffnessFactor=0.5),  # 2021 Kia K5 Steering Ratio (all trims)
    flags=HyundaiFlags.CHECKSUM_CRC8,
  )
  KIA_K5_HEV_2020 = HyundaiPlatformConfig(
    [HyundaiCarDocs("Kia K5 Hybrid 2020-22", car_parts=CarParts.common([CarHarness.hyundai_a]))],
    KIA_K5_2021.specs,
    flags=HyundaiFlags.MANDO_RADAR | HyundaiFlags.CHECKSUM_CRC8 | HyundaiFlags.HYBRID,
  )
  KIA_K8_HEV_1ST_GEN = HyundaiCanFDPlatformConfig(
    [HyundaiCarDocs("Kia K8 Hybrid (with HDA II) 2023", "Highway Driving Assist II", car_parts=CarParts.common([CarHarness.hyundai_q]))],
    # mass: https://carprices.ae/brands/kia/2023/k8/1.6-turbo-hybrid, steerRatio: guesstimate from K5 platform
    CarSpecs(mass=1630, wheelbase=2.895, steerRatio=13.27)
  )
  KIA_NIRO_EV = HyundaiPlatformConfig(
    [
      HyundaiCarDocs("Kia Niro EV 2019", "All", video_link="https://www.youtube.com/watch?v=lT7zcG6ZpGo", car_parts=CarParts.common([CarHarness.hyundai_h])),
      HyundaiCarDocs("Kia Niro EV 2020", "All", video_link="https://www.youtube.com/watch?v=lT7zcG6ZpGo", car_parts=CarParts.common([CarHarness.hyundai_f])),
      HyundaiCarDocs("Kia Niro EV 2021", "All", video_link="https://www.youtube.com/watch?v=lT7zcG6ZpGo", car_parts=CarParts.common([CarHarness.hyundai_c])),
      HyundaiCarDocs("Kia Niro EV 2022", "All", video_link="https://www.youtube.com/watch?v=lT7zcG6ZpGo", car_parts=CarParts.common([CarHarness.hyundai_h])),
    ],
    CarSpecs(mass=3543 * CV.LB_TO_KG, wheelbase=2.7, steerRatio=13.6, tireStiffnessFactor=0.385),  # average of all the cars
    flags=HyundaiFlags.MANDO_RADAR | HyundaiFlags.EV,
  )
  KIA_NIRO_EV_2ND_GEN = HyundaiCanFDPlatformConfig(
    [HyundaiCarDocs("Kia Niro EV 2023", "All", car_parts=CarParts.common([CarHarness.hyundai_a]))],
    KIA_NIRO_EV.specs,
    flags=HyundaiFlags.EV,
  )
  KIA_NIRO_PHEV = HyundaiPlatformConfig(
    [
      HyundaiCarDocs("Kia Niro Hybrid 2018", "All", min_enable_speed=10. * CV.MPH_TO_MS, car_parts=CarParts.common([CarHarness.hyundai_c])),
      HyundaiCarDocs("Kia Niro Plug-in Hybrid 2018-19", "All", min_enable_speed=10. * CV.MPH_TO_MS, car_parts=CarParts.common([CarHarness.hyundai_c])),
      HyundaiCarDocs("Kia Niro Plug-in Hybrid 2020", car_parts=CarParts.common([CarHarness.hyundai_d])),
    ],
    KIA_NIRO_EV.specs,
    flags=HyundaiFlags.MANDO_RADAR | HyundaiFlags.HYBRID | HyundaiFlags.UNSUPPORTED_LONGITUDINAL | HyundaiFlags.MIN_STEER_32_MPH,
  )
  KIA_NIRO_PHEV_2022 = HyundaiPlatformConfig(
    [
      HyundaiCarDocs("Kia Niro Plug-in Hybrid 2021", car_parts=CarParts.common([CarHarness.hyundai_d])),
      HyundaiCarDocs("Kia Niro Plug-in Hybrid 2022", car_parts=CarParts.common([CarHarness.hyundai_f])),
    ],
    KIA_NIRO_EV.specs,
    flags=HyundaiFlags.HYBRID | HyundaiFlags.MANDO_RADAR,
  )
  KIA_NIRO_HEV_2021 = HyundaiPlatformConfig(
    [
      HyundaiCarDocs("Kia Niro Hybrid 2021", car_parts=CarParts.common([CarHarness.hyundai_d])),
      HyundaiCarDocs("Kia Niro Hybrid 2022", car_parts=CarParts.common([CarHarness.hyundai_f])),
    ],
    KIA_NIRO_EV.specs,
    flags=HyundaiFlags.HYBRID,
  )
  KIA_NIRO_HEV_2ND_GEN = HyundaiCanFDPlatformConfig(
    [HyundaiCarDocs("Kia Niro Hybrid 2023", car_parts=CarParts.common([CarHarness.hyundai_a]))],
    KIA_NIRO_EV.specs,
  )
  KIA_OPTIMA_G4 = HyundaiPlatformConfig(
    [HyundaiCarDocs("Kia Optima 2017", "Advanced Smart Cruise Control",
                    car_parts=CarParts.common([CarHarness.hyundai_b]))],  # TODO: may support 2016, 2018
    CarSpecs(mass=3558 * CV.LB_TO_KG, wheelbase=2.8, steerRatio=13.75, tireStiffnessFactor=0.5),
    flags=HyundaiFlags.LEGACY | HyundaiFlags.TCU_GEARS | HyundaiFlags.MIN_STEER_32_MPH,
  )
  KIA_OPTIMA_G4_FL = HyundaiPlatformConfig(
    [HyundaiCarDocs("Kia Optima 2019-20", car_parts=CarParts.common([CarHarness.hyundai_g]))],
    CarSpecs(mass=3558 * CV.LB_TO_KG, wheelbase=2.8, steerRatio=13.75, tireStiffnessFactor=0.5),
    flags=HyundaiFlags.UNSUPPORTED_LONGITUDINAL | HyundaiFlags.TCU_GEARS,
  )
  # TODO: may support adjacent years. may have a non-zero minimum steering speed
  KIA_OPTIMA_H = HyundaiPlatformConfig(
    [HyundaiCarDocs("Kia Optima Hybrid 2017", "Advanced Smart Cruise Control", car_parts=CarParts.common([CarHarness.hyundai_c]))],
    CarSpecs(mass=3558 * CV.LB_TO_KG, wheelbase=2.8, steerRatio=13.75, tireStiffnessFactor=0.5),
    flags=HyundaiFlags.HYBRID | HyundaiFlags.LEGACY,
  )
  KIA_OPTIMA_H_G4_FL = HyundaiPlatformConfig(
    [HyundaiCarDocs("Kia Optima Hybrid 2019", car_parts=CarParts.common([CarHarness.hyundai_h]))],
    CarSpecs(mass=3558 * CV.LB_TO_KG, wheelbase=2.8, steerRatio=13.75, tireStiffnessFactor=0.5),
    flags=HyundaiFlags.HYBRID | HyundaiFlags.UNSUPPORTED_LONGITUDINAL,
  )
  KIA_SELTOS = HyundaiPlatformConfig(
    [HyundaiCarDocs("Kia Seltos 2021", car_parts=CarParts.common([CarHarness.hyundai_a]))],
    CarSpecs(mass=1337, wheelbase=2.63, steerRatio=14.56),
    flags=HyundaiFlags.CHECKSUM_CRC8,
  )
  KIA_SPORTAGE_5TH_GEN = HyundaiCanFDPlatformConfig(
    [
      HyundaiCarDocs("Kia Sportage 2023-24", car_parts=CarParts.common([CarHarness.hyundai_n])),
      HyundaiCarDocs("Kia Sportage Hybrid 2023", car_parts=CarParts.common([CarHarness.hyundai_n])),
    ],
    # weight from SX and above trims, average of FWD and AWD version, steering ratio according to Kia News https://www.kiamedia.com/us/en/models/sportage/2023/specifications
    CarSpecs(mass=1725, wheelbase=2.756, steerRatio=13.6),
  )
  KIA_SORENTO = HyundaiPlatformConfig(
    [
      HyundaiCarDocs("Kia Sorento 2018", "Advanced Smart Cruise Control & LKAS", video_link="https://www.youtube.com/watch?v=Fkh3s6WHJz8",
                     car_parts=CarParts.common([CarHarness.hyundai_e])),
      HyundaiCarDocs("Kia Sorento 2019", video_link="https://www.youtube.com/watch?v=Fkh3s6WHJz8", car_parts=CarParts.common([CarHarness.hyundai_e])),
    ],
    CarSpecs(mass=1985, wheelbase=2.78, steerRatio=14.4 * 1.1),  # 10% higher at the center seems reasonable
    flags=HyundaiFlags.CHECKSUM_6B | HyundaiFlags.UNSUPPORTED_LONGITUDINAL,
  )
  KIA_SORENTO_4TH_GEN = HyundaiCanFDPlatformConfig(
    [HyundaiCarDocs("Kia Sorento 2021-23", car_parts=CarParts.common([CarHarness.hyundai_k]))],
    CarSpecs(mass=3957 * CV.LB_TO_KG, wheelbase=2.81, steerRatio=13.5),  # average of the platforms
    flags=HyundaiFlags.RADAR_SCC,
  )
  KIA_SORENTO_HEV_4TH_GEN = HyundaiCanFDPlatformConfig(
    [
      HyundaiCarDocs("Kia Sorento Hybrid 2021-23", "All", car_parts=CarParts.common([CarHarness.hyundai_a])),
      HyundaiCarDocs("Kia Sorento Plug-in Hybrid 2022-23", "All", car_parts=CarParts.common([CarHarness.hyundai_a])),
    ],
    CarSpecs(mass=4395 * CV.LB_TO_KG, wheelbase=2.81, steerRatio=13.5),  # average of the platforms
    flags=HyundaiFlags.RADAR_SCC,
  )
  KIA_STINGER = HyundaiPlatformConfig(
    [HyundaiCarDocs("Kia Stinger 2018-20", video_link="https://www.youtube.com/watch?v=MJ94qoofYw0",
                    car_parts=CarParts.common([CarHarness.hyundai_c]))],
    CarSpecs(mass=1825, wheelbase=2.78, steerRatio=14.4 * 1.15)  # 15% higher at the center seems reasonable
  )
  KIA_STINGER_2022 = HyundaiPlatformConfig(
    [HyundaiCarDocs("Kia Stinger 2022-23", "All", car_parts=CarParts.common([CarHarness.hyundai_k]))],
    KIA_STINGER.specs,
  )
  KIA_CEED = HyundaiPlatformConfig(
    [HyundaiCarDocs("Kia Ceed 2019", car_parts=CarParts.common([CarHarness.hyundai_e]))],
    CarSpecs(mass=1450, wheelbase=2.65, steerRatio=13.75, tireStiffnessFactor=0.5),
    flags=HyundaiFlags.LEGACY,
  )
  KIA_EV6 = HyundaiCanFDPlatformConfig(
    [
      HyundaiCarDocs("Kia EV6 (Southeast Asia only) 2022-23", "All", car_parts=CarParts.common([CarHarness.hyundai_p])),
      HyundaiCarDocs("Kia EV6 (without HDA II) 2022-23", "Highway Driving Assist", car_parts=CarParts.common([CarHarness.hyundai_l])),
      HyundaiCarDocs("Kia EV6 (with HDA II) 2022-23", "Highway Driving Assist II", car_parts=CarParts.common([CarHarness.hyundai_p]))
    ],
    CarSpecs(mass=2055, wheelbase=2.9, steerRatio=16, tireStiffnessFactor=0.65),
    flags=HyundaiFlags.EV,
  )
  KIA_CARNIVAL_4TH_GEN = HyundaiCanFDPlatformConfig(
    [
      HyundaiCarDocs("Kia Carnival 2022-24", car_parts=CarParts.common([CarHarness.hyundai_a])),
      HyundaiCarDocs("Kia Carnival (China only) 2023", car_parts=CarParts.common([CarHarness.hyundai_k]))
    ],
    CarSpecs(mass=2087, wheelbase=3.09, steerRatio=14.23),
    flags=HyundaiFlags.RADAR_SCC,
  )

  # Genesis
  GENESIS_GV60_EV_1ST_GEN = HyundaiCanFDPlatformConfig(
    [
      HyundaiCarDocs("Genesis GV60 (Advanced Trim) 2023", "All", car_parts=CarParts.common([CarHarness.hyundai_a])),
      HyundaiCarDocs("Genesis GV60 (Performance Trim) 2023", "All", car_parts=CarParts.common([CarHarness.hyundai_k])),
    ],
    CarSpecs(mass=2205, wheelbase=2.9, steerRatio=12.6),  # steerRatio: https://www.motor1.com/reviews/586376/2023-genesis-gv60-first-drive/#:~:text=Relative%20to%20the%20related%20Ioniq,5%2FEV6%27s%2014.3%3A1.
    flags=HyundaiFlags.EV,
  )
  GENESIS_G70 = HyundaiPlatformConfig(
    [HyundaiCarDocs("Genesis G70 2018-19", "All", car_parts=CarParts.common([CarHarness.hyundai_f]))],
    CarSpecs(mass=1640, wheelbase=2.84, steerRatio=13.56),
    flags=HyundaiFlags.LEGACY,
  )
  GENESIS_G70_2020 = HyundaiPlatformConfig(
    [
      # TODO: 2021 MY harness is unknown
      HyundaiCarDocs("Genesis G70 2020-21", "All", car_parts=CarParts.common([CarHarness.hyundai_f])),
      # TODO: From 3.3T Sport Advanced 2022 & Prestige 2023 Trim, 2.0T is unknown
      HyundaiCarDocs("Genesis G70 2022-23", "All", car_parts=CarParts.common([CarHarness.hyundai_l])),
    ],
    CarSpecs(mass=3673 * CV.LB_TO_KG, wheelbase=2.83, steerRatio=12.9),
    flags=HyundaiFlags.MANDO_RADAR,
  )
  GENESIS_GV70_1ST_GEN = HyundaiCanFDPlatformConfig(
    [
      HyundaiCarDocs("Genesis GV70 (2.5T Trim) 2022-23", "All", car_parts=CarParts.common([CarHarness.hyundai_l])),
      HyundaiCarDocs("Genesis GV70 (3.5T Trim) 2022-23", "All", car_parts=CarParts.common([CarHarness.hyundai_m])),
    ],
    CarSpecs(mass=1950, wheelbase=2.87, steerRatio=14.6),
    flags=HyundaiFlags.RADAR_SCC,
  )
  GENESIS_G80 = HyundaiPlatformConfig(
    [HyundaiCarDocs("Genesis G80 2018-19", "All", car_parts=CarParts.common([CarHarness.hyundai_h]))],
    CarSpecs(mass=2060, wheelbase=3.01, steerRatio=16.5),
    flags=HyundaiFlags.LEGACY,
  )
  GENESIS_G90 = HyundaiPlatformConfig(
    [HyundaiCarDocs("Genesis G90 2017-20", "All", car_parts=CarParts.common([CarHarness.hyundai_c]))],
    CarSpecs(mass=2200, wheelbase=3.15, steerRatio=12.069),
  )
  GENESIS_GV80 = HyundaiCanFDPlatformConfig(
    [HyundaiCarDocs("Genesis GV80 2023", "All", car_parts=CarParts.common([CarHarness.hyundai_m]))],
    CarSpecs(mass=2258, wheelbase=2.95, steerRatio=14.14),
    flags=HyundaiFlags.RADAR_SCC,
  )


class Buttons:
  NONE = 0
  RES_ACCEL = 1
  SET_DECEL = 2
  GAP_DIST = 3
  CANCEL = 4  # on newer models, this is a pause/resume button


def get_platform_codes(fw_versions: list[bytes]) -> set[tuple[bytes, bytes | None]]:
  # Returns unique, platform-specific identification codes for a set of versions
  codes = set()  # (code-Optional[part], date)
  for fw in fw_versions:
    code_match = PLATFORM_CODE_FW_PATTERN.search(fw)
    part_match = PART_NUMBER_FW_PATTERN.search(fw)
    date_match = DATE_FW_PATTERN.search(fw)
    if code_match is not None:
      code: bytes = code_match.group()
      part = part_match.group() if part_match else None
      date = date_match.group() if date_match else None
      if part is not None:
        # part number starts with generic ECU part type, add what is specific to platform
        code += b"-" + part[-5:]

      codes.add((code, date))
  return codes


def match_fw_to_car_fuzzy(live_fw_versions, offline_fw_versions) -> set[str]:
  # Non-electric CAN FD platforms often do not have platform code specifiers needed
  # to distinguish between hybrid and ICE. All EVs so far are either exclusively
  # electric or specify electric in the platform code.
  fuzzy_platform_blacklist = {str(c) for c in (CANFD_CAR - EV_CAR - CANFD_FUZZY_WHITELIST)}
  candidates: set[str] = set()

  for candidate, fws in offline_fw_versions.items():
    # Keep track of ECUs which pass all checks (platform codes, within date range)
    valid_found_ecus = set()
    valid_expected_ecus = {ecu[1:] for ecu in fws if ecu[0] in PLATFORM_CODE_ECUS}
    for ecu, expected_versions in fws.items():
      addr = ecu[1:]
      # Only check ECUs expected to have platform codes
      if ecu[0] not in PLATFORM_CODE_ECUS:
        continue

      # Expected platform codes & dates
      codes = get_platform_codes(expected_versions)
      expected_platform_codes = {code for code, _ in codes}
      expected_dates = {date for _, date in codes if date is not None}

      # Found platform codes & dates
      codes = get_platform_codes(live_fw_versions.get(addr, set()))
      found_platform_codes = {code for code, _ in codes}
      found_dates = {date for _, date in codes if date is not None}

      # Check platform code + part number matches for any found versions
      if not any(found_platform_code in expected_platform_codes for found_platform_code in found_platform_codes):
        break

      if ecu[0] in DATE_FW_ECUS:
        # If ECU can have a FW date, require it to exist
        # (this excludes candidates in the database without dates)
        if not len(expected_dates) or not len(found_dates):
          break

        # Check any date within range in the database, format is %y%m%d
        if not any(min(expected_dates) <= found_date <= max(expected_dates) for found_date in found_dates):
          break

      valid_found_ecus.add(addr)

    # If all live ECUs pass all checks for candidate, add it as a match
    if valid_expected_ecus.issubset(valid_found_ecus):
      candidates.add(candidate)

  return candidates - fuzzy_platform_blacklist


HYUNDAI_VERSION_REQUEST_LONG = bytes([uds.SERVICE_TYPE.READ_DATA_BY_IDENTIFIER]) + \
  p16(0xf100)  # Long description

HYUNDAI_VERSION_REQUEST_ALT = bytes([uds.SERVICE_TYPE.READ_DATA_BY_IDENTIFIER]) + \
  p16(0xf110)  # Alt long description

HYUNDAI_ECU_MANUFACTURING_DATE = bytes([uds.SERVICE_TYPE.READ_DATA_BY_IDENTIFIER]) + \
  p16(uds.DATA_IDENTIFIER_TYPE.ECU_MANUFACTURING_DATE)

HYUNDAI_VERSION_RESPONSE = bytes([uds.SERVICE_TYPE.READ_DATA_BY_IDENTIFIER + 0x40])

# Regex patterns for parsing platform code, FW date, and part number from FW versions
PLATFORM_CODE_FW_PATTERN = re.compile(b'((?<=' + HYUNDAI_VERSION_REQUEST_LONG[1:] +
                                      b')[A-Z]{2}[A-Za-z0-9]{0,2})')
DATE_FW_PATTERN = re.compile(b'(?<=[ -])([0-9]{6}$)')
PART_NUMBER_FW_PATTERN = re.compile(b'(?<=[0-9][.,][0-9]{2} )([0-9]{5}[-/]?[A-Z][A-Z0-9]{3}[0-9])')

# We've seen both ICE and hybrid for these platforms, and they have hybrid descriptors (e.g. MQ4 vs MQ4H)
CANFD_FUZZY_WHITELIST = {CAR.KIA_SORENTO_4TH_GEN, CAR.KIA_SORENTO_HEV_4TH_GEN, CAR.KIA_K8_HEV_1ST_GEN,
                         # TODO: the hybrid variant is not out yet
                         CAR.KIA_CARNIVAL_4TH_GEN}

# List of ECUs expected to have platform codes, camera and radar should exist on all cars
# TODO: use abs, it has the platform code and part number on many platforms
PLATFORM_CODE_ECUS = [Ecu.fwdRadar, Ecu.fwdCamera, Ecu.eps]
# So far we've only seen dates in fwdCamera
# TODO: there are date codes in the ABS firmware versions in hex
DATE_FW_ECUS = [Ecu.fwdCamera]

FW_QUERY_CONFIG = FwQueryConfig(
  requests=[
    # TODO: add back whitelists
    # CAN queries (OBD-II port)
    Request(
      [HYUNDAI_VERSION_REQUEST_LONG],
      [HYUNDAI_VERSION_RESPONSE],
    ),

    # CAN & CAN-FD queries (from camera)
    Request(
      [HYUNDAI_VERSION_REQUEST_LONG],
      [HYUNDAI_VERSION_RESPONSE],
      bus=0,
      auxiliary=True,
    ),
    Request(
      [HYUNDAI_VERSION_REQUEST_LONG],
      [HYUNDAI_VERSION_RESPONSE],
      bus=1,
      auxiliary=True,
      obd_multiplexing=False,
    ),

    # CAN & CAN FD query to understand the three digit date code
    # HDA2 cars usually use 6 digit date codes, so skip bus 1
    Request(
      [HYUNDAI_ECU_MANUFACTURING_DATE],
      [HYUNDAI_VERSION_RESPONSE],
      bus=0,
      auxiliary=True,
      logging=True,
    ),

    # CAN-FD alt request logging queries for hvac and parkingAdas
    Request(
      [HYUNDAI_VERSION_REQUEST_ALT],
      [HYUNDAI_VERSION_RESPONSE],
      bus=0,
      auxiliary=True,
      logging=True,
    ),
    Request(
      [HYUNDAI_VERSION_REQUEST_ALT],
      [HYUNDAI_VERSION_RESPONSE],
      bus=1,
      auxiliary=True,
      logging=True,
      obd_multiplexing=False,
    ),
  ],
  # We lose these ECUs without the comma power on these cars.
  # Note that we still attempt to match with them when they are present
  non_essential_ecus={
    Ecu.abs: [CAR.HYUNDAI_PALISADE, CAR.HYUNDAI_SONATA, CAR.HYUNDAI_SANTA_FE_2022, CAR.KIA_K5_2021, CAR.HYUNDAI_ELANTRA_2021,
              CAR.HYUNDAI_SANTA_FE, CAR.HYUNDAI_KONA_EV_2022, CAR.HYUNDAI_KONA_EV, CAR.HYUNDAI_CUSTIN_1ST_GEN, CAR.KIA_SORENTO,
              CAR.KIA_CEED, CAR.KIA_SELTOS],
  },
  extra_ecus=[
    (Ecu.adas, 0x730, None),              # ADAS Driving ECU on HDA2 platforms
    (Ecu.parkingAdas, 0x7b1, None),       # ADAS Parking ECU (may exist on all platforms)
    (Ecu.hvac, 0x7b3, None),              # HVAC Control Assembly
    (Ecu.cornerRadar, 0x7b7, None),
    (Ecu.combinationMeter, 0x7c6, None),  # CAN FD Instrument cluster
  ],
  # Custom fuzzy fingerprinting function using platform codes, part numbers + FW dates:
  match_fw_to_car_fuzzy=match_fw_to_car_fuzzy,
)

CHECKSUM = {
  "crc8": CAR.with_flags(HyundaiFlags.CHECKSUM_CRC8),
  "6B": CAR.with_flags(HyundaiFlags.CHECKSUM_6B),
}

CAN_GEARS = {
  # which message has the gear. hybrid and EV use ELECT_GEAR
  "use_cluster_gears": CAR.with_flags(HyundaiFlags.CLUSTER_GEARS),
  "use_tcu_gears": CAR.with_flags(HyundaiFlags.TCU_GEARS),
}

CANFD_CAR = CAR.with_flags(HyundaiFlags.CANFD)
CANFD_RADAR_SCC_CAR = CAR.with_flags(HyundaiFlags.RADAR_SCC)

# These CAN FD cars do not accept communication control to disable the ADAS ECU,
# responds with 0x7F2822 - 'conditions not correct'
CANFD_UNSUPPORTED_LONGITUDINAL_CAR = CAR.with_flags(HyundaiFlags.CANFD_NO_RADAR_DISABLE)

# The camera does SCC on these cars, rather than the radar
CAMERA_SCC_CAR = CAR.with_flags(HyundaiFlags.CAMERA_SCC)

HYBRID_CAR = CAR.with_flags(HyundaiFlags.HYBRID)

EV_CAR = CAR.with_flags(HyundaiFlags.EV)

LEGACY_SAFETY_MODE_CAR = CAR.with_flags(HyundaiFlags.LEGACY)

UNSUPPORTED_LONGITUDINAL_CAR = CAR.with_flags(HyundaiFlags.LEGACY) | CAR.with_flags(HyundaiFlags.UNSUPPORTED_LONGITUDINAL)

DBC = CAR.create_dbc_map()

if __name__ == "__main__":
  CAR.print_debug(HyundaiFlags)
