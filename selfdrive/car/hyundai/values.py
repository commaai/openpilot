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
  # KIA_NIRO_HEV_2021
  # ('fwdCamera', 'fwdRadar'): routes: 428, dongles: {'1ea7eb816544f6c0', '92daf576a7eab465', '9c047114df2bdd4b', 'bd6b3c208fa1b9ef'}
  # ---
  # ('eps', 'fwdCamera', 'fwdRadar'): routes: 214, dongles: {'1ea7eb816544f6c0', '92daf576a7eab465', 'bd6b3c208fa1b9ef', '9c047114df2bdd4b'}
  # ---
  # ('fwdCamera',): routes: 1, dongles: {'c7c920d630c75e71'}
  # ---
  #
  # KIA_SELTOS
  # ('fwdCamera', 'fwdRadar'): routes: 56, dongles: {'b40d441148bfeb27', '54f0bbcb6527012e', '4cbcc16b655c1591'}
  # ---
  #
  # HYUNDAI_PALISADE
  # ('fwdCamera', 'fwdRadar'): routes: 7656, dongles: {'8b8d0edc013a0414', 'c5f915f8d3049fd4', '72a4e9db41db1a5c', '1ee9ef621f9077ca', 'e0c6f1daf613f27e', 'ebdfcd4184611cfd', '2156372967ebd7b8', '0af43ba62cc3ffc4', '3b75644c3b35d357', '86206fdc559dc641', '2d9700d4aa561ee5', '028e246a2b8e3144', 'c502b178b93bcc57', '0cc137ccd450204c', '858b2777504c59dd', '7b5a53e99b76e8d9', '9fd1ebeaa22616a6', 'b700d19967c0136b', '1c5b0ef6f666062d', 'e859febc78eb3372', '5d62cdce57ed5471', '31c421b9c0b2decf', 'bff778f8a1a0216d', '8e3031f91eaff400', 'b6790c2c9bfb117c', '3a781a089b7b2aac', '00b0178ab5154409', 'b64e506d0ccb4cdb', 'aa9fb1580b0e9fbf', '2c6a281b9ece6b63', '7272d111d521a7bc', '26d9f72dcde82c9c', 'd3ef9b7c8fc585a3', 'ecdd3760093058f1', '5a8e655cba86e6f1', '1d71eb3f940d3aa9', 'df5a2fe9ba59812a', '0cebe66cb649a2cd', '2e8f7c0381eb4f65', '7a6617bb650ac5f3', '4adc4b931611453a', '628935d7d3e5f4f7', 'd23a555519923793', 'a03aa1ab8800e402', 'c47c7d2bd9df6df7', '1336b7c61f4fba8b', 'fb3784f688906f36', '3f3515b88af7ab0a', 'baf9c744e8d3aedc', '8446c299c2daac81', 'b45bed1efbeb6fe3', '67a711f176acbbbf', '39070cb41aafdfc1', 'f02bf3c0f2ec038e', '075d96dd1286fdc9', 'e90590819973b0ce', '13587a8c52a6a9d0', '06c9fd1db117bd1e', '853ace5402453ce8', '43cb86fe707385e3', 'd1ef77422934c901', '32d7d577272f3de1', '9e7951200bd89ca3', '8d663db5b69a9cda', 'ebd4aa71ac92b65a', '7c1e916d88c104b0', 'f29593a9c190b384', '9a21244c25e1eb86', '4b8ae689eaa74602', '29b10f522dd0aba9', 'bade6cd7ae9bb600', 'c7d0d7dde77d49ce', 'd542a385e07a7863', '6c6bf0fba0504114', '4c6b27a2c49c555e', '829fc01ecb03f524', 'afe94a76bd8f5676', 'ea0d8e2a4cf9ce46', '6026525261343755', '757eef2aea1ab504', 'c9a00a85dd4124e0', '366df0f08bbf18b8', '221c98c2de35bec1', 'ecdc140c384bc3fb', 'ee6cdcf74e577f1f', '2c63cbbbba450ba0', '8f4ad96cc3a8c0eb', '962de50077e1e05e', '9a22857aff5535d8', 'cf101b7bfcebcbfc', '058ff2b75b84c4b6', 'ebf413d369473f7a', '325d76d16fb6cb36', '22a05b58157a0d7f', '98025ffdb4d0d3ea', 'f18ec52171747511', '5dbd6fdbc707d693', 'fd0fd9377e924058', '89ec97888874fee6'}
  # ---
  # ('eps', 'fwdCamera', 'fwdRadar'): routes: 2846, dongles: {'9a21244c25e1eb86', '4b8ae689eaa74602', '8b8d0edc013a0414', 'bade6cd7ae9bb600', 'c7d0d7dde77d49ce', 'b700d19967c0136b', '7a6617bb650ac5f3', '2e8f7c0381eb4f65', '829fc01ecb03f524', '1c5b0ef6f666062d', 'd23a555519923793', 'afe94a76bd8f5676', 'ea0d8e2a4cf9ce46', '6026525261343755', '757eef2aea1ab504', 'e20940ded141462d', 'bfd712f95f968e5d', 'f8031a15d4fded47', '1336b7c61f4fba8b', 'aea60028712346c2', 'fb3784f688906f36', '3f3515b88af7ab0a', '1ee9ef621f9077ca', '366df0f08bbf18b8', 'e0c6f1daf613f27e', 'ecdc140c384bc3fb', 'ee6cdcf74e577f1f', 'baf9c744e8d3aedc', 'ebdfcd4184611cfd', '8f4ad96cc3a8c0eb', '0af43ba62cc3ffc4', 'd29f746d6e305def', '3b75644c3b35d357', '9a22857aff5535d8', 'bff778f8a1a0216d', '8446c299c2daac81', '67a711f176acbbbf', '8e3031f91eaff400', 'cf101b7bfcebcbfc', '2d9700d4aa561ee5', '058ff2b75b84c4b6', '974bae894627779c', 'b64e506d0ccb4cdb', '075d96dd1286fdc9', '325d76d16fb6cb36', '2c6a281b9ece6b63', 'ba9a92a5c3325b3e', '028e246a2b8e3144', '7272d111d521a7bc', 'cc20792f8f18dee1', '26d9f72dcde82c9c', '32d7d577272f3de1', '9e7951200bd89ca3', 'd3ef9b7c8fc585a3', '8d663db5b69a9cda', '0cc137ccd450204c', '70f55e7f81cbf980', 'f29593a9c190b384'}
  # ---
  # ('fwdCamera',): routes: 64, dongles: {'0af43ba62cc3ffc4', '4adc4b931611453a'}
  # ---
  # (): routes: 1, dongles: {'67a711f176acbbbf'}
  # ---
  #
  # HYUNDAI_SANTA_FE
  # ('fwdCamera', 'fwdRadar'): routes: 1966, dongles: {'9ebc35731309da40', 'ee68f816a9d743ee', '32669138d261f344', '80a13eff7d046d45', '21c578cf98555ce9', '2f475f484c7e48a4', 'd0a70b78cf608a2e', '808e486fe83d0887', '4dbd55df87507948', '70a73b2fbd820de4', '238b93c156d98783', 'dec47dd6b2d62da2', '977ee92cd449d6ec', '45a8ffd448481ce2', '948abf2ebb78db3e', '6af67ee8ad45b46d', '7e7e19d892ed133b', '6c9fd2d474d55d94', 'e56ffe1d4cbc4f6c', '6ae91538768a61ff', 'c4725ef6c5755a72', 'b40cd4b04d6ada18', '05a25f3f679fe017', 'e4439ccaf28c3803', '49fd0ed02059816b', '0be63a5a6048987a', '193322ee1d7febe6', '833f9f3ab7f33b99', 'f94e3189ddb83d83', '4ea994afc80a04cf', '23d83154b19ae34b', 'fddc8d2cbb48ec6f', '00ef537a9362ae90', '124457b31149107e', '690ef0b220b8ddc3', '0e8e0a2e14e2ed6d', '6cc716ffa2eb1f32', 'ea9d3b13a4799604', '80e51b2ac7f49b7e', '0b2b36cf5a0789db', 'fc1ee927d98acc85', 'c743c23ef23e25d2'}
  # ---
  # ('eps', 'fwdCamera', 'fwdRadar'): routes: 758, dongles: {'9ebc35731309da40', 'ee68f816a9d743ee', '32669138d261f344', 'fa2c57632891fe1c', '2f475f484c7e48a4', 'd0a70b78cf608a2e', '45a8ffd448481ce2', '6af67ee8ad45b46d', '7e7e19d892ed133b', 'a647223762c0a9d5', '6c9fd2d474d55d94', 'e56ffe1d4cbc4f6c', 'c4725ef6c5755a72', 'e4439ccaf28c3803', '193322ee1d7febe6', '4ea994afc80a04cf', 'fac40c52c574f1af', '124457b31149107e', '0b2b36cf5a0789db'}
  # ---
  # ('fwdCamera',): routes: 193, dongles: {'70a73b2fbd820de4'}
  # ---
  # (): routes: 2, dongles: {'dec47dd6b2d62da2'}
  # ---
  #
  # HYUNDAI_SONATA
  # ('fwdCamera', 'fwdRadar'): routes: 6021, dongles: {'a980018466c6a475', '8d162f50f95b7128', '3bbfb3b75809e9e6', 'ffce4a1d578827ee', '0cc2c91bedd65cb1', 'b4f037b0efc74833', '83f627fc1e48c846', '23b08b05f7c0ae62', 'ba75a2db7d699895', 'ccc8ace0ff24f76f', '10b8dcf2fb80bf56', '81dd8a032a288ea7', '5307ee3c32148a5d', 'b938870bef07f799', '8eabda41519b437c', '446c796b4c55e0fe', '310cf131d28b30d8', '73ed44f1789eddd4', 'e4ad125bb7fe4ef0', 'a43df3b5d0c31a62', '026c9c73fdebe8c5', '327a12efd3455a87', '1edd7a8340016f2c', 'd8fc7efbb8e99bd3', '858b2777504c59dd', '3222c67c59b0bf40', '7dd1f8f7f731715a', '6ed00ffb9c23da95', '4b2d45ebeaf6be75', 'e25761b0b07189ce', '2bbe12792a30f61f', '38a36c2893f16967', 'b76c43f138cb09ac', '8aeee770cbcbce6c', '53cd8a9d420e11b2', '996fb8fd4c51d83d', 'aaf2f8e95bfb437f', '5b337c203559c300', '145908dabb729ad9', '92b9b216497d3ab0', '4960a9f1dd71aad1', '87aed74d45ea588f', 'e5904ea020532291', '530f1d69133b0c56', '70e370ca9294070a', 'd63936f7ce28d591', 'e7803f8f3c75dcbd', 'c86ea37c4a66ea6e', '700d37ccd12315cf', '5efd04bc2de5012a', '1caaacb72ffc04ca', 'e7631b88da08c63d', '2c8bd13a84d90687', 'a8078ee810ef83d6', '6ebd515c4f1b0911', 'e376a6928c50f02a', '62d28a68d5f04f0e', 'a15bf814a6e3acc1', '9aeb5cdf94065064', 'e6dd11f4969e1f74', 'cb7eee5264754a21', 'c11af183f22aef82', 'b43c8ab20b852edb', '5653c148eb343074', 'e425a1ba2f5f42b3', '40812bc457628b70', 'e82ec95ae4597498', '376e07c788b8072d', 'b2978e5a16b815e7', '981a14c647ac07ec', 'd3d0f4f5fa7e283d', '053bf385cdad1079', '00d6a4fc961adca2', '0a84b23d6af1b1d2', '849e52c3a7f5b687', 'a02a3dd8ad858fa3', '7ef7d92481ec0ace', '4b157737c2154617', '2c1f4ef2c30e209f', '20e010d4cd035957', 'a36807834c913deb', '04dbe1e3cbd179c6', 'df2c8cf99faefa89', 'b3a277e738da16d1', 'ee11091550c9a627', '7266220717f4c8e1', '514e0a6c4a71fa21', '287e3a3caf5f69d0', '89f56aec25cccf29', '4a612c47eb83412a', '485ae27117c7f156', '4df896aa825bd9b3', '026e1eb462c28560', '163638af4b53f555', '139b3da68ed4b976', '67cafbae8ee7f156', 'b0af2fff166d80fe', 'ad68728bc42b1357', '14118dc2aced6462'}
  # ---
  # ('eps', 'fwdCamera', 'fwdRadar'): routes: 1891, dongles: {'67cc2f476f7513d1', 'a8078ee810ef83d6', 'e376a6928c50f02a', '2c1f4ef2c30e209f', '4b2d45ebeaf6be75', 'e25761b0b07189ce', 'ffce4a1d578827ee', '2bbe12792a30f61f', '0cc2c91bedd65cb1', 'b4f037b0efc74833', '287e3a3caf5f69d0', '38a36c2893f16967', 'b76c43f138cb09ac', '9aeb5cdf94065064', '83f627fc1e48c846', 'cb7eee5264754a21', '8aeee770cbcbce6c', '53cd8a9d420e11b2', '4a612c47eb83412a', '996fb8fd4c51d83d', 'e425a1ba2f5f42b3', 'a3488ea00ac3bacc', 'aaf2f8e95bfb437f', '145908dabb729ad9', '40812bc457628b70', 'e82ec95ae4597498', '4960a9f1dd71aad1', 'b2978e5a16b815e7', '4df896aa825bd9b3', '026e1eb462c28560', 'e5904ea020532291', 'b938870bef07f799', '530f1d69133b0c56', '163638af4b53f555', '446c796b4c55e0fe', '053bf385cdad1079', 'd63936f7ce28d591', '310cf131d28b30d8', '139b3da68ed4b976', 'c86ea37c4a66ea6e', 'b0af2fff166d80fe', '700d37ccd12315cf', 'ed882557605319f9', '73ed44f1789eddd4', '0e09326bb9b6243b', '20caa585ef1501af', 'e4ad125bb7fe4ef0', '327a12efd3455a87', 'ad68728bc42b1357', 'e7631b88da08c63d', 'a02a3dd8ad858fa3', '7ef7d92481ec0ace', '4b157737c2154617', '4203853bc02e087a'}
  # ---
  # (): routes: 2, dongles: {'a980018466c6a475', 'a02a3dd8ad858fa3'}
  # ---
  # ('fwdRadar',): routes: 2, dongles: {'1edd7a8340016f2c'}
  # ---
  #
  # HYUNDAI_SANTA_FE_PHEV_2022
  # ('fwdCamera', 'fwdRadar'): routes: 727, dongles: {'1d10de75f6b8c570', 'd61540bf6f3ff5fd', '79e421cb62e1a7c3', '0a0c07ce1b42ec2a', '4e77fe0c3543bda2', '70b536268eb13ae0', '360204e2fa09bf6e'}
  # ---
  # ('eps', 'fwdCamera', 'fwdRadar'): routes: 392, dongles: {'1d10de75f6b8c570', 'd61540bf6f3ff5fd', '79e421cb62e1a7c3', '806e976752150abb', '771305d736516526', '4e77fe0c3543bda2', '70b536268eb13ae0', 'db233473abea6579'}
  # ---
  #
  # KIA_NIRO_EV
  # ('fwdCamera', 'fwdRadar'): routes: 3636, dongles: {'e0309a9e3f5ebb64', '6111bd7df5043394', '798f46c58bcb31ed', '3fd9853538e1bca9', 'c8ac983380029643', '40725113b3e20872', 'aaa581caadd328cb', '06d2b4f33202cc03', 'c04b1653edb84818', '83bbd4932913dcf7', '479079ea5ca68a43', '59d1092b5c09463c', '1ca01c0dd922973e', 'c6ea71302e1d9207', 'c575356a73bcae39', 'de9c9f27a0ecc479', '86a67c5051898f5d', 'e03bac4aa15ad36d', '6d2b75c2e67bf314', 'df1fef486ed5c0df', '28c139b629546773', '9c23d5ea49e0ab3a', 'e21fee4d299fcd99', '4a57d256c3691cb0', 'bb5b755fe317bdf5', '36abc44c4f296c7a', 'a73b0c67231730c6', '05b1a2556c5bc638', '668333f65d0bd1be', '21327590bd75de12', '89c7460182ce76eb', 'c8f86b163152d2c5', 'bc0db502f71a9b95', '3c8bfd637f561f13', 'b382b3c7a6e6c4a3', '0a682ec895905510', '512725b0cb4e6895', '5767bea945b8fc99', 'ae33e3a539c7d3d1', '6d3c0a569958cc0d', '97c0acb6a1e19087', 'b27d9eafeb61976e', 'b5cebe8c56d710c5', 'c5ae98c15b8f5509', '549bb645afc16a95', '33152b5ff07c9960', '561e5f3991916e4b', '3828b131c4c0ca83', 'a1b9acdbb4c2fa86', 'd5f923604b1b2f75', '051378e6086bc3ef', '490b5193eb790bba', 'fc4cfe20b331a575', 'b576d2ff8a193b4a', '77179c4da5d47d13'}
  # ---
  # ('eps', 'fwdCamera', 'fwdRadar'): routes: 619, dongles: {'e0309a9e3f5ebb64', '6111bd7df5043394', '06d2b4f33202cc03', 'b04d91a78f6e4fd5', 'c04b1653edb84818', '479079ea5ca68a43', '59d1092b5c09463c', 'de9c9f27a0ecc479', '86a67c5051898f5d', 'e21fee4d299fcd99', '4a57d256c3691cb0', '9c23d5ea49e0ab3a', 'bb5b755fe317bdf5', 'cc70806dd381f78a', '05b1a2556c5bc638', '89c7460182ce76eb', 'bc0db502f71a9b95', '3c8bfd637f561f13', 'b382b3c7a6e6c4a3', 'f66bfe48e9b4880a', '0a682ec895905510', '512725b0cb4e6895', '5767bea945b8fc99', 'ff097dd66b19579c', '97c0acb6a1e19087', 'b27d9eafeb61976e', 'b1f2aa9141431844', '0b552c5fc21ba264', 'c5ae98c15b8f5509', '99fa3d967af85623', '33152b5ff07c9960', '561e5f3991916e4b', 'dbda33c67462b907', 'a1b9acdbb4c2fa86', 'd5f923604b1b2f75', 'b576d2ff8a193b4a'}
  # ---
  # ('fwdCamera',): routes: 34, dongles: {'89c7460182ce76eb', 'e0309a9e3f5ebb64'}
  # ---
  # ('fwdRadar',): routes: 5, dongles: {'512725b0cb4e6895', 'e21fee4d299fcd99'}
  # ---
  # (): routes: 3, dongles: {'21327590bd75de12', '490b5193eb790bba'}
  # ---
  #
  # HYUNDAI_ELANTRA_HEV_2021
  # ('fwdCamera', 'fwdRadar'): routes: 1238, dongles: {'03c8f203de4d7d2d', '2b807b128bd53b77', 'facb73b86cf7f8d1', 'eb5a50f9e98f50b9', 'd4fe4673a1121600', '9e00d5949691495a', '258787f1c04277d4', 'b7e0416e72cba2f0', '8d49af2b041f8c5c', '97a7d053cdba9cba', '786fc028c014be71', 'ef8d357a38dc4cf2', 'd243865d2bd85bfb', 'f22818fb0dbd850d', '0c0bea43f366283f', 'e33e369bf0cac9f5', '53e9c98845d226f9', '832cc0633a84daa1', '7c85ca5f01f0889d', '616d9fb1d9ad2f82', 'e72c47871c46dde1'}
  # ---
  # ('eps', 'fwdCamera', 'fwdRadar'): routes: 508, dongles: {'8d49af2b041f8c5c', '0c0bea43f366283f', '97a7d053cdba9cba', 'eb5a50f9e98f50b9', '786fc028c014be71', '832cc0633a84daa1', 'd4fe4673a1121600', '9e00d5949691495a', 'f22818fb0dbd850d', 'e72c47871c46dde1', '43e4e476d745f104'}
  # ---
  #
  # HYUNDAI_SANTA_FE_2022
  # ('fwdCamera', 'fwdRadar'): routes: 2748, dongles: {'406bca51919f5b64', '69fb86b6677ce882', '27b7449c2dce04e6', '8304ab5048bd9189', '12b024a59cb15c68', '4983362ff1eaa006', 'dae9868756e02256', '3a6141a1328114b5', 'd26cf2f9a61ea57c', '305cf2a0bda0e51c', 'e93179f7322833f5', '2d394786f239cd0d', '95f5825be5200454', 'cc2f3c4ab6d7758b', '35c5508223abd40e', '2fcc9a43453b10de', '7e73a28494c9f94c', 'ed184604c559e2b9', '276c02aa51bc4469', '73bcac1960e3f6a4', '80f82c340254078b', '7a6ba99ba49954c0', '703a61579df52dbb', 'f3f1b2264f01f91d', '3d231d61a9f61a15', '55bc44600ca7d363', 'e37432b18654a7cc', '1ad7728ce54a161d', '2cd5c775e97b2458', 'b366a70903075727', 'f58f9a3379ede4db', '7a438df7ac795547', 'aa21d8e35f12e355', '288c2e61630f5579', 'b848ccad8ec7b386'}
  # ---
  # ('eps', 'fwdCamera', 'fwdRadar'): routes: 1037, dongles: {'406bca51919f5b64', '1493c38de09a381a', '4983362ff1eaa006', '305cf2a0bda0e51c', '42d0a73cdaa3017f', '2d394786f239cd0d', 'cc2f3c4ab6d7758b', '35c5508223abd40e', '2fcc9a43453b10de', '7e73a28494c9f94c', '73bcac1960e3f6a4', '7a6ba99ba49954c0', '3d231d61a9f61a15', '1ad7728ce54a161d', 'b366a70903075727', 'aa21d8e35f12e355', '0cb48444dd180504', 'b848ccad8ec7b386', 'd2245f33562396ab'}
  # ---
  #
  # GENESIS_G70
  # ('fwdCamera', 'fwdRadar'): routes: 208, dongles: {'edbaee4bf8d33eee', '924578fbc5ebaf56'}
  # ---
  # ('eps', 'fwdCamera', 'fwdRadar'): routes: 171, dongles: {'edbaee4bf8d33eee', '924578fbc5ebaf56'}
  # ---
  #
  # HYUNDAI_ELANTRA_2021
  # ('fwdCamera', 'fwdRadar'): routes: 1346, dongles: {'be0bb3add22aeed0', '350d89a7bad7f485', '9acd533a0f402e80', '2e2f5a9861ed1f1e', '44b7ef15064b5315', '3ea622c3c0ec3055', '1fc029d3a01de6ef', '81ba2bbf97deef94', 'b162567974b45f7f', '449474d0a79c14aa', 'd97d984dc917b497', '8ced51bb9546adb2', '6738291f59172ed5', '8a53aa70d88313f1', '6d3ba99246844a6d', '115d01262ab19696', '24a3ace7c636bf0e', 'a968db582cdd3227', '35ddbdeaf5e88f7b', '3ac08dbb3c723ffb', 'f8ae566217016356', '24053b87bbeca585', 'b0e1cdf87262c7ad', 'd74fa784003c05bb', 'b8798e927311b269', '107859f44e667d1b'}
  # ---
  # ('eps', 'fwdCamera', 'fwdRadar'): routes: 327, dongles: {'3ea622c3c0ec3055', '350d89a7bad7f485', 'be0bb3add22aeed0', '9acd533a0f402e80', 'f8ae566217016356', '81ba2bbf97deef94', 'b0e1cdf87262c7ad', 'd74fa784003c05bb', '107859f44e667d1b'}
  # ---
  #
  # HYUNDAI_SONATA_HYBRID
  # ('fwdCamera', 'fwdRadar'): routes: 2180, dongles: {'f152e524f24c58d4', '61ef897671a29f5c', '6e7a4c6df6ca11f2', '0be0dd99b51ae322', 'be44434f27731ff8', '49734429dda9bbe7', '01d4ba35f8288743', '6f322e987d684501', '8aec24dd30004113', 'a59f9751edaae273', '9beab9beed9888f0', '5dccf40c6d8d548e', '7865c2bf230bc007', '66378310df523400', '154be163a64e21f9', 'ed408cab3aa8386f', 'd28f1771a6425d69', '8b23d4b9cf8e6724', '8a97281ea26ff4a5', '8d80f957cac14e64', 'df2433ef9f6dcaa5', '9dfc658742b23506', '077309a26bf98013', 'cc820fd0720354e1', '200ee8cbc26b4dd4', 'b4e74d12c2af777c', '1d08ecde44f58620', 'ef2928662c52b5fc', 'efb951d55ea9f49c', '7a629f5c75f83fa7', '86a0e7c8fa7c90c8', '979eb0f6c3682a58', '10702b68bb2db4f7', '9b275ba13e17ad21', '2fdb3b00e818dfd7', 'a2fc24eca6e7c7c9', 'b5374bdeea21f5ce', 'ee4e9a7706b91a9a', 'dfa8dfddbd414db9'}
  # ---
  # ('eps', 'fwdCamera', 'fwdRadar'): routes: 637, dongles: {'f152e524f24c58d4', '61ef897671a29f5c', 'be44434f27731ff8', '49734429dda9bbe7', '6f322e987d684501', '01d4ba35f8288743', '9beab9beed9888f0', '66378310df523400', '154be163a64e21f9', '6e687c49358fdbba', '8d80f957cac14e64', 'df2433ef9f6dcaa5', '077309a26bf98013', 'cc820fd0720354e1', '09e134816fc26a06', '7a629f5c75f83fa7', '86a0e7c8fa7c90c8', '2fdb3b00e818dfd7', 'a2fc24eca6e7c7c9', '0b5aea50884ae68c'}
  # ---
  #
  # GENESIS_G70_2020
  # ('fwdCamera', 'fwdRadar'): routes: 251, dongles: {'a9b72ac20ef04438', '9fafce4de77b8cdf', 'dc4472af9d9d3fc5', 'cfecbc6d0172f27e', '514931078c082d88', 'bc8bbc86829fc74d'}
  # ---
  # ('eps', 'fwdCamera', 'fwdRadar'): routes: 62, dongles: {'9fafce4de77b8cdf', '26de7b3c33289e36', 'dc4472af9d9d3fc5', 'cfecbc6d0172f27e', '514931078c082d88'}
  # ---
  #
  # KIA_K5_2021
  # ('fwdCamera', 'fwdRadar'): routes: 560, dongles: {'4359171e4fcdacb6', 'e8421f268ea89d79', '0c8443518d3d85f7', '2eacfc56537568f7', '3f77febac29e35d5', 'd35187278d48ed36', 'b581747bfa733c3f', '9d4fa1c83653b90b', '5f7bdb4edf6a2565', '0e7abb6a74b095d9', '0b91b433b9332780'}
  # ---
  # ('eps', 'fwdCamera', 'fwdRadar'): routes: 93, dongles: {'0c8443518d3d85f7', '3f77febac29e35d5', 'd35187278d48ed36', 'b581747bfa733c3f', 'f05351cfc44b2d73', '9d4fa1c83653b90b', '0b91b433b9332780'}
  # ---
  #
  # HYUNDAI_KONA_EV
  # ('fwdCamera', 'fwdRadar'): routes: 989, dongles: {'f90d3cd06caeb6fa', 'bcfb37afc11dd364', '39ff090a42469a72', '74075bb321bbbe6e', 'c9508b688d2009f6', '7b97fa92ec54079a', '753ceb77b4b6ab3c', '23ee9100e02d1cbf', '8caa49f5cc8f0e97', '75ca34822294ee1e', '71e5b594f13f7227', '498b45f4a6e9645b', '45ba82b7ecf23784', '15e5d08be3f89aeb', 'fcb975ffe122a0cc', '1d123e7a239cca62', '31fdc7e819aa0dad', '0d798915363d7aa8'}
  # ---
  # ('eps', 'fwdCamera', 'fwdRadar'): routes: 338, dongles: {'f90d3cd06caeb6fa', '74f130e93415b9e4', '39ff090a42469a72', 'c9508b688d2009f6', '8caa49f5cc8f0e97', '12fdf6595e41b23a', 'b4903e8325d6c7b7', '71e5b594f13f7227', '75ca34822294ee1e', '15e5d08be3f89aeb', '1d123e7a239cca62', '0d798915363d7aa8'}
  # ---
  # ('fwdCamera',): routes: 62, dongles: {'8a817d2073d02561'}
  # ---
  # (): routes: 9, dongles: {'71e5b594f13f7227'}
  # ---
  # ('eps', 'fwdCamera'): routes: 2, dongles: {'8a817d2073d02561'}
  # ---
  #
  # GENESIS_G80
  # ('fwdCamera', 'fwdRadar'): routes: 900, dongles: {'a204ef4b8bc8b99d', '02cd169686daae6b', 'ca5d5900f2d204b4', 'c1b51be2b98f90f0', 'f2a2cd3c3791d6d0', 'c7fdc5056b318b8c', 'b52a08d19fefd49e', 'ebe2849c14e6393d', '002fafcc41c97f09', 'cf65fc70fd41c9b6', 'bcf7d921bfdfe671', '450774a9a3ffe4c8', '2eefb4d20c7c1f62', 'd720059adc848c0a', '9b17f69eefc0356a', 'c6aaf0a8faa7f18b'}
  # ---
  #
  # HYUNDAI_SANTA_FE_HEV_2022
  # ('fwdCamera', 'fwdRadar'): routes: 1322, dongles: {'7df347260f92150a', '74bfd76e8c0e22d5', 'd624493e04d36bdd', '059ec91d3175b1ac', '129db7c75bce8445', 'abf2b22fb7a842e0', '2b1631519ac72e4e', '51d2c97ba9f48ff5', '13449b2f2ad4f094', '6e9dfed306da9625', '36921882a3bf0f19', '37398f32561a23ad', 'c350c659ec0badb2', '2acc6988a1fef659', '826b9eb4ac54a1c2', 'c879647c02650645'}
  # ---
  # ('eps', 'fwdCamera', 'fwdRadar'): routes: 277, dongles: {'2acc6988a1fef659', '826b9eb4ac54a1c2', '129db7c75bce8445', 'c879647c02650645'}
  # ---
  # (): routes: 2, dongles: {'13449b2f2ad4f094'}
  # ---
  #
  # KIA_SORENTO
  # ('fwdCamera', 'fwdRadar'): routes: 221, dongles: {'bfda978d009a7a95', '725165509799e0c0', '6b111e8e45cc2d07'}
  # ---
  # (): routes: 29, dongles: {'192283cdbb7a58c2'}
  # ---
  #
  # HYUNDAI_GENESIS
  # ('fwdCamera',): routes: 981, dongles: {'b7149f3cfe1b5e18', '445c144c95eb7409', 'c1b51be2b98f90f0', '122b828d984502d5', 'b6bb1d6e9e76d5c0', '06353521b1f136dd', '85eab094426c2649', '3687c15bdfe09295', '376339dc16dca97b', 'adb9062057548380', '59eb96159f0f5308', '619d8f607590cdf4', '61b3de3cce404316', '6ae633b02be43da6', 'f76dd0d4e0f9f487', '01a6c3c6195d7686'}
  # ---
  #
  # HYUNDAI_CUSTIN_1ST_GEN
  # ('fwdCamera', 'fwdRadar'): routes: 75, dongles: {'ff1f7656e09a1f4d', '3001a63b431270f5', '9d8e67717977a4ce', '025220a830f06f8b', '0f6a323578c97922'}
  # ---
  # ('fwdCamera',): routes: 26, dongles: {'9d8e67717977a4ce', 'ff1f7656e09a1f4d'}
  # ---
  # ('eps', 'fwdCamera', 'fwdRadar'): routes: 10, dongles: {'ff1f7656e09a1f4d'}
  # ---
  # ('eps', 'fwdCamera'): routes: 1, dongles: {'ff1f7656e09a1f4d'}
  # ---
  #
  # KIA_OPTIMA_G4_FL
  # ('fwdCamera', 'fwdRadar'): routes: 319, dongles: {'fec0d754d02e943e', '0832a9399cb84370'}
  # ---
  # ('abs', 'fwdCamera', 'fwdRadar'): routes: 77, dongles: {'9c179cb211aa1609', '0832a9399cb84370'}
  # ---
  #
  # KIA_STINGER
  # ('fwdCamera', 'fwdRadar'): routes: 509, dongles: {'8c809cffb861be71', '5ccb50cda9fba8b7', 'cf93b50f6f132cb0', 'b5768b7348113531', 'd128ed82bf188c22', 'e1f176f706568ff1', 'd9e9969ca4fcda58', '0fbdf8b5634d2566', '641dc70a82bcb9c7'}
  # ---
  # ('eps', 'fwdCamera', 'fwdRadar'): routes: 225, dongles: {'8c809cffb861be71', '5ccb50cda9fba8b7', 'cf93b50f6f132cb0', 'd128ed82bf188c22', 'e1f176f706568ff1', '0fbdf8b5634d2566', '641dc70a82bcb9c7', '9aa822866b68f494'}
  # ---
  # ('fwdRadar',): routes: 3, dongles: {'641dc70a82bcb9c7'}
  # ---
  #
  # HYUNDAI_ELANTRA_GT_I30
  # ('fwdCamera', 'fwdRadar'): routes: 138, dongles: {'734ef96182ddf940'}
  # ---
  # ('abs', 'eps', 'fwdCamera', 'fwdRadar'): routes: 40, dongles: {'734ef96182ddf940'}
  # ---
  #
  # GENESIS_G90
  # ('fwdCamera', 'fwdRadar'): routes: 500, dongles: {'e0cd2f59d9bebecc', 'c3392c23a2b38aa1', '12143ca5b941c412', '83b2b58a31b4f9a6', '590a1a98ea35681c', '04d9e339e666093e', '5db006ef4c63370f', '9243432b567f9c48', '5639413f4353e72b', '0c43840e8bfa36bf', '980dde5c3e2e43c6'}
  # ---
  #
  # HYUNDAI_IONIQ_PHEV
  # ('fwdCamera', 'fwdRadar'): routes: 1719, dongles: {'bf802322d75dc7a9', '4d30f121f9c5ab4b', '6af3dfd115b055e3', 'f78ea8bac6cf6c9d', '4bcc377663bc541a', '3c7a58d85af72323', 'e1107f9d04dfb1e2', '05c9e586d6e776e3', 'ccee2cbffa0a9a14', '8fd9d7b55c053355', 'b28e9dbe4004e915', 'e84e3c263d62889f', '29b9d701376d510f', '2833de0636d26d55', '201c5aaced09a24a', '6d230127fc4de83e', '37f27727e9056e1c', '26a38e4bf36bb74a', '4ce786edc1839031', 'de59124955b921d8', '74f6ab69c669f8c5', 'dd6689a222f591ae', 'c447f3983314c453', '19486c9cbbe1a908', '78fdfacf4a805561'}
  # ---
  # ('eps', 'fwdCamera', 'fwdRadar'): routes: 328, dongles: {'26a38e4bf36bb74a', '0cebe66cb649a2cd', '19486c9cbbe1a908', '2833de0636d26d55', 'f78ea8bac6cf6c9d', 'ccee2cbffa0a9a14', '74f6ab69c669f8c5', '201c5aaced09a24a', '8fd9d7b55c053355', '3c7a58d85af72323', '862512d9f1af2a6e', 'c447f3983314c453', '37f27727e9056e1c', '78fdfacf4a805561', 'e1107f9d04dfb1e2'}
  # ---
  # ('fwdCamera',): routes: 25, dongles: {'e1107f9d04dfb1e2'}
  # ---
  # (): routes: 15, dongles: {'e1107f9d04dfb1e2'}
  # ---
  #
  # KIA_NIRO_PHEV
  # ('fwdCamera', 'fwdRadar'): routes: 166, dongles: {'2dc37a5ccd0b2813', '3d264cee10fdc8d3', '2b485b32ce9cd697', 'ae4ea35d29d156c0'}
  # ---
  # ('eps', 'fwdCamera', 'fwdRadar'): routes: 57, dongles: {'2b485b32ce9cd697'}
  # ---
  #
  # HYUNDAI_IONIQ
  # ('fwdCamera', 'fwdRadar'): routes: 456, dongles: {'0e13ee2b821302f4', '159f1d5d50452849', '95040c267966a52c', 'bd2bb42aaf9fbb35', 'bc974c38f6043e0e'}
  # ---
  # ('eps', 'fwdCamera', 'fwdRadar'): routes: 169, dongles: {'0e13ee2b821302f4', '95040c267966a52c', 'cd30864211945e4e', '61cb482c0e565685'}
  # ---
  # ('fwdRadar',): routes: 2, dongles: {'159f1d5d50452849'}
  # ---
  #
  # HYUNDAI_KONA_EV_2022
  # ('fwdCamera', 'fwdRadar'): routes: 580, dongles: {'f0709d2bc6ca451f', 'b403ac9aad30f946', 'd3a039910163c67a', '7181d034656440da', 'd4bf28312ec6d0a3', 'ecb854b94fdfa4d5', '026a62571e42f996'}
  # ---
  # ('eps', 'fwdCamera', 'fwdRadar'): routes: 200, dongles: {'f0709d2bc6ca451f', '7181d034656440da', 'd3a039910163c67a', 'd4bf28312ec6d0a3', 'e93eea88c88c9d46', 'ecb854b94fdfa4d5'}
  # ---
  # ('fwdCamera',): routes: 24, dongles: {'b81ff53f2a9a2146', 'ecb854b94fdfa4d5'}
  # ---
  # ('eps', 'fwdCamera'): routes: 1, dongles: {'ecb854b94fdfa4d5'}
  # ---
  #
  # HYUNDAI_IONIQ_PHEV_2019
  # ('fwdCamera', 'fwdRadar'): routes: 118, dongles: {'3f29334d6134fcd4', '7a28f14a1901982f', 'a12674d8cb764936'}
  # ---
  # ('eps', 'fwdCamera', 'fwdRadar'): routes: 16, dongles: {'ca5e3e5194d31b63'}
  # ---
  #
  # KIA_NIRO_PHEV_2022
  # ('fwdCamera', 'fwdRadar'): routes: 221, dongles: {'3d264cee10fdc8d3'}
  # ---
  # ('eps', 'fwdCamera', 'fwdRadar'): routes: 101, dongles: {'3d264cee10fdc8d3'}
  # ---
  #
  # HYUNDAI_SONATA_LF
  # ('fwdCamera', 'fwdRadar'): routes: 275, dongles: {'fbede21602b342e5', '7ae1c131629d96e5', 'e3500498d01af116', 'c1fc13c1b806e536', 'b477f742834b44b1', 'c6dc36fd57328148', '5460c6f74ef5201f'}
  # ---
  # ('abs', 'fwdCamera', 'fwdRadar'): routes: 202, dongles: {'fbede21602b342e5', '7ae1c131629d96e5', 'c1fc13c1b806e536', 'c6dc36fd57328148'}
  # ---
  #
  # HYUNDAI_IONIQ_EV_2020
  # ('fwdCamera', 'fwdRadar'): routes: 498, dongles: {'26e73a0e32642dc4', '555e839edbae7c0c', '085e33b2085c44f4', '2c20d3b09325e409', 'afc1609bb1584f28', '57f14d4436405cfb', '1613602610c3e789'}
  # ---
  # ('eps', 'fwdCamera', 'fwdRadar'): routes: 235, dongles: {'26e73a0e32642dc4', 'afc1609bb1584f28', '1613602610c3e789', '47ffae6b1d1580b2'}
  # ---
  #
  # KIA_FORTE
  # ('fwdCamera', 'fwdRadar'): routes: 384, dongles: {'ba50656c0849e4cd', '59d7a211a20ba2e9', 'c9f0b55647bbad0d', '753639a1c1872d8c', '7a37cc3d8b2c0416', '51532b1b25bee9a6'}
  # ---
  # ('eps', 'fwdCamera', 'fwdRadar'): routes: 28, dongles: {'753639a1c1872d8c', '8808b5e2e1e8c14f', '59d7a211a20ba2e9', 'c9f0b55647bbad0d'}
  # ---
  # ('fwdCamera',): routes: 16, dongles: {'b057fc7b43ade5f2', '84b978e6971ba8ba'}
  # ---
  # ('eps', 'fwdCamera'): routes: 4, dongles: {'bbc520c51098bebc'}
  # ---
  #
  # HYUNDAI_KONA
  # ('fwdCamera', 'fwdRadar'): routes: 98, dongles: {'f766203654763c98', '3d3ab70b52f01e67'}
  # ---
  # ('eps', 'fwdCamera', 'fwdRadar'): routes: 30, dongles: {'5e0ef2fcec8677b6'}
  # ---
  #
  # KIA_K5_HEV_2020
  # ('fwdCamera', 'fwdRadar'): routes: 69, dongles: {'f86d2325d8ec7403'}
  # ---
  #
  # KIA_STINGER_2022
  # ('fwdCamera', 'fwdRadar'): routes: 161, dongles: {'b84b4a4fbb604be1', '9238d5ce084d695e', 'b866fab9884646b0', '207859f0bcefab83', 'a0cf05907bca9fbe'}
  # ---
  # ('eps', 'fwdCamera', 'fwdRadar'): routes: 8, dongles: {'a0cf05907bca9fbe', 'a265def164d76c50'}
  # ---
  # ('fwdCamera',): routes: 6, dongles: {'b866fab9884646b0'}
  # ---
  #
  # HYUNDAI_AZERA_HEV_6TH_GEN
  # ('fwdCamera', 'fwdRadar'): routes: 49, dongles: {'3a0cde9552891b34', '9e681cd13392bf93', '844b1bd052412b1a'}
  # ---
  # ('eps', 'fwdCamera', 'fwdRadar'): routes: 27, dongles: {'3a0cde9552891b34', '844b1bd052412b1a'}
  # ---
  # ('fwdRadar',): routes: 15, dongles: {'ca9fa476e76177e5'}
  # ---
  #
  # HYUNDAI_TUCSON
  # ('fwdCamera', 'fwdRadar'): routes: 88, dongles: {'fb3fd42f0baaa2f8', '66253de0fd139602', '72c49fec56c988da', '0bbe367c98fa1538'}
  # ---
  #
  # HYUNDAI_IONIQ_EV_LTD
  # ('eps', 'fwdCamera', 'fwdRadar'): routes: 16, dongles: {'3f586748b76b8880'}
  # ---
  #
  # HYUNDAI_ELANTRA
  # ('abs', 'fwdCamera', 'fwdRadar'): routes: 25, dongles: {'dd0ea8e5bb2e3955'}
  # ---
  # ('fwdCamera', 'fwdRadar'): routes: 12, dongles: {'f6f9e1708bae2ef6'}
  # ---
  #
  # KIA_OPTIMA_H_G4_FL
  # ('fwdCamera', 'fwdRadar'): routes: 20, dongles: {'6a42c1197b2a8179'}
  # ---
  #
  # HYUNDAI_IONIQ_HEV_2022
  # ('fwdCamera', 'fwdRadar'): routes: 8, dongles: {'ab59fe909f626921', '452f244b23a48e9b'}
  # ---
  # ('eps', 'fwdCamera', 'fwdRadar'): routes: 5, dongles: {'ab59fe909f626921'}
  # ---
  #
  # HYUNDAI_AZERA_6TH_GEN
  # ('fwdCamera', 'fwdRadar'): routes: 6, dongles: {'d3d0f4f5fa7e283d'}
  # ---
  #
  # HYUNDAI_VELOSTER
  # ('fwdCamera', 'fwdRadar'): routes: 4, dongles: {'6112eef7212ce785'}
  # ---
  #
  # KIA_CEED
  # ('fwdCamera', 'fwdRadar'): routes: 4, dongles: {'99315f265be0b26c'}
  # ---
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
