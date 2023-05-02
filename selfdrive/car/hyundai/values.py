from dataclasses import dataclass
from enum import Enum, IntFlag
from typing import Dict, List, Optional, Union

from cereal import car
from panda.python import uds
from common.conversions import Conversions as CV
from selfdrive.car import dbc_dict
from selfdrive.car.docs_definitions import CarFootnote, CarInfo, Column, Harness
from selfdrive.car.fw_query_definitions import FwQueryConfig, Request, p16

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
    elif CP.carFingerprint in (CAR.GENESIS_G80, CAR.GENESIS_G90, CAR.ELANTRA, CAR.IONIQ,
                               CAR.IONIQ_EV_LTD, CAR.SANTA_FE_PHEV_2022, CAR.SONATA_LF, CAR.KIA_FORTE, CAR.KIA_NIRO_PHEV,
                               CAR.KIA_OPTIMA_H, CAR.KIA_SORENTO):
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
  CANFD_HDA2 = 1
  CANFD_ALT_BUTTONS = 2
  CANFD_ALT_GEARS = 4
  CANFD_CAMERA_SCC = 8

  ALT_LIMITS = 16
  ENABLE_BLINKERS = 32
  CANFD_ALT_GEARS_2 = 64
  SEND_LFA = 128
  USE_FCA = 256


class CAR:
  # Hyundai
  ELANTRA = "HYUNDAI ELANTRA 2017"
  ELANTRA_2021 = "HYUNDAI ELANTRA 2021"
  ELANTRA_HEV_2021 = "HYUNDAI ELANTRA HYBRID 2021"
  HYUNDAI_GENESIS = "HYUNDAI GENESIS 2015-2016"
  IONIQ = "HYUNDAI IONIQ HYBRID 2017-2019"
  IONIQ_HEV_2022 = "HYUNDAI IONIQ HYBRID 2020-2022"
  IONIQ_EV_LTD = "HYUNDAI IONIQ ELECTRIC LIMITED 2019"
  IONIQ_EV_2020 = "HYUNDAI IONIQ ELECTRIC 2020"
  IONIQ_PHEV_2019 = "HYUNDAI IONIQ PLUG-IN HYBRID 2019"
  IONIQ_PHEV = "HYUNDAI IONIQ PHEV 2020"
  KONA = "HYUNDAI KONA 2020"
  KONA_EV = "HYUNDAI KONA ELECTRIC 2019"
  KONA_EV_2022 = "HYUNDAI KONA ELECTRIC 2022"
  KONA_HEV = "HYUNDAI KONA HYBRID 2020"
  SANTA_FE = "HYUNDAI SANTA FE 2019"
  SANTA_FE_2022 = "HYUNDAI SANTA FE 2022"
  SANTA_FE_HEV_2022 = "HYUNDAI SANTA FE HYBRID 2022"
  SANTA_FE_PHEV_2022 = "HYUNDAI SANTA FE PlUG-IN HYBRID 2022"
  SONATA = "HYUNDAI SONATA 2020"
  SONATA_LF = "HYUNDAI SONATA 2019"
  TUCSON = "HYUNDAI TUCSON 2019"
  PALISADE = "HYUNDAI PALISADE 2020"
  VELOSTER = "HYUNDAI VELOSTER 2019"
  SONATA_HYBRID = "HYUNDAI SONATA HYBRID 2021"
  IONIQ_5 = "HYUNDAI IONIQ 5 2022"
  TUCSON_4TH_GEN = "HYUNDAI TUCSON 4TH GEN"
  TUCSON_HYBRID_4TH_GEN = "HYUNDAI TUCSON HYBRID 4TH GEN"
  SANTA_CRUZ_1ST_GEN = "HYUNDAI SANTA CRUZ 1ST GEN"

  # Kia
  KIA_FORTE = "KIA FORTE E 2018 & GT 2021"
  KIA_K5_2021 = "KIA K5 2021"
  KIA_K5_HEV_2020 = "KIA K5 HYBRID 2020"
  KIA_NIRO_EV = "KIA NIRO EV 2020"
  KIA_NIRO_EV_2ND_GEN = "KIA NIRO EV 2ND GEN"
  KIA_NIRO_PHEV = "KIA NIRO HYBRID 2019"
  KIA_NIRO_HEV_2021 = "KIA NIRO HYBRID 2021"
  KIA_NIRO_HEV_2ND_GEN = "KIA NIRO HYBRID 2ND GEN"
  KIA_OPTIMA_G4 = "KIA OPTIMA 4TH GEN"
  KIA_OPTIMA_G4_FL = "KIA OPTIMA 4TH GEN FACELIFT"
  KIA_OPTIMA_H = "KIA OPTIMA HYBRID 2017 & SPORTS 2019"
  KIA_SELTOS = "KIA SELTOS 2021"
  KIA_SPORTAGE_5TH_GEN = "KIA SPORTAGE 5TH GEN"
  KIA_SORENTO = "KIA SORENTO GT LINE 2018"
  KIA_SORENTO_4TH_GEN = "KIA SORENTO 4TH GEN"
  KIA_SORENTO_PHEV_4TH_GEN = "KIA SORENTO PLUG-IN HYBRID 4TH GEN"
  KIA_SPORTAGE_HYBRID_5TH_GEN = "KIA SPORTAGE HYBRID 5TH GEN"
  KIA_STINGER = "KIA STINGER GT2 2018"
  KIA_STINGER_2022 = "KIA STINGER 2022"
  KIA_CEED = "KIA CEED INTRO ED 2019"
  KIA_EV6 = "KIA EV6 2022"

  # Genesis
  GENESIS_GV60_EV_1ST_GEN = "GENESIS GV60 ELECTRIC 1ST GEN"
  GENESIS_G70 = "GENESIS G70 2018"
  GENESIS_G70_2020 = "GENESIS G70 2020"
  GENESIS_GV70_1ST_GEN = "GENESIS GV70 1ST GEN"
  GENESIS_G80 = "GENESIS G80 2017"
  GENESIS_G90 = "GENESIS G90 2017"


class Footnote(Enum):
  # footnotes which mention "red panda" will be replaced with the CAN FD panda kit on the shop page
  CANFD = CarFootnote(
    "Requires a <a href=\"https://comma.ai/shop/panda\" target=\"_blank\">red panda</a> for this <a href=\"https://en.wikipedia.org/wiki/CAN_FD\" target=\"_blank\">CAN FD car</a>. " +
    "All the hardware needed is sold in the <a href=\"https://comma.ai/shop/can-fd-panda-kit\" target=\"_blank\">CAN FD kit</a>.",
    Column.MODEL, shop_footnote=True)


@dataclass
class HyundaiCarInfo(CarInfo):
  package: str = "Smart Cruise Control (SCC)"

  def init_make(self, CP: car.CarParams):
    if CP.carFingerprint in CANFD_CAR:
      self.footnotes.insert(0, Footnote.CANFD)


CAR_INFO: Dict[str, Optional[Union[HyundaiCarInfo, List[HyundaiCarInfo]]]] = {
  CAR.ELANTRA: [
    HyundaiCarInfo("Hyundai Elantra 2017-19", min_enable_speed=19 * CV.MPH_TO_MS, harness=Harness.hyundai_b),
    HyundaiCarInfo("Hyundai Elantra GT 2017-19", harness=Harness.hyundai_e),
    HyundaiCarInfo("Hyundai i30 2017-19", harness=Harness.hyundai_e),
  ],
  CAR.ELANTRA_2021: HyundaiCarInfo("Hyundai Elantra 2021-23", video_link="https://youtu.be/_EdYQtV52-c", harness=Harness.hyundai_k),
  CAR.ELANTRA_HEV_2021: HyundaiCarInfo("Hyundai Elantra Hybrid 2021-23", video_link="https://youtu.be/_EdYQtV52-c", harness=Harness.hyundai_k),
  CAR.HYUNDAI_GENESIS: [
    HyundaiCarInfo("Hyundai Genesis 2015-16", min_enable_speed=19 * CV.MPH_TO_MS, harness=Harness.hyundai_j),  # TODO: check 2015 packages
    HyundaiCarInfo("Genesis G80 2017", "All", min_enable_speed=19 * CV.MPH_TO_MS, harness=Harness.hyundai_j),
  ],
  CAR.IONIQ: HyundaiCarInfo("Hyundai Ioniq Hybrid 2017-19", harness=Harness.hyundai_c),
  CAR.IONIQ_HEV_2022: HyundaiCarInfo("Hyundai Ioniq Hybrid 2020-22", harness=Harness.hyundai_h),  # TODO: confirm 2020-21 harness
  CAR.IONIQ_EV_LTD: HyundaiCarInfo("Hyundai Ioniq Electric 2019", harness=Harness.hyundai_c),
  CAR.IONIQ_EV_2020: HyundaiCarInfo("Hyundai Ioniq Electric 2020", "All", harness=Harness.hyundai_h),
  CAR.IONIQ_PHEV_2019: HyundaiCarInfo("Hyundai Ioniq Plug-in Hybrid 2019", harness=Harness.hyundai_c),
  CAR.IONIQ_PHEV: HyundaiCarInfo("Hyundai Ioniq Plug-in Hybrid 2020-22", "All", harness=Harness.hyundai_h),
  CAR.KONA: HyundaiCarInfo("Hyundai Kona 2020", harness=Harness.hyundai_b),
  CAR.KONA_EV: HyundaiCarInfo("Hyundai Kona Electric 2018-21", harness=Harness.hyundai_g),
  CAR.KONA_EV_2022: HyundaiCarInfo("Hyundai Kona Electric 2022", harness=Harness.hyundai_o),
  CAR.KONA_HEV: HyundaiCarInfo("Hyundai Kona Hybrid 2020", video_link="https://youtu.be/0dwpAHiZgFo", harness=Harness.hyundai_i),  # TODO: check packages
  CAR.SANTA_FE: HyundaiCarInfo("Hyundai Santa Fe 2019-20", "All", harness=Harness.hyundai_d),
  CAR.SANTA_FE_2022: HyundaiCarInfo("Hyundai Santa Fe 2021-22", "All", video_link="https://youtu.be/VnHzSTygTS4", harness=Harness.hyundai_l),
  CAR.SANTA_FE_HEV_2022: HyundaiCarInfo("Hyundai Santa Fe Hybrid 2022", "All", harness=Harness.hyundai_l),
  CAR.SANTA_FE_PHEV_2022: HyundaiCarInfo("Hyundai Santa Fe Plug-in Hybrid 2022", "All", harness=Harness.hyundai_l),
  CAR.SONATA: HyundaiCarInfo("Hyundai Sonata 2020-23", "All", video_link="https://www.youtube.com/watch?v=ix63r9kE3Fw", harness=Harness.hyundai_a),
  CAR.SONATA_LF: HyundaiCarInfo("Hyundai Sonata 2018-19", harness=Harness.hyundai_e),
  CAR.TUCSON: [
    HyundaiCarInfo("Hyundai Tucson 2021", min_enable_speed=19 * CV.MPH_TO_MS, harness=Harness.hyundai_l),
    HyundaiCarInfo("Hyundai Tucson Diesel 2019", harness=Harness.hyundai_l),
  ],
  CAR.PALISADE: [
    HyundaiCarInfo("Hyundai Palisade 2020-22", "All", video_link="https://youtu.be/TAnDqjF4fDY?t=456", harness=Harness.hyundai_h),
    HyundaiCarInfo("Kia Telluride 2020-22", "All", harness=Harness.hyundai_h),
  ],
  CAR.VELOSTER: HyundaiCarInfo("Hyundai Veloster 2019-20", min_enable_speed=5. * CV.MPH_TO_MS, harness=Harness.hyundai_e),
  CAR.SONATA_HYBRID: HyundaiCarInfo("Hyundai Sonata Hybrid 2020-22", "All", harness=Harness.hyundai_a),
  CAR.IONIQ_5: [
    HyundaiCarInfo("Hyundai Ioniq 5 (Southeast Asia only) 2022-23", "All", harness=Harness.hyundai_q),
    HyundaiCarInfo("Hyundai Ioniq 5 (without HDA II) 2022-23", "Highway Driving Assist", harness=Harness.hyundai_k),
    HyundaiCarInfo("Hyundai Ioniq 5 (with HDA II) 2022-23", "Highway Driving Assist II", harness=Harness.hyundai_q),
  ],
  CAR.TUCSON_4TH_GEN: [
    HyundaiCarInfo("Hyundai Tucson 2022", harness=Harness.hyundai_n),
    HyundaiCarInfo("Hyundai Tucson 2023", "All", harness=Harness.hyundai_n),
  ],
  CAR.TUCSON_HYBRID_4TH_GEN: HyundaiCarInfo("Hyundai Tucson Hybrid 2022-23", "All", harness=Harness.hyundai_n),
  CAR.SANTA_CRUZ_1ST_GEN: HyundaiCarInfo("Hyundai Santa Cruz 2022-23", harness=Harness.hyundai_n),

  # Kia
  CAR.KIA_FORTE: [
    HyundaiCarInfo("Kia Forte 2019-21", harness=Harness.hyundai_g),
    HyundaiCarInfo("Kia Forte 2023", harness=Harness.hyundai_e),
  ],
  CAR.KIA_K5_2021: HyundaiCarInfo("Kia K5 2021-22", harness=Harness.hyundai_a),
  CAR.KIA_K5_HEV_2020: HyundaiCarInfo("Kia K5 Hybrid 2020", harness=Harness.hyundai_a),
  CAR.KIA_NIRO_EV: [
    HyundaiCarInfo("Kia Niro EV 2019", "All", video_link="https://www.youtube.com/watch?v=lT7zcG6ZpGo", harness=Harness.hyundai_h),
    HyundaiCarInfo("Kia Niro EV 2020", "All", video_link="https://www.youtube.com/watch?v=lT7zcG6ZpGo", harness=Harness.hyundai_f),
    HyundaiCarInfo("Kia Niro EV 2021", "All", video_link="https://www.youtube.com/watch?v=lT7zcG6ZpGo", harness=Harness.hyundai_c),
    HyundaiCarInfo("Kia Niro EV 2022", "All", video_link="https://www.youtube.com/watch?v=lT7zcG6ZpGo", harness=Harness.hyundai_h),
  ],
  CAR.KIA_NIRO_EV_2ND_GEN: HyundaiCarInfo("Kia Niro EV 2023", "All", harness=Harness.hyundai_a),
  CAR.KIA_NIRO_PHEV: [
    HyundaiCarInfo("Kia Niro Plug-in Hybrid 2018-19", "All", min_enable_speed=10. * CV.MPH_TO_MS, harness=Harness.hyundai_c),
    HyundaiCarInfo("Kia Niro Plug-in Hybrid 2020", "All", harness=Harness.hyundai_d),
  ],
  CAR.KIA_NIRO_HEV_2021: [
    HyundaiCarInfo("Kia Niro Hybrid 2021-22", harness=Harness.hyundai_f),  # TODO: 2021 could be hyundai_d, verify
  ],
  CAR.KIA_NIRO_HEV_2ND_GEN: HyundaiCarInfo("Kia Niro Hybrid 2023", harness=Harness.hyundai_a),
  CAR.KIA_OPTIMA_G4: HyundaiCarInfo("Kia Optima 2017", "Advanced Smart Cruise Control", harness=Harness.hyundai_b),  # TODO: may support 2016, 2018
  CAR.KIA_OPTIMA_G4_FL: HyundaiCarInfo("Kia Optima 2019-20", harness=Harness.hyundai_g),
  CAR.KIA_OPTIMA_H: [
    HyundaiCarInfo("Kia Optima Hybrid 2017", "Advanced Smart Cruise Control"),  # TODO: may support adjacent years
    HyundaiCarInfo("Kia Optima Hybrid 2019"),
  ],
  CAR.KIA_SELTOS: HyundaiCarInfo("Kia Seltos 2021", harness=Harness.hyundai_a),
  CAR.KIA_SPORTAGE_5TH_GEN: HyundaiCarInfo("Kia Sportage 2023", harness=Harness.hyundai_n),
  CAR.KIA_SORENTO: [
    HyundaiCarInfo("Kia Sorento 2018", "Advanced Smart Cruise Control", video_link="https://www.youtube.com/watch?v=Fkh3s6WHJz8", harness=Harness.hyundai_c),
    HyundaiCarInfo("Kia Sorento 2019", video_link="https://www.youtube.com/watch?v=Fkh3s6WHJz8", harness=Harness.hyundai_e),
  ],
  CAR.KIA_SORENTO_4TH_GEN: HyundaiCarInfo("Kia Sorento 2021-23", harness=Harness.hyundai_k),
  CAR.KIA_SORENTO_PHEV_4TH_GEN: HyundaiCarInfo("Kia Sorento Plug-in Hybrid 2022-23", harness=Harness.hyundai_a),
  CAR.KIA_SPORTAGE_HYBRID_5TH_GEN: HyundaiCarInfo("Kia Sportage Hybrid 2023", harness=Harness.hyundai_n),
  CAR.KIA_STINGER: HyundaiCarInfo("Kia Stinger 2018-20", video_link="https://www.youtube.com/watch?v=MJ94qoofYw0", harness=Harness.hyundai_c),
  CAR.KIA_STINGER_2022: HyundaiCarInfo("Kia Stinger 2022", "All", harness=Harness.hyundai_k),
  CAR.KIA_CEED: HyundaiCarInfo("Kia Ceed 2019", harness=Harness.hyundai_e),
  CAR.KIA_EV6: [
    HyundaiCarInfo("Kia EV6 (Southeast Asia only) 2022-23", "All", harness=Harness.hyundai_p),
    HyundaiCarInfo("Kia EV6 (without HDA II) 2022-23", "Highway Driving Assist", harness=Harness.hyundai_l),
    HyundaiCarInfo("Kia EV6 (with HDA II) 2022-23", "Highway Driving Assist II", harness=Harness.hyundai_p)
  ],

  # Genesis
  CAR.GENESIS_GV60_EV_1ST_GEN: [
    HyundaiCarInfo("Genesis GV60 (Advanced Trim) 2023", "All", harness=Harness.hyundai_a),
    HyundaiCarInfo("Genesis GV60 (Performance Trim) 2023", "All", harness=Harness.hyundai_k),
  ],
  CAR.GENESIS_G70: HyundaiCarInfo("Genesis G70 2018-19", "All", harness=Harness.hyundai_f),
  CAR.GENESIS_G70_2020: HyundaiCarInfo("Genesis G70 2020", "All", harness=Harness.hyundai_f),
  CAR.GENESIS_GV70_1ST_GEN: HyundaiCarInfo("Genesis GV70 2022-23", "All", harness=Harness.hyundai_l),
  CAR.GENESIS_G80: HyundaiCarInfo("Genesis G80 2018-19", "All", harness=Harness.hyundai_h),
  CAR.GENESIS_G90: HyundaiCarInfo("Genesis G90 2017-18", "All", harness=Harness.hyundai_c),
}

class Buttons:
  NONE = 0
  RES_ACCEL = 1
  SET_DECEL = 2
  GAP_DIST = 3
  CANCEL = 4  # on newer models, this is a pause/resume button

FINGERPRINTS = {
  CAR.ELANTRA: [{
    66: 8, 67: 8, 68: 8, 127: 8, 273: 8, 274: 8, 275: 8, 339: 8, 356: 4, 399: 8, 512: 6, 544: 8, 593: 8, 608: 8, 688: 5, 790: 8, 809: 8, 897: 8, 832: 8, 899: 8, 902: 8, 903: 8, 905: 8, 909: 8, 916: 8, 1040: 8, 1056: 8, 1057: 8, 1078: 4, 1170: 8, 1265: 4, 1280: 1, 1282: 4, 1287: 4, 1290: 8, 1292: 8, 1294: 8, 1312: 8, 1314: 8, 1322: 8, 1345: 8, 1349: 8, 1351: 8, 1353: 8, 1363: 8, 1366: 8, 1367: 8, 1369: 8, 1407: 8, 1415: 8, 1419: 8, 1425: 2, 1427: 6, 1440: 8, 1456: 4, 1472: 8, 1486: 8, 1487: 8, 1491: 8, 1530: 8, 1532: 5, 2001: 8, 2003: 8, 2004: 8, 2009: 8, 2012: 8, 2016: 8, 2017: 8, 2024: 8, 2025: 8
  }],
  CAR.HYUNDAI_GENESIS: [{
    67: 8, 68: 8, 304: 8, 320: 8, 339: 8, 356: 4, 544: 7, 593: 8, 608: 8, 688: 5, 809: 8, 832: 8, 854: 7, 870: 7, 871: 8, 872: 5, 897: 8, 902: 8, 903: 6, 916: 8, 1024: 2, 1040: 8, 1056: 8, 1057: 8, 1078: 4, 1107: 5, 1136: 8, 1151: 6, 1168: 7, 1170: 8, 1173: 8, 1184: 8, 1265: 4, 1280: 1, 1287: 4, 1292: 8, 1312: 8, 1322: 8, 1331: 8, 1332: 8, 1333: 8, 1334: 8, 1335: 8, 1342: 6, 1345: 8, 1363: 8, 1369: 8, 1370: 8, 1371: 8, 1378: 4, 1384: 5, 1407: 8, 1419: 8, 1427: 6, 1434: 2, 1456: 4
  },
  {
    67: 8, 68: 8, 304: 8, 320: 8, 339: 8, 356: 4, 544: 7, 593: 8, 608: 8, 688: 5, 809: 8, 832: 8, 854: 7, 870: 7, 871: 8, 872: 5, 897: 8, 902: 8, 903: 6, 916: 8, 1024: 2, 1040: 8, 1056: 8, 1057: 8, 1078: 4, 1107: 5, 1136: 8, 1151: 6, 1168: 7, 1170: 8, 1173: 8, 1184: 8, 1265: 4, 1280: 1, 1281: 3, 1287: 4, 1292: 8, 1312: 8, 1322: 8, 1331: 8, 1332: 8, 1333: 8, 1334: 8, 1335: 8, 1345: 8, 1363: 8, 1369: 8, 1370: 8, 1378: 4, 1379: 8, 1384: 5, 1407: 8, 1419: 8, 1427: 6, 1434: 2, 1456: 4
  },
  {
    67: 8, 68: 8, 304: 8, 320: 8, 339: 8, 356: 4, 544: 7, 593: 8, 608: 8, 688: 5, 809: 8, 854: 7, 870: 7, 871: 8, 872: 5, 897: 8, 902: 8, 903: 6, 912: 7, 916: 8, 1040: 8, 1056: 8, 1057: 8, 1078: 4, 1107: 5, 1136: 8, 1151: 6, 1168: 7, 1170: 8, 1173: 8, 1184: 8, 1265: 4, 1268: 8, 1280: 1, 1281: 3, 1287: 4, 1292: 8, 1312: 8, 1322: 8, 1331: 8, 1332: 8, 1333: 8, 1334: 8, 1335: 8, 1345: 8, 1363: 8, 1369: 8, 1370: 8, 1371: 8, 1378: 4, 1384: 5, 1407: 8, 1419: 8, 1427: 6, 1434: 2, 1437: 8, 1456: 4
  },
  {
    67: 8, 68: 8, 304: 8, 320: 8, 339: 8, 356: 4, 544: 7, 593: 8, 608: 8, 688: 5, 809: 8, 832: 8, 854: 7, 870: 7, 871: 8, 872: 5, 897: 8, 902: 8, 903: 6, 916: 8, 1040: 8, 1056: 8, 1057: 8, 1078: 4, 1107: 5, 1136: 8, 1151: 6, 1168: 7, 1170: 8, 1173: 8, 1184: 8, 1265: 4, 1280: 1, 1287: 4, 1292: 8, 1312: 8, 1322: 8, 1331: 8, 1332: 8, 1333: 8, 1334: 8, 1335: 8, 1345: 8, 1363: 8, 1369: 8, 1370: 8, 1378: 4, 1379: 8, 1384: 5, 1407: 8, 1425: 2, 1427: 6, 1437: 8, 1456: 4
  },
  {
    67: 8, 68: 8, 304: 8, 320: 8, 339: 8, 356: 4, 544: 7, 593: 8, 608: 8, 688: 5, 809: 8, 832: 8, 854: 7, 870: 7, 871: 8, 872: 5, 897: 8, 902: 8, 903: 6, 916: 8, 1040: 8, 1056: 8, 1057: 8, 1078: 4, 1107: 5, 1136: 8, 1151: 6, 1168: 7, 1170: 8, 1173: 8, 1184: 8, 1265: 4, 1280: 1, 1287: 4, 1292: 8, 1312: 8, 1322: 8, 1331: 8, 1332: 8, 1333: 8, 1334: 8, 1335: 8, 1345: 8, 1363: 8, 1369: 8, 1370: 8, 1371: 8, 1378: 4, 1384: 5, 1407: 8, 1419: 8, 1425: 2, 1427: 6, 1437: 8, 1456: 4
  }],
  CAR.SANTA_FE: [{
    67: 8, 127: 8, 304: 8, 320: 8, 339: 8, 356: 4, 544: 8, 593: 8, 608: 8, 688: 6, 809: 8, 832: 8, 854: 7, 870: 7, 871: 8, 872: 8, 897: 8, 902: 8, 903: 8, 905: 8, 909: 8, 916: 8, 1040: 8, 1042: 8, 1056: 8, 1057: 8, 1078: 4, 1107: 5, 1136: 8, 1151: 6, 1155: 8, 1156: 8, 1162: 8, 1164: 8, 1168: 7, 1170: 8, 1173: 8, 1183: 8, 1186: 2, 1191: 2, 1227: 8, 1265: 4, 1280: 1, 1287: 4, 1290: 8, 1292: 8, 1294: 8, 1312: 8, 1322: 8, 1342: 6, 1345: 8, 1348: 8, 1363: 8, 1369: 8, 1379: 8, 1384: 8, 1407: 8, 1414: 3, 1419: 8, 1427: 6, 1456: 4, 1470: 8
  },
  {
    67: 8, 127: 8, 304: 8, 320: 8, 339: 8, 356: 4, 544: 8, 593: 8, 608: 8, 688: 6, 764: 8, 809: 8, 854: 7, 870: 7, 871: 8, 872: 8, 897: 8, 902: 8, 903: 8, 905: 8, 909: 8, 916: 8, 1040: 8, 1042: 8, 1056: 8, 1057: 8, 1064: 8, 1078: 4, 1107: 5, 1136: 8, 1151: 6, 1155: 8, 1162: 8, 1164: 8, 1168: 7, 1170: 8, 1173: 8, 1180: 8, 1183: 8, 1186: 2, 1227: 8, 1265: 4, 1280: 1, 1287: 4, 1290: 8, 1292: 8, 1294: 8, 1312: 8, 1322: 8, 1345: 8, 1348: 8, 1363: 8, 1369: 8, 1371: 8, 1378: 8, 1384: 8, 1407: 8, 1414: 3, 1419: 8, 1427: 6, 1456: 4, 1470: 8, 1988: 8, 2000: 8, 2004: 8, 2008: 8, 2012: 8
  },
  {
    67: 8, 68: 8, 80: 4, 160: 8, 161: 8, 272: 8, 288: 4, 339: 8, 356: 8, 357: 8, 399: 8, 544: 8, 608: 8, 672: 8, 688: 5, 704: 1, 790: 8, 809: 8, 848: 8, 880: 8, 898: 8, 900: 8, 901: 8, 904: 8, 1056: 8, 1064: 8, 1065: 8, 1072: 8, 1075: 8, 1087: 8, 1088: 8, 1151: 8, 1200: 8, 1201: 8, 1232: 4, 1264: 8, 1265: 8, 1266: 8, 1296: 8, 1306: 8, 1312: 8, 1322: 8, 1331: 8, 1332: 8, 1333: 8, 1348: 8, 1349: 8, 1369: 8, 1370: 8, 1371: 8, 1407: 8, 1415: 8, 1419: 8, 1440: 8, 1442: 4, 1461: 8, 1470: 8
  }],
  CAR.SONATA: [
    {67: 8, 68: 8, 127: 8, 304: 8, 320: 8, 339: 8, 356: 4, 544: 8, 546: 8, 549: 8, 550: 8, 576: 8, 593: 8, 608: 8, 688: 6, 809: 8, 832: 8, 854: 8, 865: 8, 870: 7, 871: 8, 872: 8, 897: 8, 902: 8, 903: 8, 905: 8, 908: 8, 909: 8, 912: 7, 913: 8, 916: 8, 1040: 8, 1042: 8, 1056: 8, 1057: 8, 1078: 4, 1089: 5, 1096: 8, 1107: 5, 1108: 8, 1114: 8, 1136: 8, 1145: 8, 1151: 8, 1155: 8, 1156: 8, 1157: 4, 1162: 8, 1164: 8, 1168: 8, 1170: 8, 1173: 8, 1180: 8, 1183: 8, 1184: 8, 1186: 2, 1191: 2, 1193: 8, 1210: 8, 1225: 8, 1227: 8, 1265: 4, 1268: 8, 1280: 8, 1287: 4, 1290: 8, 1292: 8, 1294: 8, 1312: 8, 1322: 8, 1330: 8, 1339: 8, 1342: 6, 1343: 8, 1345: 8, 1348: 8, 1363: 8, 1369: 8, 1371: 8, 1378: 8, 1379: 8, 1384: 8, 1394: 8, 1407: 8, 1419: 8, 1427: 6, 1446: 8, 1456: 4, 1460: 8, 1470: 8, 1485: 8, 1504: 3, 1988: 8, 1996: 8, 2000: 8, 2004: 8, 2008: 8, 2012: 8, 2015: 8},
  ],
  CAR.SONATA_LF: [
    {66: 8, 67: 8, 68: 8, 127: 8, 273: 8, 274: 8, 275: 8, 339: 8, 356: 4, 399: 8, 447: 8, 512: 6, 544: 8, 593: 8, 608: 8, 688: 5, 790: 8, 809: 8, 832: 8, 884: 8, 897: 8, 899: 8, 902: 8, 903: 6, 916: 8, 1040: 8, 1056: 8, 1057: 8, 1078: 4, 1151: 6, 1168: 7, 1170: 8, 1253: 8, 1254: 8, 1255: 8, 1265: 4, 1280: 1, 1287: 4, 1290: 8, 1292: 8, 1294: 8, 1312: 8, 1314: 8, 1322: 8, 1331: 8, 1332: 8, 1333: 8, 1342: 6, 1345: 8, 1348: 8, 1349: 8, 1351: 8, 1353: 8, 1363: 8, 1365: 8, 1366: 8, 1367: 8, 1369: 8, 1397: 8, 1407: 8, 1415: 8, 1419: 8, 1425: 2, 1427: 6, 1440: 8, 1456: 4, 1470: 8, 1472: 8, 1486: 8, 1487: 8, 1491: 8, 1530: 8, 1532: 5, 2000: 8, 2001: 8, 2004: 8, 2005: 8, 2008: 8, 2009: 8, 2012: 8, 2013: 8, 2014: 8, 2016: 8, 2017: 8, 2024: 8, 2025: 8},
  ],
  CAR.KIA_SORENTO: [{
    67: 8, 68: 8, 127: 8, 304: 8, 320: 8, 339: 8, 356: 4, 544: 8, 593: 8, 608: 8, 688: 5, 809: 8, 832: 8, 854: 7, 870: 7, 871: 8, 872: 8, 897: 8, 902: 8, 903: 8, 916: 8, 1040: 8, 1042: 8, 1056: 8, 1057: 8, 1064: 8, 1078: 4, 1107: 5, 1136: 8, 1151: 6, 1168: 7, 1170: 8, 1173: 8, 1265: 4, 1280: 1, 1287: 4, 1290: 8, 1292: 8, 1294: 8, 1312: 8, 1322: 8, 1331: 8, 1332: 8, 1333: 8, 1342: 6, 1345: 8, 1348: 8, 1363: 8, 1369: 8, 1370: 8, 1371: 8, 1384: 8, 1407: 8, 1411: 8, 1419: 8, 1425: 2, 1427: 6, 1444: 8, 1456: 4, 1470: 8, 1489: 1
  }],
  CAR.KIA_STINGER: [{
    67: 8, 127: 8, 304: 8, 320: 8, 339: 8, 356: 4, 358: 6, 359: 8, 544: 8, 576: 8, 593: 8, 608: 8, 688: 5, 809: 8, 832: 8, 854: 7, 870: 7, 871: 8, 872: 8, 897: 8, 902: 8, 909: 8, 916: 8, 1040: 8, 1042: 8, 1056: 8, 1057: 8, 1064: 8, 1078: 4, 1107: 5, 1136: 8, 1151: 6, 1168: 7, 1170: 8, 1173: 8, 1184: 8, 1265: 4, 1280: 1, 1281: 4, 1287: 4, 1290: 8, 1292: 8, 1294: 8, 1312: 8, 1322: 8, 1342: 6, 1345: 8, 1348: 8, 1363: 8, 1369: 8, 1371: 8, 1378: 4, 1379: 8, 1384: 8, 1407: 8, 1419: 8, 1425: 2, 1427: 6, 1456: 4, 1470: 8
  }],
  CAR.GENESIS_G80: [{
    67: 8, 68: 8, 127: 8, 304: 8, 320: 8, 339: 8, 356: 4, 358: 6, 544: 8, 593: 8, 608: 8, 688: 5, 809: 8, 832: 8, 854: 7, 870: 7, 871: 8, 872: 8, 897: 8, 902: 8, 903: 8, 916: 8, 1024: 2, 1040: 8, 1042: 8, 1056: 8, 1057: 8, 1078: 4, 1107: 5, 1136: 8, 1151: 6, 1156: 8, 1168: 7, 1170: 8, 1173: 8, 1184: 8, 1191: 2, 1265: 4, 1280: 1, 1287: 4, 1290: 8, 1292: 8, 1294: 8, 1312: 8, 1322: 8, 1342: 6, 1345: 8, 1348: 8, 1363: 8, 1369: 8, 1370: 8, 1371: 8, 1378: 4, 1384: 8, 1407: 8, 1419: 8, 1425: 2, 1427: 6, 1434: 2, 1456: 4, 1470: 8
  },
  {
    67: 8, 68: 8, 127: 8, 304: 8, 320: 8, 339: 8, 356: 4, 358: 6, 359: 8, 544: 8, 546: 8, 593: 8, 608: 8, 688: 5, 809: 8, 832: 8, 854: 7, 870: 7, 871: 8, 872: 8, 897: 8, 902: 8, 903: 8, 916: 8, 1040: 8, 1042: 8, 1056: 8, 1057: 8, 1064: 8, 1078: 4, 1107: 5, 1136: 8, 1151: 6, 1156: 8, 1157: 4, 1168: 7, 1170: 8, 1173: 8, 1184: 8, 1265: 4, 1280: 1, 1281: 3, 1287: 4, 1290: 8, 1292: 8, 1294: 8, 1312: 8, 1322: 8, 1342: 6, 1345: 8, 1348: 8, 1363: 8, 1369: 8, 1370: 8, 1371: 8, 1378: 4, 1384: 8, 1407: 8, 1419: 8, 1425: 2, 1427: 6, 1434: 2, 1437: 8, 1456: 4, 1470: 8
  },
  {
    67: 8, 68: 8, 127: 8, 304: 8, 320: 8, 339: 8, 356: 4, 358: 6, 544: 8, 593: 8, 608: 8, 688: 5, 809: 8, 832: 8, 854: 7, 870: 7, 871: 8, 872: 8, 897: 8, 902: 8, 903: 8, 916: 8, 1040: 8, 1042: 8, 1056: 8, 1057: 8, 1064: 8, 1078: 4, 1107: 5, 1136: 8, 1151: 6, 1156: 8, 1157: 4, 1162: 8, 1168: 7, 1170: 8, 1173: 8, 1184: 8, 1193: 8, 1265: 4, 1280: 1, 1287: 4, 1290: 8, 1292: 8, 1294: 8, 1312: 8, 1322: 8, 1342: 6, 1345: 8, 1348: 8, 1363: 8, 1369: 8, 1371: 8, 1378: 4, 1384: 8, 1407: 8, 1419: 8, 1425: 2, 1427: 6, 1437: 8, 1456: 4, 1470: 8
  }],
  CAR.GENESIS_G90: [{
    67: 8, 68: 8, 127: 8, 304: 8, 320: 8, 339: 8, 356: 4, 358: 6, 359: 8, 544: 8, 593: 8, 608: 8, 688: 5, 809: 8, 854: 7, 870: 7, 871: 8, 872: 8, 897: 8, 902: 8, 903: 8, 916: 8, 1040: 8, 1056: 8, 1057: 8, 1078: 4, 1107: 5, 1136: 8, 1151: 6, 1162: 4, 1168: 7, 1170: 8, 1173: 8, 1184: 8, 1265: 4, 1280: 1, 1281: 3, 1287: 4, 1290: 8, 1292: 8, 1294: 8, 1312: 8, 1322: 8, 1345: 8, 1348: 8, 1363: 8, 1369: 8, 1370: 8, 1371: 8, 1378: 4, 1384: 8, 1407: 8, 1419: 8, 1425: 2, 1427: 6, 1434: 2, 1456: 4, 1470: 8, 1988: 8, 2000: 8, 2003: 8, 2004: 8, 2005: 8, 2008: 8, 2011: 8, 2012: 8, 2013: 8
  }],
  CAR.IONIQ_EV_2020: [{
    127: 8, 304: 8, 320: 8, 339: 8, 352: 8, 356: 4, 524: 8, 544: 7, 593: 8, 688: 5, 832: 8, 881: 8, 882: 8, 897: 8, 902: 8, 903: 8, 905: 8, 909: 8, 916: 8, 1040: 8, 1042: 8, 1056: 8, 1057: 8, 1078: 4, 1136: 8, 1151: 6, 1155: 8, 1156: 8, 1157: 4, 1164: 8, 1168: 7, 1173: 8, 1183: 8, 1186: 2, 1191: 2, 1225: 8, 1265: 4, 1280: 1, 1287: 4, 1290: 8, 1291: 8, 1292: 8, 1294: 8, 1312: 8, 1322: 8, 1342: 6, 1345: 8, 1348: 8, 1355: 8, 1363: 8, 1369: 8, 1379: 8, 1407: 8, 1419: 8, 1426: 8, 1427: 6, 1429: 8, 1430: 8, 1456: 4, 1470: 8, 1473: 8, 1507: 8, 1535: 8, 1988: 8, 1996: 8, 2000: 8, 2004: 8, 2005: 8, 2008: 8, 2012: 8, 2013: 8
  }],
  CAR.IONIQ: [{
    68:8, 127: 8, 304: 8, 320: 8, 339: 8, 352: 8, 356: 4, 524: 8, 544: 8, 576:8, 593: 8, 688: 5, 832: 8, 881: 8, 882: 8, 897: 8, 902: 8, 903: 8, 905: 8, 909: 8, 916: 8, 1040: 8, 1042: 8, 1056: 8, 1057: 8, 1078: 4, 1136: 6, 1151: 6, 1155: 8, 1156: 8, 1157: 4, 1164: 8, 1168: 7, 1173: 8, 1183: 8, 1186: 2, 1191: 2, 1225: 8, 1265: 4, 1280: 1, 1287: 4, 1290: 8, 1291: 8, 1292: 8, 1294: 8, 1312: 8, 1322: 8, 1342: 6, 1345: 8, 1348: 8, 1355: 8, 1363: 8, 1369: 8, 1379: 8, 1407: 8, 1419: 8, 1426: 8, 1427: 6, 1429: 8, 1430: 8, 1448: 8, 1456: 4, 1470: 8, 1473: 8, 1476: 8, 1507: 8, 1535: 8, 1988: 8, 1996: 8, 2000: 8, 2004: 8, 2005: 8, 2008: 8, 2012: 8, 2013: 8
  }],
  CAR.KONA_EV: [{
    127: 8, 304: 8, 320: 8, 339: 8, 352: 8, 356: 4, 544: 8, 549: 8, 593: 8, 688: 5, 832: 8, 881: 8, 882: 8, 897: 8, 902: 8, 903: 8, 905: 8, 909: 8, 916: 8, 1040: 8, 1042: 8, 1056: 8, 1057: 8, 1078: 4, 1136: 8, 1151: 6, 1168: 7, 1173: 8, 1183: 8, 1186: 2, 1191: 2, 1225: 8, 1265: 4, 1280: 1, 1287: 4, 1290: 8, 1291: 8, 1292: 8, 1294: 8, 1307: 8, 1312: 8, 1322: 8, 1342: 6, 1345: 8, 1348: 8, 1355: 8, 1363: 8, 1369: 8, 1378: 4, 1407: 8, 1419: 8, 1426: 8, 1427: 6, 1429: 8, 1430: 8, 1456: 4, 1470: 8, 1473: 8, 1507: 8, 1535: 8, 2000: 8, 2004: 8, 2008: 8, 2012: 8, 1157: 4, 1193: 8, 1379: 8, 1988: 8, 1996: 8
  }],
  CAR.KONA_EV_2022: [{
    127: 8, 304: 8, 320: 8, 339: 8, 352: 8, 356: 4, 544: 8, 593: 8, 688: 5, 832: 8, 881: 8, 882: 8, 897: 8, 902: 8, 903: 8, 905: 8, 909: 8, 913: 8, 916: 8, 1040: 8, 1042: 8, 1056: 8, 1057: 8, 1069: 8, 1078: 4, 1136: 8, 1145: 8, 1151: 8, 1155: 8, 1156: 8, 1157: 4, 1162: 8, 1164: 8, 1168: 8, 1173: 8, 1183: 8, 1188: 8, 1191: 2, 1193: 8, 1225: 8, 1227: 8, 1265: 4, 1280: 1, 1287: 4, 1290: 8, 1291: 8, 1292: 8, 1294: 8, 1312: 8, 1322: 8, 1339: 8, 1342: 8, 1343: 8, 1345: 8, 1348: 8, 1355: 8, 1363: 8, 1369: 8, 1379: 8, 1407: 8, 1419: 8, 1426: 8, 1427: 6, 1429: 8, 1430: 8, 1446: 8, 1456: 4, 1470: 8, 1473: 8, 1485: 8, 1507: 8, 1535: 8, 1990: 8, 1998: 8
  }],
  CAR.KIA_NIRO_EV: [{
    127: 8, 304: 8, 320: 8, 339: 8, 352: 8, 356: 4, 516: 8, 544: 8, 593: 8, 688: 5, 832: 8, 881: 8, 882: 8, 897: 8, 902: 8, 903: 8, 905: 8, 909: 8, 916: 8, 1040: 8, 1042: 8, 1056: 8, 1057: 8, 1078: 4, 1136: 8, 1151: 6, 1156: 8, 1157: 4, 1168: 7, 1173: 8, 1183: 8, 1186: 2, 1191: 2, 1193: 8, 1225: 8, 1260: 8, 1265: 4, 1280: 1, 1287: 4, 1290: 8, 1291: 8, 1292: 8, 1294: 8, 1312: 8, 1322: 8, 1342: 6, 1345: 8, 1348: 8, 1355: 8, 1363: 8, 1369: 8, 1407: 8, 1419: 8, 1426: 8, 1427: 6, 1429: 8, 1430: 8, 1456: 4, 1470: 8, 1473: 8, 1507: 8, 1535: 8, 1990: 8, 1998: 8, 1996: 8, 2000: 8, 2004: 8, 2008: 8, 2012: 8, 2015: 8
  }],
  CAR.KIA_OPTIMA_H: [{
    68: 8, 127: 8, 304: 8, 320: 8, 339: 8, 352: 8, 356: 4, 544: 8, 593: 8, 688: 5, 832: 8, 881: 8, 882: 8, 897: 8, 902: 8, 903: 6, 916: 8, 1040: 8, 1056: 8, 1057: 8, 1078: 4, 1136: 6, 1151: 6, 1168: 7, 1173: 8, 1236: 2, 1265: 4, 1280: 1, 1287: 4, 1290: 8, 1291: 8, 1292: 8, 1322: 8, 1331: 8, 1332: 8, 1333: 8, 1342: 6, 1345: 8, 1348: 8, 1355: 8, 1363: 8, 1369: 8, 1371: 8, 1407: 8, 1419: 8, 1427: 6, 1429: 8, 1430: 8, 1448: 8, 1456: 4, 1470: 8, 1476: 8, 1535: 8
  },
  {
    68: 8, 127: 8, 304: 8, 320: 8, 339: 8, 352: 8, 356: 4, 544: 8, 576: 8, 593: 8, 688: 5, 881: 8, 882: 8, 897: 8, 902: 8, 903: 8, 909: 8, 912: 7, 916: 8, 1040: 8, 1056: 8, 1057: 8, 1078: 4, 1136: 6, 1151: 6, 1168: 7, 1173: 8, 1180: 8, 1186: 2, 1191: 2, 1265: 4, 1268: 8, 1280: 1, 1287: 4, 1290: 8, 1291: 8, 1292: 8, 1294: 8, 1312: 8, 1322: 8, 1342: 6, 1345: 8, 1348: 8, 1355: 8, 1363: 8, 1369: 8, 1371: 8, 1407: 8, 1419: 8, 1420: 8, 1425: 2, 1427: 6, 1429: 8, 1430: 8, 1448: 8, 1456: 4, 1470: 8, 1476: 8, 1535: 8
  }],
  CAR.PALISADE: [{
    67: 8, 127: 8, 304: 8, 320: 8, 339: 8, 356: 4, 544: 8, 546: 8, 547: 8, 548: 8, 549: 8, 576: 8, 593: 8, 608: 8, 688: 6, 809: 8, 832: 8, 854: 7, 870: 7, 871: 8, 872: 8, 897: 8, 902: 8, 903: 8, 905: 8, 909: 8, 913: 8, 916: 8, 1040: 8, 1042: 8, 1056: 8, 1057: 8, 1064: 8, 1078: 4, 1107: 5, 1123: 8, 1136: 8, 1151: 6, 1155: 8, 1156: 8, 1157: 4, 1162: 8, 1164: 8, 1168: 7, 1170: 8, 1173: 8, 1180: 8, 1186: 2, 1191: 2, 1193: 8, 1210: 8, 1225: 8, 1227: 8, 1265: 4, 1280: 8, 1287: 4, 1290: 8, 1292: 8, 1294: 8, 1312: 8, 1322: 8, 1342: 6, 1345: 8, 1348: 8, 1363: 8, 1369: 8, 1371: 8, 1378: 8, 1384: 8, 1407: 8, 1419: 8, 1427: 6, 1456: 4, 1470: 8, 1988: 8, 1996: 8, 2000: 8, 2004: 8, 2005: 8, 2008: 8, 2012: 8
  }],
}

HYUNDAI_VERSION_REQUEST_LONG = bytes([uds.SERVICE_TYPE.READ_DATA_BY_IDENTIFIER]) + \
  p16(0xf100)  # Long description

HYUNDAI_VERSION_REQUEST_ALT = bytes([uds.SERVICE_TYPE.READ_DATA_BY_IDENTIFIER]) + \
  p16(0xf110)  # Alt long description

HYUNDAI_VERSION_REQUEST_MULTI = bytes([uds.SERVICE_TYPE.READ_DATA_BY_IDENTIFIER]) + \
  p16(uds.DATA_IDENTIFIER_TYPE.VEHICLE_MANUFACTURER_SPARE_PART_NUMBER) + \
  p16(uds.DATA_IDENTIFIER_TYPE.APPLICATION_SOFTWARE_IDENTIFICATION) + \
  p16(0xf100)

HYUNDAI_VERSION_RESPONSE = bytes([uds.SERVICE_TYPE.READ_DATA_BY_IDENTIFIER + 0x40])

FW_QUERY_CONFIG = FwQueryConfig(
  requests=[
    # TODO: minimize shared whitelists for CAN and cornerRadar for CAN-FD
    # CAN queries (OBD-II port)
    Request(
      [HYUNDAI_VERSION_REQUEST_LONG],
      [HYUNDAI_VERSION_RESPONSE],
      whitelist_ecus=[Ecu.transmission, Ecu.eps, Ecu.abs, Ecu.fwdRadar, Ecu.fwdCamera],
    ),
    Request(
      [HYUNDAI_VERSION_REQUEST_MULTI],
      [HYUNDAI_VERSION_RESPONSE],
      whitelist_ecus=[Ecu.engine, Ecu.transmission, Ecu.eps, Ecu.abs, Ecu.fwdRadar],
    ),

    # CAN-FD queries (from camera)
    # TODO: combine shared whitelists with CAN requests
    Request(
      [HYUNDAI_VERSION_REQUEST_LONG],
      [HYUNDAI_VERSION_RESPONSE],
      whitelist_ecus=[Ecu.fwdCamera, Ecu.fwdRadar, Ecu.cornerRadar, Ecu.hvac],
      bus=0,
      auxiliary=True,
    ),
    Request(
      [HYUNDAI_VERSION_REQUEST_LONG],
      [HYUNDAI_VERSION_RESPONSE],
      whitelist_ecus=[Ecu.fwdCamera, Ecu.adas, Ecu.cornerRadar, Ecu.hvac],
      bus=1,
      auxiliary=True,
      obd_multiplexing=False,
    ),

    # CAN-FD debugging queries
    Request(
      [HYUNDAI_VERSION_REQUEST_ALT],
      [HYUNDAI_VERSION_RESPONSE],
      whitelist_ecus=[Ecu.parkingAdas, Ecu.hvac],
      bus=0,
      auxiliary=True,
    ),
    Request(
      [HYUNDAI_VERSION_REQUEST_ALT],
      [HYUNDAI_VERSION_RESPONSE],
      whitelist_ecus=[Ecu.parkingAdas, Ecu.hvac],
      bus=1,
      auxiliary=True,
      obd_multiplexing=False,
    ),
  ],
  extra_ecus=[
    (Ecu.adas, 0x730, None),         # ADAS Driving ECU on HDA2 platforms
    (Ecu.parkingAdas, 0x7b1, None),  # ADAS Parking ECU (may exist on all platforms)
    (Ecu.hvac, 0x7b3, None),         # HVAC Control Assembly
    (Ecu.cornerRadar, 0x7b7, None),
  ],
)

FW_VERSIONS = {
  CAR.HYUNDAI_GENESIS: {
    (Ecu.fwdCamera, 0x7c4, None): [
      b'\xf1\x00DH LKAS 1.1 -150210',
      b'\xf1\x00DH LKAS 1.4 -140110',
      b'\xf1\x00DH LKAS 1.5 -140425',
    ],
  },
  CAR.IONIQ: {
    (Ecu.fwdRadar, 0x7d0, None): [
      b'\xf1\x00AEhe SCC H-CUP      1.01 1.01 96400-G2000         ',
    ],
    (Ecu.eps, 0x7d4, None): [
      b'\xf1\x00AE  MDPS C 1.00 1.07 56310/G2301 4AEHC107',
    ],
    (Ecu.fwdCamera, 0x7c4, None): [
      b'\xf1\x00AEH MFC  AT EUR LHD 1.00 1.00 95740-G2400 180222',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'\xf1\x816H6F2051\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.transmission, 0x7e1, None): [
      b'\xf1\x816U3H1051\x00\x00\xf1\x006U3H0_C2\x00\x006U3H1051\x00\x00HAE0G16US2\x00\x00\x00\x00',
    ],
  },
  CAR.IONIQ_PHEV_2019: {
    (Ecu.fwdRadar, 0x7d0, None): [
      b'\xf1\x00AEhe SCC H-CUP      1.01 1.01 96400-G2100         ',
    ],
    (Ecu.eps, 0x7d4, None): [
      b'\xf1\x00AE  MDPS C 1.00 1.07 56310/G2501 4AEHC107',
    ],
    (Ecu.fwdCamera, 0x7c4, None): [
      b'\xf1\x00AEP MFC  AT USA LHD 1.00 1.00 95740-G2400 180222',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'\xf1\x816H6F6051\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.transmission, 0x7e1, None): [
      b'\xf1\x816U3J2051\x00\x00\xf1\x006U3H0_C2\x00\x006U3J2051\x00\x00PAE0G16NS1\xdbD\r\x81',
      b'\xf1\x816U3J2051\x00\x00\xf1\x006U3H0_C2\x00\x006U3J2051\x00\x00PAE0G16NS1\x00\x00\x00\x00',
    ],
  },
  CAR.IONIQ_PHEV: {
    (Ecu.fwdRadar, 0x7d0, None): [
      b'\xf1\x00AEhe SCC FHCUP      1.00 1.02 99110-G2100         ',
      b'\xf1\x00AEhe SCC F-CUP      1.00 1.00 99110-G2200         ',
      b'\xf1\x00AEhe SCC F-CUP      1.00 1.00 99110-G2600         ',
      b'\xf1\x00AEhe SCC F-CUP      1.00 1.02 99110-G2100         ',
      b'\xf1\x00AEhe SCC FHCUP      1.00 1.00 99110-G2600         ',
    ],
    (Ecu.eps, 0x7d4, None): [
      b'\xf1\x00AE  MDPS C 1.00 1.01 56310/G2510 4APHC101',
      b'\xf1\x00AE  MDPS C 1.00 1.01 56310/G2560 4APHC101',
      b'\xf1\x00AE  MDPS C 1.00 1.01 56310G2510\x00 4APHC101',
    ],
    (Ecu.fwdCamera, 0x7c4, None): [
      b'\xf1\x00AEP MFC  AT USA LHD 1.00 1.01 95740-G2600 190819',
      b'\xf1\x00AEP MFC  AT EUR RHD 1.00 1.01 95740-G2600 190819',
      b'\xf1\x00AEP MFC  AT USA LHD 1.00 1.00 95740-G2700 201027',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'\xf1\x816H6F6051\x00\x00\x00\x00\x00\x00\x00\x00',
      b'\xf1\x816H6G6051\x00\x00\x00\x00\x00\x00\x00\x00',
      b'\xf1\x816H6G5051\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.transmission, 0x7e1, None): [
      b'\xf1\x816U3J9051\000\000\xf1\0006U3H1_C2\000\0006U3J9051\000\000PAE0G16NL0\x82zT\xd2',
      b'\xf1\x816U3J8051\x00\x00\xf1\x006U3H1_C2\x00\x006U3J8051\x00\x00PAETG16UL0\x00\x00\x00\x00',
      b'\xf1\x816U3J9051\x00\x00\xf1\x006U3H1_C2\x00\x006U3J9051\x00\x00PAE0G16NL2\x00\x00\x00\x00',
      b'\xf1\x006U3H1_C2\x00\x006U3J9051\x00\x00PAE0G16NL0\x00\x00\x00\x00',
      b'\xf1\x006U3H1_C2\x00\x006U3J9051\x00\x00PAE0G16NL2\xad\xeb\xabt',
    ],
  },
  CAR.IONIQ_EV_2020: {
    (Ecu.fwdRadar, 0x7d0, None): [
      b'\xf1\x00AEev SCC F-CUP      1.00 1.01 99110-G7000         ',
      b'\xf1\x00AEev SCC F-CUP      1.00 1.00 99110-G7200         ',
    ],
    (Ecu.eps, 0x7d4, None): [
      b'\xf1\x00AE  MDPS C 1.00 1.01 56310/G7310 4APEC101',
      b'\xf1\x00AE  MDPS C 1.00 1.01 56310/G7560 4APEC101',
    ],
    (Ecu.fwdCamera, 0x7c4, None): [
      b'\xf1\x00AEE MFC  AT EUR LHD 1.00 1.01 95740-G2600 190819',
      b'\xf1\x00AEE MFC  AT EUR LHD 1.00 1.03 95740-G2500 190516',
      b'\xf1\x00AEE MFC  AT EUR RHD 1.00 1.01 95740-G2600 190819',
    ],
  },
  CAR.IONIQ_EV_LTD: {
    (Ecu.fwdRadar, 0x7d0, None): [
      b'\xf1\x00AEev SCC F-CUP      1.00 1.00 96400-G7000         ',
      b'\xf1\x00AEev SCC F-CUP      1.00 1.00 96400-G7100         ',
    ],
    (Ecu.eps, 0x7d4, None): [
      b'\xf1\x00AE  MDPS C 1.00 1.02 56310G7300\x00 4AEEC102',
      b'\xf1\x00AE  MDPS C 1.00 1.04 56310/G7501 4AEEC104',
      b'\xf1\x00AE  MDPS C 1.00 1.03 56310/G7300 4AEEC103',
      b'\xf1\x00AE  MDPS C 1.00 1.03 56310G7300\x00 4AEEC103',
      b'\xf1\x00AE  MDPS C 1.00 1.04 56310/G7301 4AEEC104',
    ],
    (Ecu.fwdCamera, 0x7c4, None): [
      b'\xf1\x00AEE MFC  AT EUR LHD 1.00 1.00 95740-G7200 160418',
      b'\xf1\x00AEE MFC  AT USA LHD 1.00 1.00 95740-G2400 180222',
      b'\xf1\x00AEE MFC  AT EUR LHD 1.00 1.00 95740-G2300 170703',
      b'\xf1\x00AEE MFC  AT EUR LHD 1.00 1.00 95740-G2400 180222',
    ],
  },
  CAR.IONIQ_HEV_2022: {
    (Ecu.fwdRadar, 0x7d0, None): [
      b'\xf1\x00AEhe SCC F-CUP      1.00 1.00 99110-G2600         ',
      b'\xf1\x00AEhe SCC FHCUP      1.00 1.00 99110-G2600         ',
    ],
    (Ecu.eps, 0x7d4, None): [
      b'\xf1\x00AE  MDPS C 1.00 1.01 56310G2510\x00 4APHC101',
    ],
    (Ecu.fwdCamera, 0x7c4, None): [
      b'\xf1\x00AEH MFC  AT USA LHD 1.00 1.00 95740-G2700 201027',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'\xf1\x816H6G5051\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.transmission, 0x7e1, None): [
      b'\xf1\x816U3J9051\x00\x00\xf1\x006U3H1_C2\x00\x006U3J9051\x00\x00HAE0G16NL2\x00\x00\x00\x00',
      b'\xf1\x006U3H1_C2\x00\x006U3J9051\x00\x00HAE0G16NL2\x96\xda\xd4\xee',
    ],
  },
  CAR.SONATA: {
    (Ecu.fwdRadar, 0x7d0, None): [
      b'\xf1\x00DN8 1.00 99110-L0000         \xaa\xaa\xaa\xaa\xaa\xaa\xaa     ',
      b'\xf1\x00DN8 1.00 99110-L0000         \xaa\xaa\xaa\xaa\xaa\xaa\xaa\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
      b'\xf1\x00DN8_ SCC F-CU-      1.00 1.00 99110-L0000         ',
      b'\xf1\x00DN8_ SCC F-CUP      1.00 1.00 99110-L0000         ',
      b'\xf1\x00DN8_ SCC F-CUP      1.00 1.02 99110-L1000         ',
      b'\xf1\x00DN8_ SCC FHCUP      1.00 1.00 99110-L0000         ',
      b'\xf1\x00DN8_ SCC FHCUP      1.00 1.01 99110-L1000         ',
      b'\xf1\x00DN89110-L0000         \xaa\xaa\xaa\xaa\xaa\xaa\xaa     ',
    ],
    (Ecu.abs, 0x7d1, None): [
      b'\xf1\x00DN ESC \x07 106 \x07\x01 58910-L0100',
      b'\xf1\x00DN ESC \x01 102\x19\x04\x13 58910-L1300',
      b'\xf1\x00DN ESC \x03 100 \x08\x01 58910-L0300',
      b'\xf1\x00DN ESC \x06 104\x19\x08\x01 58910-L0100',
      b'\xf1\x00DN ESC \x07 104\x19\x08\x01 58910-L0100',
      b'\xf1\x00DN ESC \x08 103\x19\x06\x01 58910-L1300',
      b'\xf1\x00DN ESC \x07 107"\x08\x07 58910-L0100',
      b'\xf1\x8758910-L0100\xf1\x00DN ESC \x07 106 \x07\x01 58910-L0100',
      b'\xf1\x8758910-L0100\xf1\x00DN ESC \x06 104\x19\x08\x01 58910-L0100',
      b'\xf1\x8758910-L0100\xf1\x00DN ESC \x06 106 \x07\x01 58910-L0100',
      b'\xf1\x8758910-L0100\xf1\x00DN ESC \x07 104\x19\x08\x01 58910-L0100',
      b'\xf1\x00DN ESC \x06 106 \x07\x01 58910-L0100',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'\xf1\x81HM6M1_0a0_F00',
      b'\xf1\x82DNBVN5GMCCXXXDCA',
      b'\xf1\x82DNBVN5GMCCXXXG2F',
      b'\xf1\x82DNBWN5TMDCXXXG2E',
      b'\xf1\x82DNCVN5GMCCXXXF0A',
      b'\xf1\x82DNCVN5GMCCXXXG2B',
      b'\xf1\x870\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf1\x81HM6M1_0a0_J10',
      b'\xf1\x870\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf1\x82DNDWN5TMDCXXXJ1A',
      b'\xf1\x87391162M003',
      b'\xf1\x87391162M013',
      b'\xf1\x87391162M023',
      b'HM6M1_0a0_F00',
      b'HM6M1_0a0_G20',
      b'HM6M2_0a0_BD0',
      b'\xf1\x8739110-2S278\xf1\x82DNDVD5GMCCXXXL5B',
      b'\xf1\x8739110-2S041\xf1\x81HM6M1_0a0_M00',
      b'\xf1\x8739110-2S042\xf1\x81HM6M1_0a0_M00',
      b'\xf1\x81HM6M1_0a0_G20',
    ],
    (Ecu.eps, 0x7d4, None): [
      b'\xf1\x00DN8 MDPS C 1,00 1,01 56310L0010\x00 4DNAC101',  # modified firmware
      b'\xf1\x8756310L0010\x00\xf1\x00DN8 MDPS C 1,00 1,01 56310L0010\x00 4DNAC101',  # modified firmware
      b'\xf1\x00DN8 MDPS C 1.00 1.01 \x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00 4DNAC101',
      b'\xf1\x00DN8 MDPS C 1.00 1.01 56310-L0010 4DNAC101',
      b'\xf1\x00DN8 MDPS C 1.00 1.01 56310L0010\x00 4DNAC101',
      b'\xf1\x00DN8 MDPS R 1.00 1.00 57700-L0000 4DNAP100',
      b'\xf1\x87\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf1\x00DN8 MDPS C 1.00 1.01 \x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00 4DNAC101',
      b'\xf1\x8756310-L0010\xf1\x00DN8 MDPS C 1.00 1.01 56310-L0010 4DNAC101',
      b'\xf1\x8756310-L0210\xf1\x00DN8 MDPS C 1.00 1.01 56310-L0210 4DNAC101',
      b'\xf1\x8756310-L1010\xf1\x00DN8 MDPS C 1.00 1.03 56310-L1010 4DNDC103',
      b'\xf1\x8756310-L1030\xf1\x00DN8 MDPS C 1.00 1.03 56310-L1030 4DNDC103',
      b'\xf1\x8756310L0010\x00\xf1\x00DN8 MDPS C 1.00 1.01 56310L0010\x00 4DNAC101',
      b'\xf1\x8756310L0210\x00\xf1\x00DN8 MDPS C 1.00 1.01 56310L0210\x00 4DNAC101',
      b'\xf1\x8757700-L0000\xf1\x00DN8 MDPS R 1.00 1.00 57700-L0000 4DNAP100',
      b'\xf1\x00DN8 MDPS R 1.00 1.00 57700-L0000 4DNAP101',
      b'\xf1\x00DN8 MDPS C 1.00 1.01 56310-L0210 4DNAC102',
      b'\xf1\x00DN8 MDPS C 1.00 1.01 56310L0200\x00 4DNAC102',
    ],
    (Ecu.fwdCamera, 0x7c4, None): [
      b'\xf1\x00DN8 MFC  AT KOR LHD 1.00 1.02 99211-L1000 190422',
      b'\xf1\x00DN8 MFC  AT RUS LHD 1.00 1.03 99211-L1000 190705',
      b'\xf1\x00DN8 MFC  AT USA LHD 1.00 1.00 99211-L0000 190716',
      b'\xf1\x00DN8 MFC  AT USA LHD 1.00 1.01 99211-L0000 191016',
      b'\xf1\x00DN8 MFC  AT USA LHD 1.00 1.03 99211-L0000 210603',
      b'\xf1\x00DN8 MFC  AT USA LHD 1.00 1.05 99211-L1000 201109',
      b'\xf1\x00DN8 MFC  AT USA LHD 1.00 1.06 99211-L1000 210325',
      b'\xf1\x00DN8 MFC  AT USA LHD 1.00 1.07 99211-L1000 211223',
    ],
    (Ecu.transmission, 0x7e1, None): [
      b'\xf1\x00bcsh8p54  U903\x00\x00\x00\x00\x00\x00SDN8T16NB0z{\xd4v',
      b'\xf1\x00bcsh8p54  U913\x00\x00\x00\x00\x00\x00SDN8T16NB1\xe3\xc10\xa1',
      b'\xf1\x00bcsh8p54  U913\x00\x00\x00\x00\x00\x00SDN8T16NB2\n\xdd^\xbc',
      b'\xf1\x00HT6TA260BLHT6TA800A1TDN8C20KS4\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
      b'\xf1\x00HT6TA260BLHT6TA810A1TDN8M25GS0\x00\x00\x00\x00\x00\x00\xaa\x8c\xd9p',
      b'\xf1\x00HT6WA250BLHT6WA910A1SDN8G25NB1\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
      b'\xf1\x00HT6WA250BLHT6WA910A1SDN8G25NB1\x00\x00\x00\x00\x00\x00\x96\xa1\xf1\x92',
      b'\xf1\x00HT6WA280BLHT6WAD10A1SDN8G25NB2\x00\x00\x00\x00\x00\x00\x08\xc9O:',
      b'\xf1\x00HT6WA280BLHT6WAD10A1SDN8G25NB4\x00\x00\x00\x00\x00\x00g!l[',
      b'\xf1\x00T02601BL  T02730A1  VDN8T25XXX730NS5\xf7_\x92\xf5',
      b'\xf1\x00T02601BL  T02832A1  VDN8T25XXX832NS8G\x0e\xfeE',
      b'\xf1\x00T02601BL  T02900A1  VDN8T25XXX900NSCF\xe4!Y',
      b'\xf1\x87954A02N060\x00\x00\x00\x00\x00\xf1\x81T02730A1  \xf1\x00T02601BL  T02730A1  VDN8T25XXX730NS5\xf7_\x92\xf5',
      b'\xf1\x87SAKFBA2926554GJ2VefVww\x87xwwwww\x88\x87xww\x87wTo\xfb\xffvUo\xff\x8d\x16\xf1\x81U903\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U903\x00\x00\x00\x00\x00\x00SDN8T16NB0z{\xd4v',
      b'\xf1\x87SAKFBA3030524GJ2UVugww\x97yx\x88\x87\x88vw\x87gww\x87wto\xf9\xfffUo\xff\xa2\x0c\xf1\x81U903\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U903\x00\x00\x00\x00\x00\x00SDN8T16NB0z{\xd4v',
      b'\xf1\x87SAKFBA3356084GJ2\x86fvgUUuWgw\x86www\x87wffvf\xb6\xcf\xfc\xffeUO\xff\x12\x19\xf1\x81U903\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U903\x00\x00\x00\x00\x00\x00SDN8T16NB0z{\xd4v',
      b'\xf1\x87SAKFBA3474944GJ2ffvgwwwwg\x88\x86x\x88\x88\x98\x88ffvfeo\xfa\xff\x86fo\xff\t\xae\xf1\x81U903\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U903\x00\x00\x00\x00\x00\x00SDN8T16NB0z{\xd4v',
      b'\xf1\x87SAKFBA3475714GJ2Vfvgvg\x96yx\x88\x97\x88ww\x87ww\x88\x87xs_\xfb\xffvUO\xff\x0f\xff\xf1\x81U903\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U903\x00\x00\x00\x00\x00\x00SDN8T16NB0z{\xd4v',
      b'\xf1\x87SALDBA3510954GJ3ww\x87xUUuWx\x88\x87\x88\x87w\x88wvfwfc_\xf9\xff\x98wO\xffl\xe0\xf1\x89HT6WA910A1\xf1\x82SDN8G25NB1\x00\x00\x00\x00\x00\x00',
      b'\xf1\x87SALDBA3573534GJ3\x89\x98\x89\x88EUuWgwvwwwwww\x88\x87xTo\xfa\xff\x86f\x7f\xffo\x0e\xf1\x89HT6WA910A1\xf1\x82SDN8G25NB1\x00\x00\x00\x00\x00\x00',
      b'\xf1\x87SALDBA3601464GJ3\x88\x88\x88\x88ffvggwvwvw\x87gww\x87wvo\xfb\xff\x98\x88\x7f\xffjJ\xf1\x89HT6WA910A1\xf1\x82SDN8G25NB1\x00\x00\x00\x00\x00\x00',
      b'\xf1\x87SALDBA3753044GJ3UUeVff\x86hwwwwvwwgvfgfvo\xf9\xfffU_\xffC\xae\xf1\x89HT6WA910A1\xf1\x82SDN8G25NB1\x00\x00\x00\x00\x00\x00',
      b'\xf1\x87SALDBA3862294GJ3vfvgvefVxw\x87\x87w\x88\x87xwwwwc_\xf9\xff\x87w\x9f\xff\xd5\xdc\xf1\x89HT6WA910A1\xf1\x82SDN8G25NB1\x00\x00\x00\x00\x00\x00',
      b'\xf1\x87SALDBA3873834GJ3fefVwuwWx\x88\x97\x88w\x88\x97xww\x87wU_\xfb\xff\x86f\x8f\xffN\x04\xf1\x89HT6WA910A1\xf1\x82SDN8G25NB1\x00\x00\x00\x00\x00\x00',
      b'\xf1\x87SALDBA4525334GJ3\x89\x99\x99\x99fevWh\x88\x86\x88fwvgw\x88\x87xfo\xfa\xffuDo\xff\xd1>\xf1\x89HT6WA910A1\xf1\x82SDN8G25NB1\x00\x00\x00\x00\x00\x00',
      b'\xf1\x87SALDBA4626804GJ3wwww\x88\x87\x88xx\x88\x87\x88wwgw\x88\x88\x98\x88\x95_\xf9\xffuDo\xff|\xe7\xf1\x89HT6WA910A1\xf1\x82SDN8G25NB1\x00\x00\x00\x00\x00\x00',
      b'\xf1\x87SALDBA4803224GJ3wwwwwvwg\x88\x88\x98\x88wwww\x87\x88\x88xu\x9f\xfc\xff\x87f\x8f\xff\xea\xea\xf1\x89HT6WA910A1\xf1\x82SDN8G25NB1\x00\x00\x00\x00\x00\x00',
      b'\xf1\x87SALDBA6212564GJ3\x87wwwUTuGg\x88\x86xx\x88\x87\x88\x87\x88\x98xu?\xf9\xff\x97f\x7f\xff\xb8\n\xf1\x89HT6WA910A1\xf1\x82SDN8G25NB1\x00\x00\x00\x00\x00\x00',
      b'\xf1\x87SALDBA6347404GJ3wwwwff\x86hx\x88\x97\x88\x88\x88\x88\x88vfgf\x88?\xfc\xff\x86Uo\xff\xec/\xf1\x89HT6WA910A1\xf1\x82SDN8G25NB1\x00\x00\x00\x00\x00\x00',
      b'\xf1\x87SALDBA6901634GJ3UUuWVeVUww\x87wwwwwvUge\x86/\xfb\xff\xbb\x99\x7f\xff]2\xf1\x89HT6WA910A1\xf1\x82SDN8G25NB1\x00\x00\x00\x00\x00\x00',
      b'\xf1\x87SALDBA7077724GJ3\x98\x88\x88\x88ww\x97ygwvwww\x87ww\x88\x87x\x87_\xfd\xff\xba\x99o\xff\x99\x01\xf1\x89HT6WA910A1\xf1\x82SDN8G25NB1\x00\x00\x00\x00\x00\x00',
      b'\xf1\x87SALFBA3525114GJ2wvwgvfvggw\x86wffvffw\x86g\x85_\xf9\xff\xa8wo\xffv\xcd\xf1\x81U903\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U903\x00\x00\x00\x00\x00\x00SDN8T16NB0z{\xd4v',
      b'\xf1\x87SALFBA3624024GJ2\x88\x88\x88\x88wv\x87hx\x88\x97\x88x\x88\x97\x88ww\x87w\x86o\xfa\xffvU\x7f\xff\xd1\xec\xf1\x81U903\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U903\x00\x00\x00\x00\x00\x00SDN8T16NB0z{\xd4v',
      b'\xf1\x87SALFBA3960824GJ2wwwwff\x86hffvfffffvfwfg_\xf9\xff\xa9\x88\x8f\xffb\x99\xf1\x81U903\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U903\x00\x00\x00\x00\x00\x00SDN8T16NB0z{\xd4v',
      b'\xf1\x87SALFBA4011074GJ2fgvwwv\x87hw\x88\x87xww\x87wwfgvu_\xfa\xffefo\xff\x87\xc0\xf1\x81U903\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U903\x00\x00\x00\x00\x00\x00SDN8T16NB0z{\xd4v',
      b'\xf1\x87SALFBA4121304GJ2x\x87xwff\x86hwwwwww\x87wwwww\x84_\xfc\xff\x98\x88\x9f\xffi\xa6\xf1\x81U903\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U903\x00\x00\x00\x00\x00\x00SDN8T16NB0z{\xd4v',
      b'\xf1\x87SALFBA4195874GJ2EVugvf\x86hgwvwww\x87wgw\x86wc_\xfb\xff\x98\x88\x8f\xff\xe23\xf1\x81U903\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U903\x00\x00\x00\x00\x00\x00SDN8T16NB0z{\xd4v',
      b'\xf1\x87SALFBA4625294GJ2eVefeUeVx\x88\x97\x88wwwwwwww\xa7o\xfb\xffvw\x9f\xff\xee.\xf1\x81U903\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U903\x00\x00\x00\x00\x00\x00SDN8T16NB0z{\xd4v',
      b'\xf1\x87SALFBA4728774GJ2vfvg\x87vwgww\x87ww\x88\x97xww\x87w\x86_\xfb\xffeD?\xffk0\xf1\x81U903\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U903\x00\x00\x00\x00\x00\x00SDN8T16NB0z{\xd4v',
      b'\xf1\x87SALFBA5129064GJ2vfvgwv\x87hx\x88\x87\x88ww\x87www\x87wd_\xfa\xffvfo\xff\x1d\x00\xf1\x81U903\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U903\x00\x00\x00\x00\x00\x00SDN8T16NB0z{\xd4v',
      b'\xf1\x87SALFBA5454914GJ2\x98\x88\x88\x88\x87vwgx\x88\x87\x88xww\x87ffvf\xa7\x7f\xf9\xff\xa8w\x7f\xff\x1b\x90\xf1\x81U903\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U903\x00\x00\x00\x00\x00\x00SDN8T16NB0z{\xd4v',
      b'\xf1\x87SALFBA5987784GJ2UVugDDtGx\x88\x87\x88w\x88\x87xwwwwd/\xfb\xff\x97fO\xff\xb0h\xf1\x81U903\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U903\x00\x00\x00\x00\x00\x00SDN8T16NB0z{\xd4v',
      b'\xf1\x87SALFBA5987864GJ2fgvwUUuWgwvw\x87wxwwwww\x84/\xfc\xff\x97w\x7f\xff\xdf\x1d\xf1\x81U903\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U903\x00\x00\x00\x00\x00\x00SDN8T16NB0z{\xd4v',
      b'\xf1\x87SALFBA6337644GJ2vgvwwv\x87hgffvwwwwwwww\x85O\xfa\xff\xa7w\x7f\xff\xc5\xfc\xf1\x81U903\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U903\x00\x00\x00\x00\x00\x00SDN8T16NB0z{\xd4v',
      b'\xf1\x87SALFBA6802004GJ2UUuWUUuWgw\x86www\x87www\x87w\x96?\xf9\xff\xa9\x88\x7f\xff\x9fK\xf1\x81U903\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U903\x00\x00\x00\x00\x00\x00SDN8T16NB0z{\xd4v',
      b'\xf1\x87SALFBA6892284GJ233S5\x87w\x87xx\x88\x87\x88vwwgww\x87w\x84?\xfb\xff\x98\x88\x8f\xff*\x9e\xf1\x81U903\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U903\x00\x00\x00\x00\x00\x00SDN8T16NB0z{\xd4v',
      b'\xf1\x87SALFBA7005534GJ2eUuWfg\x86xxww\x87x\x88\x87\x88\x88w\x88\x87\x87O\xfc\xffuUO\xff\xa3k\xf1\x81U913\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U913\x00\x00\x00\x00\x00\x00SDN8T16NB1\xe3\xc10\xa1',
      b'\xf1\x87SALFBA7152454GJ2gvwgFf\x86hx\x88\x87\x88vfWfffffd?\xfa\xff\xba\x88o\xff,\xcf\xf1\x81U913\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U913\x00\x00\x00\x00\x00\x00SDN8T16NB1\xe3\xc10\xa1',
      b'\xf1\x87SALFBA7485034GJ2ww\x87xww\x87xfwvgwwwwvfgf\xa5/\xfc\xff\xa9w_\xff40\xf1\x81U913\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U913\x00\x00\x00\x00\x00\x00SDN8T16NB2\n\xdd^\xbc',
      b'\xf1\x87SAMDBA7743924GJ3wwwwww\x87xgwvw\x88\x88\x88\x88wwww\x85_\xfa\xff\x86f\x7f\xff0\x9d\xf1\x89HT6WAD10A1\xf1\x82SDN8G25NB2\x00\x00\x00\x00\x00\x00',
      b'\xf1\x87SAMDBA7817334GJ3Vgvwvfvgww\x87wwwwwwfgv\x97O\xfd\xff\x88\x88o\xff\x8e\xeb\xf1\x89HT6WAD10A1\xf1\x82SDN8G25NB2\x00\x00\x00\x00\x00\x00',
      b'\xf1\x87SAMDBA8054504GJ3gw\x87xffvgffffwwwweUVUf?\xfc\xffvU_\xff\xddl\xf1\x89HT6WAD10A1\xf1\x82SDN8G25NB2\x00\x00\x00\x00\x00\x00',
      b'\xf1\x87SAMFB41553621GC7ww\x87xUU\x85Xvwwg\x88\x88\x88\x88wwgw\x86\xaf\xfb\xffuDo\xff\xaa\x8f\xf1\x81U913\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U913\x00\x00\x00\x00\x00\x00SDN8T16NB2\n\xdd^\xbc',
      b'\xf1\x87SAMFB42555421GC7\x88\x88\x88\x88wvwgx\x88\x87\x88wwgw\x87wxw3\x8f\xfc\xff\x98f\x8f\xffga\xf1\x81U913\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U913\x00\x00\x00\x00\x00\x00SDN8T16NB2\n\xdd^\xbc',
      b'\xf1\x87SAMFBA7978674GJ2gw\x87xgw\x97ywwwwvUGeUUeU\x87O\xfb\xff\x98w\x8f\xfffF\xf1\x81U913\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U913\x00\x00\x00\x00\x00\x00SDN8T16NB2\n\xdd^\xbc',
      b'\xf1\x87SAMFBA9283024GJ2wwwwEUuWwwgwwwwwwwww\x87/\xfb\xff\x98w\x8f\xff<\xd3\xf1\x81U913\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U913\x00\x00\x00\x00\x00\x00SDN8T16NB2\n\xdd^\xbc',
      b'\xf1\x87SAMFBA9708354GJ2wwwwVf\x86h\x88wx\x87xww\x87\x88\x88\x88\x88w/\xfa\xff\x97w\x8f\xff\x86\xa0\xf1\x81U913\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U913\x00\x00\x00\x00\x00\x00SDN8T16NB2\n\xdd^\xbc',
      b'\xf1\x87SANDB45316691GC6\x99\x99\x99\x99\x88\x88\xa8\x8avfwfwwww\x87wxwT\x9f\xfd\xff\x88wo\xff\x1c\xfa\xf1\x89HT6WAD10A1\xf1\x82SDN8G25NB3\x00\x00\x00\x00\x00\x00',
      b'\xf1\x87SALFBA7460044GJ2gx\x87\x88Vf\x86hx\x88\x87\x88wwwwgw\x86wd?\xfa\xff\x86U_\xff\xaf\x1f\xf1\x81U913\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U913\x00\x00\x00\x00\x00\x00SDN8T16NB2\n\xdd^\xbc',
      b'\xf1\x87SAMFBA8105254GJ2wx\x87\x88Vf\x86hx\x88\x87\x88wwwwwwww\x86O\xfa\xff\x99\x88\x7f\xffZG\xf1\x81U913\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U913\x00\x00\x00\x00\x00\x00SDN8T16NB2\n\xdd^\xbc',
      b'\xf1\x87SANFB45889451GC7wx\x87\x88gw\x87x\x88\x88x\x88\x87wxw\x87wxw\x87\x8f\xfc\xffeU\x8f\xff+Q\xf1\x81U913\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U913\x00\x00\x00\x00\x00\x00SDN8T16NB2\n\xdd^\xbc',
      b'\xf1\x00T02601BL  T02900A1  VDN8T25XXX900NSA\xb9\x13\xf9p',
    ],
  },
  CAR.SONATA_LF: {
    (Ecu.fwdRadar, 0x7d0, None): [
      b'\xf1\x00LF__ SCC F-CUP      1.00 1.00 96401-C2200         ',
    ],
    (Ecu.abs, 0x7d1, None): [
      b'\xf1\x00LF ESC \f 11 \x17\x01\x13 58920-C2610',
      b'\xf1\x00LF ESC \t 11 \x17\x01\x13 58920-C2610',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'\xf1\x81606D5051\x00\x00\x00\x00\x00\x00\x00\x00',
      b'\xf1\x81606D5K51\x00\x00\x00\x00\x00\x00\x00\x00',
      b'\xf1\x81606G1051\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.fwdCamera, 0x7c4, None): [
      b'\xf1\x00LFF LKAS AT USA LHD 1.00 1.01 95740-C1000 E51',
      b'\xf1\x00LFF LKAS AT USA LHD 1.01 1.02 95740-C1000 E52',
    ],
    (Ecu.transmission, 0x7e1, None): [
      b'\xf1\x006T6H0_C2\x00\x006T6B4051\x00\x00TLF0G24NL1\xb0\x9f\xee\xf5',
      b'\xf1\x87\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xf1\x816T6B4051\x00\x00\xf1\x006T6H0_C2\x00\x006T6B4051\x00\x00TLF0G24NL1\x00\x00\x00\x00',
      b'\xf1\x87\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xf1\x816T6B4051\x00\x00\xf1\x006T6H0_C2\x00\x006T6B4051\x00\x00TLF0G24NL1\xb0\x9f\xee\xf5',
      b'\xf1\x87\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xf1\x816T6B4051\x00\x00\xf1\x006T6H0_C2\x00\x006T6B4051\x00\x00TLF0G24SL2n\x8d\xbe\xd8',
      b'\xf1\x87LAHSGN012918KF10\x98\x88x\x87\x88\x88x\x87\x88\x88\x98\x88\x87w\x88w\x88\x88\x98\x886o\xf6\xff\x98w\x7f\xff3\x00\xf1\x816W3B1051\x00\x00\xf1\x006W351_C2\x00\x006W3B1051\x00\x00TLF0T20NL2\x00\x00\x00\x00',
      b'\xf1\x87LAHSGN012918KF10\x98\x88x\x87\x88\x88x\x87\x88\x88\x98\x88\x87w\x88w\x88\x88\x98\x886o\xf6\xff\x98w\x7f\xff3\x00\xf1\x816W3B1051\x00\x00\xf1\x006W351_C2\x00\x006W3B1051\x00\x00TLF0T20NL2H\r\xbdm',
      b'\xf1\x87LAJSG49645724HF0\x87x\x87\x88\x87www\x88\x99\xa8\x89\x88\x99\xa8\x89\x88\x99\xa8\x89S_\xfb\xff\x87f\x7f\xff^2\xf1\x816W3B1051\x00\x00\xf1\x006W351_C2\x00\x006W3B1051\x00\x00TLF0T20NL2H\r\xbdm',
    ],
  },
  CAR.TUCSON: {
    (Ecu.fwdRadar, 0x7d0, None): [
      b'\xf1\x00TL__ FCA F-CUP      1.00 1.01 99110-D3500         ',
      b'\xf1\x00TL__ FCA F-CUP      1.00 1.02 99110-D3510         ',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'\xf1\x8971TLC2NAIDDIR002\xf1\x8271TLC2NAIDDIR002',
      b'\xf1\x81606G3051\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.fwdCamera, 0x7c4, None): [
      b'\xf1\x00TL  MFC  AT KOR LHD 1.00 1.02 95895-D3800 180719',
      b'\xf1\x00TL  MFC  AT USA LHD 1.00 1.06 95895-D3800 190107',
    ],
    (Ecu.transmission, 0x7e1, None): [
      b'\xf1\x87LBJXAN202299KF22\x87x\x87\x88ww\x87xx\x88\x97\x88\x87\x88\x98x\x88\x99\x98\x89\x87o\xf6\xff\x87w\x7f\xff\x12\x9a\xf1\x81U083\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U083\x00\x00\x00\x00\x00\x00TTL2V20KL1\x8fRn\x8a',
      b'\xf1\x87KMLDCU585233TJ20wx\x87\x88x\x88\x98\x89vfwfwwww\x87f\x9f\xff\x98\xff\x7f\xf9\xf7s\xf1\x816T6G4051\x00\x00\xf1\x006T6J0_C2\x00\x006T6G4051\x00\x00TTL4G24NH2\x00\x00\x00\x00',
    ],
  },
  CAR.SANTA_FE: {
    (Ecu.fwdRadar, 0x7d0, None): [
      b'\xf1\x00TM__ SCC F-CUP      1.00 1.00 99110-S1210         ',
      b'\xf1\x00TM__ SCC F-CUP      1.00 1.01 99110-S2000         ',
      b'\xf1\x00TM__ SCC F-CUP      1.00 1.02 99110-S2000         ',
      b'\xf1\x00TM__ SCC F-CUP      1.00 1.03 99110-S2000         ',
    ],
    (Ecu.abs, 0x7d1, None): [
      b'\xf1\x00TM ESC \r 100\x18\x031 58910-S2650',
      b'\xf1\x00TM ESC \r 105\x19\x05# 58910-S1500',
      b'\xf1\x00TM ESC \r 103\x18\x11\x08 58910-S2650',
      b'\xf1\x00TM ESC \r 104\x19\x07\x08 58910-S2650',
      b'\xf1\x00TM ESC \x02 100\x18\x030 58910-S2600',
      b'\xf1\x00TM ESC \x02 102\x18\x07\x01 58910-S2600',
      b'\xf1\x00TM ESC \x02 103\x18\x11\x07 58910-S2600',
      b'\xf1\x00TM ESC \x02 104\x19\x07\x07 58910-S2600',
      b'\xf1\x00TM ESC \x03 103\x18\x11\x07 58910-S2600',
      b'\xf1\x00TM ESC \x0c 103\x18\x11\x08 58910-S2650',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'\xf1\x81606EA051\x00\x00\x00\x00\x00\x00\x00\x00',
      b'\xf1\x81606G1051\x00\x00\x00\x00\x00\x00\x00\x00',
      b'\xf1\x81606G3051\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.eps, 0x7d4, None): [
      b'\xf1\x00TM  MDPS C 1.00 1.00 56340-S2000 8409',
      b'\xf1\x00TM  MDPS C 1.00 1.00 56340-S2000 8A12',
      b'\xf1\x00TM  MDPS C 1.00 1.01 56340-S2000 9129',
      b'\xf1\x00TM  MDPS R 1.00 1.02 57700-S1100 4TMDP102'
    ],
    (Ecu.fwdCamera, 0x7c4, None): [
      b'\xf1\x00TM  MFC  AT EUR LHD 1.00 1.01 99211-S1010 181207',
      b'\xf1\x00TM  MFC  AT USA LHD 1.00 1.00 99211-S2000 180409',
    ],
    (Ecu.transmission, 0x7e1, None): [
      b'\xf1\x00bcsh8p54  U833\x00\x00\x00\x00\x00\x00TTM4V22US3_<]\xf1',
      b'\xf1\x006W351_C2\x00\x006W3E1051\x00\x00TTM4T20NS5\x00\x00\x00\x00',
      b'\xf1\x87LBJSGA7082574HG0\x87www\x98\x88\x88\x88\x99\xaa\xb9\x9afw\x86gx\x99\xa7\x89co\xf8\xffvU_\xffR\xaf\xf1\x816W3C2051\x00\x00\xf1\x006W351_C2\x00\x006W3C2051\x00\x00TTM2T20NS1\x00\xa6\xe0\x91',
      b'\xf1\x87LBKSGA0458404HG0vfvg\x87www\x89\x99\xa8\x99y\xaa\xa7\x9ax\x88\xa7\x88t_\xf9\xff\x86w\x8f\xff\x15x\xf1\x816W3C2051\x00\x00\xf1\x006W351_C2\x00\x006W3C2051\x00\x00TTM2T20NS1\x00\x00\x00\x00',
      b'\xf1\x87LDJUEA6010814HG1\x87w\x87x\x86gvw\x88\x88\x98\x88gw\x86wx\x88\x97\x88\x85o\xf8\xff\x86f_\xff\xd37\xf1\x816W3C2051\x00\x00\xf1\x006W351_C2\x00\x006W3C2051\x00\x00TTM4T20NS0\xf8\x19\x92g',
      b'\xf1\x87LDJUEA6458264HG1ww\x87x\x97x\x87\x88\x88\x99\x98\x89g\x88\x86xw\x88\x97x\x86o\xf7\xffvw\x8f\xff3\x9a\xf1\x816W3C2051\x00\x00\xf1\x006W351_C2\x00\x006W3C2051\x00\x00TTM4T20NS0\xf8\x19\x92g',
      b'\xf1\x87LDKUEA2045844HG1wwww\x98\x88x\x87\x88\x88\xa8\x88x\x99\x97\x89x\x88\xa7\x88U\x7f\xf8\xffvfO\xffC\x1e\xf1\x816W3E0051\x00\x00\xf1\x006W351_C2\x00\x006W3E0051\x00\x00TTM4T20NS3\x00\x00\x00\x00',
      b'\xf1\x87LDKUEA9993304HG1\x87www\x97x\x87\x88\x99\x99\xa9\x99x\x99\xa7\x89w\x88\x97x\x86_\xf7\xffwwO\xffl#\xf1\x816W3C2051\x00\x00\xf1\x006W351_C2\x00\x006W3C2051\x00\x00TTM4T20NS1R\x7f\x90\n',
      b'\xf1\x87LDLUEA6061564HG1\xa9\x99\x89\x98\x87wwwx\x88\x97\x88x\x99\xa7\x89x\x99\xa7\x89sO\xf9\xffvU_\xff<\xde\xf1\x816W3E1051\x00\x00\xf1\x006W351_C2\x00\x006W3E1051\x00\x00TTM4T20NS50\xcb\xc3\xed',
      b'\xf1\x87LDLUEA6159884HG1\x88\x87hv\x99\x99y\x97\x89\xaa\xb8\x9ax\x99\x87\x89y\x99\xb7\x99\xa7?\xf7\xff\x97wo\xff\xf3\x05\xf1\x816W3E1051\x00\x00\xf1\x006W351_C2\x00\x006W3E1051\x00\x00TTM4T20NS5\x00\x00\x00\x00',
      b'\xf1\x87LDLUEA6852664HG1\x97wWu\x97www\x89\xaa\xc8\x9ax\x99\x97\x89x\x99\xa7\x89SO\xf7\xff\xa8\x88\x7f\xff\x03z\xf1\x816W3E1051\x00\x00\xf1\x006W351_C2\x00\x006W3E1051\x00\x00TTM4T20NS50\xcb\xc3\xed',
      b'\xf1\x87LDLUEA6898374HG1fevW\x87wwwx\x88\x97\x88h\x88\x96\x88x\x88\xa7\x88ao\xf9\xff\x98\x99\x7f\xffD\xe2\xf1\x816W3E1051\x00\x00\xf1\x006W351_C2\x00\x006W3E1051\x00\x00TTM4T20NS5\x00\x00\x00\x00',
      b'\xf1\x87LDLUEA6898374HG1fevW\x87wwwx\x88\x97\x88h\x88\x96\x88x\x88\xa7\x88ao\xf9\xff\x98\x99\x7f\xffD\xe2\xf1\x816W3E1051\x00\x00\xf1\x006W351_C2\x00\x006W3E1051\x00\x00TTM4T20NS50\xcb\xc3\xed',
      b'\xf1\x87SBJWAA5842214GG0\x88\x87\x88xww\x87x\x89\x99\xa8\x99\x88\x99\x98\x89w\x88\x87xw_\xfa\xfffU_\xff\xd1\x8d\xf1\x816W3C2051\x00\x00\xf1\x006W351_C2\x00\x006W3C2051\x00\x00TTM2G24NS1\x98{|\xe3',
      b'\xf1\x87SBJWAA5890864GG0\xa9\x99\x89\x98\x98\x87\x98y\x89\x99\xa8\x99w\x88\x87xww\x87wvo\xfb\xffuD_\xff\x9f\xb5\xf1\x816W3C2051\x00\x00\xf1\x006W351_C2\x00\x006W3C2051\x00\x00TTM2G24NS1\x98{|\xe3',
      b'\xf1\x87SBJWAA6562474GG0ffvgeTeFx\x88\x97\x88ww\x87www\x87w\x84o\xfa\xff\x87fO\xff\xc2 \xf1\x816W3C2051\x00\x00\xf1\x006W351_C2\x00\x006W3C2051\x00\x00TTM2G24NS1\x00\x00\x00\x00',
      b'\xf1\x87SBJWAA6562474GG0ffvgeTeFx\x88\x97\x88ww\x87www\x87w\x84o\xfa\xff\x87fO\xff\xc2 \xf1\x816W3C2051\x00\x00\xf1\x006W351_C2\x00\x006W3C2051\x00\x00TTM2G24NS1\x98{|\xe3',
      b'\xf1\x87SBJWAA7780564GG0wvwgUUeVwwwwx\x88\x87\x88wwwwd_\xfc\xff\x86f\x7f\xff\xd7*\xf1\x816W3C2051\x00\x00\xf1\x006W351_C2\x00\x006W3C2051\x00\x00TTM2G24NS2F\x84<\xc0',
      b'\xf1\x87SBJWAA8278284GG0ffvgUU\x85Xx\x88\x87\x88x\x88w\x88ww\x87w\x96o\xfd\xff\xa7U_\xff\xf2\xa0\xf1\x816W3C2051\x00\x00\xf1\x006W351_C2\x00\x006W3C2051\x00\x00TTM2G24NS2F\x84<\xc0',
      b'\xf1\x87SBLWAA4363244GG0wvwgwv\x87hgw\x86ww\x88\x87xww\x87wdo\xfb\xff\x86f\x7f\xff3$\xf1\x816W3E1051\x00\x00\xf1\x006W351_C2\x00\x006W3E1051\x00\x00TTM2G24NS6\x00\x00\x00\x00',
      b'\xf1\x87SBLWAA4363244GG0wvwgwv\x87hgw\x86ww\x88\x87xww\x87wdo\xfb\xff\x86f\x7f\xff3$\xf1\x816W3E1051\x00\x00\xf1\x006W351_C2\x00\x006W3E1051\x00\x00TTM2G24NS6x0\x17\xfe',
      b'\xf1\x87SBLWAA4899564GG0VfvgUU\x85Xx\x88\x87\x88vfgf\x87wxwvO\xfb\xff\x97f\xb1\xffSB\xf1\x816W3E1051\x00\x00\xf1\x006W351_C2\x00\x006W3E1051\x00\x00TTM2G24NS7\x00\x00\x00\x00',
      b'\xf1\x87SBLWAA6622844GG0wwwwff\x86hwwwwx\x88\x87\x88\x88\x88\x88\x88\x98?\xfd\xff\xa9\x88\x7f\xffn\xe5\xf1\x816W3E1051\x00\x00\xf1\x006W351_C2\x00\x006W3E1051\x00\x00TTM2G24NS7u\x1e{\x1c',
      b'\xf1\x87SDJXAA7656854GG1DEtWUU\x85X\x88\x88\x98\x88w\x88\x87xx\x88\x87\x88\x96o\xfb\xff\x86f\x7f\xff.\xca\xf1\x816W3C2051\x00\x00\xf1\x006W351_C2\x00\x006W3C2051\x00\x00TTM4G24NS2\x00\x00\x00\x00',
      b'\xf1\x87SDJXAA7656854GG1DEtWUU\x85X\x88\x88\x98\x88w\x88\x87xx\x88\x87\x88\x96o\xfb\xff\x86f\x7f\xff.\xca\xf1\x816W3C2051\x00\x00\xf1\x006W351_C2\x00\x006W3C2051\x00\x00TTM4G24NS2K\xdaV0',
      b'\xf1\x87SDKXAA2443414GG1vfvgwv\x87h\x88\x88\x88\x88ww\x87wwwww\x99_\xfc\xffvD?\xffl\xd2\xf1\x816W3E1051\x00\x00\xf1\x006W351_C2\x00\x006W3E1051\x00\x00TTM4G24NS6\x00\x00\x00\x00',
    ],
  },
  CAR.SANTA_FE_2022: {
    (Ecu.fwdRadar, 0x7d0, None): [
      b'\xf1\x00TM__ SCC F-CUP      1.00 1.00 99110-S1500         ',
      b'\xf1\x8799110S1500\xf1\x00TM__ SCC F-CUP      1.00 1.00 99110-S1500         ',
      b'\xf1\x00TM__ SCC FHCUP      1.00 1.00 99110-S1500         ',
    ],
    (Ecu.abs, 0x7d1, None): [
      b'\xf1\x00TM ESC \x02 101 \x08\x04 58910-S2GA0',
      b'\xf1\x00TM ESC \x03 101 \x08\x02 58910-S2DA0',
      b'\xf1\x8758910-S2DA0\xf1\x00TM ESC \x03 101 \x08\x02 58910-S2DA0',
      b'\xf1\x8758910-S2GA0\xf1\x00TM ESC \x02 101 \x08\x04 58910-S2GA0',
      b'\xf1\x8758910-S1DA0\xf1\x00TM ESC \x1e 102 \x08\x08 58910-S1DA0',
      b'\xf1\x8758910-S2GA0\xf1\x00TM ESC \x04 102!\x04\x05 58910-S2GA0',
      b'\xf1\x00TM ESC \x04 102!\x04\x05 58910-S2GA0',
      b'\xf1\x00TM ESC \x04 101 \x08\x04 58910-S2GA0',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'\xf1\x82TACVN5GMI3XXXH0A',
      b'\xf1\x82TMBZN5TMD3XXXG2E',
      b'\xf1\x82TACVN5GSI3XXXH0A',
      b'\xf1\x82TMCFD5MMCXXXXG0A',
      b'\xf1\x81HM6M1_0a0_G20',
      b'\xf1\x870\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf1\x82TMDWN5TMD3TXXJ1A',
      b'\xf1\x81HM6M2_0a0_G00',
      b'\xf1\x870\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf1\x81HM6M1_0a0_J10',
    ],
    (Ecu.eps, 0x7d4, None): [
      b'\xf1\x00TM  MDPS C 1.00 1.02 56370-S2AA0 0B19',
      b'\xf1\x00TM  MDPS C 1.00 1.01 56310-S1AB0 4TSDC101',
    ],
    (Ecu.fwdCamera, 0x7c4, None): [
      b'\xf1\x00TMA MFC  AT MEX LHD 1.00 1.01 99211-S2500 210205',
      b'\xf1\x00TMA MFC  AT USA LHD 1.00 1.00 99211-S2500 200720',
      b'\xf1\x00TM  MFC  AT EUR LHD 1.00 1.03 99211-S1500 210224',
      b'\xf1\x00TMA MFC  AT USA LHD 1.00 1.01 99211-S2500 210205',
    ],
    (Ecu.transmission, 0x7e1, None): [
      b'\xf1\x00T02601BL  T02900A1  VTMPT25XXX900NSA\xf3\xf4Uj',
      b'\xf1\x87SDMXCA9087684GN1VfvgUUeVwwgwwwwwffffU?\xfb\xff\x97\x88\x7f\xff+\xa4\xf1\x89HT6WAD00A1\xf1\x82STM4G25NH1\x00\x00\x00\x00\x00\x00',
      b'\xf1\x00T02601BL  T02730A1  VTMPT25XXX730NS2\xa6\x06\x88\xf7',
      b'\xf1\x87SDMXCA8653204GN1EVugEUuWwwwwww\x87wwwwwv/\xfb\xff\xa8\x88\x9f\xff\xa5\x9c\xf1\x89HT6WAD00A1\xf1\x82STM4G25NH1\x00\x00\x00\x00\x00\x00',
      b'\xf1\x87954A02N250\x00\x00\x00\x00\x00\xf1\x81T02730A1  \xf1\x00T02601BL  T02730A1  VTMPT25XXX730NS2\xa6\x06\x88\xf7',
      b'\xf1\x87KMMYBU034207SB72x\x89\x88\x98h\x88\x98\x89\x87fhvvfWf33_\xff\x87\xff\x8f\xfa\x81\xe5\xf1\x89HT6TAF00A1\xf1\x82STM0M25GS1\x00\x00\x00\x00\x00\x00',
      b'\xf1\x87954A02N250\x00\x00\x00\x00\x00\xf1\x81T02730A1  \xf1\x00T02601BL  T02730A1  VTMPT25XXX730NS2\xa6',
      b'\xf1\x00HT6TA290BLHT6TAF00A1STM0M25GS1\x00\x00\x00\x00\x00\x006\xd8\x97\x15',
      b'\xf1\x00T02601BL  T02900A1  VTMPT25XXX900NS8\xb7\xaa\xfe\xfc',
      b'\xf1\x87954A02N250\x00\x00\x00\x00\x00\xf1\x81T02900A1  \xf1\x00T02601BL  T02900A1  VTMPT25XXX900NS8\xb7\xaa\xfe\xfc',
      b'\xf1\x00T02601BL  T02800A1  VTMPT25XXX800NS4\xed\xaf\xed\xf5',
    ],
  },
  CAR.SANTA_FE_HEV_2022: {
    (Ecu.fwdRadar, 0x7d0, None): [
      b'\xf1\x00TMhe SCC FHCUP      1.00 1.00 99110-CL500         ',
    ],
    (Ecu.eps, 0x7d4, None): [
      b'\xf1\x00TM  MDPS C 1.00 1.02 56310-CLAC0 4TSHC102',
      b'\xf1\x00TM  MDPS R 1.00 1.05 57700-CL000 4TSHP105',
    ],
    (Ecu.fwdCamera, 0x7c4, None): [
      b'\xf1\x00TMH MFC  AT EUR LHD 1.00 1.06 99211-S1500 220727',
      b'\xf1\x00TMH MFC  AT USA LHD 1.00 1.03 99211-S1500 210224',
    ],
    (Ecu.transmission, 0x7e1, None): [
      b'\xf1\x00PSBG2333  E16\x00\x00\x00\x00\x00\x00\x00TTM2H16UA3I\x94\xac\x8f',
      b'\xf1\x87959102T250\x00\x00\x00\x00\x00\xf1\x81E14\x00\x00\x00\x00\x00\x00\x00\xf1\x00PSBG2333  E14\x00\x00\x00\x00\x00\x00\x00TTM2H16SA2\x80\xd7l\xb2',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'\xf1\x87391312MTC1',
      b'\xf1\x87391312MTE0',
    ],
  },
  CAR.SANTA_FE_PHEV_2022: {
    (Ecu.fwdRadar, 0x7d0, None): [
      b'\xf1\x8799110CL500\xf1\x00TMhe SCC FHCUP      1.00 1.00 99110-CL500         ',
    ],
    (Ecu.eps, 0x7d4, None): [
      b'\xf1\x00TM  MDPS C 1.00 1.02 56310-CLAC0 4TSHC102',
      b'\xf1\x00TM  MDPS C 1.00 1.02 56310-CLEC0 4TSHC102',
    ],
    (Ecu.fwdCamera, 0x7c4, None): [
      b'\xf1\x00TMP MFC  AT USA LHD 1.00 1.03 99211-S1500 210224',
    ],
    (Ecu.transmission, 0x7e1, None): [
      b'\xf1\x8795441-3D121\x00\xf1\x81E16\x00\x00\x00\x00\x00\x00\x00\xf1\x00PSBG2333  E16\x00\x00\x00\x00\x00\x00\x00TTM2P16SA0o\x88^\xbe',
      b'\xf1\x8795441-3D121\x00\xf1\x81E16\x00\x00\x00\x00\x00\x00\x00\xf1\x00PSBG2333  E16\x00\x00\x00\x00\x00\x00\x00TTM2P16SA1\x0b\xc5\x0f\xea',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'\xf1\x87391312MTF0',
    ],
  },
  CAR.KIA_STINGER: {
    (Ecu.fwdRadar, 0x7d0, None): [
      b'\xf1\x00CK__ SCC F_CUP      1.00 1.01 96400-J5100         ',
      b'\xf1\x00CK__ SCC F_CUP      1.00 1.03 96400-J5100         ',
      b'\xf1\x00CK__ SCC F_CUP      1.00 1.01 96400-J5000         ',
      b'\xf1\x00CK__ SCC F_CUP      1.00 1.02 96400-J5100         ',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'\xf1\x81606DE051\x00\x00\x00\x00\x00\x00\x00\x00',
      b'\xf1\x81640E0051\x00\x00\x00\x00\x00\x00\x00\x00',
      b'\xf1\x82CKJN3TMSDE0B\x00\x00\x00\x00',
      b'\xf1\x82CKKN3TMD_H0A\x00\x00\x00\x00',
      b'\xe0\x19\xff\xe7\xe7g\x01\xa2\x00\x0f\x00\x9e\x00\x06\x00\xff\xff\xff\xff\xff\xff\x00\x00\xff\xff\xff\xff\xff\xff\x00\x00\x0f\x0e\x0f\x0f\x0e\r\x00\x00\x7f\x02.\xff\x00\x00~p\x00\x00\x00\x00u\xff\xf9\xff\x00\x00\x00\x00V\t\xd5\x01\xc0\x00\x00\x00\x007\xfb\xfc\x0b\x8d\x00',
      b'\xf1\x81640H0051\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.eps, 0x7d4, None): [
      b'\xf1\x00CK  MDPS R 1.00 1.04 57700-J5200 4C2CL104',
      b'\xf1\x00CK  MDPS R 1.00 1.04 57700-J5220 4C2VL104',
      b'\xf1\x00CK  MDPS R 1.00 1.04 57700-J5420 4C4VL104',
      b'\xf1\x00CK  MDPS R 1.00 1.06 57700-J5420 4C4VL106',
      b'\xf1\x00CK  MDPS R 1.00 1.07 57700-J5220 4C2VL107',
      b'\xf1\x00CK  MDPS R 1.00 1.06 57700-J5220 4C2VL106',
    ],
    (Ecu.fwdCamera, 0x7c4, None): [
      b'\xf1\x00CK  MFC  AT USA LHD 1.00 1.03 95740-J5000 170822',
      b'\xf1\x00CK  MFC  AT USA LHD 1.00 1.04 95740-J5000 180504',
      b'\xf1\x00CK  MFC  AT EUR LHD 1.00 1.03 95740-J5000 170822',
    ],
    (Ecu.transmission, 0x7e1, None): [
      b'\xf1\x00bcsh8p54  E25\x00\x00\x00\x00\x00\x00\x00SCK0T33NB2\xb3\xee\xba\xdc',
      b'\xf1\x87VCJLE17622572DK0vd6D\x99\x98y\x97vwVffUfvfC%CuT&Dx\x87o\xff{\x1c\xf1\x81E21\x00\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  E21\x00\x00\x00\x00\x00\x00\x00SCK0T33NB0\x88\xa2\xe6\xf0',
      b'\xf1\x87VDHLG17000192DK2xdFffT\xa5VUD$DwT\x86wveVeeD&T\x99\xba\x8f\xff\xcc\x99\xf1\x81E21\x00\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  E21\x00\x00\x00\x00\x00\x00\x00SCK0T33NB0\x88\xa2\xe6\xf0',
      b'\xf1\x87VDHLG17000192DK2xdFffT\xa5VUD$DwT\x86wveVeeD&T\x99\xba\x8f\xff\xcc\x99\xf1\x89E21\x00\x00\x00\x00\x00\x00\x00\xf1\x82SCK0T33NB0',
      b'\xf1\x87VDHLG17034412DK2vD6DfVvVTD$D\x99w\x88\x98EDEDeT6DgfO\xff\xc3=\xf1\x81E21\x00\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  E21\x00\x00\x00\x00\x00\x00\x00SCK0T33NB0\x88\xa2\xe6\xf0',
      b'\xf1\x87VDHLG17118862DK2\x8awWwgu\x96wVfUVwv\x97xWvfvUTGTx\x87o\xff\xc9\xed\xf1\x81E21\x00\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  E21\x00\x00\x00\x00\x00\x00\x00SCK0T33NB0\x88\xa2\xe6\xf0',
      b'\xf1\x87VDKLJ18675252DK6\x89vhgwwwwveVU\x88w\x87w\x99vgf\x97vXfgw_\xff\xc2\xfb\xf1\x89E25\x00\x00\x00\x00\x00\x00\x00\xf1\x82TCK0T33NB2',
      b'\xf1\x87WAJTE17552812CH4vfFffvfVeT5DwvvVVdFeegeg\x88\x88o\xff\x1a]\xf1\x81E21\x00\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  E21\x00\x00\x00\x00\x00\x00\x00TCK2T20NB1\x19\xd2\x00\x94',
      b'\xf1\x87VDHLG17274082DK2wfFf\x89x\x98wUT5T\x88v\x97xgeGefTGTVvO\xff\x1c\x14\xf1\x81E19\x00\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  E19\x00\x00\x00\x00\x00\x00\x00SCK0T33UB2\xee[\x97S',
      b'\xf1\x87VDHLG17000192DK2xdFffT\xa5VUD$DwT\x86wveVeeD&T\x99\xba\x8f\xff\xcc\x99\xf1\x81E21\x00\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  E21\x00\x00\x00\x00\x00\x00\x00SCK0T33NB0\t\xb7\x17\xf5',
      b'\xf1\x00bcsh8p54  E21\x00\x00\x00\x00\x00\x00\x00SCK0T33NB0\t\xb7\x17\xf5',
      b'\xf1\x00bcsh8p54  E21\x00\x00\x00\x00\x00\x00\x00SCK0T33NB0\x88\xa2\xe6\xf0',
    ],
  },
  CAR.KIA_STINGER_2022: {
    (Ecu.fwdRadar, 0x7d0, None): [
      b'\xf1\x00CK__ SCC F-CUP      1.00 1.00 99110-J5500         ',
      b'\xf1\x00CK__ SCC FHCUP      1.00 1.00 99110-J5500         ',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'\xf1\x81640R0051\x00\x00\x00\x00\x00\x00\x00\x00',
      b'\xf1\x81HM6M1_0a0_H00',
    ],
    (Ecu.eps, 0x7d4, None): [
      b'\xf1\x00CK  MDPS R 1.00 5.03 57700-J5380 4C2VR503',
      b'\xf1\x00CK  MDPS R 1.00 5.03 57700-J5300 4C2CL503',
    ],
    (Ecu.fwdCamera, 0x7c4, None): [
      b'\xf1\x00CK  MFC  AT AUS RHD 1.00 1.00 99211-J5500 210622',
      b'\xf1\x00CK  MFC  AT KOR LHD 1.00 1.00 99211-J5500 210622',
    ],
    (Ecu.transmission, 0x7e1, None): [
      b'\xf1\x87VCNLF11383972DK1vffV\x99\x99\x89\x98\x86eUU\x88wg\x89vfff\x97fff\x99\x87o\xff"\xc1\xf1\x81E30\x00\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  E30\x00\x00\x00\x00\x00\x00\x00SCK0T33GH0\xbe`\xfb\xc6',
      b'\xf1\x00bcsh8p54  E31\x00\x00\x00\x00\x00\x00\x00SCK0T25KH2B\xfbI\xe2',
    ],
  },
  CAR.PALISADE: {
    (Ecu.fwdRadar, 0x7d0, None): [
      b'\xf1\x00LX2_ SCC F-CUP      1.00 1.04 99110-S8100         ',
      b'\xf1\x00LX2_ SCC F-CUP      1.00 1.05 99110-S8100         ',
      b'\xf1\x00LX2 SCC FHCUP      1.00 1.04 99110-S8100         ',
      b'\xf1\x00LX2_ SCC FHCU-      1.00 1.05 99110-S8100         ',
      b'\xf1\x00LX2_ SCC FHCUP      1.00 1.00 99110-S8110         ',
      b'\xf1\x00LX2_ SCC FHCUP      1.00 1.04 99110-S8100         ',
      b'\xf1\x00LX2_ SCC FHCUP      1.00 1.05 99110-S8100         ',
      b'\xf1\x00ON__ FCA FHCUP      1.00 1.02 99110-S9100         ',
      b'\xf1\x00ON__ FCA FHCUP      1.00 1.01 99110-S9110         ',
    ],
    (Ecu.abs, 0x7d1, None): [
      b'\xf1\x00LX ESC \x01 103\x19\t\x10 58910-S8360',
      b'\xf1\x00LX ESC \x01 1031\t\x10 58910-S8360',
      b'\xf1\x00LX ESC \x0b 101\x19\x03\x17 58910-S8330',
      b'\xf1\x00LX ESC \x0b 102\x19\x05\x07 58910-S8330',
      b'\xf1\x00LX ESC \x0b 103\x19\t\t 58910-S8350',
      b'\xf1\x00LX ESC \x0b 103\x19\t\x07 58910-S8330',
      b'\xf1\x00LX ESC \x0b 103\x19\t\x10 58910-S8360',
      b'\xf1\x00LX ESC \x0b 104 \x10\x16 58910-S8360',
      b'\xf1\x00ON ESC \x0b 100\x18\x12\x18 58910-S9360',
      b'\xf1\x00ON ESC \x0b 101\x19\t\x08 58910-S9360',
      b'\xf1\x00ON ESC \x0b 101\x19\t\x05 58910-S9320',
      b'\xf1\x00ON ESC \x01 101\x19\t\x08 58910-S9360',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'\xf1\x81640J0051\x00\x00\x00\x00\x00\x00\x00\x00',
      b'\xf1\x81640K0051\x00\x00\x00\x00\x00\x00\x00\x00',
      b'\xf1\x81640S1051\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.eps, 0x7d4, None): [
      b'\xf1\x00LX2 MDPS C 1,00 1,03 56310-S8020 4LXDC103',
      b'\xf1\x00LX2 MDPS C 1.00 1.03 56310-S8000 4LXDC103',
      b'\xf1\x00LX2 MDPS C 1.00 1.03 56310-S8020 4LXDC103',
      b'\xf1\x00LX2 MDPS C 1.00 1.04 56310-S8020 4LXDC104',
      b'\xf1\x00ON  MDPS C 1.00 1.00 56340-S9000 8B13',
      b'\xf1\x00ON  MDPS C 1.00 1.01 56340-S9000 9201',
    ],
    (Ecu.fwdCamera, 0x7c4, None): [
      b'\xf1\x00LX2 MFC  AT USA LHD 1.00 1.03 99211-S8100 190125',
      b'\xf1\x00LX2 MFC  AT USA LHD 1.00 1.05 99211-S8100 190909',
      b'\xf1\x00LX2 MFC  AT USA LHD 1.00 1.07 99211-S8100 200422',
      b'\xf1\x00LX2 MFC  AT USA LHD 1.00 1.08 99211-S8100 200903',
      b'\xf1\x00ON  MFC  AT USA LHD 1.00 1.01 99211-S9100 181105',
      b'\xf1\x00ON  MFC  AT USA LHD 1.00 1.03 99211-S9100 200720',
      b'\xf1\x00LX2 MFC  AT USA LHD 1.00 1.00 99211-S8110 210226',
      b'\xf1\x00ON  MFC  AT USA LHD 1.00 1.04 99211-S9100 211227',
    ],
    (Ecu.transmission, 0x7e1, None): [
      b'\xf1\x00bcsh8p54  U872\x00\x00\x00\x00\x00\x00TON4G38NB1\x96z28',
      b'\xf1\x00bcsh8p54  U891\x00\x00\x00\x00\x00\x00SLX4G38NB3X\xa8\xc08',
      b'\xf1\x00bcsh8p54  U903\x00\x00\x00\x00\x00\x00TON4G38NB2[v\\\xb6',
      b'\xf1\x00bcsh8p54  U922\x00\x00\x00\x00\x00\x00TON2G38NB5j\x94.\xde',
      b'\xf1\x87LBLUFN591307KF25vgvw\x97wwwy\x99\xa7\x99\x99\xaa\xa9\x9af\x88\x96h\x95o\xf7\xff\x99f/\xff\xe4c\xf1\x81U891\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U891\x00\x00\x00\x00\x00\x00SLX2G38NB2\xd7\xc1/\xd1',
      b'\xf1\x87LBLUFN650868KF36\xa9\x98\x89\x88\xa8\x88\x88\x88h\x99\xa6\x89fw\x86gw\x88\x97x\xaa\x7f\xf6\xff\xbb\xbb\x8f\xff+\x82\xf1\x81U891\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U891\x00\x00\x00\x00\x00\x00SLX2G38NB3\xd1\xc3\xf8\xa8',
      b'\xf1\x87LBLUFN655162KF36\x98\x88\x88\x88\x98\x88\x88\x88x\x99\xa7\x89x\x99\xa7\x89x\x99\x97\x89g\x7f\xf7\xffwU_\xff\xe9!\xf1\x81U891\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U891\x00\x00\x00\x00\x00\x00SLX2G38NB3\xd1\xc3\xf8\xa8',
      b'\xf1\x87LBLUFN731381KF36\xb9\x99\x89\x98\x98\x88\x88\x88\x89\x99\xa8\x99\x88\x99\xa8\x89\x88\x88\x98\x88V\x7f\xf6\xff\x99w\x8f\xff\xad\xd8\xf1\x81U891\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U891\x00\x00\x00\x00\x00\x00SLX2G38NB3\xd1\xc3\xf8\xa8',
      b'\xf1\x87LDKVAA0028604HH1\xa8\x88x\x87vgvw\x88\x99\xa8\x89gw\x86ww\x88\x97x\x97o\xf9\xff\x97w\x7f\xffo\x02\xf1\x81U872\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U872\x00\x00\x00\x00\x00\x00TON4G38NB1\x96z28',
      b'\xf1\x87LDKVAA3068374HH1wwww\x87xw\x87y\x99\xa7\x99w\x88\x87xw\x88\x97x\x85\xaf\xfa\xffvU/\xffU\xdc\xf1\x81U872\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U872\x00\x00\x00\x00\x00\x00TON4G38NB1\x96z28',
      b'\xf1\x87LDKVBN382172KF26\x98\x88\x88\x88\xa8\x88\x88\x88x\x99\xa7\x89\x87\x88\x98x\x98\x99\xa9\x89\xa5_\xf6\xffDDO\xff\xcd\x16\xf1\x81U891\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U891\x00\x00\x00\x00\x00\x00SLX4G38NB2\xafL]\xe7',
      b'\xf1\x87LDKVBN424201KF26\xba\xaa\x9a\xa9\x99\x99\x89\x98\x89\x99\xa8\x99\x88\x99\x98\x89\x88\x99\xa8\x89v\x7f\xf7\xffwf_\xffq\xa6\xf1\x81U891\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U891\x00\x00\x00\x00\x00\x00SLX4G38NB2\xafL]\xe7',
      b'\xf1\x87LDKVBN540766KF37\x87wgv\x87w\x87xx\x99\x97\x89v\x88\x97h\x88\x88\x88\x88x\x7f\xf6\xffvUo\xff\xd3\x01\xf1\x81U891\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U891\x00\x00\x00\x00\x00\x00SLX4G38NB2\xafL]\xe7',
      b'\xf1\x87LDLVAA4225634HH1\x98\x88\x88\x88eUeVx\x88\x87\x88g\x88\x86xx\x88\x87\x88\x86o\xf9\xff\x87w\x7f\xff\xf2\xf7\xf1\x81U903\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U903\x00\x00\x00\x00\x00\x00TON4G38NB2[v\\\xb6',
      b'\xf1\x87LDLVAA4777834HH1\x98\x88x\x87\x87wwwx\x88\x87\x88x\x99\x97\x89x\x88\x97\x88\x86o\xfa\xff\x86fO\xff\x1d9\xf1\x81U903\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U903\x00\x00\x00\x00\x00\x00TON4G38NB2[v\\\xb6',
      b'\xf1\x87LDLVAA5194534HH1ffvguUUUx\x88\xa7\x88h\x99\x96\x89x\x88\x97\x88ro\xf9\xff\x98wo\xff\xaaM\xf1\x81U903\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U903\x00\x00\x00\x00\x00\x00TON4G38NB2[v\\\xb6',
      b'\xf1\x87LDLVAA5949924HH1\xa9\x99y\x97\x87wwwx\x99\x97\x89x\x99\xa7\x89x\x99\xa7\x89\x87_\xfa\xffeD?\xff\xf1\xfd\xf1\x81U903\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U903\x00\x00\x00\x00\x00\x00TON4G38NB2[v\\\xb6',
      b'\xf1\x87LDLVBN560098KF26\x86fff\x87vgfg\x88\x96xfw\x86gfw\x86g\x95\xf6\xffeU_\xff\x92c\xf1\x81U891\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U891\x00\x00\x00\x00\x00\x00SLX4G38NB2\xafL]\xe7',
      b'\xf1\x87LDLVBN602045KF26\xb9\x99\x89\x98\x97vwgy\xaa\xb7\x9af\x88\x96hw\x99\xa7y\xa9\x7f\xf5\xff\x99w\x7f\xff,\xd3\xf1\x81U891\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U891\x00\x00\x00\x00\x00\x00SLX4G38NB3X\xa8\xc08',
      b'\xf1\x87LDLVBN628911KF26\xa9\x99\x89\x98\x98\x88\x88\x88y\x99\xa7\x99fw\x86gw\x88\x87x\x83\x7f\xf6\xff\x98wo\xff2\xda\xf1\x81U891\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U891\x00\x00\x00\x00\x00\x00SLX4G38NB3X\xa8\xc08',
      b'\xf1\x87LDLVBN645817KF37\x87www\x98\x87xwx\x99\x97\x89\x99\x99\x99\x99g\x88\x96x\xb6_\xf7\xff\x98fo\xff\xe2\x86\xf1\x81U891\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U891\x00\x00\x00\x00\x00\x00SLX4G38NB3X\xa8\xc08',
      b'\xf1\x87LDLVBN662115KF37\x98\x88\x88\x88\xa8\x88\x88\x88x\x99\x97\x89x\x99\xa7\x89\x88\x99\xa8\x89\x88\x7f\xf7\xfffD_\xff\xdc\x84\xf1\x81U891\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U891\x00\x00\x00\x00\x00\x00SLX4G38NB3X\xa8\xc08',
      b'\xf1\x87LDLVBN667933KF37\xb9\x99\x89\x98\xb9\x99\x99\x99x\x88\x87\x88w\x88\x87x\x88\x88\x98\x88\xcbo\xf7\xffe3/\xffQ!\xf1\x81U891\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U891\x00\x00\x00\x00\x00\x00SLX4G38NB3X\xa8\xc08',
      b'\xf1\x87LDLVBN673087KF37\x97www\x86fvgx\x99\x97\x89\x99\xaa\xa9\x9ag\x88\x86x\xe9_\xf8\xff\x98w\x7f\xff"\xad\xf1\x81U891\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U891\x00\x00\x00\x00\x00\x00SLX4G38NB3X\xa8\xc08',
      b'\xf1\x87LDLVBN673841KF37\x98\x88x\x87\x86g\x86xy\x99\xa7\x99\x88\x99\xa8\x89w\x88\x97xdo\xf5\xff\x98\x88\x8f\xffT\xec\xf1\x81U891\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U891\x00\x00\x00\x00\x00\x00SLX4G38NB3X\xa8\xc08',
      b'\xf1\x87LDLVBN681363KF37\x98\x88\x88\x88\x97x\x87\x88y\xaa\xa7\x9a\x88\x88\x98\x88\x88\x88\x88\x88vo\xf6\xffvD\x7f\xff%v\xf1\x81U891\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U891\x00\x00\x00\x00\x00\x00SLX4G38NB3X\xa8\xc08',
      b'\xf1\x87LDLVBN713782KF37\x99\x99y\x97\x98\x88\x88\x88x\x88\x97\x88\x88\x99\x98\x89\x88\x99\xa8\x89\x87o\xf7\xffeU?\xff7,\xf1\x81U891\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U891\x00\x00\x00\x00\x00\x00SLX4G38NB3X\xa8\xc08',
      b'\xf1\x87LDLVBN713890KF26\xb9\x99\x89\x98\xa9\x99\x99\x99x\x99\x97\x89\x88\x99\xa8\x89\x88\x99\xb8\x89Do\xf7\xff\xa9\x88o\xffs\r\xf1\x81U891\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U891\x00\x00\x00\x00\x00\x00SLX4G38NB3X\xa8\xc08',
      b'\xf1\x87LDLVBN733215KF37\x99\x98y\x87\x97wwwi\x99\xa6\x99x\x99\xa7\x89V\x88\x95h\x86o\xf7\xffeDO\xff\x12\xe7\xf1\x81U891\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U891\x00\x00\x00\x00\x00\x00SLX4G38NB3X\xa8\xc08',
      b'\xf1\x87LDLVBN750044KF37\xca\xa9\x8a\x98\xa7wwwy\xaa\xb7\x9ag\x88\x96x\x88\x99\xa8\x89\xb9\x7f\xf6\xff\xa8w\x7f\xff\xbe\xde\xf1\x81U891\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U891\x00\x00\x00\x00\x00\x00SLX4G38NB3X\xa8\xc08',
      b'\xf1\x87LDLVBN752612KF37\xba\xaa\x8a\xa8\x87w\x87xy\xaa\xa7\x9a\x88\x99\x98\x89x\x88\x97\x88\x96o\xf6\xffvU_\xffh\x1b\xf1\x81U891\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U891\x00\x00\x00\x00\x00\x00SLX4G38NB3X\xa8\xc08',
      b'\xf1\x87LDLVBN755553KF37\x87xw\x87\x97w\x87xy\x99\xa7\x99\x99\x99\xa9\x99Vw\x95gwo\xf6\xffwUO\xff\xb5T\xf1\x81U891\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U891\x00\x00\x00\x00\x00\x00SLX4G38NB3X\xa8\xc08',
      b'\xf1\x87LDLVBN757883KF37\x98\x87xw\x98\x87\x88xy\xaa\xb7\x9ag\x88\x96x\x89\x99\xa8\x99e\x7f\xf6\xff\xa9\x88o\xff5\x15\xf1\x81U922\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U922\x00\x00\x00\x00\x00\x00SLX4G38NB4\xd6\xe8\xd7\xa6',
      b'\xf1\x87LDMVBN778156KF37\x87vWe\xa9\x99\x99\x99y\x99\xb7\x99\x99\x99\x99\x99x\x99\x97\x89\xa8\x7f\xf8\xffwf\x7f\xff\x82_\xf1\x81U922\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U922\x00\x00\x00\x00\x00\x00SLX4G38NB4\xd6\xe8\xd7\xa6',
      b'\xf1\x87LDMVBN780576KF37\x98\x87hv\x97x\x97\x89x\x99\xa7\x89\x88\x99\x98\x89w\x88\x97x\x98\x7f\xf7\xff\xba\x88\x8f\xff\x1e0\xf1\x81U922\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U922\x00\x00\x00\x00\x00\x00SLX4G38NB4\xd6\xe8\xd7\xa6',
      b'\xf1\x87LDMVBN783485KF37\x87www\x87vwgy\x99\xa7\x99\x99\x99\xa9\x99Vw\x95g\x89_\xf6\xff\xa9w_\xff\xc5\xd6\xf1\x81U922\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U922\x00\x00\x00\x00\x00\x00SLX4G38NB4\xd6\xe8\xd7\xa6',
      b'\xf1\x87LDMVBN811844KF37\x87vwgvfffx\x99\xa7\x89Vw\x95gg\x88\xa6xe\x8f\xf6\xff\x97wO\xff\t\x80\xf1\x81U922\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U922\x00\x00\x00\x00\x00\x00SLX4G38NB4\xd6\xe8\xd7\xa6',
      b'\xf1\x87LDMVBN830601KF37\xa7www\xa8\x87xwx\x99\xa7\x89Uw\x85Ww\x88\x97x\x88o\xf6\xff\x8a\xaa\x7f\xff\xe2:\xf1\x81U922\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U922\x00\x00\x00\x00\x00\x00SLX4G38NB4\xd6\xe8\xd7\xa6',
      b'\xf1\x87LDMVBN848789KF37\x87w\x87x\x87w\x87xy\x99\xb7\x99\x87\x88\x98x\x88\x99\xa8\x89\x87\x7f\xf6\xfffUo\xff\xe3!\xf1\x81U922\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U922\x00\x00\x00\x00\x00\x00SLX4G38NB5\xb9\x94\xe8\x89',
      b'\xf1\x87LDMVBN851595KF37\x97wgvvfffx\x99\xb7\x89\x88\x99\x98\x89\x87\x88\x98x\x99\x7f\xf7\xff\x97w\x7f\xff@\xf3\xf1\x81U922\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U922\x00\x00\x00\x00\x00\x00SLX4G38NB5\xb9\x94\xe8\x89',
      b'\xf1\x87LDMVBN873175KF26\xa8\x88\x88\x88vfVex\x99\xb7\x89\x88\x99\x98\x89x\x88\x97\x88f\x7f\xf7\xff\xbb\xaa\x8f\xff,\x04\xf1\x81U922\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U922\x00\x00\x00\x00\x00\x00SLX4G38NB5\xb9\x94\xe8\x89',
      b'\xf1\x87LDMVBN879401KF26veVU\xa8\x88\x88\x88g\x88\xa6xVw\x95gx\x88\xa7\x88v\x8f\xf9\xff\xdd\xbb\xbf\xff\xb3\x99\xf1\x81U922\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U922\x00\x00\x00\x00\x00\x00SLX4G38NB5\xb9\x94\xe8\x89',
      b'\xf1\x87LDMVBN881314KF37\xa8\x88h\x86\x97www\x89\x99\xa8\x99w\x88\x97xx\x99\xa7\x89\xca\x7f\xf8\xff\xba\x99\x8f\xff\xd8v\xf1\x81U922\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U922\x00\x00\x00\x00\x00\x00SLX4G38NB5\xb9\x94\xe8\x89',
      b'\xf1\x87LDMVBN888651KF37\xa9\x99\x89\x98vfff\x88\x99\x98\x89w\x99\xa7y\x88\x88\x98\x88D\x8f\xf9\xff\xcb\x99\x8f\xff\xa5\x1e\xf1\x81U922\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U922\x00\x00\x00\x00\x00\x00SLX4G38NB5\xb9\x94\xe8\x89',
      b'\xf1\x87LDMVBN889419KF37\xa9\x99y\x97\x87w\x87xx\x88\x97\x88w\x88\x97x\x88\x99\x98\x89e\x9f\xf9\xffeUo\xff\x901\xf1\x81U922\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U922\x00\x00\x00\x00\x00\x00SLX4G38NB5\xb9\x94\xe8\x89',
      b'\xf1\x87LDMVBN895969KF37vefV\x87vgfx\x99\xa7\x89\x99\x99\xb9\x99f\x88\x96he_\xf7\xffxwo\xff\x14\xf9\xf1\x81U922\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U922\x00\x00\x00\x00\x00\x00SLX4G38NB5\xb9\x94\xe8\x89',
      b'\xf1\x87LDMVBN899222KF37\xa8\x88x\x87\x97www\x98\x99\x99\x89\x88\x99\x98\x89f\x88\x96hdo\xf7\xff\xbb\xaa\x9f\xff\xe2U\xf1\x81U922\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U922\x00\x00\x00\x00\x00\x00SLX4G38NB5\xb9\x94\xe8\x89',
      b"\xf1\x87LBLUFN622950KF36\xa8\x88\x88\x88\x87w\x87xh\x99\x96\x89\x88\x99\x98\x89\x88\x99\x98\x89\x87o\xf6\xff\x98\x88o\xffx'\xf1\x81U891\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U891\x00\x00\x00\x00\x00\x00SLX2G38NB3\xd1\xc3\xf8\xa8",
      b'\xf1\x87LDMVBN950669KF37\x97www\x96fffy\x99\xa7\x99\xa9\x99\xaa\x99g\x88\x96x\xb8\x8f\xf9\xffTD/\xff\xa7\xcb\xf1\x81U922\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U922\x00\x00\x00\x00\x00\x00SLX4G38NB5\xb9\x94\xe8\x89',
      b'\xf1\x87LDLVAA4478824HH1\x87wwwvfvg\x89\x99\xa8\x99w\x88\x87x\x89\x99\xa8\x99\xa6o\xfa\xfffU/\xffu\x92\xf1\x81U903\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U903\x00\x00\x00\x00\x00\x00TON4G38NB2[v\\\xb6',
      b'\xf1\x87LDMVBN871852KF37\xb9\x99\x99\x99\xa8\x88\x88\x88y\x99\xa7\x99x\x99\xa7\x89\x88\x88\x98\x88\x89o\xf7\xff\xaa\x88o\xff\x0e\xed\xf1\x81U922\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U922\x00\x00\x00\x00\x00\x00SLX4G38NB5\xb9\x94\xe8\x89',
    ],
  },
  CAR.VELOSTER: {
    (Ecu.fwdRadar, 0x7d0, None): [
      b'\xf1\x00JS__ SCC H-CUP      1.00 1.02 95650-J3200         ',
      b'\xf1\x00JS__ SCC HNCUP      1.00 1.02 95650-J3100         ',
    ],
    (Ecu.abs, 0x7d1, None): [
      b'\xf1\x00\x00\x00\x00\x00\x00\x00',
      b'\xf1\x816V8RAC00121.ELF\xf1\x00\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'\x01TJS-JNU06F200H0A',
      b'\x01TJS-JDK06F200H0A',
      b'391282BJF5 ',
    ],
    (Ecu.eps, 0x7d4, None): [b'\xf1\x00JSL MDPS C 1.00 1.03 56340-J3000 8308', ],
    (Ecu.fwdCamera, 0x7c4, None): [
      b'\xf1\x00JS  LKAS AT USA LHD 1.00 1.02 95740-J3000 K32',
      b'\xf1\x00JS  LKAS AT KOR LHD 1.00 1.03 95740-J3000 K33',
    ],
    (Ecu.transmission, 0x7e1, None): [
      b'\xf1\x816U2V8051\x00\x00\xf1\x006U2V0_C2\x00\x006U2V8051\x00\x00DJS0T16NS1\xba\x02\xb8\x80',
      b'\xf1\x816U2V8051\x00\x00\xf1\x006U2V0_C2\x00\x006U2V8051\x00\x00DJS0T16NS1\x00\x00\x00\x00',
      b'\xf1\x816U2V8051\x00\x00\xf1\x006U2V0_C2\x00\x006U2V8051\x00\x00DJS0T16KS2\016\xba\036\xa2',
    ],
  },
  CAR.GENESIS_G70: {
    (Ecu.fwdRadar, 0x7d0, None): [
      b'\xf1\x00IK__ SCC F-CUP      1.00 1.02 96400-G9100         ',
      b'\xf1\x00IK__ SCC F-CUP      1.00 1.01 96400-G9100         ',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'\xf1\x81640F0051\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.eps, 0x7d4, None): [
      b'\xf1\x00IK  MDPS R 1.00 1.06 57700-G9420 4I4VL106',
    ],
    (Ecu.fwdCamera, 0x7c4, None): [
      b'\xf1\x00IK  MFC  AT USA LHD 1.00 1.01 95740-G9000 170920',
    ],
    (Ecu.transmission, 0x7e1, None): [
      b'\xf1\x00bcsh8p54  E25\x00\x00\x00\x00\x00\x00\x00SIK0T33NB2\x11\x1am\xda',
      b'\xf1\x87VDJLT17895112DN4\x88fVf\x99\x88\x88\x88\x87fVe\x88vhwwUFU\x97eFex\x99\xff\xb7\x82\xf1\x81E25\x00\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  E25\x00\x00\x00\x00\x00\x00\x00SIK0T33NB2\x11\x1am\xda',
    ],
  },
  CAR.GENESIS_G70_2020: {
    (Ecu.eps, 0x7d4, None): [
      b'\xf1\x00IK  MDPS R 1.00 1.07 57700-G9220 4I2VL107',
      b'\xf1\x00IK  MDPS R 1.00 1.07 57700-G9420 4I4VL107',
      b'\xf1\x00IK  MDPS R 1.00 1.08 57700-G9420 4I4VL108',
    ],
    (Ecu.transmission, 0x7e1, None): [
      b'\xf1\x87VCJLP18407832DN3\x88vXfvUVT\x97eFU\x87d7v\x88eVeveFU\x89\x98\x7f\xff\xb2\xb0\xf1\x81E25\x00\x00\x00',
      b'\x00\x00\x00\x00\xf1\x00bcsh8p54  E25\x00\x00\x00\x00\x00\x00\x00SIK0T33NB4\xecE\xefL',
      b'\xf1\x87VDKLT18912362DN4wfVfwefeveVUwfvw\x88vWfvUFU\x89\xa9\x8f\xff\x87w\xf1\x81E25\x00\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  E25\x00\x00\x00\x00\x00\x00\x00SIK0T33NB4\xecE\xefL',
      b'\xf1\x87VDJLC18480772DK9\x88eHfwfff\x87eFUeDEU\x98eFe\x86T5DVyo\xff\x87s\xf1\x81E25\x00\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  E25\x00\x00\x00\x00\x00\x00\x00SIK0T33KB5\x9f\xa5&\x81',
    ],
    (Ecu.fwdRadar, 0x7d0, None): [
      b'\xf1\x00IK__ SCC F-CUP      1.00 1.02 96400-G9100         ',
      b'\xf1\x00IK__ SCC F-CUP      1.00 1.02 96400-G9100         \xf1\xa01.02',
      b'\xf1\x00IK__ SCC FHCUP      1.00 1.02 96400-G9000         ',
    ],
    (Ecu.fwdCamera, 0x7c4, None): [
      b'\xf1\x00IK  MFC  AT USA LHD 1.00 1.01 95740-G9000 170920',
      b'\xf1\x00IK  MFC  AT KOR LHD 1.00 1.01 95740-G9000 170920',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'\xf1\x81640J0051\x00\x00\x00\x00\x00\x00\x00\x00',
      b'\xf1\x81640H0051\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
  },
  CAR.GENESIS_G80: {
    (Ecu.fwdRadar, 0x7d0, None): [
      b'\xf1\x00DH__ SCC F-CUP      1.00 1.01 96400-B1120         ',
    ],
    (Ecu.fwdCamera, 0x7c4, None): [
      b'\xf1\x00DH  LKAS AT USA LHD 1.01 1.03 95895-B1500 180713',
      b'\xf1\x00DH  LKAS AT USA LHD 1.01 1.02 95895-B1500 170810',
      b'\xf1\x00DH  LKAS AT USA LHD 1.01 1.01 95895-B1500 161014',
    ],
    (Ecu.transmission, 0x7e1, None): [
      b'\xf1\x00bcsh8p54  E21\x00\x00\x00\x00\x00\x00\x00SDH0T33NH4\xd7O\x9e\xc9',
      b'\xf1\x00bcsh8p54  E18\x00\x00\x00\x00\x00\x00\x00TDH0G38NH3:-\xa9n',
      b'\xf1\x00bcsh8p54  E18\x00\x00\x00\x00\x00\x00\x00SDH0G38NH2j\x9dA\x1c',
      b'\xf1\x00bcsh8p54  E18\x00\x00\x00\x00\x00\x00\x00SDH0T33NH3\x97\xe6\xbc\xb8',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'\xf1\x81640F0051\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
  },
  CAR.GENESIS_G90: {
    (Ecu.transmission, 0x7e1, None): [b'\xf1\x87VDGMD15866192DD3x\x88x\x89wuFvvfUf\x88vWwgwwwvfVgx\x87o\xff\xbc^\xf1\x81E14\x00\x00\x00\x00\x00\x00\x00\xf1\x00bcshcm49  E14\x00\x00\x00\x00\x00\x00\x00SHI0G50NB1tc5\xb7'],
    (Ecu.fwdRadar, 0x7d0, None): [b'\xf1\x00HI__ SCC F-CUP      1.00 1.01 96400-D2100         '],
    (Ecu.fwdCamera, 0x7c4, None): [b'\xf1\x00HI  LKAS AT USA LHD 1.00 1.00 95895-D2020 160302'],
    (Ecu.engine, 0x7e0, None): [b'\xf1\x810000000000\x00'],
  },
  CAR.KONA: {
    (Ecu.fwdRadar, 0x7d0, None): [b'\xf1\x00OS__ SCC F-CUP      1.00 1.00 95655-J9200         ', ],
    (Ecu.abs, 0x7d1, None): [b'\xf1\x816V5RAK00018.ELF\xf1\x00\x00\x00\x00\x00\x00\x00', ],
    (Ecu.engine, 0x7e0, None): [b'"\x01TOS-0NU06F301J02', ],
    (Ecu.eps, 0x7d4, None): [b'\xf1\x00OS  MDPS C 1.00 1.05 56310J9030\x00 4OSDC105', ],
    (Ecu.fwdCamera, 0x7c4, None): [b'\xf1\x00OS9 LKAS AT USA LHD 1.00 1.00 95740-J9300 g21', ],
    (Ecu.transmission, 0x7e1, None): [b'\xf1\x816U2VE051\x00\x00\xf1\x006U2V0_C2\x00\x006U2VE051\x00\x00DOS4T16NS3\x00\x00\x00\x00', ],
  },
  CAR.KIA_CEED:  {
    (Ecu.fwdRadar, 0x7D0, None): [b'\xf1\000CD__ SCC F-CUP      1.00 1.02 99110-J7000         ', ],
    (Ecu.eps, 0x7D4, None): [b'\xf1\000CD  MDPS C 1.00 1.06 56310-XX000 4CDEC106', ],
    (Ecu.fwdCamera, 0x7C4, None): [b'\xf1\000CD  LKAS AT EUR LHD 1.00 1.01 99211-J7000 B40', ],
    (Ecu.engine, 0x7E0, None): [b'\001TCD-JECU4F202H0K', ],
    (Ecu.transmission, 0x7E1, None): [
      b'\xf1\x816U2V7051\000\000\xf1\0006U2V0_C2\000\0006U2V7051\000\000DCD0T14US1\000\000\000\000',
      b'\xf1\x816U2V7051\x00\x00\xf1\x006U2V0_C2\x00\x006U2V7051\x00\x00DCD0T14US1U\x867Z',
    ],
    (Ecu.abs, 0x7D1, None): [b'\xf1\000CD ESC \003 102\030\b\005 58920-J7350', ],
  },
  CAR.KIA_FORTE: {
    (Ecu.eps, 0x7D4, None): [
      b'\xf1\x00BD  MDPS C 1.00 1.02 56310-XX000 4BD2C102',
      b'\xf1\x00BD  MDPS C 1.00 1.08 56310/M6300 4BDDC108',
      b'\xf1\x00BD  MDPS C 1.00 1.08 56310M6300\x00 4BDDC108',
      b'\xf1\x00BDm MDPS C A.01 1.03 56310M7800\x00 4BPMC103',
    ],
    (Ecu.fwdCamera, 0x7C4, None): [
      b'\xf1\x00BD  LKAS AT USA LHD 1.00 1.04 95740-M6000 J33',
      b'\xf1\x00BDP LKAS AT USA LHD 1.00 1.05 99211-M6500 744',
    ],
    (Ecu.fwdRadar, 0x7D0, None): [
      b'\xf1\x00BD__ SCC H-CUP      1.00 1.02 99110-M6000         ',
      b'\xf1\x00BDPE_SCC FHCUPC     1.00 1.04 99110-M6500\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'\x01TBDM1NU06F200H01',
      b'391182B945\x00',
      b'\xf1\x81616F2051\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.abs, 0x7d1, None): [
      b'\xf1\x816VGRAH00018.ELF\xf1\x00\x00\x00\x00\x00\x00\x00',
      b'\xf1\x8758900-M7AB0 \xf1\x816VQRAD00127.ELF\xf1\x00\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.transmission, 0x7e1, None): [
      b'\xf1\x006V2B0_C2\x00\x006V2C6051\x00\x00CBD0N20NL1\x00\x00\x00\x00',
      b'\xf1\x816U2VC051\x00\x00\xf1\x006U2V0_C2\x00\x006U2VC051\x00\x00DBD0T16SS0\x00\x00\x00\x00',
      b"\xf1\x816U2VC051\x00\x00\xf1\x006U2V0_C2\x00\x006U2VC051\x00\x00DBD0T16SS0\xcf\x1e'\xc3",
    ],
  },
  CAR.KIA_K5_2021: {
    (Ecu.fwdRadar, 0x7D0, None): [
      b'\xf1\000DL3_ SCC FHCUP      1.00 1.03 99110-L2000         ',
      b'\xf1\x8799110L2000\xf1\000DL3_ SCC FHCUP      1.00 1.03 99110-L2000         ',
      b'\xf1\x8799110L2100\xf1\x00DL3_ SCC F-CUP      1.00 1.03 99110-L2100         ',
      b'\xf1\x8799110L2100\xf1\x00DL3_ SCC FHCUP      1.00 1.03 99110-L2100         ',
      b'\xf1\x00DL3_ SCC F-CUP      1.00 1.03 99110-L2100         ',
    ],
    (Ecu.eps, 0x7D4, None): [
      b'\xf1\x8756310-L3110\xf1\000DL3 MDPS C 1.00 1.01 56310-L3110 4DLAC101',
      b'\xf1\x8756310-L3220\xf1\x00DL3 MDPS C 1.00 1.01 56310-L3220 4DLAC101',
      b'\xf1\x8757700-L3000\xf1\x00DL3 MDPS R 1.00 1.02 57700-L3000 4DLAP102',
      b'\xf1\x00DL3 MDPS C 1.00 1.01 56310-L3220 4DLAC101',
    ],
    (Ecu.fwdCamera, 0x7C4, None): [
      b'\xf1\x00DL3 MFC  AT USA LHD 1.00 1.03 99210-L3000 200915',
      b'\xf1\x00DL3 MFC  AT USA LHD 1.00 1.04 99210-L3000 210208',
    ],
    (Ecu.abs, 0x7D1, None): [
      b'\xf1\000DL ESC \006 101 \004\002 58910-L3200',
      b'\xf1\x8758910-L3200\xf1\000DL ESC \006 101 \004\002 58910-L3200',
      b'\xf1\x8758910-L3800\xf1\x00DL ESC \t 101 \x07\x02 58910-L3800',
      b'\xf1\x8758910-L3600\xf1\x00DL ESC \x03 100 \x08\x02 58910-L3600',
      b'\xf1\x00DL ESC \t 100 \x06\x02 58910-L3800',
    ],
    (Ecu.engine, 0x7E0, None): [
      b'\xf1\x87391212MKT0',
      b'\xf1\x87391212MKV0',
      b'\xf1\x870\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf1\x82DLDWN5TMDCXXXJ1B',
    ],
    (Ecu.transmission, 0x7E1, None): [
      b'\xf1\000bcsh8p54  U913\000\000\000\000\000\000TDL2T16NB1ia\v\xb8',
      b'\xf1\x87SALFEA5652514GK2UUeV\x88\x87\x88xxwg\x87ww\x87wwfwvd/\xfb\xffvU_\xff\x93\xd3\xf1\x81U913\000\000\000\000\000\000\xf1\000bcsh8p54  U913\000\000\000\000\000\000TDL2T16NB1ia\v\xb8',
      b'\xf1\x87SALFEA6046104GK2wvwgeTeFg\x88\x96xwwwwffvfe?\xfd\xff\x86fo\xff\x97A\xf1\x81U913\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U913\x00\x00\x00\x00\x00\x00TDL2T16NB1ia\x0b\xb8',
      b'\xf1\x87SCMSAA8572454GK1\x87x\x87\x88Vf\x86hgwvwvwwgvwwgT?\xfb\xff\x97fo\xffH\xb8\xf1\x81U913\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U913\x00\x00\x00\x00\x00\x00TDL4T16NB05\x94t\x18',
      b'\xf1\x87954A02N300\x00\x00\x00\x00\x00\xf1\x81T02730A1  \xf1\x00T02601BL  T02730A1  WDL3T25XXX730NS2b\x1f\xb8%',
      b'\xf1\x00bcsh8p54  U913\x00\x00\x00\x00\x00\x00TDL4T16NB05\x94t\x18',
    ],
  },
  CAR.KIA_K5_HEV_2020: {
    (Ecu.fwdRadar, 0x7D0, None): [
      b'\xf1\x00DLhe SCC FHCUP      1.00 1.02 99110-L7000         ',
    ],
    (Ecu.eps, 0x7D4, None): [
      b'\xf1\x00DL3 MDPS C 1.00 1.02 56310-L7000 4DLHC102',
    ],
    (Ecu.fwdCamera, 0x7C4, None): [
      b'\xf1\x00DL3HMFC  AT KOR LHD 1.00 1.02 99210-L2000 200309',
    ],
    (Ecu.engine, 0x7E0, None): [
      b'\xf1\x87391162JLA0',
    ],
    (Ecu.transmission, 0x7E1, None): [
      b'\xf1\x00PSBG2323  E08\x00\x00\x00\x00\x00\x00\x00TDL2H20KA2\xe3\xc6cz',
    ],
  },
  CAR.KONA_EV: {
    (Ecu.abs, 0x7D1, None): [
      b'\xf1\x00OS IEB \r 105\x18\t\x18 58520-K4000',
      b'\xf1\x00OS IEB \x01 212 \x11\x13 58520-K4000',
      b'\xf1\x00OS IEB \x02 212 \x11\x13 58520-K4000',
      b'\xf1\x00OS IEB \x03 210 \x02\x14 58520-K4000',
      b'\xf1\x00OS IEB \x03 212 \x11\x13 58520-K4000',
    ],
    (Ecu.fwdCamera, 0x7C4, None): [
      b'\xf1\x00OE2 LKAS AT EUR LHD 1.00 1.00 95740-K4200 200',
      b'\xf1\x00OSE LKAS AT EUR LHD 1.00 1.00 95740-K4100 W40',
      b'\xf1\x00OSE LKAS AT EUR RHD 1.00 1.00 95740-K4100 W40',
      b'\xf1\x00OSE LKAS AT KOR LHD 1.00 1.00 95740-K4100 W40',
      b'\xf1\x00OSE LKAS AT USA LHD 1.00 1.00 95740-K4300 W50',
    ],
    (Ecu.eps, 0x7D4, None): [
      b'\xf1\x00OS  MDPS C 1.00 1.03 56310/K4550 4OEDC103',
      b'\xf1\x00OS  MDPS C 1.00 1.04 56310K4000\x00 4OEDC104',
      b'\xf1\x00OS  MDPS C 1.00 1.04 56310K4050\x00 4OEDC104',
    ],
    (Ecu.fwdRadar, 0x7D0, None): [
      b'\xf1\x00OSev SCC F-CUP      1.00 1.00 99110-K4000         ',
      b'\xf1\x00OSev SCC F-CUP      1.00 1.00 99110-K4100         ',
      b'\xf1\x00OSev SCC F-CUP      1.00 1.01 99110-K4000         ',
      b'\xf1\x00OSev SCC FNCUP      1.00 1.01 99110-K4000         ',
    ],
  },
  CAR.KONA_EV_2022: {
    (Ecu.abs, 0x7D1, None): [
      b'\xf1\x8758520-K4010\xf1\x00OS IEB \x02 101 \x11\x13 58520-K4010',
      b'\xf1\x8758520-K4010\xf1\x00OS IEB \x04 101 \x11\x13 58520-K4010',
      b'\xf1\x8758520-K4010\xf1\x00OS IEB \x03 101 \x11\x13 58520-K4010',
      b'\xf1\x00OS IEB \r 102"\x05\x16 58520-K4010',
      # TODO: these return from the MULTI request, above return from LONG
      b'\x01\x04\x7f\xff\xff\xf8\xff\xff\x00\x00\x01\xd3\x00\x00\x00\x00\xff\xb7\xff\xee\xff\xe0\x00\xc0\xc0\xfc\xd5\xfc\x00\x00U\x10\xffP\xf5\xff\xfd\x00\x00\x00\x00\xfc\x00\x01',
      b'\x01\x04\x7f\xff\xff\xf8\xff\xff\x00\x00\x01\xdb\x00\x00\x00\x00\xff\xb1\xff\xd9\xff\xd2\x00\xc0\xc0\xfc\xd5\xfc\x00\x00U\x10\xff\xd6\xf5\x00\x06\x00\x00\x00\x14\xfd\x00\x04',
      b'\x01\x04\x7f\xff\xff\xf8\xff\xff\x00\x00\x01\xd3\x00\x00\x00\x00\xff\xb7\xff\xf4\xff\xd9\x00\xc0',
    ],
    (Ecu.fwdCamera, 0x7C4, None): [
      b'\xf1\x00OSP LKA  AT CND LHD 1.00 1.02 99211-J9110 802',
      b'\xf1\x00OSP LKA  AT EUR RHD 1.00 1.02 99211-J9110 802',
      b'\xf1\x00OSP LKA  AT AUS RHD 1.00 1.04 99211-J9200 904',
      b'\xf1\x00OSP LKA  AT EUR LHD 1.00 1.04 99211-J9200 904',
    ],
    (Ecu.eps, 0x7D4, None): [
      b'\xf1\x00OSP MDPS C 1.00 1.02 56310K4260\x00 4OEPC102',
      b'\xf1\x00OSP MDPS C 1.00 1.02 56310/K4970 4OEPC102',
      b'\xf1\x00OSP MDPS C 1.00 1.02 56310/K4271 4OEPC102',
    ],
    (Ecu.fwdRadar, 0x7D0, None): [
      b'\xf1\x00YB__ FCA -----      1.00 1.01 99110-K4500      \x00\x00\x00',
    ],
  },
  CAR.KIA_NIRO_EV: {
    (Ecu.fwdRadar, 0x7D0, None): [
      b'\xf1\x00DEev SCC F-CUP      1.00 1.00 99110-Q4000         ',
      b'\xf1\x00DEev SCC F-CUP      1.00 1.02 96400-Q4000         ',
      b'\xf1\x00DEev SCC F-CUP      1.00 1.02 96400-Q4100         ',
      b'\xf1\x00DEev SCC F-CUP      1.00 1.03 96400-Q4100         ',
      b'\xf1\x00DEev SCC FHCUP      1.00 1.03 96400-Q4000         ',
      b'\xf1\x8799110Q4000\xf1\x00DEev SCC F-CUP      1.00 1.00 99110-Q4000         ',
      b'\xf1\x8799110Q4100\xf1\x00DEev SCC F-CUP      1.00 1.00 99110-Q4100         ',
      b'\xf1\x8799110Q4500\xf1\x00DEev SCC F-CUP      1.00 1.00 99110-Q4500         ',
      b'\xf1\x8799110Q4600\xf1\x00DEev SCC F-CUP      1.00 1.00 99110-Q4600         ',
      b'\xf1\x8799110Q4600\xf1\x00DEev SCC FNCUP      1.00 1.00 99110-Q4600         ',
      b'\xf1\x8799110Q4600\xf1\x00DEev SCC FHCUP      1.00 1.00 99110-Q4600         ',
    ],
    (Ecu.eps, 0x7D4, None): [
      b'\xf1\x00DE  MDPS C 1.00 1.05 56310Q4000\x00 4DEEC105',
      b'\xf1\x00DE  MDPS C 1.00 1.05 56310Q4100\x00 4DEEC105',
      b'\xf1\x00DE  MDPS C 1.00 1.04 56310Q4100\x00 4DEEC104',
    ],
    (Ecu.fwdCamera, 0x7C4, None): [
      b'\xf1\x00DEE MFC  AT EUR LHD 1.00 1.00 99211-Q4100 200706',
      b'\xf1\x00DEE MFC  AT EUR LHD 1.00 1.00 99211-Q4000 191211',
      b'\xf1\x00DEE MFC  AT USA LHD 1.00 1.00 99211-Q4000 191211',
      b'\xf1\x00DEE MFC  AT USA LHD 1.00 1.03 95740-Q4000 180821',
      b'\xf1\x00DEE MFC  AT USA LHD 1.00 1.01 99211-Q4500 210428',
      b'\xf1\x00DEE MFC  AT EUR LHD 1.00 1.03 95740-Q4000 180821',
      b'\xf1\x00DEE MFC  AT KOR LHD 1.00 1.03 95740-Q4000 180821',
    ],
  },
  CAR.KIA_NIRO_EV_2ND_GEN: {
    (Ecu.fwdRadar, 0x7d0, None): [
      b'\xf1\x00SG2_ RDR -----      1.00 1.01 99110-AT000         ',
    ],
    (Ecu.fwdCamera, 0x7c4, None): [
      b'\xf1\x00SG2EMFC  AT EUR LHD 1.01 1.09 99211-AT000 220801',
      b'\xf1\x00SG2EMFC  AT USA LHD 1.01 1.09 99211-AT000 220801',
    ],
  },
  CAR.KIA_NIRO_PHEV: {
    (Ecu.engine, 0x7e0, None): [
      b'\xf1\x816H6F4051\x00\x00\x00\x00\x00\x00\x00\x00',
      b'\xf1\x816H6D1051\x00\x00\x00\x00\x00\x00\x00\x00',
      b'\xf1\x816H6F6051\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.transmission, 0x7e1, None): [
      b"\xf1\x816U3J2051\x00\x00\xf1\x006U3H0_C2\x00\x006U3J2051\x00\x00PDE0G16NS2\xf4'\\\x91",
      b'\xf1\x816U3J2051\x00\x00\xf1\x006U3H0_C2\x00\x006U3J2051\x00\x00PDE0G16NS2\x00\x00\x00\x00',
      b'\xf1\x816U3H3051\x00\x00\xf1\x006U3H0_C2\x00\x006U3H3051\x00\x00PDE0G16NS1\x00\x00\x00\x00',
      b'\xf1\x816U3H3051\x00\x00\xf1\x006U3H0_C2\x00\x006U3H3051\x00\x00PDE0G16NS1\x13\xcd\x88\x92',
      b'\xf1\x006U3H1_C2\x00\x006U3J9051\x00\x00PDE0G16NL2&[\xc3\x01',
    ],
    (Ecu.eps, 0x7D4, None): [
      b'\xf1\x00DE  MDPS C 1.00 1.09 56310G5301\x00 4DEHC109',
      b'\xf1\x00DE  MDPS C 1.00 1.01 56310G5520\x00 4DEPC101',
    ],
    (Ecu.fwdCamera, 0x7C4, None): [
      b'\xf1\x00DEP MFC  AT USA LHD 1.00 1.01 95740-G5010 170424',
      b'\xf1\x00DEP MFC  AT USA LHD 1.00 1.00 95740-G5010 170117',
      b'\xf1\x00DEP MFC  AT USA LHD 1.00 1.05 99211-G5000 190826',
    ],
    (Ecu.fwdRadar, 0x7D0, None): [
      b'\xf1\x00DEhe SCC H-CUP      1.01 1.02 96400-G5100         ',
      b'\xf1\x00DEhe SCC F-CUP      1.00 1.02 99110-G5100         ',
    ],
  },
  CAR.KIA_NIRO_HEV_2021: {
    (Ecu.engine, 0x7e0, None): [
      b'\xf1\x816H6G5051\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.transmission, 0x7e1, None): [
      b'\xf1\x816U3J9051\x00\x00\xf1\x006U3H1_C2\x00\x006U3J9051\x00\x00HDE0G16NL3\x00\x00\x00\x00',
      b'\xf1\x816U3J9051\x00\x00\xf1\x006U3H1_C2\x00\x006U3J9051\x00\x00HDE0G16NL3\xb9\xd3\xfaW',
    ],
    (Ecu.eps, 0x7d4, None): [
      b'\xf1\x00DE  MDPS C 1.00 1.01 56310G5520\x00 4DEPC101',
    ],
    (Ecu.fwdCamera, 0x7c4, None): [
      b'\xf1\x00DEH MFC  AT USA LHD 1.00 1.07 99211-G5000 201221',
      b'\xf1\x00DEH MFC  AT USA LHD 1.00 1.00 99211-G5500 210428',
    ],
    (Ecu.fwdRadar, 0x7d0, None): [
      b'\xf1\x00DEhe SCC FHCUP      1.00 1.00 99110-G5600         ',
    ],
  },
  CAR.KIA_SELTOS: {
    (Ecu.fwdRadar, 0x7d0, None): [b'\xf1\x8799110Q5100\xf1\000SP2_ SCC FHCUP      1.01 1.05 99110-Q5100         ',],
    (Ecu.abs, 0x7d1, None): [
      b'\xf1\x8758910-Q5450\xf1\000SP ESC \a 101\031\t\005 58910-Q5450',
      b'\xf1\x8758910-Q5450\xf1\000SP ESC \t 101\031\t\005 58910-Q5450',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'\xf1\x81616D2051\000\000\000\000\000\000\000\000',
      b'\xf1\x81616D5051\000\000\000\000\000\000\000\000',
      b'\001TSP2KNL06F100J0K',
      b'\001TSP2KNL06F200J0K',
    ],
    (Ecu.eps, 0x7d4, None): [
      b'\xf1\000SP2 MDPS C 1.00 1.04 56300Q5200          ',
      b'\xf1\000SP2 MDPS C 1.01 1.05 56300Q5200          ',
    ],
    (Ecu.fwdCamera, 0x7c4, None): [
      b'\xf1\000SP2 MFC  AT USA LHD 1.00 1.04 99210-Q5000 191114',
      b'\xf1\000SP2 MFC  AT USA LHD 1.00 1.05 99210-Q5000 201012',
    ],
    (Ecu.transmission, 0x7e1, None): [
      b'\xf1\x87CZLUB49370612JF7h\xa8y\x87\x99\xa7hv\x99\x97fv\x88\x87x\x89x\x96O\xff\x88\xff\xff\xff.@\xf1\x816V2C2051\000\000\xf1\0006V2B0_C2\000\0006V2C2051\000\000CSP4N20NS3\000\000\000\000',
      b'\xf1\x87954A22D200\xf1\x81T01950A1  \xf1\000T0190XBL  T01950A1  DSP2T16X4X950NS6\xd30\xa5\xb9',
      b'\xf1\x87954A22D200\xf1\x81T01950A1  \xf1\000T0190XBL  T01950A1  DSP2T16X4X950NS8\r\xfe\x9c\x8b',
    ],
  },
  CAR.KIA_OPTIMA_G4: {
    (Ecu.fwdRadar, 0x7d0, None): [
      b'\xf1\x00JF__ SCC F-CUP      1.00 1.00 96400-D4100         ',
    ],
    (Ecu.abs, 0x7d1, None): [
      b'\xf1\x00JF ESC \x0f 16 \x16\x06\x17 58920-D5080',
    ],
    (Ecu.fwdCamera, 0x7c4, None): [
      b'\xf1\x00JFWGN LDWS AT USA LHD 1.00 1.02 95895-D4100 G21',
    ],
    (Ecu.transmission, 0x7e1, None): [
      b'\xf1\x87\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xf1\x816T6J0051\x00\x00\xf1\x006T6J0_C2\x00\x006T6J0051\x00\x00TJF0T20NSB\x00\x00\x00\x00',
    ],
  },
  CAR.KIA_OPTIMA_G4_FL: {
    (Ecu.fwdRadar, 0x7d0, None): [
      b'\xf1\x00JF__ SCC F-CUP      1.00 1.00 96400-D4110         ',
    ],
    (Ecu.abs, 0x7d1, None): [
      b'\xf1\x00JF ESC \x0b 11 \x18\x030 58920-D5180',
      b"\xf1\x00JF ESC \t 11 \x18\x03' 58920-D5260",
    ],
    (Ecu.fwdCamera, 0x7c4, None): [
      b'\xf1\x00JFA LKAS AT USA LHD 1.00 1.00 95895-D5001 h32',
      b'\xf1\x00JFA LKAS AT USA LHD 1.00 1.00 95895-D5100 h32',
    ],
    (Ecu.transmission, 0x7e1, None): [
      b'\xf1\x006U2V0_C2\x00\x006U2V8051\x00\x00DJF0T16NL0\t\xd2GW',
      b'\xf1\x006U2V0_C2\x00\x006U2VA051\x00\x00DJF0T16NL1\xca3\xeb.',
      b'\xf1\x006U2V0_C2\x00\x006U2VC051\x00\x00DJF0T16NL2\x9eA\x80\x01',
      b'\xf1\x006U2V0_C2\x00\x006U2VA051\x00\x00DJF0T16NL1\x00\x00\x00\x00',
      b'\xf1\x816U2V8051\x00\x00\xf1\x006U2V0_C2\x00\x006U2V8051\x00\x00DJF0T16NL0\t\xd2GW',
      b'\xf1\x816U2VA051\x00\x00\xf1\x006U2V0_C2\x00\x006U2VA051\x00\x00DJF0T16NL1\xca3\xeb.',
      b'\xf1\x816U2VC051\x00\x00\xf1\x006U2V0_C2\x00\x006U2VC051\x00\x00DJF0T16NL2\x9eA\x80\x01',
      b'\xf1\x816U2VA051\x00\x00\xf1\x006U2V0_C2\x00\x006U2VA051\x00\x00DJF0T16NL1\x00\x00\x00\x00',
      b'\xf1\x87\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xf1\x816T6B8051\x00\x00\xf1\x006T6H0_C2\x00\x006T6B8051\x00\x00TJFSG24NH27\xa7\xc2\xb4',
    ],
  },
  CAR.ELANTRA: {
    (Ecu.fwdCamera, 0x7c4, None): [
      b'\xf1\x00PD  LKAS AT USA LHD 1.01 1.01 95740-G3100 A54',
      b'\xf1\x00PD  LKAS AT KOR LHD 1.00 1.02 95740-G3000 A51',
    ],
    (Ecu.transmission, 0x7e1, None): [
      b'\xf1\x006U2V0_C2\x00\x006U2VA051\x00\x00DPD0H16NS0e\x0e\xcd\x8e',
      b'\xf1\x006U2U0_C2\x00\x006U2T0051\x00\x00DPD0D16KS0u\xce\x1fk',
    ],
    (Ecu.eps, 0x7d4, None): [
      b'\xf1\x00PD  MDPS C 1.00 1.04 56310/G3300 4PDDC104',
      b'\xf1\x00PD  MDPS C 1.00 1.00 56310G3300\x00 4PDDC100',
    ],
    (Ecu.abs, 0x7d1, None): [
      b'\xf1\x00PD ESC \x0b 104\x18\t\x03 58920-G3350',
      b'\xf1\x00PD ESC \t 104\x18\t\x03 58920-G3350',
    ],
    (Ecu.fwdRadar, 0x7d0, None): [
      b'\xf1\x00PD__ SCC F-CUP      1.00 1.00 96400-G3300         ',
      b'\xf1\x00PD__ SCC FNCUP      1.01 1.00 96400-G3000         ',
    ],
  },
  CAR.ELANTRA_2021: {
    (Ecu.fwdRadar, 0x7d0, None): [
      b'\xf1\x00CN7_ SCC F-CUP      1.00 1.01 99110-AA000         ',
      b'\xf1\x00CN7_ SCC FHCUP      1.00 1.01 99110-AA000         ',
      b'\xf1\x00CN7_ SCC FNCUP      1.00 1.01 99110-AA000         ',
      b'\xf1\x8799110AA000\xf1\x00CN7_ SCC FHCUP      1.00 1.01 99110-AA000         ',
      b'\xf1\x8799110AA000\xf1\x00CN7_ SCC F-CUP      1.00 1.01 99110-AA000         ',
    ],
    (Ecu.eps, 0x7d4, None): [
      b'\xf1\x87\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf1\x00CN7 MDPS C 1.00 1.06 \x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00 4CNDC106',
      b'\xf1\x8756310/AA070\xf1\x00CN7 MDPS C 1.00 1.06 56310/AA070 4CNDC106',
      b'\xf1\x8756310AA050\x00\xf1\x00CN7 MDPS C 1.00 1.06 56310AA050\x00 4CNDC106\xf1\xa01.06',
      b'\xf1\x00CN7 MDPS C 1.00 1.06 56310AA050\x00 4CNDC106',
    ],
    (Ecu.fwdCamera, 0x7c4, None): [
      b'\xf1\x00CN7 MFC  AT USA LHD 1.00 1.00 99210-AB000 200819',
      b'\xf1\x00CN7 MFC  AT USA LHD 1.00 1.03 99210-AA000 200819',
      b'\xf1\x00CN7 MFC  AT USA LHD 1.00 1.01 99210-AB000 210205',
      b'\xf1\x00CN7 MFC  AT USA LHD 1.00 1.06 99210-AA000 220111',
      b'\xf1\x00CN7 MFC  AT USA LHD 1.00 1.03 99210-AB000 220426',
    ],
    (Ecu.abs, 0x7d1, None): [
      b'\xf1\x00CN ESC \t 101 \x10\x03 58910-AB800',
      b'\xf1\x8758910-AA800\xf1\x00CN ESC \t 104 \x08\x03 58910-AA800',
      b'\xf1\x8758910-AA800\xf1\x00CN ESC \t 105 \x10\x03 58910-AA800',
      b'\xf1\x8758910-AB800\xf1\x00CN ESC \t 101 \x10\x03 58910-AB800\xf1\xa01.01',
    ],
    (Ecu.transmission, 0x7e1, None): [
      b'\xf1\x00HT6WA280BLHT6VA640A1CCN0N20NS5\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
      b'\xf1\x00HT6WA280BLHT6VA640A1CCN0N20NS5\x00\x00\x00\x00\x00\x00\xe8\xba\xce\xfa',
      b'\xf1\x87CXMQFM2135005JB2E\xb9\x89\x98W\xa9y\x97h\xa9\x98\x99wxvwh\x87\177\xffx\xff\xff\xff,,\xf1\x89HT6VA640A1\xf1\x82CCN0N20NS5\x00\x00\x00\x00\x00\x00',
      b'\xf1\x87CXMQFM1916035JB2\x88vvgg\x87Wuwgev\xa9\x98\x88\x98h\x99\x9f\xffh\xff\xff\xff\xa5\xee\xf1\x89HT6VA640A1\xf1\x82CCN0N20NS5\x00\x00\x00\x00\x00\x00',
      b'\xf1\x87CXLQF40189012JL2f\x88\x86\x88\x88vUex\xb8\x88\x88\x88\x87\x88\x89fh?\xffz\xff\xff\xff\x08z\xf1\x89HT6VA640A1\xf1\x82CCN0N20NS5\x00\x00\x00\x00\x00\x00',
      b'\xf1\x87CXMQFM2728305JB2E\x97\x87xw\x87vwgw\x84x\x88\x88w\x89EI\xbf\xff{\xff\xff\xff\xe6\x0e\xf1\x89HT6VA640A1\xf1\x82CCN0N20NS5\x00\x00\x00\x00\x00\x00',
      b'\xf1\x87CXMQFM3806705JB2\x89\x87wwx\x88g\x86\x99\x87\x86xwwv\x88yv\x7f\xffz\xff\xff\xffV\x15\xf1\x89HT6VA640A1\xf1\x82CCN0N20NS5\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'\xf1\x82CNCWD0AMFCXCSFFA',
      b'\xf1\x81HM6M2_0a0_FF0',
      b'\xf1\x82CNCVD0AMFCXCSFFB',
      b'\xf1\x870\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf1\x81HM6M2_0a0_G80',
      b'\xf1\x870\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf1\x81HM6M2_0a0_HC0',
    ],
  },
  CAR.ELANTRA_HEV_2021: {
    (Ecu.fwdCamera, 0x7c4, None): [
      b'\xf1\x00CN7HMFC  AT USA LHD 1.00 1.05 99210-AA000 210930',
      b'\xf1\000CN7HMFC  AT USA LHD 1.00 1.03 99210-AA000 200819',
      b'\xf1\x00CN7HMFC  AT USA LHD 1.00 1.07 99210-AA000 220426',
      b'\xf1\x00CN7HMFC  AT USA LHD 1.00 1.08 99210-AA000 220728',
      b'\xf1\x00CN7HMFC  AT USA LHD 1.00 1.09 99210-AA000 221108',
    ],
    (Ecu.fwdRadar, 0x7d0, None): [
      b'\xf1\x00CNhe SCC FHCUP      1.00 1.01 99110-BY000         ',
      b'\xf1\x8799110BY000\xf1\x00CNhe SCC FHCUP      1.00 1.01 99110-BY000         ',
    ],
    (Ecu.eps, 0x7d4, None): [
      b'\xf1\x00CN7 MDPS C 1.00 1.03 56310BY0500 4CNHC103',
      b'\xf1\x8756310/BY050\xf1\x00CN7 MDPS C 1.00 1.03 56310/BY050 4CNHC103',
      b'\xf1\x8756310/BY050\xf1\000CN7 MDPS C 1.00 1.02 56310/BY050 4CNHC102',
      b'\xf1\x00CN7 MDPS C 1.00 1.04 56310BY050\x00 4CNHC104',
    ],
    (Ecu.transmission, 0x7e1, None): [
      b'\xf1\0006U3L0_C2\000\0006U3K3051\000\000HCN0G16NS0\xb9?A\xaa',
      b'\xf1\0006U3L0_C2\000\0006U3K3051\000\000HCN0G16NS0\000\000\000\000',
      b'\xf1\x816U3K3051\000\000\xf1\0006U3L0_C2\000\0006U3K3051\000\000HCN0G16NS0\xb9?A\xaa',
      b'\xf1\x816U3K3051\x00\x00\xf1\x006U3L0_C2\x00\x006U3K3051\x00\x00HCN0G16NS0\x00\x00\x00\x00',
      b'\xf1\x006U3L0_C2\x00\x006U3K9051\x00\x00HCN0G16NS1\x00\x00\x00\x00',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'\xf1\x816H6G5051\x00\x00\x00\x00\x00\x00\x00\x00',
      b'\xf1\x816H6G6051\x00\x00\x00\x00\x00\x00\x00\x00',
      b'\xf1\x816H6G8051\x00\x00\x00\x00\x00\x00\x00\x00',
    ]
  },
  CAR.KONA_HEV: {
    (Ecu.abs, 0x7d1, None): [
      b'\xf1\x00OS IEB \x01 104 \x11  58520-CM000',
    ],
    (Ecu.fwdRadar, 0x7d0, None): [
      b'\xf1\x00OShe SCC FNCUP      1.00 1.01 99110-CM000         ',
    ],
    (Ecu.eps, 0x7d4, None): [
      b'\xf1\x00OS  MDPS C 1.00 1.00 56310CM030\x00 4OHDC100',
    ],
    (Ecu.fwdCamera, 0x7c4, None): [
      b'\xf1\x00OSH LKAS AT KOR LHD 1.00 1.01 95740-CM000 l31',
    ],
    (Ecu.transmission, 0x7e1, None): [
      b'\xf1\x816U3J9051\x00\x00\xf1\x006U3H1_C2\x00\x006U3J9051\x00\x00HOS0G16DS1\x16\xc7\xb0\xd9',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'\xf1\x816H6F6051\x00\x00\x00\x00\x00\x00\x00\x00',
    ]
  },
  CAR.SONATA_HYBRID: {
    (Ecu.fwdRadar, 0x7d0, None): [
      b'\xf1\000DNhe SCC FHCUP      1.00 1.02 99110-L5000         ',
      b'\xf1\x8799110L5000\xf1\000DNhe SCC FHCUP      1.00 1.02 99110-L5000         ',
      b'\xf1\000DNhe SCC F-CUP      1.00 1.02 99110-L5000         ',
      b'\xf1\x8799110L5000\xf1\000DNhe SCC F-CUP      1.00 1.02 99110-L5000         ',
    ],
    (Ecu.eps, 0x7d4, None): [
      b'\xf1\x8756310-L5500\xf1\x00DN8 MDPS C 1.00 1.02 56310-L5500 4DNHC102',
      b'\xf1\x8756310-L5450\xf1\x00DN8 MDPS C 1.00 1.02 56310-L5450 4DNHC102',
      b'\xf1\x8756310-L5450\xf1\000DN8 MDPS C 1.00 1.03 56310-L5450 4DNHC103',
    ],
    (Ecu.fwdCamera, 0x7c4, None): [
      b'\xf1\x00DN8HMFC  AT USA LHD 1.00 1.04 99211-L1000 191016',
      b'\xf1\x00DN8HMFC  AT USA LHD 1.00 1.05 99211-L1000 201109',
      b'\xf1\000DN8HMFC  AT USA LHD 1.00 1.06 99211-L1000 210325',
    ],
    (Ecu.transmission, 0x7e1, None): [
      b'\xf1\000PSBG2333  E14\x00\x00\x00\x00\x00\x00\x00TDN2H20SA6N\xc2\xeeW',
      b'\xf1\x87959102T250\x00\x00\x00\x00\x00\xf1\x81E09\x00\x00\x00\x00\x00\x00\x00\xf1\x00PSBG2323  E09\x00\x00\x00\x00\x00\x00\x00TDN2H20SA5\x97R\x88\x9e',
      b'\xf1\000PSBG2323  E09\000\000\000\000\000\000\000TDN2H20SA5\x97R\x88\x9e',
      b'\xf1\000PSBG2333  E16\000\000\000\000\000\000\000TDN2H20SA7\0323\xf9\xab',
      b'\xf1\x87PCU\000\000\000\000\000\000\000\000\000\xf1\x81E16\000\000\000\000\000\000\000\xf1\000PSBG2333  E16\000\000\000\000\000\000\000TDN2H20SA7\0323\xf9\xab',
      b'\xf1\x87959102T250\x00\x00\x00\x00\x00\xf1\x81E14\x00\x00\x00\x00\x00\x00\x00\xf1\x00PSBG2333  E14\x00\x00\x00\x00\x00\x00\x00TDN2H20SA6N\xc2\xeeW',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'\xf1\x87391162J012',
      b'\xf1\x87391162J013',
      b'\xf1\x87391062J002',
    ],
  },
  CAR.KIA_SORENTO: {
    (Ecu.fwdCamera, 0x7c4, None): [
      b'\xf1\x00UMP LKAS AT USA LHD 1.01 1.01 95740-C6550 d01'
    ],
    (Ecu.abs, 0x7d1, None): [
      b'\xf1\x00UM ESC \x0c 12 \x18\x05\x06 58910-C6330'
    ],
    (Ecu.fwdRadar, 0x7D0, None): [
      b'\xf1\x00UM__ SCC F-CUP      1.00 1.00 96400-C6500         '
    ],
    (Ecu.transmission, 0x7e1, None): [
      b'\xf1\x87LDKUAA0348164HE3\x87www\x87www\x88\x88\xa8\x88w\x88\x97xw\x88\x97x\x86o\xf8\xff\x87f\x7f\xff\x15\xe0\xf1\x81U811\x00\x00\x00\x00\x00\x00\xf1\x00bcsh8p54  U811\x00\x00\x00\x00\x00\x00TUM4G33NL3V|DG'
    ],
    (Ecu.engine, 0x7e0, None): [
      b'\xf1\x81640F0051\x00\x00\x00\x00\x00\x00\x00\x00'
    ],
  },
  CAR.KIA_SORENTO_PHEV_4TH_GEN: {
    (Ecu.fwdRadar, 0x7d0, None): [
      b'\xf1\x00MQhe SCC FHCUP      1.00 1.06 99110-P4000         ',
    ],
    (Ecu.fwdCamera, 0x7c4, None): [
      b'\xf1\x00MQ4HMFC  AT USA LHD 1.00 1.11 99210-P2000 211217',
    ]
  },
  CAR.KIA_EV6: {
    (Ecu.fwdRadar, 0x7d0, None): [
      b'\xf1\x00CV1_ RDR -----      1.00 1.01 99110-CV000         ',
    ],
    (Ecu.fwdCamera, 0x7c4, None): [
      b'\xf1\x00CV1 MFC  AT USA LHD 1.00 1.05 99210-CV000 211027',
      b'\xf1\x00CV1 MFC  AT USA LHD 1.00 1.06 99210-CV000 220328',
      b'\xf1\x00CV1 MFC  AT EUR LHD 1.00 1.05 99210-CV000 211027',
      b'\xf1\x00CV1 MFC  AT EUR LHD 1.00 1.06 99210-CV000 220328',
      b'\xf1\x00CV1 MFC  AT EUR RHD 1.00 1.00 99210-CV100 220630',
      b'\xf1\x00CV1 MFC  AT USA LHD 1.00 1.00 99210-CV100 220630',
      b'\xf1\x00CV1 MFC  AT KOR LHD 1.00 1.04 99210-CV000 210823',
      b'\xf1\x00CV1 MFC  AT KOR LHD 1.00 1.05 99210-CV000 211027',
      b'\xf1\x00CV1 MFC  AT KOR LHD 1.00 1.06 99210-CV000 220328',
    ],
  },
  CAR.IONIQ_5: {
    (Ecu.fwdRadar, 0x7d0, None): [
      b'\xf1\x00NE1_ RDR -----      1.00 1.00 99110-GI000         ',
    ],
    (Ecu.fwdCamera, 0x7c4, None): [
      b'\xf1\x00NE1 MFC  AT USA LHD 1.00 1.02 99211-GI010 211206',
      b'\xf1\x00NE1 MFC  AT EUR LHD 1.00 1.06 99211-GI000 210813',
      b'\xf1\x00NE1 MFC  AT USA LHD 1.00 1.05 99211-GI010 220614',
      b'\xf1\x00NE1 MFC  AT KOR LHD 1.00 1.05 99211-GI010 220614',
      b'\xf1\x00NE1 MFC  AT EUR RHD 1.00 1.01 99211-GI010 211007',
      b'\xf1\x00NE1 MFC  AT USA LHD 1.00 1.01 99211-GI010 211007',
      b'\xf1\x00NE1 MFC  AT EUR RHD 1.00 1.02 99211-GI010 211206',
      b'\xf1\x00NE1 MFC  AT USA LHD 1.00 1.03 99211-GI010 220401',
    ],
  },
  CAR.TUCSON_4TH_GEN: {
    (Ecu.fwdCamera, 0x7c4, None): [
      b'\xf1\x00NX4 FR_CMR AT USA LHD 1.00 1.00 99211-N9210 14G',
      b'\xf1\x00NX4 FR_CMR AT USA LHD 1.00 1.01 99211-N9240 14T',
      b'\xf1\x00NX4 FR_CMR AT USA LHD 1.00 1.00 99211-CW010 14X',
    ],
    (Ecu.fwdRadar, 0x7d0, None): [
      b'\xf1\x00NX4__               1.00 1.00 99110-N9100         ',
      b'\xf1\x00NX4__               1.01 1.00 99110-N9100         ',
    ],
  },
  CAR.TUCSON_HYBRID_4TH_GEN: {
    (Ecu.fwdCamera, 0x7c4, None): [
      b'\xf1\x00NX4 FR_CMR AT USA LHD 1.00 1.00 99211-N9240 14Q',
      b'\xf1\x00NX4 FR_CMR AT USA LHD 1.00 1.00 99211-N9220 14K',
      b'\xf1\x00NX4 FR_CMR AT USA LHD 1.00 1.01 99211-N9100 14A',
      b'\xf1\x00NX4 FR_CMR AT USA LHD 1.00 1.00 99211-N9250 14W',
    ],
    (Ecu.fwdRadar, 0x7d0, None): [
      b'\xf1\x00NX4__               1.00 1.00 99110-N9100         ',
      b'\xf1\x00NX4__               1.01 1.00 99110-N9100         ',
    ],
  },
  CAR.KIA_SPORTAGE_HYBRID_5TH_GEN: {
    (Ecu.fwdCamera, 0x7c4, None): [
      b'\xf1\x00NQ5 FR_CMR AT GEN LHD 1.00 1.00 99211-P1060 665',
      b'\xf1\x00NQ5 FR_CMR AT USA LHD 1.00 1.00 99211-P1060 665',
    ],
    (Ecu.fwdRadar, 0x7d0, None): [
      b'\xf1\x00NQ5__               1.01 1.03 99110-CH000         ',
    ],
  },
  CAR.SANTA_CRUZ_1ST_GEN: {
    (Ecu.fwdCamera, 0x7c4, None): [
      b'\xf1\x00NX4 FR_CMR AT USA LHD 1.00 1.00 99211-CW000 14M',
      b'\xf1\x00NX4 FR_CMR AT USA LHD 1.00 1.00 99211-CW010 14X',
    ],
    (Ecu.fwdRadar, 0x7d0, None): [
      b'\xf1\x00NX4__               1.00 1.00 99110-K5000         ',
      b'\xf1\x00NX4__               1.01 1.00 99110-K5000         ',
    ],
  },
  CAR.KIA_SPORTAGE_5TH_GEN: {
    (Ecu.fwdCamera, 0x7c4, None): [
      b'\xf1\x00NQ5 FR_CMR AT USA LHD 1.00 1.00 99211-P1030 662',
      b'\xf1\x00NQ5 FR_CMR AT USA LHD 1.00 1.00 99211-P1040 663',
    ],
    (Ecu.fwdRadar, 0x7d0, None): [
      b'\xf1\x00NQ5__               1.00 1.02 99110-P1000         ',
      b'\xf1\x00NQ5__               1.00 1.03 99110-P1000         ',
    ],
  },
  CAR.GENESIS_GV70_1ST_GEN: {
    (Ecu.fwdCamera, 0x7c4, None): [
      b'\xf1\x00JK1 MFC  AT USA LHD 1.00 1.04 99211-AR000 210204',
      b'\xf1\x00JK1 MFC  AT USA LHD 1.00 1.01 99211-AR200 220125',
    ],
    (Ecu.fwdRadar, 0x7d0, None): [
      b'\xf1\x00JK1_ SCC FHCUP      1.00 1.02 99110-AR000         ',
      b'\xf1\x00JK1_ SCC FHCUP      1.00 1.00 99110-AR200         ',
    ],
  },
  CAR.GENESIS_GV60_EV_1ST_GEN: {
    (Ecu.fwdCamera, 0x7c4, None): [
      b'\xf1\x00JW1 MFC  AT USA LHD 1.00 1.02 99211-CU100 211215',
      b'\xf1\x00JW1 MFC  AT USA LHD 1.00 1.02 99211-CU000 211215',
      b'\xf1\x00JW1 MFC  AT USA LHD 1.00 1.03 99211-CU000 221118',
    ],
    (Ecu.fwdRadar, 0x7d0, None): [
      b'\xf1\x00JW1_ RDR -----      1.00 1.00 99110-CU000         ',
    ],
  },
  CAR.KIA_SORENTO_4TH_GEN: {
    (Ecu.fwdCamera, 0x7c4, None): [
      b'\xf1\x00MQ4 MFC  AT USA LHD 1.00 1.05 99210-R5000 210623',
      b'\xf1\x00MQ4 MFC  AT USA LHD 1.00 1.03 99210-R5000 200903',
    ],
    (Ecu.fwdRadar, 0x7d0, None): [
      b'\xf1\x00MQ4_ SCC FHCUP      1.00 1.06 99110-P2000         ',
      b'\xf1\x00MQ4_ SCC F-CUP      1.00 1.06 99110-P2000         ',
    ],
  },
  CAR.KIA_NIRO_HEV_2ND_GEN: {
    (Ecu.fwdCamera, 0x7c4, None): [
      b'\xf1\x00SG2HMFC  AT USA LHD 1.01 1.08 99211-AT000 220531',
      b'\xf1\x00SG2HMFC  AT USA LHD 1.01 1.09 99211-AT000 220801',
    ],
    (Ecu.fwdRadar, 0x7d0, None): [
      b'\xf1\x00SG2_ RDR -----      1.00 1.01 99110-AT000         ',
    ],
  },
}

CHECKSUM = {
  "crc8": [CAR.SANTA_FE, CAR.SONATA, CAR.PALISADE, CAR.KIA_SELTOS, CAR.ELANTRA_2021, CAR.ELANTRA_HEV_2021, CAR.SONATA_HYBRID, CAR.SANTA_FE_2022, CAR.KIA_K5_2021, CAR.SANTA_FE_HEV_2022, CAR.SANTA_FE_PHEV_2022, CAR.KIA_K5_HEV_2020],
  "6B": [CAR.KIA_SORENTO, CAR.HYUNDAI_GENESIS],
}

CAN_GEARS = {
  # which message has the gear
  "use_cluster_gears": {CAR.ELANTRA, CAR.KONA},
  "use_tcu_gears": {CAR.KIA_OPTIMA_G4, CAR.KIA_OPTIMA_G4_FL, CAR.SONATA_LF, CAR.VELOSTER, CAR.TUCSON},
  "use_elect_gears": {CAR.KIA_NIRO_EV, CAR.KIA_NIRO_PHEV, CAR.KIA_NIRO_HEV_2021, CAR.KIA_OPTIMA_H, CAR.IONIQ_EV_LTD, CAR.KONA_EV, CAR.IONIQ, CAR.IONIQ_EV_2020, CAR.IONIQ_PHEV, CAR.ELANTRA_HEV_2021, CAR.SONATA_HYBRID, CAR.KONA_HEV, CAR.IONIQ_HEV_2022, CAR.SANTA_FE_HEV_2022, CAR.SANTA_FE_PHEV_2022, CAR.IONIQ_PHEV_2019, CAR.KONA_EV_2022, CAR.KIA_K5_HEV_2020},
}

CANFD_CAR = {CAR.KIA_EV6, CAR.IONIQ_5, CAR.TUCSON_4TH_GEN, CAR.TUCSON_HYBRID_4TH_GEN, CAR.KIA_SPORTAGE_HYBRID_5TH_GEN, CAR.SANTA_CRUZ_1ST_GEN, CAR.KIA_SPORTAGE_5TH_GEN, CAR.GENESIS_GV70_1ST_GEN, CAR.KIA_SORENTO_PHEV_4TH_GEN, CAR.GENESIS_GV60_EV_1ST_GEN, CAR.KIA_SORENTO_4TH_GEN, CAR.KIA_NIRO_HEV_2ND_GEN, CAR.KIA_NIRO_EV_2ND_GEN}

# The radar does SCC on these cars when HDA I, rather than the camera
CANFD_RADAR_SCC_CAR = {CAR.GENESIS_GV70_1ST_GEN, CAR.KIA_SORENTO_PHEV_4TH_GEN, CAR.KIA_SORENTO_4TH_GEN}

# The camera does SCC on these cars, rather than the radar
CAMERA_SCC_CAR = {CAR.KONA_EV_2022, }

HYBRID_CAR = {CAR.IONIQ_PHEV, CAR.ELANTRA_HEV_2021, CAR.KIA_NIRO_PHEV, CAR.KIA_NIRO_HEV_2021, CAR.SONATA_HYBRID, CAR.KONA_HEV, CAR.IONIQ, CAR.IONIQ_HEV_2022, CAR.SANTA_FE_HEV_2022, CAR.SANTA_FE_PHEV_2022, CAR.IONIQ_PHEV_2019, CAR.TUCSON_HYBRID_4TH_GEN, CAR.KIA_SPORTAGE_HYBRID_5TH_GEN, CAR.KIA_SORENTO_PHEV_4TH_GEN, CAR.KIA_K5_HEV_2020, CAR.KIA_NIRO_HEV_2ND_GEN}  # these cars use a different gas signal
EV_CAR = {CAR.IONIQ_EV_2020, CAR.IONIQ_EV_LTD, CAR.KONA_EV, CAR.KIA_NIRO_EV, CAR.KIA_NIRO_EV_2ND_GEN, CAR.KONA_EV_2022, CAR.KIA_EV6, CAR.IONIQ_5, CAR.GENESIS_GV60_EV_1ST_GEN}

# these cars require a special panda safety mode due to missing counters and checksums in the messages
LEGACY_SAFETY_MODE_CAR = {CAR.HYUNDAI_GENESIS, CAR.IONIQ_EV_2020, CAR.IONIQ_EV_LTD, CAR.IONIQ_PHEV, CAR.IONIQ, CAR.KONA_EV, CAR.KIA_SORENTO, CAR.SONATA_LF, CAR.KIA_OPTIMA_G4, CAR.KIA_OPTIMA_G4_FL, CAR.VELOSTER,
                          CAR.GENESIS_G70, CAR.GENESIS_G80, CAR.KIA_CEED, CAR.ELANTRA, CAR.IONIQ_HEV_2022}

# If 0x500 is present on bus 1 it probably has a Mando radar outputting radar points.
# If no points are outputted by default it might be possible to turn it on using  selfdrive/debug/hyundai_enable_radar_points.py
DBC = {
  CAR.ELANTRA: dbc_dict('hyundai_kia_generic', None),
  CAR.ELANTRA_2021: dbc_dict('hyundai_kia_generic', None),
  CAR.ELANTRA_HEV_2021: dbc_dict('hyundai_kia_generic', None),
  CAR.GENESIS_G70: dbc_dict('hyundai_kia_generic', None),
  CAR.GENESIS_G70_2020: dbc_dict('hyundai_kia_generic', 'hyundai_kia_mando_front_radar_generated'),
  CAR.GENESIS_G80: dbc_dict('hyundai_kia_generic', None),
  CAR.GENESIS_G90: dbc_dict('hyundai_kia_generic', None),
  CAR.HYUNDAI_GENESIS: dbc_dict('hyundai_kia_generic', None),
  CAR.IONIQ_PHEV_2019: dbc_dict('hyundai_kia_generic', None),
  CAR.IONIQ_PHEV: dbc_dict('hyundai_kia_generic', None),
  CAR.IONIQ_EV_2020: dbc_dict('hyundai_kia_generic', None),
  CAR.IONIQ_EV_LTD: dbc_dict('hyundai_kia_generic', 'hyundai_kia_mando_front_radar_generated'),
  CAR.IONIQ: dbc_dict('hyundai_kia_generic', None),
  CAR.IONIQ_HEV_2022: dbc_dict('hyundai_kia_generic', None),
  CAR.KIA_FORTE: dbc_dict('hyundai_kia_generic', None),
  CAR.KIA_K5_2021: dbc_dict('hyundai_kia_generic', None),
  CAR.KIA_K5_HEV_2020: dbc_dict('hyundai_kia_generic', 'hyundai_kia_mando_front_radar_generated'),
  CAR.KIA_NIRO_EV: dbc_dict('hyundai_kia_generic', 'hyundai_kia_mando_front_radar_generated'),
  CAR.KIA_NIRO_PHEV: dbc_dict('hyundai_kia_generic', 'hyundai_kia_mando_front_radar_generated'),
  CAR.KIA_NIRO_HEV_2021: dbc_dict('hyundai_kia_generic', None),
  CAR.KIA_OPTIMA_G4: dbc_dict('hyundai_kia_generic', None),
  CAR.KIA_OPTIMA_G4_FL: dbc_dict('hyundai_kia_generic', None),
  CAR.KIA_OPTIMA_H: dbc_dict('hyundai_kia_generic', None),
  CAR.KIA_SELTOS: dbc_dict('hyundai_kia_generic', None),
  CAR.KIA_SORENTO: dbc_dict('hyundai_kia_generic', None), # Has 0x5XX messages, but different format
  CAR.KIA_STINGER: dbc_dict('hyundai_kia_generic', None),
  CAR.KIA_STINGER_2022: dbc_dict('hyundai_kia_generic', None),
  CAR.KONA: dbc_dict('hyundai_kia_generic', None),
  CAR.KONA_EV: dbc_dict('hyundai_kia_generic', None),
  CAR.KONA_EV_2022: dbc_dict('hyundai_kia_generic', None),
  CAR.KONA_HEV: dbc_dict('hyundai_kia_generic', None),
  CAR.SANTA_FE: dbc_dict('hyundai_kia_generic', 'hyundai_kia_mando_front_radar_generated'),
  CAR.SANTA_FE_2022: dbc_dict('hyundai_kia_generic', None),
  CAR.SANTA_FE_HEV_2022: dbc_dict('hyundai_kia_generic', None),
  CAR.SANTA_FE_PHEV_2022: dbc_dict('hyundai_kia_generic', None),
  CAR.SONATA: dbc_dict('hyundai_kia_generic', 'hyundai_kia_mando_front_radar_generated'),
  CAR.SONATA_LF: dbc_dict('hyundai_kia_generic', None), # Has 0x5XX messages, but different format
  CAR.TUCSON: dbc_dict('hyundai_kia_generic', None),
  CAR.PALISADE: dbc_dict('hyundai_kia_generic', 'hyundai_kia_mando_front_radar_generated'),
  CAR.VELOSTER: dbc_dict('hyundai_kia_generic', None),
  CAR.KIA_CEED: dbc_dict('hyundai_kia_generic', None),
  CAR.KIA_EV6: dbc_dict('hyundai_canfd', None),
  CAR.SONATA_HYBRID: dbc_dict('hyundai_kia_generic', 'hyundai_kia_mando_front_radar_generated'),
  CAR.TUCSON_4TH_GEN: dbc_dict('hyundai_canfd', None),
  CAR.TUCSON_HYBRID_4TH_GEN: dbc_dict('hyundai_canfd', None),
  CAR.IONIQ_5: dbc_dict('hyundai_canfd', None),
  CAR.SANTA_CRUZ_1ST_GEN: dbc_dict('hyundai_canfd', None),
  CAR.KIA_SPORTAGE_5TH_GEN: dbc_dict('hyundai_canfd', None),
  CAR.KIA_SPORTAGE_HYBRID_5TH_GEN: dbc_dict('hyundai_canfd', None),
  CAR.GENESIS_GV70_1ST_GEN: dbc_dict('hyundai_canfd', None),
  CAR.KIA_SORENTO_PHEV_4TH_GEN: dbc_dict('hyundai_canfd', None),
  CAR.GENESIS_GV60_EV_1ST_GEN: dbc_dict('hyundai_canfd', None),
  CAR.KIA_SORENTO_4TH_GEN: dbc_dict('hyundai_canfd', None),
  CAR.KIA_NIRO_HEV_2ND_GEN: dbc_dict('hyundai_canfd', None),
  CAR.KIA_NIRO_EV_2ND_GEN: dbc_dict('hyundai_canfd', None),
}
