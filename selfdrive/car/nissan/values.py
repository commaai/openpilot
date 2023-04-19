from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from enum import Enum

from cereal import car
from panda.python import uds
from selfdrive.car import AngleRateLimit, dbc_dict
from selfdrive.car.docs_definitions import CarInfo, Harness
from selfdrive.car.fw_query_definitions import FwQueryConfig, Request, StdQueries

Ecu = car.CarParams.Ecu


class CarControllerParams:
  ANGLE_RATE_LIMIT_UP = AngleRateLimit(speed_bp=[0., 5., 15.], angle_v=[5., .8, .15])
  ANGLE_RATE_LIMIT_DOWN = AngleRateLimit(speed_bp=[0., 5., 15.], angle_v=[5., 3.5, 0.4])
  LKAS_MAX_TORQUE = 1               # A value of 1 is easy to overpower
  STEER_THRESHOLD = 1.0

  def __init__(self, CP):
    pass


class CAR:
  XTRAIL = "NISSAN X-TRAIL 2017"
  LEAF = "NISSAN LEAF 2018"
  # Leaf with ADAS ECU found behind instrument cluster instead of glovebox
  # Currently the only known difference between them is the inverted seatbelt signal.
  LEAF_IC = "NISSAN LEAF 2018 Instrument Cluster"
  ROGUE = "NISSAN ROGUE 2019"
  ALTIMA = "NISSAN ALTIMA 2020"


@dataclass
class NissanCarInfo(CarInfo):
  package: str = "ProPILOT Assist"
  harness: Enum = Harness.nissan_a


CAR_INFO: Dict[str, Optional[Union[NissanCarInfo, List[NissanCarInfo]]]] = {
  CAR.XTRAIL: NissanCarInfo("Nissan X-Trail 2017"),
  CAR.LEAF: NissanCarInfo("Nissan Leaf 2018-23", video_link="https://youtu.be/vaMbtAh_0cY"),
  CAR.LEAF_IC: None,  # same platforms
  CAR.ROGUE: NissanCarInfo("Nissan Rogue 2018-20"),
  CAR.ALTIMA: NissanCarInfo("Nissan Altima 2019-20", harness=Harness.nissan_b),
}

FINGERPRINTS = {
  CAR.XTRAIL: [
    {
      2: 5, 42: 6, 346: 6, 347: 5, 348: 8, 349: 7, 361: 8, 386: 8, 389: 8, 397: 8, 398: 8, 403: 8, 520: 2, 523: 6, 548: 8, 645: 8, 658: 8, 665: 8, 666: 8, 674: 2, 682: 8, 683: 8, 689: 8, 723: 8, 758: 3, 768: 2, 783: 3, 851: 8, 855: 8, 1041: 8, 1055: 2, 1104: 4, 1105: 6, 1107: 4, 1108: 8, 1111: 4, 1227: 8, 1228: 8, 1247: 4, 1266: 8, 1273: 7, 1342: 1, 1376: 6, 1401: 8, 1474: 2, 1497: 3, 1821: 8, 1823: 8, 1837: 8, 2015: 8, 2016: 8, 2024: 8
    },
    {
      2: 5, 42: 6, 346: 6, 347: 5, 348: 8, 349: 7, 361: 8, 386: 8, 389: 8, 397: 8, 398: 8, 403: 8, 520: 2, 523: 6, 527: 1, 548: 8, 637: 4, 645: 8, 658: 8, 665: 8, 666: 8, 674: 2, 682: 8, 683: 8, 689: 8, 723: 8, 758: 3, 768: 6, 783: 3, 851: 8, 855: 8, 1041: 8, 1055: 2, 1104: 4, 1105: 6, 1107: 4, 1108: 8, 1111: 4, 1227: 8, 1228: 8, 1247: 4, 1266: 8, 1273: 7, 1342: 1, 1376: 6, 1401: 8, 1474: 8, 1497: 3, 1534: 6, 1792: 8, 1821: 8, 1823: 8, 1837: 8, 1872: 8, 1937: 8, 1953: 8, 1968: 8, 2015: 8, 2016: 8, 2024: 8
    },
  ],
  CAR.LEAF: [
    {
      2: 5, 42: 6, 264: 3, 361: 8, 372: 8, 384: 8, 389: 8, 403: 8, 459: 7, 460: 4, 470: 8, 520: 1, 569: 8, 581: 8, 634: 7, 640: 8, 644: 8, 645: 8, 646: 5, 658: 8, 682: 8, 683: 8, 689: 8, 724: 6, 758: 3, 761: 2, 783: 3, 852: 8, 853: 8, 856: 8, 861: 8, 944: 1, 976: 6, 1008: 7, 1011: 7, 1057: 3, 1227: 8, 1228: 8, 1261: 5, 1342: 1, 1354: 8, 1361: 8, 1459: 8, 1477: 8, 1497: 3, 1549: 8, 1573: 6, 1821: 8, 1837: 8, 1856: 8, 1859: 8, 1861: 8, 1864: 8, 1874: 8, 1888: 8, 1891: 8, 1893: 8, 1906: 8, 1947: 8, 1949: 8, 1979: 8, 1981: 8, 2016: 8, 2017: 8, 2021: 8, 643: 5, 1792: 8, 1872: 8, 1937: 8, 1953: 8, 1968: 8, 1988: 8, 2000: 8, 2001: 8, 2004: 8, 2005: 8, 2015: 8
    },
    # 2020 Leaf SV Plus
    {
      2: 5, 42: 8, 264: 3, 361: 8, 372: 8, 384: 8, 389: 8, 403: 8, 459: 7, 460: 4, 470: 8, 520: 1, 569: 8, 581: 8, 634: 7, 640: 8, 643: 5, 644: 8, 645: 8, 646: 5, 658: 8, 682: 8, 683: 8, 689: 8, 724: 6, 758: 3, 761: 2, 772: 8, 773: 6, 774: 7, 775: 8, 776: 6, 777: 7, 778: 6, 783: 3, 852: 8, 853: 8, 856: 8, 861: 8, 943: 8, 944: 1, 976: 6, 1008: 7, 1009: 8, 1010: 8, 1011: 7, 1012: 8, 1013: 8, 1019: 8, 1020: 8, 1021: 8, 1022: 8, 1057: 3, 1227: 8, 1228: 8, 1261: 5, 1342: 1, 1354: 8, 1361: 8, 1402: 8, 1459: 8, 1477: 8, 1497: 3, 1549: 8, 1573: 6, 1821: 8, 1837: 8
    },
  ],
  CAR.LEAF_IC: [
    {
      2: 5, 42: 6, 264: 3, 282: 8, 361: 8, 372: 8, 384: 8, 389: 8, 403: 8, 459: 7, 460: 4, 470: 8, 520: 1, 569: 8, 581: 8, 634: 7, 640: 8, 643: 5, 644: 8, 645: 8, 646: 5, 658: 8, 682: 8, 683: 8, 689: 8, 756: 5, 758: 3, 761: 2, 783: 3, 830: 2, 852: 8, 853: 8, 856: 8, 861: 8, 943: 8, 944: 1, 1001: 6, 1057: 3, 1227: 8, 1228: 8, 1229: 8, 1342: 1, 1354: 8, 1361: 8, 1459: 8, 1477: 8, 1497: 3, 1514: 6, 1549: 8, 1573: 6, 1792: 8, 1821: 8, 1822: 8, 1837: 8, 1838: 8, 1872: 8, 1937: 8, 1953: 8, 1968: 8, 1988: 8, 2000: 8, 2001: 8, 2004: 8, 2005: 8, 2015: 8, 2016: 8, 2017: 8
    },
  ],
  CAR.ROGUE: [
    {
      2: 5, 42: 6, 346: 6, 347: 5, 348: 8, 349: 7, 361: 8, 386: 8, 389: 8, 397: 8, 398: 8, 403: 8, 520: 2, 523: 6, 548: 8, 634: 7, 643: 5, 645: 8, 658: 8, 665: 8, 666: 8, 674: 2, 682: 8, 683: 8, 689: 8, 723: 8, 758: 3, 772: 8, 773: 6, 774: 7, 775: 8, 776: 6, 777: 7, 778: 6, 783: 3, 851: 8, 855: 8, 1041: 8, 1042: 8, 1055: 2, 1104: 4, 1105: 6, 1107: 4, 1108: 8, 1110: 7, 1111: 7, 1227: 8, 1228: 8, 1247: 4, 1266: 8, 1273: 7, 1342: 1, 1376: 6, 1401: 8, 1474: 2, 1497: 3, 1534: 7, 1792: 8, 1821: 8, 1823: 8, 1837: 8, 1839: 8, 1872: 8, 1937: 8, 1953: 8, 1968: 8, 1988: 8, 2000: 8, 2001: 8, 2004: 8, 2005: 8, 2015: 8, 2016: 8, 2017: 8, 2024: 8, 2025: 8
    },
  ],
  CAR.ALTIMA: [
    {
      2: 5, 42: 6, 346: 6, 347: 5, 348: 8, 349: 7, 361: 8, 386: 8, 389: 8, 397: 8, 398: 8, 403: 8, 438: 8, 451: 8, 517: 8, 520: 2, 522: 8, 523: 6, 539: 8, 541: 7, 542: 8, 543: 8, 544: 8, 545: 8, 546: 8, 547: 8, 548: 8, 570: 8, 576: 8, 577: 8, 582: 8, 583: 8, 584: 8, 586: 8, 587: 8, 588: 8, 589: 8, 590: 8, 591: 8, 592: 8, 600: 8, 601: 8, 610: 8, 611: 8, 612: 8, 614: 8, 615: 8, 616: 8, 617: 8, 622: 8, 623: 8, 634: 7, 638: 8, 645: 8, 648: 5, 654: 6, 658: 8, 659: 8, 660: 8, 661: 8, 665: 8, 666: 8, 674: 2, 675: 8, 676: 8, 682: 8, 683: 8, 684: 8, 685: 8, 686: 8, 687: 8, 689: 8, 690: 8, 703: 8, 708: 7, 709: 7, 711: 7, 712: 7, 713: 7, 714: 8, 715: 8, 716: 8, 717: 7, 718: 7, 719: 7, 720: 7, 723: 8, 726: 7, 727: 7, 728: 7, 735: 8, 746: 8, 748: 6, 749: 6, 750: 8, 758: 3, 772: 8, 773: 6, 774: 7, 775: 8, 776: 6, 777: 7, 778: 6, 779: 7, 781: 7, 782: 7, 783: 3, 851: 8, 855: 5, 1001: 6, 1041: 8, 1042: 8, 1055: 3, 1100: 7, 1104: 4, 1105: 6, 1107: 4, 1108: 8, 1110: 7, 1111: 7, 1144: 7, 1145: 7, 1227: 8, 1228: 8, 1229: 8, 1232: 8, 1247: 4, 1258: 8, 1259: 8, 1266: 8, 1273: 7, 1306: 1, 1314: 8, 1323: 8, 1324: 8, 1342: 1, 1376: 8, 1401: 8, 1454: 8, 1497: 3, 1514: 6, 1526: 8, 1527: 5, 1792: 8, 1821: 8, 1823: 8, 1837: 8, 1872: 8, 1937: 8, 1953: 8, 1968: 8, 1988: 8, 2000: 8, 2001: 8, 2004: 8, 2005: 8, 2015: 8, 2016: 8, 2017: 8, 2024: 8, 2025: 8
    },
  ]
}

NISSAN_DIAGNOSTIC_REQUEST_KWP = bytes([uds.SERVICE_TYPE.DIAGNOSTIC_SESSION_CONTROL, 0xc0])
NISSAN_DIAGNOSTIC_RESPONSE_KWP = bytes([uds.SERVICE_TYPE.DIAGNOSTIC_SESSION_CONTROL + 0x40, 0xc0])

NISSAN_VERSION_REQUEST_KWP = b'\x21\x83'
NISSAN_VERSION_RESPONSE_KWP = b'\x61\x83'

NISSAN_RX_OFFSET = 0x20

FW_QUERY_CONFIG = FwQueryConfig(
  requests=[
    Request(
      [NISSAN_DIAGNOSTIC_REQUEST_KWP, NISSAN_VERSION_REQUEST_KWP],
      [NISSAN_DIAGNOSTIC_RESPONSE_KWP, NISSAN_VERSION_RESPONSE_KWP],
    ),
    Request(
      [NISSAN_DIAGNOSTIC_REQUEST_KWP, NISSAN_VERSION_REQUEST_KWP],
      [NISSAN_DIAGNOSTIC_RESPONSE_KWP, NISSAN_VERSION_RESPONSE_KWP],
      rx_offset=NISSAN_RX_OFFSET,
    ),
    Request(
      [StdQueries.MANUFACTURER_SOFTWARE_VERSION_REQUEST],
      [StdQueries.MANUFACTURER_SOFTWARE_VERSION_RESPONSE],
      rx_offset=NISSAN_RX_OFFSET,
    ),
  ],
)

FW_VERSIONS = {
  CAR.ALTIMA: {
    (Ecu.fwdCamera, 0x707, None): [
      b'284N86CA1D',
    ],
    (Ecu.eps, 0x742, None): [
      b'6CA2B\xa9A\x02\x02G8A89P90D6A\x00\x00\x01\x80',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'237109HE2B',
    ],
    (Ecu.gateway, 0x18dad0f1, None): [
      b'284U29HE0A',
    ],
  },
  CAR.LEAF_IC: {
    (Ecu.fwdCamera, 0x707, None): [
      b'5SH1BDB\x04\x18\x00\x00\x00\x00\x00_-?\x04\x91\xf2\x00\x00\x00\x80',
      b'5SK0ADB\x04\x18\x00\x00\x00\x00\x00_(5\x07\x9aQ\x00\x00\x00\x80',
      b'5SH4BDB\x04\x18\x00\x00\x00\x00\x00_-?\x04\x91\xf2\x00\x00\x00\x80',
    ],
    (Ecu.abs, 0x740, None): [
      b'476605SH1D',
      b'476605SK2A',
      b'476605SD2E',
    ],
    (Ecu.eps, 0x742, None): [
      b'5SH2A\x99A\x05\x02N123F\x15\x81\x00\x00\x00\x00\x00\x00\x00\x80',
      b'5SK3A\x99A\x05\x02N123F\x15u\x00\x00\x00\x00\x00\x00\x00\x80',
      b'5SH2C\xb7A\x05\x02N123F\x15\xa3\x00\x00\x00\x00\x00\x00\x00\x80',
    ],
    (Ecu.gateway, 0x18dad0f1, None): [
      b'284U25SH3A',
      b'284U25SK2D',
      b'284U25SF0C',
    ],
  },
  CAR.XTRAIL: {
    (Ecu.fwdCamera, 0x707, None): [
      b'284N86FR2A',
    ],
    (Ecu.abs, 0x740, None): [
      b'6FU1BD\x11\x02\x00\x02e\x95e\x80iX#\x01\x00\x00\x00\x00\x00\x80',
      b'6FU0AD\x11\x02\x00\x02e\x95e\x80iQ#\x01\x00\x00\x00\x00\x00\x80',
    ],
    (Ecu.eps, 0x742, None): [
      b'6FP2A\x99A\x05\x02N123F\x18\x02\x00\x00\x00\x00\x00\x00\x00\x80',
    ],
    (Ecu.combinationMeter, 0x743, None): [
      b'6FR2A\x18B\x05\x17\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'6FU9B\xa0A\x06\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80',
      b'6FR9A\xa0A\x06\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80',
    ],
    (Ecu.gateway, 0x18dad0f1, None): [
      b'284U26FR0E',
    ],
  },
}

DBC = {
  CAR.XTRAIL: dbc_dict('nissan_x_trail_2017', None),
  CAR.LEAF: dbc_dict('nissan_leaf_2018', None),
  CAR.LEAF_IC: dbc_dict('nissan_leaf_2018', None),
  CAR.ROGUE: dbc_dict('nissan_x_trail_2017', None),
  CAR.ALTIMA: dbc_dict('nissan_x_trail_2017', None),
}
