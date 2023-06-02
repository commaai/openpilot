from collections import namedtuple
from typing import Dict, List, Union

from cereal import car
from selfdrive.car import AngleRateLimit, dbc_dict
from selfdrive.car.docs_definitions import CarInfo
from selfdrive.car.fw_query_definitions import FwQueryConfig, Request, StdQueries

Ecu = car.CarParams.Ecu

Button = namedtuple('Button', ['event_type', 'can_addr', 'can_msg', 'values'])


class CAR:
  AP1_MODELS = 'TESLA AP1 MODEL S'
  AP2_MODELS = 'TESLA AP2 MODEL S'


CAR_INFO: Dict[str, Union[CarInfo, List[CarInfo]]] = {
  CAR.AP1_MODELS: CarInfo("Tesla AP1 Model S", "All"),
  CAR.AP2_MODELS: CarInfo("Tesla AP2 Model S", "All"),
}

FINGERPRINTS = {
  CAR.AP1_MODELS: [
    {
      1: 8, 3: 8, 14: 8, 21: 4, 69: 8, 109: 4, 257: 3, 264: 8, 267: 5, 277: 6, 280: 6, 283: 5, 293: 4, 296: 4, 309: 5, 325: 8, 328: 5, 336: 8, 341: 8, 360: 7, 373: 8, 389: 8, 415: 8, 513: 5, 516: 8, 520: 4, 522: 8, 524: 8, 526: 8, 532: 3, 536: 8, 537: 3, 542: 8, 551: 5, 552: 2, 556: 8, 558: 8, 568: 8, 569: 8, 574: 8, 577: 8, 582: 5, 584: 4, 585: 8, 590: 8, 606: 8, 622: 8, 627: 6, 638: 8, 641: 8, 643: 8, 660: 5, 693: 8, 696: 8, 697: 8, 712: 8, 728: 8, 744: 8, 760: 8, 772: 8, 775: 8, 776: 8, 777: 8, 778: 8, 782: 8, 788: 8, 791: 8, 792: 8, 796: 2, 797: 8, 798: 6, 799: 8, 804: 8, 805: 8, 807: 8, 808: 1, 809: 8, 812: 8, 813: 8, 814: 5, 815: 8, 820: 8, 823: 8, 824: 8, 829: 8, 830: 5, 836: 8, 840: 8, 841: 8, 845: 8, 846: 5, 852: 8, 856: 4, 857: 6, 861: 8, 862: 5, 872: 8, 873: 8, 877: 8, 878: 8, 879: 8, 880: 8, 884: 8, 888: 8, 889: 8, 893: 8, 896: 8, 901: 6, 904: 3, 905: 8, 908: 2, 909: 8, 920: 8, 921: 8, 925: 4, 936: 8, 937: 8, 941: 8, 949: 8, 952: 8, 953: 6, 957: 8, 968: 8, 973: 8, 984: 8, 987: 8, 989: 8, 990: 8, 1000: 8, 1001: 8, 1006: 8, 1016: 8, 1026: 8, 1028: 8, 1029: 8, 1030: 8, 1032: 1, 1033: 1, 1034: 8, 1048: 1, 1064: 8, 1070: 8, 1080: 8, 1160: 4, 1281: 8, 1329: 8, 1332: 8, 1335: 8, 1337: 8, 1368: 8, 1412: 8, 1436: 8, 1465: 8, 1476: 8, 1497: 8, 1524: 8, 1527: 8, 1601: 8, 1605: 8, 1611: 8, 1614: 8, 1617: 8, 1621: 8, 1627: 8, 1630: 8, 1800: 4, 1804: 8, 1812: 8, 1815: 8, 1816: 8, 1828: 8, 1831: 8, 1832: 8, 1840: 8, 1848: 8, 1864: 8, 1880: 8, 1892: 8, 1896: 8, 1912: 8, 1960: 8, 1992: 8, 2008: 3, 2043: 5, 2045: 4
    },
  ],
}

DBC = {
  CAR.AP2_MODELS: dbc_dict('tesla_powertrain', 'tesla_radar', chassis_dbc='tesla_can'),
  CAR.AP1_MODELS: dbc_dict('tesla_powertrain', 'tesla_radar', chassis_dbc='tesla_can'),
}

FW_QUERY_CONFIG = FwQueryConfig(
  requests=[
    Request(
      [StdQueries.TESTER_PRESENT_REQUEST, StdQueries.UDS_VERSION_REQUEST],
      [StdQueries.TESTER_PRESENT_RESPONSE, StdQueries.UDS_VERSION_RESPONSE],
      whitelist_ecus=[Ecu.eps],
      rx_offset=0x08,
      bus=0,
    ),
    Request(
      [StdQueries.TESTER_PRESENT_REQUEST, StdQueries.UDS_VERSION_REQUEST],
      [StdQueries.TESTER_PRESENT_RESPONSE, StdQueries.UDS_VERSION_RESPONSE],
      whitelist_ecus=[Ecu.adas, Ecu.electricBrakeBooster, Ecu.fwdRadar],
      rx_offset=0x10,
      bus=0,
    ),
  ]
)

FW_VERSIONS = {
  CAR.AP2_MODELS: {
    (Ecu.adas, 0x649, None): [
      b'\x01\x00\x8b\x07\x01\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x11',
    ],
    (Ecu.electricBrakeBooster, 0x64d, None): [
      b'1037123-00-A',
    ],
    (Ecu.fwdRadar, 0x671, None): [
      b'\x01\x00W\x00\x00\x00\x07\x00\x00\x00\x00\x08\x01\x00\x00\x00\x07\xff\xfe',
    ],
    (Ecu.eps, 0x730, None): [
      b'\x10#\x01',
    ],
  },
}

class CANBUS:
  # Lateral harness
  chassis = 0
  radar = 1
  autopilot_chassis = 2

  # Longitudinal harness
  powertrain = 4
  private = 5
  autopilot_powertrain = 6

GEAR_MAP = {
  "DI_GEAR_INVALID": car.CarState.GearShifter.unknown,
  "DI_GEAR_P": car.CarState.GearShifter.park,
  "DI_GEAR_R": car.CarState.GearShifter.reverse,
  "DI_GEAR_N": car.CarState.GearShifter.neutral,
  "DI_GEAR_D": car.CarState.GearShifter.drive,
  "DI_GEAR_SNA": car.CarState.GearShifter.unknown,
}

DOORS = ["DOOR_STATE_FL", "DOOR_STATE_FR", "DOOR_STATE_RL", "DOOR_STATE_RR", "DOOR_STATE_FrontTrunk", "BOOT_STATE"]

# Make sure the message and addr is also in the CAN parser!
BUTTONS = [
  Button(car.CarState.ButtonEvent.Type.leftBlinker, "STW_ACTN_RQ", "TurnIndLvr_Stat", [1]),
  Button(car.CarState.ButtonEvent.Type.rightBlinker, "STW_ACTN_RQ", "TurnIndLvr_Stat", [2]),
  Button(car.CarState.ButtonEvent.Type.accelCruise, "STW_ACTN_RQ", "SpdCtrlLvr_Stat", [4, 16]),
  Button(car.CarState.ButtonEvent.Type.decelCruise, "STW_ACTN_RQ", "SpdCtrlLvr_Stat", [8, 32]),
  Button(car.CarState.ButtonEvent.Type.cancel, "STW_ACTN_RQ", "SpdCtrlLvr_Stat", [1]),
  Button(car.CarState.ButtonEvent.Type.resumeCruise, "STW_ACTN_RQ", "SpdCtrlLvr_Stat", [2]),
]

class CarControllerParams:
  ANGLE_RATE_LIMIT_UP = AngleRateLimit(speed_bp=[0., 5., 15.], angle_v=[10., 1.6, .3])
  ANGLE_RATE_LIMIT_DOWN = AngleRateLimit(speed_bp=[0., 5., 15.], angle_v=[10., 7.0, 0.8])
  JERK_LIMIT_MAX = 8
  JERK_LIMIT_MIN = -8
  ACCEL_TO_SPEED_MULTIPLIER = 3

  def __init__(self, CP):
    pass
