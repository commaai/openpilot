# flake8: noqa

from collections import namedtuple
from selfdrive.car import dbc_dict
from cereal import car

Button = namedtuple('Button', ['event_type', 'can_addr', 'can_msg', 'values'])
AngleRateLimit = namedtuple('AngleRateLimit', ['speed_points', 'max_angle_diff_points'])

class CAR:
  AP2_MODELS = 'TESLA AP2 MODEL S'

FINGERPRINTS = {
  CAR.AP2_MODELS: [
    {
      1: 8, 3: 8, 14: 8, 21: 4, 69: 8, 109: 4, 257: 3, 264: 8, 277: 6, 280: 6, 293: 4, 296: 4, 309: 5, 325: 8, 328: 5, 336: 8, 341: 8, 360: 7, 373: 8, 389: 8, 415: 8, 513: 5, 516: 8, 518: 8, 520: 4, 522: 8, 524: 8, 526: 8, 532: 3, 536: 8, 537: 3, 542: 8, 551: 5, 552: 2, 556: 8, 558: 8, 568: 8, 569: 8, 574: 8, 577: 8, 582: 5, 583: 8, 584: 4, 585: 8, 590: 8, 601: 8, 606: 8, 608: 1, 622: 8, 627: 6, 638: 8, 641: 8, 643: 8, 692: 8, 693: 8, 695: 8, 696: 8, 697: 8, 699: 8, 700: 8, 701: 8, 702: 8, 703: 8, 704: 8, 708: 8, 709: 8, 710: 8, 711: 8, 712: 8, 728: 8, 744: 8, 760: 8, 772: 8, 775: 8, 776: 8, 777: 8, 778: 8, 782: 8, 788: 8, 791: 8, 792: 8, 796: 2, 797: 8, 798: 6, 799: 8, 804: 8, 805: 8, 807: 8, 808: 1, 811: 8, 812: 8, 813: 8, 814: 5, 815: 8, 820: 8, 823: 8, 824: 8, 829: 8, 830: 5, 836: 8, 840: 8, 845: 8, 846: 5, 848: 8, 852: 8, 853: 8, 856: 4, 857: 6, 861: 8, 862: 5, 872: 8, 876: 8, 877: 8, 879: 8, 880: 8, 882: 8, 884: 8, 888: 8, 893: 8, 894: 8, 901: 6, 904: 3, 905: 8, 906: 8, 908: 2, 909: 8, 910: 8, 912: 8, 920: 8, 921: 8, 925: 4, 926: 6, 936: 8, 941: 8, 949: 8, 952: 8, 953: 6, 968: 8, 969: 6, 970: 8, 971: 8, 977: 8, 984: 8, 987: 8, 990: 8, 1000: 8, 1001: 8, 1006: 8, 1007: 8, 1008: 8, 1010: 6, 1014: 1, 1015: 8, 1016: 8, 1017: 8, 1018: 8, 1020: 8, 1026: 8, 1028: 8, 1029: 8, 1030: 8, 1032: 1, 1033: 1, 1034: 8, 1048: 1, 1049: 8, 1061: 8, 1064: 8, 1065: 8, 1070: 8, 1080: 8, 1081: 8, 1097: 8, 1113: 8, 1129: 8, 1145: 8, 1160: 4, 1177: 8, 1281: 8, 1328: 8, 1329: 8, 1332: 8, 1335: 8, 1337: 8, 1353: 8, 1368: 8, 1412: 8, 1436: 8, 1476: 8, 1481: 8, 1497: 8, 1513: 8, 1519: 8, 1601: 8, 1605: 8, 1617: 8, 1621: 8, 1800: 4, 1804: 8, 1812: 8, 1815: 8, 1816: 8, 1824: 8, 1828: 8, 1831: 8, 1832: 8, 1864: 8, 1880: 8, 1892: 8, 1896: 8, 1912: 8, 1960: 8, 1992: 8, 2008: 3, 2043: 5, 2045: 4
    },
  ],
}

DBC = {
  CAR.AP2_MODELS: dbc_dict(None, 'tesla_radar', chassis_dbc='tesla_can'),
}

class CANBUS:
  chassis = 0
  autopilot = 2
  radar = 1

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
  Button(car.CarState.ButtonEvent.Type.cancel, "STW_ACTN_RQ", "SpdCtrlLvr_Stat", [2]),
  Button(car.CarState.ButtonEvent.Type.resumeCruise, "STW_ACTN_RQ", "SpdCtrlLvr_Stat", [1]),
]

class CarControllerParams:
  RATE_LIMIT_UP = AngleRateLimit(speed_points=[0., 5., 15.], max_angle_diff_points=[5., .8, .15])
  RATE_LIMIT_DOWN = AngleRateLimit(speed_points=[0., 5., 15.], max_angle_diff_points=[5., 3.5, 0.4])