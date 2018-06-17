from selfdrive.car import dbc_dict

class CAR:
  MODELS = "TESLA MODEL S"


FINGERPRINTS = {
  CAR.MODELS: [{
    1: 8, 3: 8, 14: 8, 21: 4, 69: 8, 109: 4, 257: 3, 264: 8, 267: 5, 277: 4, 280: 6, 283: 5, 293: 4, 296: 4, 309: 5, 336: 8, 341: 8, 357: 8, 360: 7, 415: 8, 513: 5, 516: 8, 520: 4, 522: 8, 524: 8, 527: 8, 536: 8, 551: 4, 552: 2, 556: 8, 568: 8, 582: 5, 638: 8, 643: 8, 693: 8, 696: 8, 712: 8, 728: 8, 744: 8, 760: 8, 771: 2, 772: 8, 775: 8, 776: 8, 778: 8, 785: 8, 780: 2, 788: 8, 791: 8, 792: 8, 796: 2, 798: 6, 799: 8, 804: 8, 807: 8, 808: 1, 812: 8, 814: 5, 815: 8, 820: 8, 823: 8, 824: 8, 830: 5, 836: 8, 840: 8, 856: 4, 872: 8, 879: 8, 880: 8, 888: 8, 896: 8, 904: 3, 920: 8, 936: 8, 952: 8, 953: 6, 984: 8, 1026: 8, 1028: 8, 1029: 8, 1030: 8, 1032: 1, 1034: 8, 1048: 1, 1281: 8, 1332: 8, 1335: 8, 1368: 8, 1436: 8, 1456: 8, 1519: 8, 1804: 8, 1812: 8, 1815: 8, 1824: 1, 1828: 8, 1831: 8, 1832: 8, 1864: 8, 1880: 8, 1892: 8, 1912: 8, 1960: 8, 1992: 8, 2008: 3, 2043: 5
  }],
}


DBC = {
  CAR.MODELS: dbc_dict('tesla_can', None),
}


# Car button codes
class CruiseButtons:
  # VAL_ 69 SpdCtrlLvr_Stat 32 "DN_1ST" 16 "UP_1ST" 8 "DN_2ND" 4 "UP_2ND" 2 "RWD" 1 "FWD" 0 "IDLE" ;
  RES_ACCEL   = 16
  DECEL_SET   = 32
  CANCEL      = 1
  MAIN        = 2


#car chimes: enumeration from dbc file. Chimes are for alerts and warnings
class CM:
  MUTE = 0
  SINGLE = 3
  DOUBLE = 4
  REPEATED = 1
  CONTINUOUS = 2


#car beepss: enumeration from dbc file. Beeps are for activ and deactiv
class BP:
  MUTE = 0
  SINGLE = 3
  TRIPLE = 2
  REPEATED = 1

class AH:
  #[alert_idx, value]
  # See dbc files for info on values"
  NONE           = [0, 0]
  FCW            = [1, 1]
  STEER          = [2, 1]
  BRAKE_PRESSED  = [3, 10]
  GEAR_NOT_D     = [4, 6]
  SEATBELT       = [5, 5]
  SPEED_TOO_HIGH = [6, 8]
