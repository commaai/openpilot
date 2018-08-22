from selfdrive.car import dbc_dict

class CAR:
  MODELS = "TESLA MODEL S"


FINGERPRINTS = {
  CAR.MODELS: [{
    1: 8, 3: 8, 14: 8, 21: 4, 69: 8, 109: 4, 257: 3, 264: 8, 267: 5, 277: 4, 280: 6, 283: 5, 293: 4, 296: 4, 309: 5, 336: 8, 341: 8, 357: 8, 360: 7, 415: 8, 513: 5, 516: 8, 520: 4, 522: 8, 524: 8, 526: 8, 527: 8, 536: 8, 542: 8, 551: 4, 552: 2, 556: 8, 558: 8, 568: 8, 574: 8, 582: 5, 590: 8, 606: 8, 622: 8, 638: 8, 643: 8, 693: 8, 696: 8, 712: 8, 728: 8, 744: 8, 760: 8, 771: 2, 772: 8, 775: 8, 776: 8, 778: 8, 780: 2, 782: 8, 783: 8, 785: 8, 787: 8, 788: 8, 791: 8, 792: 8, 796: 2, 798: 6, 799: 8, 804: 8, 807: 8, 808: 1, 812: 8, 814: 5, 815: 8, 820: 8, 823: 8, 824: 8, 830: 5, 831: 8, 836: 8, 840: 8, 846: 5, 856: 4, 862: 5, 863: 8, 872: 8, 878: 8, 879: 8, 880: 8, 888: 8, 896: 8, 904: 3, 920: 8, 936: 8, 952: 8, 953: 6, 984: 8, 1026: 8, 1028: 8, 1029: 8, 1030: 8, 1032: 1, 1034: 8, 1048: 1, 1281: 8, 1332: 8, 1335: 8, 1361: 6, 1362: 6, 1368: 8, 1412: 8, 1436: 8, 1456: 8, 1486: 8, 1519: 8, 1524: 8, 1527: 8, 1601: 8, 1605: 8, 1611: 8, 1614: 8, 1617: 8, 1621: 8, 1627: 8, 1630: 8, 1800: 4, 1804: 8, 1812: 8, 1815: 8, 1816: 8, 1824: 1, 1828: 8, 1831: 8, 1832: 8, 1840: 8, 1848: 8, 1864: 8, 1880: 8, 1892: 8, 1896: 8, 1912: 8, 1960: 8, 1992: 8, 2008: 3, 2043: 5
  }],
}


DBC = {
  CAR.MODELS: dbc_dict('tesla_can', None),
}


# Car button codes
class CruiseButtons:
  # VAL_ 69 SpdCtrlLvr_Stat 32 "DN_1ST" 16 "UP_1ST" 8 "DN_2ND" 4 "UP_2ND" 2 "RWD" 1 "FWD" 0 "IDLE" ;
  RES_ACCEL     = 16
  RES_ACCEL_2ND = 4
  DECEL_SET     = 32
  DECEL_2ND     = 8
  CANCEL        = 1
  MAIN          = 2
  IDLE          = 0
  
  @classmethod
  def is_accel(cls, btn):
    return btn in [cls.RES_ACCEL, cls.RES_ACCEL_2ND]
    
  @classmethod
  def is_decel(cls, btn):
    return btn in [cls.DECEL_SET, cls.DECEL_2ND]
    
  @classmethod
  def should_be_throttled(cls, btn):
    # Some buttons should not be spammed or they may overwhelm the SCCM.
    return btn not in [cls.MAIN, cls.IDLE]


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

class CruiseState:
  # DI_cruiseState from the DBC
  OFF = 0
  STANDBY = 1
  ENABLED = 2
  STANDSTILL = 3
  OVERRIDE = 4
  FAULT = 5
  PRE_FAULT = 6
  PRE_CANCEL = 7
  
  @classmethod
  def is_enabled_or_standby(cls, state):
    return state in [cls.ENABLED, cls.STANDBY]
    
  @classmethod
  def is_faulted(cls, state):
    return state in [cls.PRE_FAULT, cls.FAULT]
