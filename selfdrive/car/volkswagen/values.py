# flake8: noqa

from selfdrive.car import dbc_dict
from cereal import car
Ecu = car.CarParams.Ecu

class CarControllerParams:
  HCA_STEP = 2                   # HCA_01 message frequency 50Hz
  LDW_STEP = 10                  # LDW_02 message frequency 10Hz
  GRA_ACC_STEP = 3               # GRA_ACC_01 message frequency 33Hz

  GRA_VBP_STEP = 100             # Send ACC virtual button presses once a second
  GRA_VBP_COUNT = 16             # Send VBP messages for ~0.5s (GRA_ACC_STEP * 16)

  # Observed documented MQB limits: 3.00 Nm max, rate of change 5.00 Nm/sec.
  # Limiting rate-of-change based on real-world testing and Comma's safety
  # requirements for minimum time to lane departure.
  STEER_MAX = 300                # Max heading control assist torque 3.00 Nm
  STEER_DELTA_UP = 4             # Max HCA reached in 1.50s (STEER_MAX / (50Hz * 1.50))
  STEER_DELTA_DOWN = 10          # Min HCA reached in 0.60s (STEER_MAX / (50Hz * 0.60))
  STEER_DRIVER_ALLOWANCE = 80
  STEER_DRIVER_MULTIPLIER = 3    # weight driver torque heavily
  STEER_DRIVER_FACTOR = 1        # from dbc

class CANBUS:
  pt = 0
  cam = 2

BUTTON_STATES = {
  "accelCruise": False,
  "decelCruise": False,
  "cancel": False,
  "setCruise": False,
  "resumeCruise": False,
  "gapAdjustCruise": False
}

MQB_LDW_MESSAGES = {
  "none": 0,                            # Nothing to display
  "laneAssistUnavailChime": 1,          # "Lane Assist currently not available." with chime
  "laneAssistUnavailNoSensorChime": 3,  # "Lane Assist not available. No sensor view." with chime
  "laneAssistTakeOverUrgent": 4,        # "Lane Assist: Please Take Over Steering" with urgent beep
  "emergencyAssistUrgent": 6,           # "Emergency Assist: Please Take Over Steering" with urgent beep
  "laneAssistTakeOverChime": 7,         # "Lane Assist: Please Take Over Steering" with chime
  "laneAssistTakeOverSilent": 8,        # "Lane Assist: Please Take Over Steering" silent
  "emergencyAssistChangingLanes": 9,    # "Emergency Assist: Changing lanes..." with urgent beep
  "laneAssistDeactivated": 10,          # "Lane Assist deactivated." silent with persistent icon afterward
}

class CAR:
  AU_A3_MK3 = "AUDI A3 3RD GEN"
  VW_GOLF_MK7 = "VOLKSWAGEN GOLF 7TH GEN"

MQB_CARS = {
  CAR.AU_A3_MK3,             # Chassis 8V, 2013-2019, includes A3, A3 e-tron, A3 g-tron, S3, RS3
  CAR.VW_GOLF_MK7            # Chassis AU, 2013-2020, includes Golf, Alltrack, Sportwagen, GTI, GTI TCR, GTE, GTD, Clubsport, Golf R
}

FINGERPRINTS = {
  CAR.AU_A3_MK3: [{
    64: 8, 134: 8, 159: 8, 173: 8, 178: 8, 253: 8, 257: 8, 260: 8, 262: 8, 278: 8, 279: 8, 283: 8, 285: 8, 286: 8, 288: 8, 289: 8, 290: 8, 294: 8, 295: 8, 299: 8, 302: 8, 346: 8, 418: 8, 427: 8, 506: 8, 679: 8, 681: 8, 695: 8, 779: 8, 780: 8, 783: 8, 787: 8, 788: 8, 789: 8, 792: 8, 802: 8, 804: 8, 806: 8, 807: 8, 808: 8, 809: 8, 846: 8, 847: 8, 870: 8, 896: 8, 897:  8, 898: 8, 901: 8, 917: 8, 919: 8, 949: 8, 958: 8, 960: 4, 981: 8, 987: 8, 988: 8, 991: 8, 997: 8, 1000: 8, 1019: 8, 1122: 8, 1123: 8, 1124: 8, 1153: 8, 1162: 8, 1175: 8, 1312: 8, 1385: 8, 1413: 8, 1440: 5, 1514: 8, 1515: 8, 1520: 8, 1600: 8, 1601: 8, 1603: 8, 1624: 8, 1629: 8, 1631: 8, 1646: 8, 1648: 8, 1712: 6, 1714: 8, 1716: 8, 1717: 8, 1719: 8, 1720: 8, 1721: 8, 1792: 8, 1872: 8, 1976: 8, 1977: 8, 1982: 8, 1985: 8
  }],
  CAR.VW_GOLF_MK7: [{
    64: 8, 134: 8, 159: 8, 173: 8, 178: 8, 253: 8, 257: 8, 260: 8, 262: 8, 264: 8, 278: 8, 279: 8, 283: 8, 286: 8, 288: 8, 289: 8, 290: 8, 294: 8, 299: 8, 302: 8, 346: 8, 385: 8, 418: 8, 427: 8, 668: 8, 679: 8, 681: 8, 695: 8, 779: 8, 780: 8, 783: 8, 792: 8, 795: 8, 804: 8, 806: 8, 807: 8, 808: 8, 809: 8, 870: 8, 896: 8, 897: 8, 898: 8, 901: 8, 917: 8, 919: 8, 927: 8, 949: 8, 958: 8, 960: 4, 981: 8, 987: 8, 988: 8, 991: 8, 997: 8, 1000: 8, 1019: 8, 1120: 8, 1122: 8, 1123: 8, 1124: 8, 1153: 8, 1162: 8, 1175: 8, 1312: 8, 1385: 8, 1413: 8, 1440: 5, 1514: 8, 1515: 8, 1520: 8, 1529: 8, 1600: 8, 1601: 8, 1603: 8, 1605: 8, 1624: 8, 1626: 8, 1629: 8, 1631: 8, 1646: 8, 1648: 8, 1712: 6, 1714: 8, 1716: 8, 1717: 8, 1719: 8, 1720: 8, 1721: 8
  }],
}

FW_VERSIONS = {
  CAR.VW_GOLF_MK7: {
    # Mk7 2013-2017 and Mk7.5 facelift 2018-2020
    (Ecu.engine, 0x7e0, None): [
      b'8V0906259P ',  # 2018 Golf R (DJJA)
    ],
    (Ecu.transmission, 0x7e1, None): [
      b'0GC300012A ',  # 2018 Golf R (DQ381)
    ],
    (Ecu.esp, 0x713, None): [
      b'5Q0614517CR',  # 2018 Golf R
    ],
    (Ecu.srs, 0x715, None): [
      b'5Q0959655J ',  # 2018 Golf R
    ],
    (Ecu.eps, 0x712, None): [
      b'3Q0909144L ',  # Shared
    ],
    (Ecu.fwdRadar, 0x757, None): [
      b'5Q0907572J ',  # Shared
    ],
    (Ecu.fwdCamera, 0x74f, None): [
      b'3Q0980654H ',  # Shared
    ],
  }
}

DBC = {
  CAR.AU_A3_MK3: dbc_dict('vw_mqb_2010', None),
  CAR.VW_GOLF_MK7: dbc_dict('vw_mqb_2010', None),
}
