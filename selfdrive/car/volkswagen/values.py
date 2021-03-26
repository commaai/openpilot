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

TransmissionType = car.CarParams.TransmissionType
GearShifter = car.CarState.GearShifter

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

# Check the 7th and 8th characters of the VIN before adding a new CAR. If the
# chassis code is already listed below, don't add a new CAR, just add to the
# FW_VERSIONS for that existing CAR.

class CAR:
  GOLF = "VOLKSWAGEN GOLF"                    # Chassis 5G/AU/BA/BE, Mk7 VW Golf and variants
  JETTA_MK7 = "VOLKSWAGEN JETTA 7TH GEN"      # Chassis BU, Mk7 Jetta
  TIGUAN_MK2 = "VOLKSWAGEN TIGUAN 2ND GEN"    # Chassis AD/BW, Mk2 VW Tiguan and variants
  SEAT_ATECA_MK1 = "SEAT ATECA 1ST GEN"       # Chassis 5F, Mk1 SEAT Ateca and CUPRA Ateca
  SKODA_KODIAQ_MK1 = "SKODA KODIAQ 1ST GEN"   # Chassis NS, Mk1 Skoda Kodiaq
  AUDI_A3 = "AUDI A3"                         # Chassis 8V/FF, Mk3 Audi A3 and variants

FINGERPRINTS = {
  CAR.GOLF: [{
    64: 8, 134: 8, 159: 8, 173: 8, 178: 8, 253: 8, 257: 8, 260: 8, 262: 8, 264: 8, 278: 8, 279: 8, 283: 8, 286: 8, 288: 8, 289: 8, 290: 8, 294: 8, 299: 8, 302: 8, 346: 8, 385: 8, 418: 8, 427: 8, 668: 8, 679: 8, 681: 8, 695: 8, 779: 8, 780: 8, 783: 8, 792: 8, 795: 8, 804: 8, 806: 8, 807: 8, 808: 8, 809: 8, 870: 8, 896: 8, 897: 8, 898: 8, 901: 8, 917: 8, 919: 8, 927: 8, 949: 8, 958: 8, 960: 4, 981: 8, 987: 8, 988: 8, 991: 8, 997: 8, 1000: 8, 1019: 8, 1120: 8, 1122: 8, 1123: 8, 1124: 8, 1153: 8, 1162: 8, 1175: 8, 1312: 8, 1385: 8, 1413: 8, 1440: 5, 1514: 8, 1515: 8, 1520: 8, 1529: 8, 1600: 8, 1601: 8, 1603: 8, 1605: 8, 1624: 8, 1626: 8, 1629: 8, 1631: 8, 1646: 8, 1648: 8, 1712: 6, 1714: 8, 1716: 8, 1717: 8, 1719: 8, 1720: 8, 1721: 8
  }],
  CAR.JETTA_MK7: [{
    64: 8, 134: 8, 159: 8, 173: 8, 178: 8, 253: 8, 257: 8, 260: 8, 262: 8, 264: 8, 278: 8, 279: 8, 283: 8, 286: 8, 288: 8, 289: 8, 290: 8, 294: 8, 299: 8, 302: 8, 346: 8, 376: 8, 418: 8, 427: 8, 679: 8, 681: 8, 695: 8, 779: 8, 780: 8, 783: 8, 792: 8, 795: 8, 804: 8, 806: 8, 807: 8, 808: 8, 809: 8, 828: 8, 870: 8, 879: 8, 884: 8, 888: 8, 891: 8, 901: 8, 913: 8, 919: 8, 949: 8, 958: 8, 960: 4, 981: 8, 987: 8, 988: 8, 991: 8, 997: 8, 1000: 8, 1019: 8, 1122: 8, 1123: 8, 1124: 8, 1153: 8, 1156: 8, 1157: 8, 1158: 8, 1162: 8, 1312: 8, 1343: 8, 1385: 8, 1413: 8, 1440: 5, 1471: 4, 1514: 8, 1515: 8, 1520: 8, 1600: 8, 1601: 8, 1603: 8, 1605: 8, 1624: 8, 1626: 8, 1629: 8, 1631: 8, 1635: 8, 1646: 8, 1648: 8, 1712: 6, 1714: 8, 1716: 8, 1717: 8, 1719: 8, 1720: 8
  }],
  CAR.TIGUAN_MK2: [{
    64: 8, 134: 8, 159: 8, 173: 8, 178: 8, 253: 8, 257: 8, 260: 8, 262: 8, 278: 8, 279: 8, 283: 8, 286: 8, 288: 8, 289: 8, 290: 8, 294: 8, 299: 8, 302: 8, 346: 8, 376: 8, 418: 8, 427: 8, 573: 8, 679: 8, 681: 8, 684: 8, 695: 8, 779: 8, 780: 8, 783: 8, 787: 8, 788: 8, 789: 8, 792: 8, 795: 8, 804: 8, 806: 8, 807: 8, 808: 8, 809: 8, 828: 8, 870: 8, 879: 8, 884: 8, 888: 8, 891: 8, 896: 8, 897: 8, 898: 8, 901: 8, 913: 8, 917: 8, 919: 8, 949: 8, 958: 8, 960: 4, 981: 8, 987: 8, 988: 8, 991: 8, 997: 8, 1000: 8, 1019: 8, 1122: 8, 1123: 8, 1124: 8, 1153: 8, 1156: 8, 1157: 8, 1158: 8, 1162: 8, 1175: 8, 1312: 8, 1343: 8, 1385: 8, 1413: 8, 1440: 5, 1471: 4, 1514: 8, 1515: 8, 1520: 8, 1600: 8, 1601: 8, 1603: 8, 1605: 8, 1624: 8, 1626: 8, 1629: 8, 1631: 8, 1635: 8, 1646: 8, 1648: 8, 1712: 6, 1714: 8, 1716: 8, 1717: 8, 1719: 8, 1720: 8, 1721: 8
  }],
  CAR.AUDI_A3: [{
    64: 8, 134: 8, 159: 8, 173: 8, 178: 8, 253: 8, 257: 8, 260: 8, 262: 8, 278: 8, 279: 8, 283: 8, 285: 8, 286: 8, 288: 8, 289: 8, 290: 8, 294: 8, 295: 8, 299: 8, 302: 8, 346: 8, 418: 8, 427: 8, 506: 8, 679: 8, 681: 8, 695: 8, 779: 8, 780: 8, 783: 8, 787: 8, 788: 8, 789: 8, 792: 8, 802: 8, 804: 8, 806: 8, 807: 8, 808: 8, 809: 8, 846: 8, 847: 8, 870: 8, 896: 8, 897: 8, 898: 8, 901: 8, 917: 8, 919: 8, 949: 8, 958: 8, 960: 4, 981: 8, 987: 8, 988: 8, 991: 8, 997: 8, 1000: 8, 1019: 8, 1122: 8, 1123: 8, 1124: 8, 1153: 8, 1162: 8, 1175: 8, 1312: 8, 1385: 8, 1413: 8, 1440: 5, 1514: 8, 1515: 8, 1520: 8, 1600: 8, 1601: 8, 1603: 8, 1624: 8, 1629: 8, 1631: 8, 1646: 8, 1648: 8, 1712: 6, 1714: 8, 1716: 8, 1717: 8, 1719: 8, 1720: 8, 1721: 8, 1792: 8, 1872: 8, 1976: 8, 1977: 8, 1982: 8, 1985: 8
  }],
  CAR.SEAT_ATECA_MK1: [{
    64: 8, 134: 8, 159: 8, 173: 8, 178: 8, 253: 8, 257: 8, 260: 8, 262: 8, 278: 8, 279: 8, 283: 8, 286: 8, 288: 8, 289: 8, 290: 8, 294: 8, 299: 8, 302: 8, 346: 8, 385: 8, 418: 8, 427: 8, 668: 8, 679: 8, 681: 8, 684: 8, 779: 8, 780: 8, 792: 8, 795: 8, 804: 8, 806: 8, 807: 8, 808: 8, 809: 8, 870: 8, 901: 8, 917: 8, 919: 8, 927: 8, 949: 8, 958: 8, 960: 4, 981: 8, 987: 8, 988: 8, 991: 8, 997: 8, 1000: 8, 1019: 8, 1120: 8, 1122: 8, 1123: 8, 1124: 8, 1153: 8, 1162: 8, 1175: 8, 1312: 8, 1385: 8, 1413: 8, 1440: 5, 1514: 8, 1515: 8, 1520: 8, 1600: 8, 1601: 8, 1603: 8, 1605: 8, 1624: 8, 1626: 8, 1629: 8, 1631: 8, 1646: 8, 1648: 8, 1712: 6, 1714: 8, 1716: 8, 1717: 8, 1719: 8, 1720: 8, 1721: 8
  }],
  CAR.SKODA_KODIAQ_MK1: [{
    64: 8, 134: 8, 159: 8, 173: 8, 178: 8, 253: 8, 257: 8, 260: 8, 262: 8, 278: 8, 279: 8, 283: 8, 286: 8, 288: 8, 289: 8, 290: 8, 294: 8, 299: 8, 302: 8, 346: 8, 385: 8, 418: 8, 427: 8, 573: 8, 668: 8, 679: 8, 681: 8, 684: 8, 695: 8, 779: 8, 780: 8, 783: 8, 787: 8, 788: 8, 789: 8, 792: 8, 795: 8, 802: 8, 804: 8, 806: 8, 807: 8, 808: 8, 809: 8, 828: 8, 870: 8, 896: 8, 897: 8, 898: 8, 901: 8, 917: 8, 919: 8, 949: 8, 958: 8, 960: 4, 981: 8, 987: 8, 988: 8, 991: 8, 997: 8, 1000: 8, 1019: 8, 1120: 8, 1153: 8, 1162: 8, 1175: 8, 1312: 8, 1385: 8, 1413: 8, 1440: 5, 1514: 8, 1515: 8, 1520: 8, 1529: 8, 1600: 8, 1601: 8, 1603: 8, 1605: 8, 1624: 8, 1626: 8, 1629: 8, 1631: 8, 1646: 8, 1648: 8, 1712: 6, 1714: 8, 1716: 8, 1717: 8, 1719: 8, 1720: 8, 1721: 8, 1792: 8, 1871: 8, 1872: 8, 1879: 8, 1909: 8, 1976: 8, 1977: 8, 1985: 8
  }],
}

IGNORED_FINGERPRINTS = [CAR.JETTA_MK7, CAR.TIGUAN_MK2, CAR.SEAT_ATECA_MK1, CAR.SKODA_KODIAQ_MK1]

FW_VERSIONS = {
  CAR.AUDI_A3: {
    (Ecu.engine, 0x7e0, None): [
      b'\xf1\x875G0906259L \xf1\x890002',
    ],
    (Ecu.transmission, 0x7e1, None): [
      b'\xf1\x870D9300013B \xf1\x894931',
    ],
    (Ecu.srs, 0x715, None): [
      b'\xf1\x875Q0959655J \xf1\x890830\xf1\x82\023121111111211--261117141112231291163221',
    ],
    (Ecu.eps, 0x712, None): [
      b'\xf1\x875Q0909144T \xf1\x891072\xf1\x82\00521G00807A1',
    ],
    (Ecu.fwdRadar, 0x757, None): [
      b'\xf1\x875Q0907572G \xf1\x890571',
    ],
  },
  CAR.GOLF: {
    (Ecu.engine, 0x7e0, None): [
      b'\xf1\x878V0906259P \xf1\x890001',
    ],
    (Ecu.transmission, 0x7e1, None): [
      b'\xf1\x870GC300012A \xf1\x891403',
    ],
    (Ecu.srs, 0x715, None): [
      b'\xf1\x875Q0959655J \xf1\x890830\xf1\x82\x13271212111312--071104171838103891131211',
    ],
    (Ecu.eps, 0x712, None): [
      b'\xf1\x873Q0909144L \xf1\x895081\xf1\x82\x0571A0JA15A1',
    ],
    (Ecu.fwdRadar, 0x757, None): [
      b'\xf1\x875Q0907572J \xf1\x890654',
    ],
  },
  CAR.JETTA_MK7: {
    (Ecu.engine, 0x7e0, None): [
      b'\xf1\x8704E906024B \xf1\x895594',
      b'\xf1\x8704E906024L \xf1\x895595',
      b'\xf1\x8704E906024AK\xf1\x899937',
      b'\xf1\x875G0906259T \xf1\x890003',
    ],
    (Ecu.transmission, 0x7e1, None): [
      b'\xf1\x8709S927158R \xf1\x893552',
      b'\xf1\x8709S927158R \xf1\x893587',
      b'\xf1\x870GC300020N \xf1\x892803',
    ],
    (Ecu.srs, 0x715, None): [
      b'\xf1\x875Q0959655AG\xf1\x890336\xf1\x82\02314171231313500314611011630169333463100',
      b'\xf1\x875Q0959655BR\xf1\x890403\xf1\x82\02311170031313300314240011150119333433100',
      b'\xf1\x875Q0959655BM\xf1\x890403\xf1\x82\02314171231313500314643011650169333463100',
    ],
    (Ecu.eps, 0x712, None): [
      b'\xf1\x875QM909144B \xf1\x891081\xf1\x82\00521A10A01A1',
      b'\xf1\x875QM909144B \xf1\x891081\xf1\x82\x0521B00404A1',
      b'\xf1\x875QM909144C \xf1\x891082\xf1\x82\00521A10A01A1',
      b'\xf1\x875QN909144B \xf1\x895082\xf1\x82\00571A10A11A1',
    ],
    (Ecu.fwdRadar, 0x757, None): [
      b'\xf1\x875Q0907572N \xf1\x890681',
      b'\xf1\x875Q0907572R \xf1\x890771',
    ],
  },
  CAR.TIGUAN_MK2: {
    (Ecu.engine, 0x7e0, None): [
      b'\xf1\x8783A907115B \xf1\x890005',
    ],
    (Ecu.transmission, 0x7e1, None): [
      b'\xf1\x8709G927158DT\xf1\x893698',
    ],
    (Ecu.srs, 0x715, None): [
      b'\xf1\x875Q0959655BM\xf1\x890403\xf1\x82\02316143231313500314641011750179333423100',
    ],
    (Ecu.eps, 0x712, None): [
      b'\xf1\x875QM909144C \xf1\x891082\xf1\x82\00521A60804A1',
    ],
    (Ecu.fwdRadar, 0x757, None): [
      b'\xf1\x872Q0907572R \xf1\x890372',
    ],
  },
  CAR.SEAT_ATECA_MK1: {
    (Ecu.engine, 0x7e0, None): [
      b'\xf1\x8704E906027KA\xf1\x893749',
    ],
    (Ecu.transmission, 0x7e1, None): [
      b'\xf1\x870D9300014S \xf1\x895202',
    ],
    (Ecu.srs, 0x715, None): [
      b'\xf1\x873Q0959655BH\xf1\x890703\xf1\x82\0161212001211001305121211052900',
    ],
    (Ecu.eps, 0x712, None): [
      b'\xf1\x873Q0909144L \xf1\x895081\xf1\x82\00571N60511A1',
    ],
    (Ecu.fwdRadar, 0x757, None): [
      b'\xf1\x872Q0907572M \xf1\x890233',
    ],
  },
  CAR.SKODA_KODIAQ_MK1: {
    (Ecu.engine, 0x7e0, None): [
      b'\xf1\x8704E906027DD\xf1\x893123',
      b'\xf1\x875NA907115E \xf1\x890003',
    ],
    (Ecu.transmission, 0x7e1, None): [
      b'\xf1\x870D9300043  \xf1\x895202',
      b'\xf1\x870DL300012M \xf1\x892107',
    ],
    (Ecu.srs, 0x715, None): [
      b'\xf1\x873Q0959655BJ\xf1\x890703\xf1\x82\0161213001211001205212111052100',
    ],
    (Ecu.eps, 0x712, None): [
      b'\xf1\x875Q0909143P \xf1\x892051\xf1\x820527T6050405',
      b'\xf1\x875Q0909143P \xf1\x892051\xf1\x820527T6060405',
    ],
    (Ecu.fwdRadar, 0x757, None): [
      b'\xf1\x872Q0907572R \xf1\x890372',
    ],
  },
}

DBC = {
  CAR.GOLF: dbc_dict('vw_mqb_2010', None),
  CAR.JETTA_MK7: dbc_dict('vw_mqb_2010', None),
  CAR.TIGUAN_MK2: dbc_dict('vw_mqb_2010', None),
  CAR.AUDI_A3: dbc_dict('vw_mqb_2010', None),
  CAR.SEAT_ATECA_MK1: dbc_dict('vw_mqb_2010', None),
  CAR.SKODA_KODIAQ_MK1: dbc_dict('vw_mqb_2010', None),
}
