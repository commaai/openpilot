from selfdrive.car import dbc_dict

class CAR:
  ELANTRA = "HYUNDAI ELANTRA 2017"
  GENESIS = "HYUNDAI GENESIS 2018"


class ECU:
  CAM = 0 # camera
  DSU = 1 # driving support unit
  APGS = 2 # advanced parking guidance system


# addr: (ecu, cars, bus, 1/freq*100, vl)
# STATIC_MSGS = [(0x141, ECU.DSU, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RXH, CAR.RAV4, CAR.COROLLA), 1,   2, '\x00\x00\x00\x46'),
#                (0x128, ECU.DSU, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RXH, CAR.RAV4, CAR.COROLLA), 1,   3, '\xf4\x01\x90\x83\x00\x37'),
#
#                (0x292, ECU.APGS, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RXH, CAR.RAV4, CAR.COROLLA), 0,   3, '\x00\x00\x00\x00\x00\x00\x00\x9e'),
#                (0x283, ECU.DSU, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RXH, CAR.RAV4, CAR.COROLLA), 0,   3, '\x00\x00\x00\x00\x00\x00\x8c'),
#                (0x2E6, ECU.DSU, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RXH), 0,   3, '\xff\xf8\x00\x08\x7f\xe0\x00\x4e'),
#                (0x2E7, ECU.DSU, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RXH), 0,   3, '\xa8\x9c\x31\x9c\x00\x00\x00\x02'),
#
#                (0x240, ECU.CAM, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RXH, CAR.RAV4, CAR.COROLLA), 1,   5, '\x00\x10\x01\x00\x10\x01\x00'),
#                (0x241, ECU.CAM, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RXH, CAR.RAV4, CAR.COROLLA), 1,   5, '\x00\x10\x01\x00\x10\x01\x00'),
#                (0x244, ECU.CAM, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RXH, CAR.RAV4, CAR.COROLLA), 1,   5, '\x00\x10\x01\x00\x10\x01\x00'),
#                (0x245, ECU.CAM, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RXH, CAR.RAV4, CAR.COROLLA), 1,   5, '\x00\x10\x01\x00\x10\x01\x00'),
#                (0x248, ECU.CAM, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RXH, CAR.RAV4, CAR.COROLLA), 1,   5, '\x00\x00\x00\x00\x00\x00\x01'),
#                (0x344, ECU.DSU, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RXH, CAR.RAV4, CAR.COROLLA), 0,   5, '\x00\x00\x01\x00\x00\x00\x00\x50'),
#
#                (0x160, ECU.DSU, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RXH, CAR.RAV4, CAR.COROLLA), 1,   7, '\x00\x00\x08\x12\x01\x31\x9c\x51'),
#                (0x161, ECU.DSU, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RXH, CAR.RAV4, CAR.COROLLA), 1,   7, '\x00\x1e\x00\x00\x00\x80\x07'),
#
#                (0x32E, ECU.APGS, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RXH, CAR.RAV4, CAR.COROLLA), 0,  20, '\x00\x00\x00\x00\x00\x00\x00\x00'),
#                (0x33E, ECU.DSU, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RXH), 0,  20, '\x0f\xff\x26\x40\x00\x1f\x00'),
#                (0x365, ECU.DSU, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RXH), 0,  20, '\x00\x00\x00\x80\x03\x00\x08'),
#                (0x365, ECU.DSU, (CAR.RAV4, CAR.COROLLA), 0,  20, '\x00\x00\x00\x80\xfc\x00\x08'),
#                (0x366, ECU.DSU, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RXH), 0,  20, '\x00\x00\x4d\x82\x40\x02\x00'),
#                (0x366, ECU.DSU, (CAR.RAV4, CAR.COROLLA), 0,  20, '\x00\x72\x07\xff\x09\xfe\x00'),
#
#                (0x367, ECU.CAM, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RXH, CAR.RAV4, CAR.COROLLA), 0,  40, '\x06\x00'),
#
#                (0x414, ECU.CAM, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RXH, CAR.RAV4, CAR.COROLLA), 0, 100, '\x00\x00\x00\x00\x00\x00\x17\x00'),
#                (0x489, ECU.CAM, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RXH, CAR.RAV4, CAR.COROLLA), 0, 100, '\x00\x00\x00\x00\x00\x00\x00'),
#                (0x48a, ECU.CAM, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RXH, CAR.RAV4, CAR.COROLLA), 0, 100, '\x00\x00\x00\x00\x00\x00\x00'),
#                (0x48b, ECU.CAM, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RXH, CAR.RAV4, CAR.COROLLA), 0, 100, '\x66\x06\x08\x0a\x02\x00\x00\x00'),
#                (0x4d3, ECU.CAM, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RXH, CAR.RAV4, CAR.COROLLA), 0, 100, '\x1C\x00\x00\x01\x00\x00\x00\x00'),
#                (0x130, ECU.CAM, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RXH, CAR.RAV4, CAR.COROLLA), 1, 100, '\x00\x00\x00\x00\x00\x00\x38'),
#                (0x466, ECU.CAM, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RXH, CAR.RAV4), 1, 100, '\x20\x20\xAD'),
#                (0x466, ECU.CAM, (CAR.COROLLA), 1, 100, '\x24\x20\xB1'),
#                (0x396, ECU.APGS, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RXH, CAR.RAV4, CAR.COROLLA), 0, 100, '\xBD\x00\x00\x00\x60\x0F\x02\x00'),
#                (0x43A, ECU.APGS, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RXH, CAR.RAV4, CAR.COROLLA), 0, 100, '\x84\x00\x00\x00\x00\x00\x00\x00'),
#                (0x43B, ECU.APGS, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RXH, CAR.RAV4, CAR.COROLLA), 0, 100, '\x00\x00\x00\x00\x00\x00\x00\x00'),
#                (0x497, ECU.APGS, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RXH, CAR.RAV4, CAR.COROLLA), 0, 100, '\x00\x00\x00\x00\x00\x00\x00\x00'),
#                (0x4CC, ECU.APGS, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RXH, CAR.RAV4, CAR.COROLLA), 0, 100, '\x0D\x00\x00\x00\x00\x00\x00\x00'),
#                (0x4CB, ECU.DSU, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RXH, CAR.RAV4, CAR.COROLLA), 0, 100, '\x0c\x00\x00\x00\x00\x00\x00\x00'),
#                (0x470, ECU.DSU, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RXH), 1, 100, '\x00\x00\x02\x7a'),
#               ]


def check_ecu_msgs(fingerprint, candidate, ecu):
  # return True if fingerprint contains messages normally sent by a given ecu
  ecu_msgs = [x[0] for x in STATIC_MSGS if (x[1] == ecu and
                                            candidate in x[2] and
                                            x[3] == 0)]

  return any(msg for msg in fingerprint if msg in ecu_msgs)


FINGERPRINTS = {
  #REAL CAR.ELANTRA: [{
  #   66: 4, 67: 8, 68: 5, 273: 8, 274: 8, 275: 8, 339: 8, 356: 4, 399: 8, 512: 3, 544: 8, 593: 8, 608: 8, 688: 5, 790: 8, 809: 8, 832: 8, 897: 8, 899: 8, 902: 8, 903: 7, 905: 3, 909: 8, 916: 8, 1040: 5, 1056: 7, 1057: 8, 1170: 8, 1265: 4, 1280: 1, 1290: 1, 1292: 8, 1314: 5, 1322: 5, 1345: 8, 1349: 4, 1351: 8, 1353: 8, 1363: 7, 1366: 6, 1367: 6, 1369: 8, 1407: 1, 1415: 8, 1419: 3, 1425: 1, 1427: 6, 1440: 5, 1456: 3, 1486: 8, 1487: 7, 1491: 1, 1530: 6
  # }],
  CAR.ELANTRA: [{
    66: 8, 67: 8, 68: 8, 127: 8, 273: 8, 274: 8, 275: 8, 339: 8, 356: 8, 399: 8, 512: 8, 544: 8, 593: 8, 608: 8, 688: 8, 790: 8, 809: 8, 832: 8, 897: 8, 899: 8, 902: 8, 903: 8, 905: 8, 909: 8, 916: 8, 1040: 8, 1056: 8, 1057: 8, 1078: 8, 1170: 8, 1265: 8, 1280: 8, 1282: 8, 1287: 8, 1290: 8, 1292: 8, 1294: 8, 1312: 8, 1314: 8, 1322: 8, 1345: 8, 1349: 8, 1351: 8, 1353: 8, 1363: 8, 1366: 8, 1367: 8, 1369: 8, 1407: 8, 1415: 8, 1419: 8, 1425: 8, 1427: 8, 1440: 8, 1456: 8, 1472: 8, 1486: 8, 1487: 8, 1491: 8, 1530: 8,
  }]
  CAR.GENESIS: [{
    67: 8, 68: 8, 304: 8, 320: 8, 339: 8, 356: 4, 544: 7, 593: 8, 608: 8, 688: 5, 809: 8, 832: 8, 854: 7, 870: 7, 871: 8, 872: 5, 897: 8, 902: 8, 903: 6, 916: 8, 1024: 2, 1040: 8, 1056: 8, 1057: 8, 1078: 4, 1107: 5, 1136: 8, 1151: 6, 1168: 7, 1170: 8, 1173: 8, 1265: 4, 1280: 1, 1292: 8, 1312: 8, 1322: 8, 1331: 8, 1332: 8, 1333: 8, 1334: 8, 1335: 8, 1345: 8, 1363: 8, 1369: 8, 1370: 8, 1371: 8, 1378: 4, 1384: 5, 1407: 8, 1419: 8, 1427: 6, 1434: 2, 1456: 4, 2016: 8, 2017: 8, 2024: 8, 2025: 8,
  }],
}


DBC = {
  CAR.ELANTRA: dbc_dict('hyundai_2015_ccan', None), ## TODO: find radar dbc
  CAR.GENESIS: dbc_dict('hyundai_2015_ccan', None),
}
