from selfdrive.car import dbc_dict

def get_hud_alerts(visual_alert, audble_alert):
  if visual_alert == "steerRequired":
    return 4 if audble_alert != "none" else 5
  else:
    return 0

class CAR:
  SANTA_FE = "HYUNDAI SANTA FE LIMITED 2019"
  SORENTO = "KIA SORENTO GT LINE 2018"         # Top Trim Kia Sorento for Australian Market, AWD Diesel 8sp Auto

class Buttons:
  NONE = 0
  RES_ACCEL = 1
  SET_DECEL = 2
  CANCEL = 4

FINGERPRINTS = {
  CAR.SORENTO: [{
    67: 8, 68: 8, 127: 8, 304: 8, 320: 8, 339: 8, 356: 4, 544: 8, 593: 8, 608: 8, 688: 5, 809: 8, 832: 8, 854: 7, 870: 7, 871: 8, 872: 8, 897: 8, 902: 8, 903: 8, 916: 8, 1040: 8, 1042: 8, 1056: 8, 1057: 8, 1064: 8, 1078: 4, 1107: 5, 1136: 8, 1151: 6, 1168: 7, 1170: 8, 1173: 8, 1265: 4, 1280: 1, 1287: 4, 1290: 8, 1292: 8, 1294: 8, 1312: 8, 1322: 8, 1331: 8, 1332: 8, 1333: 8, 1342: 6, 1345: 8, 1348: 8, 1363: 8, 1369: 8, 1370: 8, 1371: 8, 1384: 8, 1407: 8, 1411: 8, 1419: 8, 1425: 2, 1427: 6, 1444: 8, 1456: 4, 1470: 8, 1489: 1
  }],
  CAR.SANTA_FE: [{
    67: 8, 127: 8, 304: 8, 320: 8, 339: 8, 356: 4, 544: 8, 593: 8, 608: 8, 688: 6, 809: 8, 832: 8, 854: 7, 870: 7, 871: 8, 872: 8, 897: 8, 902: 8, 903: 8, 905: 8, 909: 8, 916: 8, 1040: 8, 1042: 8, 1056: 8, 1057: 8, 1078: 4, 1107: 5, 1136: 8, 1151: 6, 1155: 8, 1156: 8, 1162: 8, 1164: 8, 1168: 7, 1170: 8, 1173: 8, 1183: 8, 1186: 2, 1191: 2, 1227: 8, 1265: 4, 1280: 1, 1287: 4, 1290: 8, 1292: 8, 1294: 8, 1312: 8, 1322: 8, 1342: 6, 1345: 8, 1348: 8, 1363: 8, 1369: 8, 1379: 8, 1384: 8, 1407: 8, 1414: 3, 1419: 8, 1427: 6, 1456: 4, 1470: 8
  }],
}

#CAMERA_MSGS = {
#  CAR.SANTA_FE: [832, 1156, 1191, 1342],   # msgs sent by the camera
CAMERA_MSGS = [832, 1156]
#}

CHECKSUM = {
  CAR.SANTA_FE: 0,
  CAR.SORENTO: 6,
}

DBC = {
  CAR.SANTA_FE: dbc_dict('hyundai_santa_fe_2019_ccan', None),
  CAR.SORENTO: dbc_dict('hyundai_santa_fe_2019_ccan', None),
}

