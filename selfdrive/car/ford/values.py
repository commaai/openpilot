from selfdrive.car import dbc_dict
from cereal import car
Ecu = car.CarParams.Ecu

MAX_ANGLE = 87.  # make sure we never command the extremes (0xfff) which cause latching fault

class CAR:
  #Unsupported Ford Models
  #CMAX = "FORD C-MAX" C-MAX is not supported on any year or trim. 
  #ECOSPORT = "FORD ECOSPORT" EcoSport is not supported on any year or trim. 
  
  #SG denotes Stop/Go, where op can take longitudinal control. 
  
  #Ford models w/ lkas and acc
  EDGE = "FORD EDGE" # 2013+ 2011-2012 has acc only Shares arch with 2013+. May be able to add IPMA to HS2.
  ESCAPE = "FORD ESCAPE" # 2017+
  EXPEDITION = "FORD EXPEDITION" # 2018+
  EXPLORER = "FORD EXPLORER" # 2013+. 2011-2012 has acc only. Shares arch with Explorer 2013+. May be able to add IPMA to HS2.
  F150 = "FORD F150" # 2015-2017
  F150SG = "FORD F150 STOP/GO" # 2018+
  FUSION = "FORD FUSION" # 2013-2017
  FUSIONSG = "FORD FUSION STOP/GO" # 2018+
  MUSTANG = "FORD MUSTANG" # 2018+ 2015-2017 has acc only. 2015-2017 shares arch with Mustang 2018+. May be able to add IPMA to HS2. Not available on Shelby GT350 or GT500.
  MONDEO = "FORD MONDEO" # euro spec fusion. maintained by tambetm
  RANGER = "FORD RANGER" # 2019+
  TAURUS = "FORD TAURUS" # 2014+ 2010-2013 has acc only. 2013 shares arch with Taurus 2014+. May be able to add IPMA to HS2.
  #Ford models with lkas or acc but not both
  FLEX = "FORD FLEX" # Flex does not have LKAS on any year or trim. ACC only 2013+. Uses same arch as Explorer 2013-2015. May be able to add IPMA to HS2.
  FOCUS = "FORD FOCUS" # Focus does not have ACC on any year or trim. Steering only 2015+. Uses same arch as Escape. May be able to add CCM to HS2.
  TRANSIT = "FORD TRANSIT" # 2020+ 2013-2017 has LKAS only.
  #Lincoln models w/ lkas and acc
  AVIATOR = "LINCOLN AVIATOR" # 2020+. 
  CONTINENTAL = "LINCOLN CONTINENTAL" # 2017+
  CORSAIR = "LINCOLN CORSAIR" # Corsair is a rebranded MKC. 2020+
  MKC = "LINCOLN MKC" # 2015+. 
  MKS = "LINCOLN MKS" # 2013+. 2010-2012 is ACC only. Uses Taurus DBC. 
  MKT = "LINCOLN MKT" # 2013+. 2011-2012 is ACC only. Uses Explorer DBC. 
  MKX = "LINCOLN MKX" # 2016+. 2011-2015 is ACC only. May be able to add IPMA to HS2. Uses Edge DBC. 
  MKZ = "LINCOLN MKZ" # 2013+ Uses Fusion DBC
  NAVIGATOR = "LINCOLN NAVIGATOR" # 2018+. Uses Expedition DBC. 
  NAUTILUS = "LINCOLN NAUTILUS" # Nautilus is a rebranded MKX. 2019+

FINGERPRINTS = {
  #Ford
  CAR.FUSION: [{
    71: 8, 74: 8, 75: 8, 76: 8, 90: 8, 92: 8, 93: 8, 118: 8, 119: 8, 120: 8, 125: 8, 129: 8, 130: 8, 131: 8, 132: 8, 133: 8, 145: 8, 146: 8, 357: 8, 359: 8, 360: 8, 361: 8, 376: 8, 390: 8, 391: 8, 392: 8, 394: 8, 512: 8, 514: 8, 516: 8, 531: 8, 532: 8, 534: 8, 535: 8, 560: 8, 578: 8, 604: 8, 613: 8, 673: 8, 827: 8, 848: 8, 934: 8, 935: 8, 936: 8, 947: 8, 963: 8, 970: 8, 972: 8, 973: 8, 984: 8, 992: 8, 994: 8, 997: 8, 998: 8, 1003: 8, 1034: 8, 1045: 8, 1046: 8, 1053: 8, 1054: 8, 1058: 8, 1059: 8, 1068: 8, 1072: 8, 1073: 8, 1082: 8, 1107: 8, 1108: 8, 1109: 8, 1110: 8, 1200: 8, 1427: 8, 1430: 8, 1438: 8, 1459: 8
  }],
  #CAR.EDGE: [{
  #}], 
  #CAR.ESCAPE: [{
  #}], 
  #CAR.EXPEDITION: [{
  #}], 
  #CAR.EXPLORER: [{
  #}], 
  CAR.F150: [{ 
    #2015 Lariat 502A
    74: 8, 75: 8, 76: 8, 118: 8, 119: 8, 120: 8, 125: 8, 129: 8,130: 8, 131: 8, 133: 8, 145: 8, 146: 8, 357: 8, 359: 8, 376: 8, 390: 8, 391: 8, 392: 8, 394: 8, 512: 8, 514: 8, 516: 8, 531: 8, 532: 8, 534: 8, 535: 8, 560: 8, 578: 8, 611: 8, 613: 8, 673: 8, 827: 8, 848: 8, 934: 8, 935: 8, 936: 8, 938: 8, 939: 8, 947: 8, 963: 8, 970: 8, 972: 8, 984: 8, 992: 8, 994: 8, 997: 8, 998: 8, 1003: 8, 1006: 8, 1034: 8, 1042: 8, 1045: 8,1046: 8, 1047: 8, 1054: 8, 1058: 8, 1059: 8, 1068: 8, 1072: 8, 1090: 8, 1091: 8, 1093: 8, 1105: 8, 1107: 8, 1108: 8, 1109: 8, 1114: 8, 1186: 8, 1200: 8, 1430: 8, 1438: 8, 1441: 8, 1459: 8, 1461: 8, 1472: 8, 1824: 8
  }], 
  #CAR.F150SG [{
    #2018 Stop/Go
  #  74: 8, 75: 8, 76: 8, 92: 8, 118: 8, 119: 8, 120: 8, 125: 8, 129: 8, 130: 8, 131: 8, 133: 8, 145: 8, 146: 8, 357: 8, 359: 8, 370: 8, 376: 8, 390: 8, 391: 8, 392: 8, 394: 8, 512: 8, 514: 8, 516: 8, 531: 8, 532: 8, 534: 8, 535: 8, 560: 8, 578: 8, 611: 8, 613: 8, 673: 8, 827: 8, 845: 8, 848: 8, 850: 8, 934: 8, 935: 8, 936: 8, 938: 8, 939: 8, 945: 8, 947: 8, 961: 8, 962: 8, 963: 8, 970: 8, 972: 8, 973: 8, 979: 8, 980: 8, 983: 8, 984: 8, 985: 8, 992: 8, 994: 8, 997: 8, 998: 8, 1003: 8, 1006: 8, 1034: 8, 1042: 8, 1045: 8, 1046: 8, 1047: 8, 1053: 8, 1054: 8, 1056: 8, 1058: 8, 1059: 8, 1068: 8, 1072: 8, 1073: 8, 1080: 8, 1082: 8, 1090: 8, 1091: 8, 1093: 8, 1102: 8, 1105: 8, 1107: 8, 1108: 8, 1109: 8, 1110: 8, 1114: 8, 1122: 8, 1126: 8, 1186: 8, 1200: 8, 1430: 8, 1438: 8, 1441: 8, 1459: 8, 1461: 8, 1472: 8, 1609: 8, 1611: 8, 1798: 8, 1824: 8, 1888: 8, 1896: 8
  #}],
  CAR.F150SG: [{
    #2018 Stop/Go
    74: 8, 75: 8, 76: 8, 92: 8, 118: 8, 119: 8, 120: 8, 125: 8, 129: 8, 130: 8, 131: 8, 133: 8, 145: 8, 146: 8, 357: 8, 359: 8, 370: 8, 376: 8, 390: 8, 391: 8, 392: 8, 394: 8, 512: 8, 514: 8, 516: 8, 531: 8, 532: 8, 534: 8, 535: 8, 560: 8, 578: 8, 611: 8, 613: 8, 673: 8, 827: 8, 845: 8, 848: 8, 850: 8, 934: 8, 935: 8, 936: 8, 938: 8, 939: 8, 945: 8, 947: 8, 961: 8, 962: 8, 963: 8, 970: 8, 972: 8, 973: 8, 979: 8, 980: 8, 983: 8, 984: 8, 985: 8, 992: 8, 994: 8, 997: 8, 998: 8, 1003: 8, 1006: 8, 1034: 8, 1042: 8, 1045: 8, 1046: 8, 1047: 8, 1053: 8, 1054: 8, 1056: 8, 1058: 8, 1059: 8, 1068: 8, 1072: 8, 1073: 8, 1080: 8, 1082: 8, 1090: 8, 1091: 8, 1093: 8, 1102: 8, 1105: 8, 1107: 8, 1108: 8, 1109: 8, 1110: 8, 1114: 8, 1122: 8, 1126: 8, 1186: 8, 1200: 8, 1430: 8, 1438: 8, 1441: 8, 1459: 8, 1461: 8, 1472: 8, 1609: 8, 1611: 8, 1798: 8, 1824: 8, 1888: 8, 1896: 8
  }],
  #CAR.MUSTANG: [{
  #}], 
  CAR.MONDEO: [{
    71: 8, 73: 8, 74: 8, 75: 8, 76: 8, 118: 8, 119: 8, 120: 8, 125: 8, 129: 8, 130: 8, 131: 8, 132: 8, 133: 8, 145: 8, 146: 8, 357: 8, 359: 8, 376: 8, 512: 8, 514: 8, 516: 8, 531: 8, 532: 8, 534: 8, 535: 8, 560: 8, 578: 8, 609: 8, 673: 8, 825: 8, 827: 8, 848: 8, 936: 8, 947: 8, 963: 8, 972: 8, 992: 8, 994: 8, 998: 8, 1003: 8, 1034: 8, 1045: 8, 1046: 8, 1058: 8, 1059: 8, 1068: 8, 1071: 8, 1072: 8, 1073: 8, 1108: 8, 1109: 8, 1200: 8, 1214: 8, 1215: 8, 1430: 8, 1438: 8, 1459: 8
  }],
  #CAR.RANGER: [{
  #}], 
  #CAR.TAURUS: [{
  #}], 
  #CAR.FLEX: [{
  #}], 
  #CAR.FOCUS: [{
  #}], 
  #CAR.TRANSIT: [{
  #}], 
  #Lincoln
  #CAR.AVIATOR: [{
  #}], 
  #CAR.CONTINENTAL: [{
  #}], 
  #CAR.CORSAIR: [{
  #}], 
  #CAR.MKC: [{
  #}], 
  #CAR.MKS: [{
  #}], 
  #CAR.MKT: [{
  #}], 
  #CAR.MKX: [{
  #}], 
  #CAR.MKZ: [{
  #}], 
  #CAR.NAVIGATOR: [{
  #}], 
  #CAR.NAUTILUS: [{
  #}], 
}

ECU_FINGERPRINT = {
  Ecu.fwdCamera: [970, 972, 984]
}

FEATURES = {
  "use_op_longitudinal": [CAR.F150SG, CAR.FUSIONSG],
}

DBC = {
  #Ford
  CAR.FUSION: dbc_dict('ford_newcan_pt', 'ford_fusion_2018_adas'), #ford_fusion_2018_pt
  CAR.FUSIONSG: dbc_dict('ford_newcan_pt', 'ford_fusion_2018_adas'), #ford_fusion_2018_pt
  #CAR.EDGE: dbc_dict('placeholder', 'placeholder'),
  #CAR.ESCAPE: dbc_dict('placeholder', 'placeholder'),
  #CAR.EXPEDITION: dbc_dict('placeholder', 'placeholder'),
  #CAR.EXPLORER: dbc_dict('placeholder', 'placeholder'),
  CAR.F150: dbc_dict('ford_newcan_pt', None),
  CAR.F150SG: dbc_dict('ford_newcan_pt', None),
  #CAR.MUSTANG: dbc_dict('placeholder', 'placeholder'),
  CAR.MONDEO: dbc_dict('ford_mondeo_2016_pt', 'ford_fusion_2018_adas'), #ford_mondeo_2016_pt
  #CAR.RANGER: dbc_dict('placeholder', 'placeholder'),
  #CAR.TAURUS: dbc_dict('placeholder', 'placeholder'),
  #CAR.FLEX: dbc_dict('placeholder', 'placeholder'),
  #CAR.FOCUS: dbc_dict('placeholder', 'placeholder'),
  #CAR.TRANSIT: dbc_dict('placeholder', 'placeholder'),
  #Lincoln
  #CAR.AVIATOR: dbc_dict('placeholder', 'placeholder'),
  #CAR.CONTINENTAL: dbc_dict('placeholder', 'placeholder'),
  #CAR.CORSAIR: dbc_dict('placeholder', 'placeholder'),
  #CAR.MKC: dbc_dict('placeholder', 'placeholder'),
  #CAR.MKS: dbc_dict('placeholder', 'placeholder'),
  #CAR.MKT: dbc_dict('placeholder', 'placeholder'),
  #CAR.MKX: dbc_dict('placeholder', 'placeholder'),
  #CAR.MKZ: dbc_dict('placeholder', 'placeholder'),
  #CAR.NAVIGATOR: dbc_dict('placeholder', 'placeholder'),
  #CAR.NAUTILUS: dbc_dict('placeholder', 'placeholder'),
}
