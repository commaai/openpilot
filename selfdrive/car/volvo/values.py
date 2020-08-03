from selfdrive.car import dbc_dict
from cereal import car
Ecu = car.CarParams.Ecu

"""
Volvo Electronic Control Units abbreviations and network topology
Platforms C1/EUCD

Three main CAN network buses
  1. Powertrain
  2. Chassis (also called MS* CAN) *MS=Medium Speed
  3. Extended
Only mentioning control units of interest on the network buses.

Powertrain CAN
  BCM - Brake Control Module
  CEM - Central Electronic Module
  CVM - Closing Velocity Module (low speed auto emergency braking <30kph)
  FSM - Forward Sensing Module (camera mounted in windscreen)
  PPM - Pedestrian Protection Module (controls pedestrian airbag under the engine hood)
  PSCM - Power Steering Control Module (EPS - Electronic Power Steering)
  SAS - Steering Angle Sensor Module
  SRS - Supplemental Restraint System Module (seatbelts, airbags...)
  TCM - Transmission Control Module

Chassis CAN
  CEM - Central Electronic Module
  DIM - Driver Information Module (the instrument cluster with odo and speedometer, relayed thru CEM)
  PAM - Parking Assistance Module (automatic parking, relayed thru CEM)

Extended CAN
  CEM - Central Electronic Module
  SODL - Side Object Detection Left (relayed thru CEM)
  SODR - Side Object Detection Right (relayed thru CEM)
"""

class CarControllerParams():
  # constants, collected from v40 dbc lka_direction.
  STEER_NO = 0
  STEER_RIGHT = 1
  STEER_LEFT = 2
  STEER = 3

  # maximum degress offset/rate of change on request from current/last steering angle
  MAX_ACT_ANGLE_REQUEST_DIFF = 25   # A bigger angle difference will trigger disengage.
  STEER_ANGLE_DELTA_REQ_DIFF = 0.25

  # Limits  
  ANGLE_DELTA_BP = [0., 5., 15., 27., 36.]   # 0, 18, 54, 97.2, 129.6 km/h
  ANGLE_DELTA_V = [2, 1.2, .15, .1, .08]     # windup limit
  ANGLE_DELTA_VU = [3, 1.8, 0.4, .2, .1]   # unwind limit

  # number of 0 torque samples in a row before trying to restore steering.
  N_ZERO_TRQ = 10

  # EUCD
  # When changing steer direction steering request need to be blocked.
  # This calibration sets the number of samples to block it and no steering instead.
  BLOCK_LEN = 8
  # don't change steer direction inside deadzone, 
  # might not be needed in future after discovering STEER command.
  DEADZONE = 0.1


BUTTON_STATES = {
  "altButton1": False, # On/Off button
  #"cancel": False, Not present in V60
  "setCruise": False,
  "resumeCruise": False,
  "accelCruise": False,
  "decelCruise": False,
  "gapAdjustCruise": False,
}   


class CAR:
  V40 = "VOLVO V40 2017"
  V60 = "VOLVO V60 2015"

class PLATFORM:
  C1 = [CAR.V40]
  EUCD = [CAR.V60]

ECU_ADDRESS = { 
  CAR.V40: {"BCM": 0x760, "ECM": 0x7E0, "DIM": 0x720, "CEM": 0x726, "FSM": 0x764, "PSCM": 0x730, "TCM": 0x7E1, "CVM": 0x793},
  }

# TODO: Find good DID for identifying SW version
# 0xf1a1, 0xf1a3 not on CVM.
# Possible options 0xf1a2, 0xf1a4, 0xf1a5 
# Decided to use DID 0xf1a2 as a start. 
#
# Response is 27 bytes. But panda only has 24 bytes in response.
# Probably some timeout or max byte limit.
#  
# Create byte string (pad with 00 until length=24)
# s='32233863 AA'
# b = bytes([ord(s[a]) for a in range(len(s))]) + b'\x00' * 13
#
FW_VERSIONS = {
  CAR.V40: {
  (Ecu.unknown, ECU_ADDRESS[CAR.V40]["CEM"], None): [b'31453061 AA\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'],  # 0xf1a2
  #(Ecu.unknown, ECU_ADDRESS[CAR.V40]["CEM"], None): [b'31453132 AA\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'], # 0xf1a4
  #(Ecu.unknown, ECU_ADDRESS[CAR.V40]["CEM"], None): [b'32233863 AA\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'], # 0xf1a5
  #(Ecu.eps, ECU_ADDRESS[CAR.V40]["PSCM"], None): [b'31288595 AE\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'],           # 0xf1a2
  (Ecu.eps, ECU_ADDRESS[CAR.V40]["PSCM"], None): [b'31288595 AE\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'],            # 0xf1a2
  #(Ecu.eps, ECU_ADDRESS[CAR.V40]["PSCM"], None): [b'31678017\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'],  # 0xf1a4
  #(Ecu.eps, ECU_ADDRESS[CAR.V40]["PSCM"], None): [b'31681147 AA\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'],           # 0xf1a5
  #(Ecu.fwdCamera, ECU_ADDRESS[CAR.V40]["FSM"], None): [b'31400454 AA\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'],  # 0xf1a2
  (Ecu.fwdCamera, ECU_ADDRESS[CAR.V40]["FSM"], None): [b'31400454 AA\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'],  # 0xf1a2
  #(Ecu.fwdCamera, ECU_ADDRESS[CAR.V40]["FSM"], None): [b'31660982 AA\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'], # 0xf1a4
  #(Ecu.fwdCamera, ECU_ADDRESS[CAR.V40]["FSM"], None): [b'31660983 AA\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'], # 0xf1a5
  #(Ecu.cvm, ECU_ADDRESS[CAR.V40]["CVM"], None): [b'31360093 AA\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'], # 0xf1a2 Could be used in future.
  #(Ecu.cvm, ECU_ADDRESS[CAR.V40]["CVM"], None): [b'31360888 AA\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'], # 0xf1a4 Could be used in future.
  #(Ecu.cvm, ECU_ADDRESS[CAR.V40]["CVM"], None): [b'31360340 AA\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'], # 0xf1a5 Could be used in future.
  }
}

# Result from black panda
#[<car.capnp:CarParams.CarFw builder (ecu = engine, fwVersion = "31432422 CA\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00", address = 2016, subAddress = 0)>, 
# <car.capnp:CarParams.CarFw builder (ecu = eps, fwVersion = "31288595 AE\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00", address = 1840, subAddress = 0)>, 
# <car.capnp:CarParams.CarFw builder (ecu = unknown, fwVersion = "31453061 AA\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00", address = 1830, subAddress = 0)>, 
# <car.capnp:CarParams.CarFw builder (ecu = fwdCamera, fwVersion = "31400454 AA\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00", address = 1892, subAddress = 0)>]


FINGERPRINTS = {
  CAR.V40: [
    # V40 2017
    {8: 8, 16: 8, 48: 8, 64: 8, 85: 8, 101: 8, 112: 8, 114: 8, 117: 8, 128: 8, 176: 8, 192: 8, 208: 8, 224: 8, 240: 8, 245: 8, 256: 8, 272: 8, 288: 8, 291: 8, 293: 8, 304: 8, 325: 8, 336: 8, 352: 8, 424: 8, 432: 8, 437: 8, 464: 8, 472: 8, 480: 8, 528: 8, 608: 8, 624: 8, 640: 8, 648: 8, 652: 8, 656: 8, 657: 8, 681: 8, 693: 8, 704: 8, 707: 8, 709: 8, 816: 8, 832: 8, 848: 8, 853: 8, 864: 8, 880: 8, 912: 8, 928: 8, 943: 8, 944: 8, 968: 8, 970: 8, 976: 8, 992: 8, 997: 8, 1024: 8, 1029: 8, 1061: 8, 1072: 8, 1409: 8},
    # V40 2015
    {8: 8, 16: 8, 64: 8, 85: 8, 101: 8, 112: 8, 114: 8, 117: 8, 128: 8, 176: 8, 192: 8, 224: 8, 240: 8, 245: 8, 256: 8, 272: 8, 288: 8, 291: 8, 293: 8, 304: 8, 325: 8, 336: 8, 424: 8, 432: 8, 437: 8, 464: 8, 472: 8, 480: 8, 528: 8, 608: 8, 648: 8, 652: 8, 656: 8, 657: 8, 681: 8, 693: 8, 704: 8, 707: 8, 709: 8, 816: 8, 832: 8, 864: 8, 880: 8, 912: 8, 928: 8, 943: 8, 944: 8, 968: 8, 970: 8, 976: 8, 992: 8, 997: 8, 1024: 8, 1029: 8, 1061: 8, 1072: 8, 1409: 8},
    # V40 2014
    {8: 8, 16: 8, 64: 8, 85: 8, 101: 8, 112: 8, 114: 8, 117: 8, 128: 8, 176: 8, 192: 8, 224: 8, 240: 8, 245: 8, 256: 8, 272: 8, 288: 8, 291: 8, 293: 8, 304: 8, 325: 8, 336: 8, 424: 8, 432: 8, 437: 8, 464: 8, 472: 8, 480: 8, 528: 8, 608: 8, 648: 8, 652: 8, 657: 8, 681: 8, 693: 8, 704: 8, 707: 8, 709: 8, 816: 8, 864: 8, 880: 8, 912: 8, 928: 8, 943: 8, 944: 8, 968: 8, 970: 8, 976: 8, 992: 8, 997: 8, 1024: 8, 1029: 8, 1072: 8, 1409: 8},
    
    # Black Remove in future.
    #{272: 8, 432: 8, 437: 8, 240: 8, 424: 8, 472: 8, 480: 8, 624: 8, 8: 8, 707: 8, 224: 8, 64: 8, 291: 8, 16: 8, 114: 8, 464: 8, 48: 8, 528: 8, 640: 8, 85: 8, 101: 8, 112: 8, 117: 8, 128: 8, 176: 8, 325: 8, 652: 8, 693: 8, 832: 8, 293: 8, 192: 8, 245: 8, 288: 8, 304: 8, 336: 8, 208: 8, 853: 8, 256: 8, 657: 8, 816: 8, 1061: 8, 352: 8, 656: 8, 648: 8, 608: 8, 880: 8, 970: 8, 681: 8, 864: 8, 912: 8, 704: 8, 709: 8, 943: 8, 944: 8, 976: 8, 1072: 8},
    #{8: 8, 16: 8, 64: 8, 85: 8, 101: 8, 112: 8, 114: 8, 117: 8, 128: 8, 176: 8, 192: 8, 224: 8, 240: 8, 245: 8, 256: 8, 272: 8, 288: 8, 291: 8, 293: 8, 304: 8, 325: 8, 336: 8, 424: 8, 432: 8, 437: 8, 464: 8, 472: 8, 480: 8, 528: 8, 608: 8, 648: 8, 652: 8, 656: 8, 657: 8, 681: 8, 693: 8, 704: 8, 707: 8, 709: 8, 816: 8, 832: 8, 848: 8, 864: 8, 880: 8, 912: 8, 928: 8, 943: 8, 944: 8, 968: 8, 970: 8, 976: 8, 992: 8, 997: 8, 1024: 8, 1029: 8, 1061: 8, 1072: 8, 1409: 8, 1838: 8, 1848: 8}
  ],
  CAR.V60: [
    {0: 8, 16: 8, 32: 8, 81: 8, 99: 8, 104: 8, 112: 8, 144: 8, 277: 8, 295: 8, 298: 8, 307: 8, 320: 8, 328: 8, 336: 8, 343: 8, 352: 8, 359: 8, 384: 8, 465: 8, 511: 8, 522: 8, 544: 8, 565: 8, 582: 8, 608: 8, 609: 8, 610: 8, 612: 8, 613: 8, 624: 8, 626: 8, 635: 8, 648: 8, 665: 8, 673: 8, 704: 8, 706: 8, 708: 8, 750: 8, 751: 8, 778: 8, 788: 8, 794: 8, 797: 8, 802: 8, 803: 8, 805: 8, 807: 8, 819: 8, 820: 8, 821: 8, 913: 8, 923: 8, 978: 8, 979: 8, 1006: 8, 1021: 8, 1024: 8, 1029: 8, 1039: 8, 1042: 8, 1045: 8, 1137: 8, 1141: 8, 1152: 8, 1174: 8, 1187: 8, 1198: 8, 1214: 8, 1217: 8, 1226: 8, 1240: 8, 1409: 8},
  ],
}


DBC = {
  # dbc_dict( powertrain_dbc, radar_dbc )
  CAR.V40: dbc_dict('volvo_v40_2017_pt', None),
  CAR.V60: dbc_dict('volvo_v60_2015_pt', None),
}
