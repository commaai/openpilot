from typing import Dict, List, Union

from selfdrive.car import dbc_dict
from selfdrive.car.docs_definitions import CarInfo
from cereal import car
Ecu = car.CarParams.Ecu


class CarControllerParams:
  ANGLE_DELTA_BP = [0., 5., 15.]
  ANGLE_DELTA_V = [5., .8, .15]     # windup limit
  ANGLE_DELTA_VU = [5., 3.5, 0.4]   # unwind limit
  LKAS_MAX_TORQUE = 1               # A value of 1 is easy to overpower
  STEER_THRESHOLD = 1.0


class CAR:
  XTRAIL = "NISSAN X-TRAIL 2017"
  LEAF = "NISSAN LEAF 2018"
  # Leaf with ADAS ECU found behind instrument cluster instead of glovebox
  # Currently the only known difference between them is the inverted seatbelt signal.
  LEAF_IC = "NISSAN LEAF 2018 Instrument Cluster"
  ROGUE = "NISSAN ROGUE 2019"
  ALTIMA = "NISSAN ALTIMA 2020"


CAR_INFO: Dict[str, Union[CarInfo, List[CarInfo]]] = {
  CAR.XTRAIL: CarInfo("Nissan X-Trail 2017", "ProPILOT"),
  CAR.LEAF: CarInfo("Nissan Leaf 2018-22", "ProPILOT"),
  CAR.ROGUE: CarInfo("Nissan Rogue 2018-20", "ProPILOT"),
  CAR.ALTIMA: CarInfo("Nissan Altima 2019-20", "ProPILOT"),
}

FW_VERSIONS = {
  CAR.ALTIMA: {
    (Ecu.fwdCamera, 0x707, None): [
      b'284N86CA1D',
    ],
    (Ecu.eps, 0x742, None): [
      b'6CA2B\xa9A\x02\x02G8A89P90D6A\x00\x00\x01\x80',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'237109HE2B',
    ],
    (Ecu.gateway, 0x18dad0f1, None): [
      b'284U29HE0A',
    ],
  },
  CAR.LEAF: {
    (Ecu.eps, 0x742, None): [
      b'5SA2A\x99A\x05\x02N123F\x15b\x00\x00\x00\x00\x00\x00\x00\x80',
      b'5SN2A\xb7A\x05\x02N126F\x15\xb2\x00\x00\x00\x00\x00\x00\x00\x80',
      b'5SN2A\xb7A\x05\x02N123F\x15\xa2\x00\x00\x00\x00\x00\x00\x00\x80',
    ],
    (Ecu.fwdCamera, 0x707, None): [
      b'5SA0ADB\x04\x18\x00\x00\x00\x00\x00_*6\x04\x94a\x00\x00\x00\x80',
      b'6WK2BDB\x04\x18\x00\x00\x00\x00\x00R;1\x18\x99\x10\x00\x00\x00\x80',
      b'5SA2ADB\x04\x18\x00\x00\x00\x00\x00_*6\x04\x94a\x00\x00\x00\x80',
    ],
    (Ecu.gateway, 0x18dad0f1, None): [
      b'284U25SA3C',
      b'284U26WK0C',
      b'284U25SP1C',
      b'284U26WK0A',
    ],
    (Ecu.esp, 0x740, None): [
      b'476605SA1C',
      b'476606WK9B',
      b'476605SC2D',
    ],
  },
  CAR.LEAF_IC: {
    (Ecu.fwdCamera, 0x707, None): [
      b'5SH1BDB\x04\x18\x00\x00\x00\x00\x00_-?\x04\x91\xf2\x00\x00\x00\x80',
      b'5SH3BDB\x04\x18\x00\x00\x00\x00\x00_-?\x04\x91\xf2\x00\x00\x00\x80',
      b'5SK0ADB\x04\x18\x00\x00\x00\x00\x00_(5\x07\x9aQ\x00\x00\x00\x80',
      b'5SK2ADB\x04\x18\x00\x00\x00\x00\x00_(5\x07\x9aQ\x00\x00\x00\x80',
    ],
    (Ecu.esp, 0x740, None): [
      b'476605SB2D',
      b'476605SH1D',
      b'476605SK2A',
    ],
    (Ecu.eps, 0x742, None): [
      b'5SB3A\xb7A\x05\x02N123F\x15\xa1\x00\x00\x00\x00\x00\x00\x00\x80',
      b'5SH2A\x99A\x05\x02N123F\x15\x81\x00\x00\x00\x00\x00\x00\x00\x80',
      b'5SH3A\x99A\x05\x02N123F\x15\x81\x00\x00\x00\x00\x00\x00\x00\x80',
      b'5SK3A\x99A\x05\x02N123F\x15u\x00\x00\x00\x00\x00\x00\x00\x80',
    ],
    (Ecu.gateway, 0x18dad0f1, None): [
      b'284U25SH3A',
      b'284U25SH3B',
      b'284U25SK2D',
      b'284U25SN0D',
    ],
  },
  CAR.XTRAIL: {
    (Ecu.fwdCamera, 0x707, None): [
      b'284N86FR2A',
    ],
    (Ecu.esp, 0x740, None): [
      b'6FU1BD\x11\x02\x00\x02e\x95e\x80iX#\x01\x00\x00\x00\x00\x00\x80',
      b'6FU0AD\x11\x02\x00\x02e\x95e\x80iQ#\x01\x00\x00\x00\x00\x00\x80',
    ],
    (Ecu.eps, 0x742, None): [
      b'6FP2A\x99A\x05\x02N123F\x18\x02\x00\x00\x00\x00\x00\x00\x00\x80',
    ],
    (Ecu.combinationMeter, 0x743, None): [
      b'6FR2A\x18B\x05\x17\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'6FU9B\xa0A\x06\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80',
      b'6FR9A\xa0A\x06\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80',
    ],
    (Ecu.gateway, 0x18dad0f1, None): [
      b'284U26FR0E',
    ],
  },
  CAR.ROGUE: {
    (Ecu.engine, 0x7e0, None): [
      b'7FM1A\x00A\x06\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80',
      b'7FW4A\x00A\x06\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80',
      b'7FL5A\x00A\x06\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80',
      b'7FW5A\x00A\x06\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80',
      b'7FL2A\x00A\x06\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80',
    ],
    (Ecu.esp, 0x740, None): [
      b'7FM1CD\x11\x01\x00\x02e\x95e\x80\x80c#\x01\x00\x00\x00\x00\x00\x80',
      b'7FW1CD\x11\x01\x00\x02e\x95e\x80iV#\x01\x00\x00\x00\x00\x00\x80',
      b'7FL6BD\x11\x02\x00\x02e\x95e\x80\x80f#\x01\x00\x00\x00\x00\x00\x80',
      b'7FW5CD\x11\x02\x00\x02e\x95e\x80iW#\x01\x00\x00\x00\x00\x00\x80',
      b'7FL5BD\x11\x01\x00\x02e\x95e\x80\x80e#\x01\x00\x00\x00\x00\x00\x80',
    ],
    (Ecu.eps, 0x742, None): [
      b'7FR2A\x99A\x05\x02N123F\x18\x12\x00\x00\x00\x00\x00\x00\x00\x80',
      b'9TG0A\x99A\x05\x02N123F\x1a \x00\x00\x00\x00\x00\x00\x00\x80',
    ],
    (Ecu.combinationMeter, 0x743, None): [
      b'7FM0A\x18B\x05\x17\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80',
      b'7FW1A\x18B\x05\x17\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80',
      b'7FL0A\x18B\x05\x17\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80',
    ],
    (Ecu.fwdCamera, 0x707, None): [
      b'284N87FW1A',
      b'284N87FL0B',
    ],
    (Ecu.gateway, 0x18dad0f1, None): [
      b'284U27FM0A',
      b'284U27FW0A',
      b'284U27FL0A',
    ],
  }
}

DBC = {
  CAR.XTRAIL: dbc_dict('nissan_x_trail_2017', None),
  CAR.LEAF: dbc_dict('nissan_leaf_2018', None),
  CAR.LEAF_IC: dbc_dict('nissan_leaf_2018', None),
  CAR.ROGUE: dbc_dict('nissan_x_trail_2017', None),
  CAR.ALTIMA: dbc_dict('nissan_x_trail_2017', None),
}
