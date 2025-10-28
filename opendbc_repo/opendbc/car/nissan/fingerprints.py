""" AUTO-FORMATTED USING opendbc/car/debug/format_fingerprints.py, EDIT STRUCTURE THERE."""
from opendbc.car.structs import CarParams
from opendbc.car.nissan.values import CAR

Ecu = CarParams.Ecu

FW_VERSIONS = {
  CAR.NISSAN_ALTIMA: {
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
  CAR.NISSAN_LEAF: {
    (Ecu.abs, 0x740, None): [
      b'476605SA1C',
      b'476605SA7D',
      b'476605SC2D',
      b'476606WK7B',
      b'476606WK9B',
    ],
    (Ecu.eps, 0x742, None): [
      b'5SA2A\x99A\x05\x02N123F\x15b\x00\x00\x00\x00\x00\x00\x00\x80',
      b'5SA2A\xb7A\x05\x02N123F\x15\xa2\x00\x00\x00\x00\x00\x00\x00\x80',
      b'5SN2A\xb7A\x05\x02N123F\x15\xa2\x00\x00\x00\x00\x00\x00\x00\x80',
      b'5SN2A\xb7A\x05\x02N126F\x15\xb2\x00\x00\x00\x00\x00\x00\x00\x80',
    ],
    (Ecu.fwdCamera, 0x707, None): [
      b'5SA0ADB\x04\x18\x00\x00\x00\x00\x00_*6\x04\x94a\x00\x00\x00\x80',
      b'5SA2ADB\x04\x18\x00\x00\x00\x00\x00_*6\x04\x94a\x00\x00\x00\x80',
      b'6WK2ADB\x04\x18\x00\x00\x00\x00\x00R;1\x18\x99\x10\x00\x00\x00\x80',
      b'6WK2BDB\x04\x18\x00\x00\x00\x00\x00R;1\x18\x99\x10\x00\x00\x00\x80',
      b'6WK2CDB\x04\x18\x00\x00\x00\x00\x00R=1\x18\x99\x10\x00\x00\x00\x80',
    ],
    (Ecu.gateway, 0x18dad0f1, None): [
      b'284U25SA3C',
      b'284U25SP0C',
      b'284U25SP1C',
      b'284U26WK0A',
      b'284U26WK0C',
    ],
  },
  CAR.NISSAN_LEAF_IC: {
    (Ecu.fwdCamera, 0x707, None): [
      b'5SH1BDB\x04\x18\x00\x00\x00\x00\x00_-?\x04\x91\xf2\x00\x00\x00\x80',
      b'5SH3BDB\x04\x18\x00\x00\x00\x00\x00_-?\x04\x91\xf2\x00\x00\x00\x80',
      b'5SH4BDB\x04\x18\x00\x00\x00\x00\x00_-?\x04\x91\xf2\x00\x00\x00\x80',
      b'5SK0ADB\x04\x18\x00\x00\x00\x00\x00_(5\x07\x9aQ\x00\x00\x00\x80',
    ],
    (Ecu.abs, 0x740, None): [
      b'476605SD2E',
      b'476605SH1D',
      b'476605SH7D',
      b'476605SH7E',
      b'476605SK2A',
    ],
    (Ecu.eps, 0x742, None): [
      b'5SH2A\x99A\x05\x02N123F\x15\x81\x00\x00\x00\x00\x00\x00\x00\x80',
      b'5SH2A\xb7A\x05\x02N123F\x15\xa3\x00\x00\x00\x00\x00\x00\x00\x80',
      b'5SH2C\xb7A\x05\x02N123F\x15\xa3\x00\x00\x00\x00\x00\x00\x00\x80',
      b'5SK3A\x99A\x05\x02N123F\x15u\x00\x00\x00\x00\x00\x00\x00\x80',
    ],
    (Ecu.gateway, 0x18dad0f1, None): [
      b'284U25SF0C',
      b'284U25SH3A',
      b'284U25SK2D',
      b'284U25SR0B',
    ],
  },
  CAR.NISSAN_XTRAIL: {
    (Ecu.fwdCamera, 0x707, None): [
      b'284N86FR2A',
    ],
    (Ecu.abs, 0x740, None): [
      b'6FU0AD\x11\x02\x00\x02e\x95e\x80iQ#\x01\x00\x00\x00\x00\x00\x80',
      b'6FU1BD\x11\x02\x00\x02e\x95e\x80iX#\x01\x00\x00\x00\x00\x00\x80',
    ],
    (Ecu.eps, 0x742, None): [
      b'6FP2A\x99A\x05\x02N123F\x18\x02\x00\x00\x00\x00\x00\x00\x00\x80',
    ],
    (Ecu.combinationMeter, 0x743, None): [
      b'6FR2A\x18B\x05\x17\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'6FR9A\xa0A\x06\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80',
      b'6FU9B\xa0A\x06\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80',
    ],
    (Ecu.gateway, 0x18dad0f1, None): [
      b'284U26FR0E',
    ],
  },
}
