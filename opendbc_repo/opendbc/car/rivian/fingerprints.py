""" AUTO-FORMATTED USING opendbc/car/debug/format_fingerprints.py, EDIT STRUCTURE THERE."""
from opendbc.car.structs import CarParams
from opendbc.car.rivian.values import CAR

Ecu = CarParams.Ecu

FW_VERSIONS = {
  CAR.RIVIAN_R1_GEN1: {
    (Ecu.eps, 0x733, None): [
      b'R1TS_v3.4.1(51),3.4.1\x00',
      b'R1TS_v4.4.1(63),4.4.1\x00',
    ],
  },
}
