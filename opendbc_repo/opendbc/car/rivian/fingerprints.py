from opendbc.car.structs import CarParams
from opendbc.car.rivian.values import CAR

Ecu = CarParams.Ecu

FW_VERSIONS = {
  CAR.RIVIAN_R1_GEN1: {
    (Ecu.eps, 0x733, None): [
      b'R1TS_v3.4.1(51),3.4.1\x00',
    ],
  },
}
