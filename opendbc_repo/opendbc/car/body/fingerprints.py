# ruff: noqa: E501
from opendbc.car.structs import CarParams
from opendbc.car.body.values import CAR

Ecu = CarParams.Ecu

# debug ecu fw version is the git hash of the firmware


FINGERPRINTS = {
  CAR.COMMA_BODY: [{
    513: 8, 516: 8, 514: 3, 515: 4
  }],
}

FW_VERSIONS = {
  CAR.COMMA_BODY: {
    (Ecu.engine, 0x720, None): [
      b'0.0.01',
      b'0.3.00a',
      b'02/27/2022',
    ],
    (Ecu.debug, 0x721, None): [
      b'166bd860',
      b'dc780f85',
    ],
  },
}
