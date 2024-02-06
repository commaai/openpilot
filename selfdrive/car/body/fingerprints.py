# ruff: noqa: E501
from cereal import car
from openpilot.selfdrive.car.body.values import CAR

Ecu = car.CarParams.Ecu

# debug ecu fw version is the git hash of the firmware


FINGERPRINTS = {
  CAR.BODY: [{
    513: 8, 516: 8, 514: 3, 515: 4
  }],
}

FW_VERSIONS = {
  CAR.BODY: {
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
