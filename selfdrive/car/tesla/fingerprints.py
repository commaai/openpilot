from cereal import car
from openpilot.selfdrive.car.tesla.values import CAR

Ecu = car.CarParams.Ecu

FW_VERSIONS = {
  CAR.TESLA_AP2_MODELS: {
    (Ecu.adas, 0x649, None): [
      b'\x01\x00\x8b\x07\x01\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x11',
    ],
    (Ecu.electricBrakeBooster, 0x64d, None): [
      b'1037123-00-A',
    ],
    (Ecu.fwdRadar, 0x671, None): [
      b'\x01\x00W\x00\x00\x00\x07\x00\x00\x00\x00\x08\x01\x00\x00\x00\x07\xff\xfe',
    ],
    (Ecu.eps, 0x730, None): [
      b'\x10#\x01',
    ],
  },
  CAR.TESLA_MODELS_RAVEN: {
    (Ecu.electricBrakeBooster, 0x64d, None): [
      b'1037123-00-A',
    ],
    (Ecu.fwdRadar, 0x671, None): [
      b'\x01\x00\x99\x02\x01\x00\x10\x00\x00AP8.3.03\x00\x10',
    ],
    (Ecu.eps, 0x730, None): [
      b'SX_0.0.0 (99),SR013.7',
    ],
  },
}
