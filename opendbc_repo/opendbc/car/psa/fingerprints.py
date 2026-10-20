""" AUTO-FORMATTED USING opendbc/car/debug/format_fingerprints.py, EDIT STRUCTURE THERE."""
from opendbc.car.structs import CarParams
from opendbc.car.psa.values import CAR

Ecu = CarParams.Ecu

FW_VERSIONS = {
  CAR.PSA_PEUGEOT_208: {
    (Ecu.fwdRadar, 0x6b6, None): [
      b'212053276',
    ],
  },
}
