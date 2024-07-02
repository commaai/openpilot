from cereal import car
from openpilot.selfdrive.car.fca_giorgio.values import CAR

Ecu = car.CarParams.Ecu

FW_VERSIONS = {
  CAR.ALFA_ROMEO_STELVIO_1ST_GEN: {
    (Ecu.engine, 0x7e0, None): [
      b'PLACEHOLDER',
    ],
  }
}
