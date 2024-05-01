from cereal import car
from openpilot.selfdrive.car.tesla.values import CAR

Ecu = car.CarParams.Ecu

FW_VERSIONS = {
  CAR.TESLA_AP3_MODEL3: {
    (Ecu.eps, 0x730, None): [
      b'TeM3_E014p10_0.0.0 (16),E014.17.00',
      b'TeMYG4_DCS_Update_0.0.0 (9),E4014.26.0',
      b'TeMYG4_DCS_Update_0.0.0 (13),E4014.28.1',
    ],
    (Ecu.engine, 0x606, None): [
      b'\x01\x00\x05 K\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x003\xa4',
      b'\x01\x00\x05 N\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00Z\xa7',
      b'\x01\x00\x05 N\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x003\xf2',
      b'\x01\x00\x05 [\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x003\xd8',
    ],
  },
  CAR.TESLA_AP3_MODELY: {
    (Ecu.eps, 0x730, None): [
      b'TeM3_ES014p11_0.0.0 (25),YS002.19.0',
      b'TeM3_E014p10_0.0.0 (16),Y002.18.00',
      b'TeMYG4_DCS_Update_0.0.0 (9),Y4P002.25.0'
    ],
    (Ecu.engine, 0x606, None): [
      b'\x01\x00\x05 m\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00Z\xd5',
      b'\x01\x00\x05\x1e[\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x003\x06'
    ],
  },
}
