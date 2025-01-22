from opendbc.car.structs import CarParams
from opendbc.car.tesla.values import CAR

Ecu = CarParams.Ecu

FW_VERSIONS = {
  CAR.TESLA_MODEL_3: {
    (Ecu.eps, 0x730, None): [
      b'TeM3_E014p10_0.0.0 (16),E014.17.00',
      b'TeM3_E014p10_0.0.0 (16),EL014.17.00',
      b'TeMYG4_DCS_Update_0.0.0 (13),E4014.28.1',
      b'TeMYG4_DCS_Update_0.0.0 (9),E4014.26.0',
      b'TeMYG4_Main_0.0.0 (59),E4H014.29.0',
      b'TeMYG4_SingleECU_0.0.0 (33),E4S014.27',
    ],
  },
  CAR.TESLA_MODEL_Y: {
    (Ecu.eps, 0x730, None): [
      b'TeM3_E014p10_0.0.0 (16),Y002.18.00',
      b'TeM3_ES014p11_0.0.0 (16),YS002.17',
      b'TeM3_ES014p11_0.0.0 (25),YS002.19.0',
      b'TeMYG4_DCS_Update_0.0.0 (13),Y4002.27.1',
      b'TeMYG4_DCS_Update_0.0.0 (13),Y4P002.27.1',
      b'TeMYG4_DCS_Update_0.0.0 (9),Y4P002.25.0',
      b'TeMYG4_SingleECU_0.0.0 (33),Y4S002.26',
    ],
  },
}
