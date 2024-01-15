from cereal import car
import unittest
from openpilot.selfdrive.car.subaru.fingerprints import FW_VERSIONS
from openpilot.selfdrive.car.subaru.values import GLOBAL_GEN2, LKAS_ANGLE, CAR, PREGLOBAL_CARS

Ecu = car.CarParams.Ecu

ECU_NAME = {v: k for k, v in Ecu.schema.enumerants.items()}

def get_fw_sizes(platform):
  return {
    Ecu.abs: 4 if platform in PREGLOBAL_CARS else 5,
    Ecu.eps: 5 if platform in {CAR.OUTBACK_2023} else 4,
    Ecu.fwdCamera: 10 if platform in (LKAS_ANGLE | GLOBAL_GEN2) else 8,
    Ecu.engine: 5,
    Ecu.transmission: 5
  }

class TestSubaruFingerprint(unittest.TestCase):
  def test_fw_version_format(self):
    for platform, fws_per_ecu in FW_VERSIONS.items():
      for (ecu, _, _), fws in fws_per_ecu.items():
        for fw in fws:
          print(len(fw), get_fw_sizes(platform)[ecu], ECU_NAME[ecu])
          self.assertEqual(len(fw), get_fw_sizes(platform)[ecu], f"{platform}, {ECU_NAME[ecu]}")


if __name__ == "__main__":
  unittest.main()
