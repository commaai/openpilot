from cereal import car
import unittest
from openpilot.selfdrive.car.subaru.fingerprints import FW_VERSIONS

Ecu = car.CarParams.Ecu

ECU_NAME = {v: k for k, v in Ecu.schema.enumerants.items()}


class TestSubaruFingerprint(unittest.TestCase):
  def test_fw_version_format(self):
    for platform, fws_per_ecu in FW_VERSIONS.items():
      for (ecu, _, _), fws in fws_per_ecu.items():
        fw_size = len(fws[0])
        for fw in fws:
          self.assertEqual(len(fw), fw_size, f"{platform} {ecu}: {len(fw)} {fw_size}")


if __name__ == "__main__":
  unittest.main()
