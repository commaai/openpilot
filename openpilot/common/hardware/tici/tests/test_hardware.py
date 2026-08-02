from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from openpilot.common.hardware.tici import hardware as hardware_module
from openpilot.common.hardware.tici.hardware import NetworkType, Tici


class TestTiciHardware(TestCase):
  def test_canonical_wifi_profile_metered(self):
    profile = """\
[connection]
metered=1

[802-11-wireless]
ssid=TestNet
"""

    with (
      patch.object(hardware_module, "wpa_supplicant_cmd", return_value={"ssid": "TestNet"}),
      patch.object(Path, "glob", return_value=[Path("profile.nmconnection")]),
      patch.object(hardware_module, "sudo_read", return_value=profile),
    ):
      assert Tici().get_network_metered(NetworkType.wifi)
