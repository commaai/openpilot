from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock, patch

from openpilot.common.hardware.comma import hardware as hardware_module
from openpilot.common.hardware.comma.hardware import HardwareComma, NetworkType


class TestHardwareComma(TestCase):
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
      assert HardwareComma().get_network_metered(NetworkType.wifi)

  def test_escaped_wifi_profile_metered(self):
    profile = """\
[connection]
metered=1

[wifi]
ssid=\\sGuest\\s
"""

    with (
      patch.object(hardware_module, "wpa_supplicant_cmd", return_value={"ssid": " Guest "}),
      patch.object(Path, "glob", return_value=[Path("profile.nmconnection")]),
      patch.object(hardware_module, "sudo_read", return_value=profile),
    ):
      assert HardwareComma().get_network_metered(NetworkType.wifi)
  def test_selected_profile_uuid_controls_metering(self):
    first_uuid = "11111111-1111-1111-1111-111111111111"
    second_uuid = "22222222-2222-2222-2222-222222222222"
    profiles = {
      "first.nmconnection": f"""\\
[connection]
uuid={first_uuid}
metered=1

[wifi]
ssid=Duplicate
""",
      "second.nmconnection": f"""\\
[connection]
uuid={second_uuid}
metered=2

[wifi]
ssid=Duplicate
""",
    }

    with (
      patch.object(hardware_module, "wpa_supplicant_cmd", return_value={"ssid": "Duplicate", "id_str": second_uuid}),
      patch.object(Path, "glob", return_value=[Path(name) for name in profiles]),
      patch.object(hardware_module, "sudo_read", side_effect=lambda path: profiles[path]),
    ):
      assert not HardwareComma().get_network_metered(NetworkType.wifi)

  def test_hardware_uses_owned_control_socket(self):
    sock = MagicMock()
    sock.__enter__.return_value = sock
    sock.recv.return_value = b"FAIL\\n"

    with patch.object(hardware_module.socket, "socket", return_value=sock):
      assert hardware_module.wpa_supplicant_cmd("STATUS") == {}

    sock.connect.assert_called_once_with("/run/openpilot-wpa/wlan0")
