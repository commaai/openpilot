from openpilot.common.test import OpenpilotTestCase
from openpilot.system.ui.lib import wifi_manager
from openpilot.system.ui.lib.networkmanager import (NM_802_11_AP_FLAGS_PRIVACY,
                                                    NM_802_11_AP_SEC_KEY_MGMT_802_1X,
                                                    NM_802_11_AP_SEC_KEY_MGMT_PSK,
                                                    NM_802_11_AP_SEC_KEY_MGMT_SAE)
from openpilot.system.ui.lib.wifi_manager import SecurityType


class TestGetSecurityType(OpenpilotTestCase):
  def test_wpa3_is_known_but_unsupported(self, mocker):
    warning = mocker.patch.object(wifi_manager.cloudlog, "warning")

    security_type = wifi_manager.get_security_type(NM_802_11_AP_FLAGS_PRIVACY, 0, 0x488)

    self.assertEqual(security_type, SecurityType.UNSUPPORTED)
    warning.assert_not_called()

  def test_wpa2_wpa3_transition_network_remains_supported(self, mocker):
    warning = mocker.patch.object(wifi_manager.cloudlog, "warning")
    rsn_flags = NM_802_11_AP_SEC_KEY_MGMT_PSK | NM_802_11_AP_SEC_KEY_MGMT_SAE

    security_type = wifi_manager.get_security_type(NM_802_11_AP_FLAGS_PRIVACY, 0, rsn_flags)

    self.assertEqual(security_type, SecurityType.WPA)
    warning.assert_not_called()

  def test_unknown_security_type_still_warns(self, mocker):
    warning = mocker.patch.object(wifi_manager.cloudlog, "warning")

    security_type = wifi_manager.get_security_type(NM_802_11_AP_FLAGS_PRIVACY, 0, 0x2000)

    self.assertEqual(security_type, SecurityType.UNSUPPORTED)
    warning.assert_called_once()

  def test_8021x_is_still_unsupported(self, mocker):
    warning = mocker.patch.object(wifi_manager.cloudlog, "warning")
    rsn_flags = NM_802_11_AP_SEC_KEY_MGMT_PSK | NM_802_11_AP_SEC_KEY_MGMT_802_1X

    security_type = wifi_manager.get_security_type(NM_802_11_AP_FLAGS_PRIVACY, 0, rsn_flags)

    self.assertEqual(security_type, SecurityType.UNSUPPORTED)
    warning.assert_called_once()
