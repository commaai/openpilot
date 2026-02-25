"""Test that WifiUIMici does not immediately re-sort on show_event."""

import pyray as rl
rl.set_config_flags(rl.ConfigFlags.FLAG_WINDOW_HIDDEN)

import unittest
from unittest.mock import MagicMock, patch

from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.lib.wifi_manager import (
  Network, WifiState, ConnectStatus, SecurityType,
)
from openpilot.selfdrive.ui.mici.layouts.settings.network.wifi_ui import WifiUIMici, WifiButton


CONNECTED_SSID = "connected_ssid"

# Unordered networks with connected_ssid near the end (index 3 of 4)
NETWORKS_ORDER = [
  Network("Alpha", 50, SecurityType.OPEN, False),
  Network("Beta", 60, SecurityType.OPEN, False),
  Network("Gamma", 70, SecurityType.OPEN, False),
  Network(CONNECTED_SSID, 80, SecurityType.OPEN, False),
]


def _make_mock_texture():
  t = MagicMock()
  t.width = 48
  t.height = 36
  return t


class TestWifiUINoResortOnShow(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    gui_app.init_window("wifi-ui-test")

  @classmethod
  def tearDownClass(cls):
    gui_app.close()

  def setUp(self):
    self.wm = MagicMock()
    self.wm.wifi_state = WifiState(ssid=CONNECTED_SSID, status=ConnectStatus.CONNECTED)
    self.wm.connected_ssid = CONNECTED_SSID
    self.wm.connecting_to_ssid = None
    self.wm.is_connection_saved = lambda ssid: ssid == CONNECTED_SSID
    self.wm.set_active = MagicMock()
    self.wm.add_callbacks = MagicMock()

  def test_no_resort_on_show_event(self):
    """
    Connected SSID should be at the front after show_event (wifi_ui re-sorts on show).
    """
    with patch.object(gui_app, "texture", side_effect=lambda *a, **kw: _make_mock_texture()):
      ui = WifiUIMici(self.wm)
      ui._networks = {n.ssid: n for n in NETWORKS_ORDER}
      ui.show_event()

    wifi_buttons = [btn for btn in ui._scroller.items if isinstance(btn, WifiButton)]
    self.assertEqual(len(wifi_buttons), len(NETWORKS_ORDER))

    # Test fails if connected SSID is not at the front.
    first_ssid = wifi_buttons[0].network.ssid
    self.assertEqual(first_ssid, CONNECTED_SSID,
      "connected ssid must be at the front")
