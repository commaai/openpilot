import unittest
from unittest.mock import Mock, patch

from openpilot.system.ui import mici_setup
from openpilot.system.ui.lib.wifi_manager import ConnectStatus, WifiState


class TestNetworkSetupNavigation(unittest.TestCase):
  def setUp(self):
    self.gui = self.enterContext(patch.object(mici_setup, "gui_app"))
    self.clock = self.enterContext(patch.object(mici_setup.rl, "get_time", return_value=100.0))
    self.monitor = mici_setup.NetworkConnectivityMonitor()

    self.page = mici_setup.NetworkSetupPageBase.__new__(mici_setup.NetworkSetupPageBase)
    self.page._network_monitor = self.monitor
    self.wifi_manager = Mock(wifi_state=WifiState())
    self.page._wifi_manager = self.wifi_manager
    self.page._wifi_ui = Mock(any_network_forgetting=False)
    self.page._continue_button = Mock()
    self.page._waiting_button = Mock()
    self.page._prev_has_internet = False
    self.page._prev_wifi_connected = False
    self.page._pending_has_internet_scroll = None
    self.page._scroll_to_end_and_grow = Mock()

    self.gui.get_active_widget.return_value = self.page._wifi_ui
    self.gui.widget_in_stack.return_value = True
    self.password_dialog = mici_setup.BigInputDialog.__new__(mici_setup.BigInputDialog)

  def tick(self, time):
    self.clock.return_value = time
    self.page._nav_stack_tick()

  def test_cellular_connection_keeps_password_dialog(self):
    self.gui.get_active_widget.return_value = self.password_dialog
    self.monitor.network_connected.set()

    self.tick(100.0)
    self.tick(100.8)

    self.gui.pop_widgets_to.assert_not_called()
    self.page._continue_button.set_visible.assert_called_with(True)
    self.page._waiting_button.set_visible.assert_called_with(False)

  def test_password_dialog_cancels_queued_return(self):
    self.monitor.network_connected.set()
    self.tick(100.0)

    self.gui.get_active_widget.return_value = self.password_dialog
    self.tick(100.2)
    self.tick(100.8)
    self.gui.get_active_widget.return_value = self.page._wifi_ui
    self.tick(101.0)

    self.gui.pop_widgets_to.assert_not_called()

  def test_wifi_connection_after_password_entry_returns(self):
    self.gui.get_active_widget.return_value = self.password_dialog
    self.monitor.network_connected.set()
    self.tick(100.0)

    self.gui.get_active_widget.return_value = self.page._wifi_ui
    self.wifi_manager.wifi_state = WifiState(status=ConnectStatus.CONNECTED)
    self.tick(101.0)
    self.tick(101.8)

    self.gui.pop_widgets_to.assert_called_once_with(self.page, self.page._scroll_to_end_and_grow)

  def test_cellular_connection_returns_from_wifi_list(self):
    self.monitor.network_connected.set()
    self.tick(100.0)
    self.gui.pop_widgets_to.assert_not_called()

    self.tick(100.8)

    self.gui.pop_widgets_to.assert_called_once_with(self.page, self.page._scroll_to_end_and_grow)

  def test_wifi_connection_returns_without_internet(self):
    self.wifi_manager.wifi_state = WifiState(status=ConnectStatus.CONNECTED)

    self.tick(100.0)
    self.tick(100.8)

    self.gui.pop_widgets_to.assert_called_once_with(self.page, self.page._scroll_to_end_and_grow)

  def test_cellular_connection_scrolls_current_setup_page(self):
    self.gui.get_active_widget.return_value = self.page
    self.monitor.network_connected.set()

    self.tick(100.0)

    self.gui.pop_widgets_to.assert_called_once_with(self.page, self.page._scroll_to_end_and_grow)

  def test_offline_keeps_password_dialog(self):
    self.gui.get_active_widget.return_value = self.password_dialog

    self.tick(100.0)
    self.tick(100.8)

    self.gui.pop_widgets_to.assert_not_called()


if __name__ == "__main__":
  unittest.main()
