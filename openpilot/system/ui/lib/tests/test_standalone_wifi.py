from importlib import import_module
import os
from unittest import SkipTest, skipUnless, TestCase
from unittest.mock import MagicMock, patch

previous_scale = os.environ.get("SCALE")
os.environ["SCALE"] = "1"
try:
  try:
    import_module("pyray")
  except ImportError:
    pyray_available = False
  else:
    pyray_available = True
    from openpilot.system.ui.widgets import network as network_module
    from openpilot.system.ui.widgets.network import UIState, WifiManagerUI
finally:
  if previous_scale is None:
    del os.environ["SCALE"]
  else:
    os.environ["SCALE"] = previous_scale


@skipUnless(pyray_available, "pyray is unavailable")
class TestStandaloneWifi(TestCase):
  def test_forget_failure_releases_wifi_controls(self):
    wifi_ui = WifiManagerUI.__new__(WifiManagerUI)
    wifi_ui.state = UIState.FORGETTING
    wifi_ui._page_shown = True
    wifi_ui._panel_active = True
    dialog = MagicMock()

    with (
      patch.object(network_module, "alert_dialog", return_value=dialog),
      patch.object(network_module.gui_app, "push_widget") as push_widget,
    ):
      wifi_ui._on_forget_failed("SavedNet")

    assert wifi_ui.state == UIState.IDLE
    push_widget.assert_called_once_with(dialog)

  def test_forget_failure_on_advanced_panel_does_not_open_dialog(self):
    wifi_ui = WifiManagerUI.__new__(WifiManagerUI)
    wifi_ui.state = UIState.FORGETTING
    wifi_ui._page_shown = True
    wifi_ui._panel_active = False

    with (
      patch.object(network_module, "alert_dialog", return_value=MagicMock()),
      patch.object(network_module.gui_app, "push_widget") as push_widget,
    ):
      wifi_ui._on_forget_failed("SavedNet")

    assert wifi_ui.state == UIState.IDLE
    push_widget.assert_not_called()

  def test_network_ui_tracks_active_wifi_panel(self):
    network_ui = network_module.NetworkUI.__new__(network_module.NetworkUI)
    network_ui._wifi_panel = MagicMock()

    network_ui._set_current_panel(network_module.PanelType.ADVANCED)

    assert network_ui._current_panel == network_module.PanelType.ADVANCED
    network_ui._wifi_panel.set_panel_active.assert_called_once_with(False)

  def test_mici_forget_failure_restores_button_and_opens_dialog(self):
    try:
      from openpilot.selfdrive.ui.mici.layouts.settings.network import wifi_ui as wifi_ui_module
      from openpilot.selfdrive.ui.mici.layouts.settings.network.wifi_ui import WifiUIMici
    except ImportError as e:
      raise SkipTest("mici UI dependencies are unavailable") from e

    class WifiButton:
      def __init__(self):
        self.network = MagicMock(ssid="SavedNet")
        self.on_forgotten = MagicMock()

    button = WifiButton()
    wifi_ui = WifiUIMici.__new__(WifiUIMici)
    wifi_ui._scroller = MagicMock(items=[button])
    wifi_ui._shown = True
    dialog = MagicMock()

    with (
      patch.object(wifi_ui_module, "WifiButton", WifiButton),
      patch.object(wifi_ui_module, "BigDialog", return_value=dialog),
      patch.object(wifi_ui_module.gui_app, "push_widget") as push_widget,
    ):
      wifi_ui._on_forget_failed("SavedNet")

    button.on_forgotten.assert_called_once()
    push_widget.assert_called_once_with(dialog)

  def test_mici_hidden_forget_failure_does_not_open_dialog(self):
    try:
      from openpilot.selfdrive.ui.mici.layouts.settings.network import wifi_ui as wifi_ui_module
      from openpilot.selfdrive.ui.mici.layouts.settings.network.wifi_ui import WifiUIMici
    except ImportError as e:
      raise SkipTest("mici UI dependencies are unavailable") from e

    class WifiButton:
      def __init__(self):
        self.network = MagicMock(ssid="SavedNet")
        self.on_forgotten = MagicMock()

    button = WifiButton()
    wifi_ui = WifiUIMici.__new__(WifiUIMici)
    wifi_ui._scroller = MagicMock(items=[button])
    wifi_ui._shown = False

    with (
      patch.object(wifi_ui_module, "WifiButton", WifiButton),
      patch.object(wifi_ui_module, "BigDialog", return_value=MagicMock()),
      patch.object(wifi_ui_module.gui_app, "push_widget") as push_widget,
    ):
      wifi_ui._on_forget_failed("SavedNet")

    button.on_forgotten.assert_called_once()
    push_widget.assert_not_called()

  def test_mici_wrong_password_opens_password_dialog(self):
    try:
      from openpilot.selfdrive.ui.mici.layouts.settings.network import wifi_ui as wifi_ui_module
      from openpilot.selfdrive.ui.mici.layouts.settings.network.wifi_ui import WifiUIMici
    except ImportError as e:
      raise SkipTest("mici UI dependencies are unavailable") from e

    class WifiButton:
      def __init__(self):
        self.network = MagicMock(ssid="SavedNet")
        self.set_wrong_password = MagicMock()

    button = WifiButton()
    wifi_ui = WifiUIMici.__new__(WifiUIMici)
    wifi_ui._scroller = MagicMock(items=[button])
    wifi_ui._shown = True
    dialog = MagicMock()

    with (
      patch.object(wifi_ui_module, "WifiButton", WifiButton),
      patch.object(wifi_ui_module, "BigInputDialog", return_value=dialog),
      patch.object(wifi_ui_module.gui_app, "push_widget") as push_widget,
    ):
      wifi_ui._on_need_auth("SavedNet")

    button.set_wrong_password.assert_called_once()
    push_widget.assert_called_once_with(dialog)

  def test_mici_hidden_wrong_password_defers_password_dialog(self):
    try:
      from openpilot.selfdrive.ui.mici.layouts.settings.network import wifi_ui as wifi_ui_module
      from openpilot.selfdrive.ui.mici.layouts.settings.network.wifi_ui import WifiUIMici
    except ImportError as e:
      raise SkipTest("mici UI dependencies are unavailable") from e

    class WifiButton:
      def __init__(self):
        self.network = MagicMock(ssid="SavedNet")
        self.set_wrong_password = MagicMock()

    button = WifiButton()
    wifi_ui = WifiUIMici.__new__(WifiUIMici)
    wifi_ui._scroller = MagicMock(items=[button])
    wifi_ui._shown = False

    with (
      patch.object(wifi_ui_module, "WifiButton", WifiButton),
      patch.object(wifi_ui_module.gui_app, "push_widget") as push_widget,
    ):
      wifi_ui._on_need_auth("SavedNet")

    button.set_wrong_password.assert_called_once()
    push_widget.assert_not_called()

  def test_mici_wrong_password_opens_auth_on_network_tap(self):
    try:
      from openpilot.selfdrive.ui.mici.layouts.settings.network import wifi_ui as wifi_ui_module
      from openpilot.selfdrive.ui.mici.layouts.settings.network.wifi_ui import WifiUIMici
    except ImportError as e:
      raise SkipTest("mici UI dependencies are unavailable") from e

    class WifiButton:
      def __init__(self):
        self.network = MagicMock(ssid="SavedNet")
        self.wrong_password = True

    button = WifiButton()
    wifi_ui = WifiUIMici.__new__(WifiUIMici)
    wifi_ui._wifi_manager = MagicMock()
    wifi_ui._wifi_manager.is_tethering_active.return_value = False
    wifi_ui._wifi_manager.is_connection_saved.return_value = True
    wifi_ui._networks = {"SavedNet": MagicMock(ssid="SavedNet")}
    wifi_ui._scroller = MagicMock(items=[button])
    wifi_ui._on_need_auth = MagicMock()
    wifi_ui._move_network_to_front = MagicMock()

    with patch.object(wifi_ui_module, "WifiButton", WifiButton):
      wifi_ui._connect_to_network("SavedNet")

    wifi_ui._on_need_auth.assert_called_once_with("SavedNet", False)
    wifi_ui._wifi_manager.activate_connection.assert_not_called()

  def test_mici_wifi_page_leaves_manager_active_for_parent(self):
    try:
      from openpilot.selfdrive.ui.mici.layouts.settings.network import wifi_ui as wifi_ui_module
      from openpilot.selfdrive.ui.mici.layouts.settings.network.wifi_ui import WifiUIMici
    except ImportError as e:
      raise SkipTest("mici UI dependencies are unavailable") from e

    wifi_ui = WifiUIMici.__new__(WifiUIMici)
    wifi_ui._wifi_manager = MagicMock(networks=[])
    wifi_ui._update_buttons = MagicMock()

    with (
      patch.object(wifi_ui_module.NavScroller, "show_event"),
      patch.object(wifi_ui_module.NavScroller, "hide_event"),
    ):
      wifi_ui.show_event()
      wifi_ui.hide_event()

    wifi_ui._wifi_manager.set_active.assert_not_called()
