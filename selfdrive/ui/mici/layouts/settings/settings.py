import pyray as rl

from openpilot.common.params import Params
from openpilot.system.ui.widgets.scroller import Scroller
from openpilot.selfdrive.ui.mici.widgets.button import BigButton
from openpilot.selfdrive.ui.mici.layouts.settings.toggles import TogglesLayoutMici
from openpilot.selfdrive.ui.mici.layouts.settings.network import NetworkLayoutMici
from openpilot.selfdrive.ui.mici.layouts.settings.device import DeviceLayoutMici, PairBigButton
from openpilot.selfdrive.ui.mici.layouts.settings.developer import DeveloperLayoutMici
from openpilot.selfdrive.ui.mici.layouts.settings.firehose import FirehoseLayout
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.widgets.nav_widget import NavWidget


class SettingsBigButton(BigButton):
  def _get_label_font_size(self):
    return 64


class SettingsLayout(NavWidget):
  def __init__(self):
    super().__init__()
    self._params = Params()

    toggles_panel = TogglesLayoutMici()
    toggles_btn = SettingsBigButton("toggles", "", "icons_mici/settings.png")
    # toggles_btn.set_click_callback(lambda: gui_app.push_widget(toggles_panel))
    toggles_btn.set_click_callback(lambda: self._scroller.move_item(self._scroller.items.index(toggles_btn), 0))

    network_panel = NetworkLayoutMici()
    network_btn = SettingsBigButton("network", "", "icons_mici/settings/network/wifi_strength_full.png", icon_size=(76, 56))
    # network_btn.set_click_callback(lambda: gui_app.push_widget(network_panel))
    network_btn.set_click_callback(lambda: self._scroller.move_item(self._scroller.items.index(network_btn), 0))

    device_panel = DeviceLayoutMici()
    device_btn = SettingsBigButton("device", "", "icons_mici/settings/device_icon.png", icon_size=(74, 60))
    # device_btn.set_click_callback(lambda: gui_app.push_widget(device_panel))
    device_btn.set_click_callback(lambda: self._scroller.move_item(self._scroller.items.index(device_btn), 0))

    developer_panel = DeveloperLayoutMici()
    developer_btn = SettingsBigButton("developer", "", "icons_mici/settings/developer_icon.png", icon_size=(64, 60))
    # developer_btn.set_click_callback(lambda: gui_app.push_widget(developer_panel))
    developer_btn.set_click_callback(lambda: self._scroller.move_item(self._scroller.items.index(developer_btn), 0))

    firehose_panel = FirehoseLayout()
    firehose_btn = SettingsBigButton("firehose", "", "icons_mici/settings/firehose.png", icon_size=(52, 62))
    # firehose_btn.set_click_callback(lambda: gui_app.push_widget(firehose_panel))
    firehose_btn.set_click_callback(lambda: self._scroller.move_item(self._scroller.items.index(firehose_btn), 0))

    self._scroller = Scroller([
      toggles_btn,
      network_btn,
      device_btn,
      # PairBigButton(),
      #BigDialogButton("manual", "", "icons_mici/settings/manual_icon.png", "Check out the mici user\nmanual at comma.ai/setup"),
      firehose_btn,
      developer_btn,
    ])

    # Set up back navigation
    self.set_back_callback(gui_app.pop_widget)

    self._font_medium = gui_app.font(FontWeight.MEDIUM)

  def show_event(self):
    super().show_event()
    self._scroller.show_event()

  def hide_event(self):
    super().hide_event()
    self._scroller.hide_event()

  def _render(self, rect: rl.Rectangle):
    self._scroller.render(rect)
