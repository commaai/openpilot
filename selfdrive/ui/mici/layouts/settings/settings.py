import pyray as rl
from dataclasses import dataclass
from enum import IntEnum

from openpilot.common.params import Params
from openpilot.system.ui.widgets.scroller import Scroller
from openpilot.selfdrive.ui.mici.widgets.button import BigButton
from openpilot.selfdrive.ui.mici.layouts.settings.toggles import TogglesLayoutMici
from openpilot.selfdrive.ui.mici.layouts.settings.network import NetworkLayoutMici
from openpilot.selfdrive.ui.mici.layouts.settings.device import DeviceLayoutMici, PairBigButton
from openpilot.selfdrive.ui.mici.layouts.settings.developer import DeveloperLayoutMici
from openpilot.selfdrive.ui.mici.layouts.settings.firehose import FirehoseLayout
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.widgets import Widget, NavWidget


class SettingsLayout(NavWidget):
  def __init__(self):
    super().__init__()
    self._params = Params()
    # self._current_panel = None  # PanelType.DEVICE

    toggles_panel = TogglesLayoutMici()
    network_panel = NetworkLayoutMici()
    device_panel = DeviceLayoutMici()
    developer_panel = DeveloperLayoutMici()
    firehose_panel = FirehoseLayout()

    toggles_btn = BigButton("toggles", "", "icons_mici/settings.png")
    toggles_btn.set_click_callback(lambda: gui_app.push_widget(toggles_panel))

    network_btn = BigButton("network", "", "icons_mici/settings/network/wifi_strength_full.png", icon_size=(76, 56))
    network_btn.set_click_callback(lambda: gui_app.push_widget(network_panel))

    device_btn = BigButton("device", "", "icons_mici/settings/device_icon.png", icon_size=(74, 60))
    device_btn.set_click_callback(lambda: gui_app.push_widget(device_panel))

    developer_btn = BigButton("developer", "", "icons_mici/settings/developer_icon.png", icon_size=(64, 60))
    developer_btn.set_click_callback(lambda: gui_app.push_widget(developer_panel))

    firehose_btn = BigButton("firehose", "", "icons_mici/settings/firehose.png", icon_size=(52, 62))
    firehose_btn.set_click_callback(lambda: gui_app.push_widget(firehose_panel))

    self._scroller = Scroller([
      toggles_btn,
      network_btn,
      device_btn,
      PairBigButton(),
      #BigDialogButton("manual", "", "icons_mici/settings/manual_icon.png", "Check out the mici user\nmanual at comma.ai/setup"),
      firehose_btn,
      developer_btn,
    ], snap_items=False)

    # Set up back navigation
    self.set_back_callback(gui_app.pop_widget)
    # self.set_back_enabled(lambda: self._current_panel is None)

    self._font_medium = gui_app.font(FontWeight.MEDIUM)

  def show_event(self):
    super().show_event()
    # self._set_current_panel(None)
    self._scroller.show_event()
    # if self._current_panel is not None:
    #   self._panels[self._current_panel].instance.show_event()

  def hide_event(self):
    super().hide_event()
    self._scroller.hide_event()
    # if self._current_panel is not None:
    #   self._panels[self._current_panel].instance.hide_event()

  def _render(self, rect: rl.Rectangle):
    # if self._current_panel is not None:
    #   self._draw_current_panel()
    # else:
    self._scroller.render(rect)
    return -1

  # def _draw_current_panel(self):
  #   panel = self._panels[self._current_panel]
  #   panel.instance.render(self._rect)

  # def _set_current_panel(self, panel_type: PanelType | None):
  #   if panel_type is None:
  #     # TODO: move this into each layout's class above
  #     gui_app.pop_widget()
  #   else:
  #     gui_app.push_widget(self._panels[panel_type].instance)
  #   # if panel_type != self._current_panel:
  #   #   if self._current_panel is not None:
  #   #     self._panels[self._current_panel].instance.hide_event()
  #   #   self._current_panel = panel_type
  #   #   if self._current_panel is not None:
  #   #     self._panels[self._current_panel].instance.show_event()
