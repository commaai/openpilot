import pyray as rl
from dataclasses import dataclass
from enum import IntEnum
from collections.abc import Callable

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


class PanelType(IntEnum):
  TOGGLES = 0
  NETWORK = 1
  DEVICE = 2
  DEVELOPER = 3
  USER_MANUAL = 4
  FIREHOSE = 5


@dataclass
class PanelInfo:
  name: str
  instance: Widget


class SettingsLayout(NavWidget):
  def __init__(self):
    super().__init__()
    self._params = Params()
    self._current_panel = None  # PanelType.DEVICE

    toggles_btn = BigButton("toggles", "", "icons_mici/settings/toggles_icon.png")
    toggles_btn.set_click_callback(lambda: self._set_current_panel(PanelType.TOGGLES))
    network_btn = BigButton("network", "", "icons_mici/settings/network/wifi_strength_full.png")
    network_btn.set_click_callback(lambda: self._set_current_panel(PanelType.NETWORK))
    device_btn = BigButton("device", "", "icons_mici/settings/device_icon.png")
    device_btn.set_click_callback(lambda: self._set_current_panel(PanelType.DEVICE))
    developer_btn = BigButton("developer", "", "icons_mici/settings/developer_icon.png")
    developer_btn.set_click_callback(lambda: self._set_current_panel(PanelType.DEVELOPER))

    firehose_btn = BigButton("firehose", "", "icons_mici/settings/comma_icon.png")
    firehose_btn.set_click_callback(lambda: self._set_current_panel(PanelType.FIREHOSE))

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
    self.set_back_callback(self.close_settings)
    self.set_back_enabled(lambda: self._current_panel is None)

    self._panels = {
      PanelType.TOGGLES: PanelInfo("Toggles", TogglesLayoutMici(back_callback=lambda: self._set_current_panel(None))),
      PanelType.NETWORK: PanelInfo("Network", NetworkLayoutMici(back_callback=lambda: self._set_current_panel(None))),
      PanelType.DEVICE: PanelInfo("Device", DeviceLayoutMici(back_callback=lambda: self._set_current_panel(None))),
      PanelType.DEVELOPER: PanelInfo("Developer", DeveloperLayoutMici(back_callback=lambda: self._set_current_panel(None))),
      PanelType.FIREHOSE: PanelInfo("Firehose", FirehoseLayout(back_callback=lambda: self._set_current_panel(None))),
    }

    self._font_medium = gui_app.font(FontWeight.MEDIUM)

    # Callbacks
    self._close_callback: Callable | None = None

  def show_event(self):
    super().show_event()
    self._set_current_panel(None)
    self._scroller.show_event()
    if self._current_panel is not None:
      self._panels[self._current_panel].instance.show_event()

  def hide_event(self):
    super().hide_event()
    if self._current_panel is not None:
      self._panels[self._current_panel].instance.hide_event()

  def set_callbacks(self, on_close: Callable):
    self._close_callback = on_close

  def _render(self, rect: rl.Rectangle):
    if self._current_panel is not None:
      self._draw_current_panel()
    else:
      self._scroller.render(rect)

  def _draw_current_panel(self):
    panel = self._panels[self._current_panel]
    panel.instance.render(self._rect)

  def _set_current_panel(self, panel_type: PanelType | None):
    if panel_type != self._current_panel:
      if self._current_panel is not None:
        self._panels[self._current_panel].instance.hide_event()
      self._current_panel = panel_type
      if self._current_panel is not None:
        self._panels[self._current_panel].instance.show_event()

  def close_settings(self):
    if self._close_callback:
      self._close_callback()
