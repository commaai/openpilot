import math
import numpy as np
import pyray as rl
from enum import IntEnum
from collections.abc import Callable

from openpilot.common.swaglog import cloudlog
from openpilot.system.ui.widgets.scroller import Scroller
from openpilot.system.ui.widgets.label import UnifiedLabel
from openpilot.selfdrive.ui.mici.layouts.settings.network.wifi_ui import WifiUIMici
from openpilot.selfdrive.ui.mici.widgets.button import BigButton, BigMultiToggle, BigToggle
from openpilot.selfdrive.ui.mici.widgets.dialog import BigMultiOptionDialog, BigInputDialog, BigDialogOptionButton, BigConfirmationDialogV2
from openpilot.system.ui.lib.application import gui_app, MousePos, FontWeight
from openpilot.system.ui.widgets import Widget, NavWidget
from openpilot.system.ui.lib.wifi_manager import WifiManager, Network, SecurityType, MeteredType


class NetworkPanelType(IntEnum):
  NONE = 0
  WIFI = 1


class NetworkLayoutMici(NavWidget):
  def __init__(self, back_callback: Callable):
    super().__init__()

    self._current_panel = NetworkPanelType.WIFI
    self.set_back_enabled(lambda: self._current_panel == NetworkPanelType.NONE)

    self._wifi_manager = WifiManager()
    self._wifi_manager.set_active(False)
    self._wifi_ui = WifiUIMici(self._wifi_manager, back_callback=lambda: self._switch_to_panel(NetworkPanelType.NONE))

    self._wifi_manager.add_callbacks(
      networks_updated=self._on_network_updated,
    )

    _tethering_icon = "icons_mici/settings/network/tethering.png"

    # ******** Tethering ********
    def tethering_toggle_callback(checked: bool):
      self._tethering_toggle_btn.set_enabled(False)
      self._network_metered_btn.set_enabled(False)
      self._wifi_manager.set_tethering_active(checked)

    self._tethering_toggle_btn = BigToggle("enable tethering", "", toggle_callback=tethering_toggle_callback)

    def tethering_password_callback(password: str):
      if password:
        self._wifi_manager.set_tethering_password(password)

    def tethering_password_clicked():
      tethering_password = self._wifi_manager.tethering_password
      dlg = BigInputDialog("enter password...", tethering_password, minimum_length=8,
                           confirm_callback=tethering_password_callback)
      gui_app.set_modal_overlay(dlg)

    txt_tethering = gui_app.texture(_tethering_icon, 64, 53)
    self._tethering_password_btn = BigButton("tethering password", "", txt_tethering)
    self._tethering_password_btn.set_click_callback(tethering_password_clicked)

    # ******** IP Address ********
    self._ip_address_btn = BigButton("IP Address", "Not connected")

    # ******** Network Metered ********
    def network_metered_callback(value: str):
      self._network_metered_btn.set_enabled(False)
      metered = {
        'default': MeteredType.UNKNOWN,
        'metered': MeteredType.YES,
        'unmetered': MeteredType.NO
      }.get(value, MeteredType.UNKNOWN)
      self._wifi_manager.set_current_network_metered(metered)

    # TODO: signal for current network metered type when changing networks, this is wrong until you press it once
    # TODO: disable when not connected
    self._network_metered_btn = BigMultiToggle("network usage", ["default", "metered", "unmetered"], select_callback=network_metered_callback)
    self._network_metered_btn.set_enabled(False)

    wifi_button = BigButton("wi-fi")
    wifi_button.set_click_callback(lambda: self._switch_to_panel(NetworkPanelType.WIFI))

    # Main scroller ----------------------------------
    self._scroller = Scroller([
      wifi_button,
      self._network_metered_btn,
      self._tethering_toggle_btn,
      self._tethering_password_btn,
      self._ip_address_btn,
    ], snap_items=False)

    # Set up back navigation
    self.set_back_callback(back_callback)

  def show_event(self):
    super().show_event()
    self._current_panel = NetworkPanelType.NONE
    self._wifi_ui.show_event()
    self._scroller.show_event()

  def hide_event(self):
    super().hide_event()
    self._wifi_ui.hide_event()

  def _on_network_updated(self, networks: list[Network]):
    # Update tethering state
    tethering_active = self._wifi_manager.is_tethering_active()
    # TODO: use real signals (like activated/settings changed, etc.) to speed up re-enabling buttons
    self._tethering_toggle_btn.set_enabled(True)
    self._network_metered_btn.set_enabled(lambda: not tethering_active and bool(self._wifi_manager.ipv4_address))
    self._tethering_toggle_btn.set_checked(tethering_active)

    # Update IP address
    self._ip_address_btn.set_value(self._wifi_manager.ipv4_address or "Not connected")

    # Update network metered
    self._network_metered_btn.set_value(
      {
        MeteredType.UNKNOWN: 'default',
        MeteredType.YES: 'metered',
        MeteredType.NO: 'unmetered'
      }.get(self._wifi_manager.current_network_metered, 'default'))

  def _switch_to_panel(self, panel_type: NetworkPanelType):
    self._current_panel = panel_type

  def _render(self, rect: rl.Rectangle):
    self._wifi_manager.process_callbacks()

    if self._current_panel == NetworkPanelType.WIFI:
      self._wifi_ui.render(rect)
    else:
      self._scroller.render(rect)
