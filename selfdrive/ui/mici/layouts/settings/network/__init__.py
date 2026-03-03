import pyray as rl

from openpilot.selfdrive.ui.mici.layouts.settings.network.wifi_ui import WifiIcon
from openpilot.selfdrive.ui.mici.widgets.button import BigButton
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.lib.wifi_manager import WifiManager, ConnectStatus, SecurityType, normalize_ssid


class WifiNetworkButton(BigButton):
  def __init__(self, wifi_manager: WifiManager):
    self._wifi_manager = wifi_manager
    self._lock_txt = gui_app.texture("icons_mici/settings/network/new/lock.png", 28, 36)
    self._draw_lock = False

    self._wifi_slash_txt = gui_app.texture("icons_mici/settings/network/wifi_strength_slash.png", 64, 56)
    self._wifi_low_txt = gui_app.texture("icons_mici/settings/network/wifi_strength_low.png", 64, 47)
    self._wifi_medium_txt = gui_app.texture("icons_mici/settings/network/wifi_strength_medium.png", 64, 47)
    self._wifi_full_txt = gui_app.texture("icons_mici/settings/network/wifi_strength_full.png", 64, 47)

    super().__init__("wi-fi", "not connected", self._wifi_slash_txt, scroll=True)

  def _update_state(self):
    super()._update_state()

    # Update wi-fi button with ssid and ip address
    # TODO: make sure we handle hidden ssids
    wifi_state = self._wifi_manager.wifi_state
    display_network = next((n for n in self._wifi_manager.networks if n.ssid == wifi_state.ssid), None)
    if wifi_state.status == ConnectStatus.CONNECTING:
      self.set_text(normalize_ssid(wifi_state.ssid or "wi-fi"))
      self.set_value("starting" if self._wifi_manager.is_tethering_active() else "connecting...")
    elif wifi_state.status == ConnectStatus.CONNECTED:
      self.set_text(normalize_ssid(wifi_state.ssid or "wi-fi"))
      self.set_value(self._wifi_manager.ipv4_address or "obtaining IP...")
    else:
      display_network = None
      self.set_text("wi-fi")
      self.set_value("not connected")

    if display_network is not None:
      strength = WifiIcon.get_strength_icon_idx(display_network.strength)
      self.set_icon(self._wifi_full_txt if strength == 2 else self._wifi_medium_txt if strength == 1 else self._wifi_low_txt)
      self._draw_lock = display_network.security_type not in (SecurityType.OPEN, SecurityType.UNSUPPORTED)
    elif self._wifi_manager.is_tethering_active():
      # takes a while to get Network
      self.set_icon(self._wifi_full_txt)
      self._draw_lock = True
    else:
      self.set_icon(self._wifi_slash_txt)
      self._draw_lock = False

  def _draw_content(self, btn_y: float):
    super()._draw_content(btn_y)
    # Render lock icon at lower right of wifi icon if secured
    if self._draw_lock:
      icon_x = self._rect.x + self._rect.width - 30 - self._txt_icon.width
      icon_y = btn_y + 30
      lock_x = icon_x + self._txt_icon.width - self._lock_txt.width + 7
      lock_y = icon_y + self._txt_icon.height - self._lock_txt.height + 8
      rl.draw_texture_ex(self._lock_txt, (lock_x, lock_y), 0.0, 1.0, rl.WHITE)
