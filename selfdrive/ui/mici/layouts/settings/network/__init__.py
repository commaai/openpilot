import pyray as rl

from cereal import log
from openpilot.system.hardware import HARDWARE
from openpilot.system.ui.lib.cellular_manager import CellularManager
from openpilot.selfdrive.ui.mici.layouts.settings.network.wifi_ui import WifiIcon
from openpilot.selfdrive.ui.mici.widgets.button import BigButton
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.lib.wifi_manager import WifiManager, ConnectStatus, SecurityType, normalize_ssid

NetworkType = log.DeviceState.NetworkType
NetworkStrength = log.DeviceState.NetworkStrength


class ESimNetworkButton(BigButton):
  def __init__(self, cellular_manager: CellularManager):
    self._cellular_manager = cellular_manager
    self._cell_none_icon = gui_app.texture("icons_mici/settings/network/cell_strength_none.png", 64, 47)
    self._cell_low_icon = gui_app.texture("icons_mici/settings/network/cell_strength_low.png", 64, 47)
    self._cell_medium_icon = gui_app.texture("icons_mici/settings/network/cell_strength_medium.png", 64, 47)
    self._cell_high_icon = gui_app.texture("icons_mici/settings/network/cell_strength_high.png", 64, 47)
    self._cell_full_icon = gui_app.texture("icons_mici/settings/network/cell_strength_full.png", 64, 47)
    super().__init__("esim", "no active profile", self._cell_none_icon, scroll=True)

  def _update_state(self):
    super()._update_state()

    if self._cellular_manager.is_euicc is False:
      ms = self._cellular_manager.modem_state
      iccid = ms.get("iccid") or ""
      if iccid:
        self.set_text(f"sim (...{iccid[-4:]})")
        self.set_value(self._cellular_manager.modem_ip or ms.get("mcc_mnc") or "no IP")
      else:
        self.set_text("sim")
        self.set_value("no sim")
      self.set_icon(self._get_cell_icon() if iccid else self._cell_none_icon)
      return

    if self._cellular_manager.switching_iccid is not None:
      self.set_text("esim")
      self.set_value("switching...")
      self.set_icon(self._cell_none_icon)
    else:
      active = next((p for p in self._cellular_manager.profiles if p.enabled), None)
      if active:
        self.set_text(active.display_name)
        self.set_value(self._cellular_manager.modem_ip or "obtaining IP...")
        self.set_icon(self._get_cell_icon())
      else:
        self.set_text("esim")
        self.set_value("no active profile")
        self.set_icon(self._cell_none_icon)

  def _get_cell_icon(self):
    # always read cell strength from HARDWARE so it reflects modem state even when wifi is the active connection
    strength = HARDWARE.get_network_strength(NetworkType.cell4G)
    icons = {
      NetworkStrength.unknown: self._cell_none_icon,
      NetworkStrength.poor: self._cell_low_icon,
      NetworkStrength.moderate: self._cell_medium_icon,
      NetworkStrength.good: self._cell_high_icon,
      NetworkStrength.great: self._cell_full_icon,
    }
    return icons.get(strength, self._cell_none_icon)


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
