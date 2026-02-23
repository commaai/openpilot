import pyray as rl

from openpilot.system.ui.widgets.scroller import Scroller
from openpilot.selfdrive.ui.mici.layouts.settings.network.wifi_ui import WifiUIMici, WifiIcon
from openpilot.selfdrive.ui.mici.widgets.button import BigButton, BigMultiToggle, BigParamControl, BigToggle
from openpilot.selfdrive.ui.mici.widgets.dialog import BigInputDialog
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.selfdrive.ui.lib.prime_state import PrimeType
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.widgets.nav_widget import NavWidget
from openpilot.system.ui.lib.wifi_manager import WifiManager, Network, MeteredType, ConnectStatus, SecurityType, normalize_ssid


class WifiNetworkButton(BigButton):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._lock_txt = gui_app.texture("icons_mici/settings/network/new/lock.png", 28, 36)
    self._draw_lock = False

  def set_draw_lock(self, draw: bool):
    self._draw_lock = draw

  def _draw_content(self, btn_y: float):
    super()._draw_content(btn_y)
    # Render lock icon at lower right of wifi icon if secured
    if self._draw_lock:
      icon_x = self._rect.x + self._rect.width - 30 - self._txt_icon.width
      icon_y = btn_y + 30
      lock_x = icon_x + self._txt_icon.width - self._lock_txt.width + 7
      lock_y = icon_y + self._txt_icon.height - self._lock_txt.height + 8
      rl.draw_texture_ex(self._lock_txt, (lock_x, lock_y), 0.0, 1.0, rl.WHITE)


class NetworkLayoutMici(NavWidget):
  def __init__(self):
    super().__init__()

    self._wifi_manager = WifiManager()
    self._wifi_manager.set_active(False)
    self._wifi_ui = WifiUIMici(self._wifi_manager)

    self._wifi_manager.add_callbacks(
      networks_updated=self._on_network_updated,
    )

    # ******** Tethering ********
    def tethering_toggle_callback(checked: bool):
      self._tethering_toggle_btn.set_enabled(False)
      self._tethering_password_btn.set_enabled(False)
      self._network_metered_btn.set_enabled(False)
      self._wifi_manager.set_tethering_active(checked)

    self._tethering_toggle_btn = BigToggle("enable tethering", "", toggle_callback=tethering_toggle_callback)

    def tethering_password_callback(password: str):
      if password:
        self._tethering_toggle_btn.set_enabled(False)
        self._tethering_password_btn.set_enabled(False)
        self._wifi_manager.set_tethering_password(password)

    def tethering_password_clicked():
      tethering_password = self._wifi_manager.tethering_password
      dlg = BigInputDialog("enter password...", tethering_password, minimum_length=8,
                           confirm_callback=tethering_password_callback)
      gui_app.push_widget(dlg)

    txt_tethering = gui_app.texture("icons_mici/settings/network/tethering.png", 64, 54)
    self._tethering_password_btn = BigButton("tethering password", "", txt_tethering)
    self._tethering_password_btn.set_click_callback(tethering_password_clicked)

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

    self._wifi_slash_txt = gui_app.texture("icons_mici/settings/network/wifi_strength_slash.png", 64, 56)
    self._wifi_low_txt = gui_app.texture("icons_mici/settings/network/wifi_strength_low.png", 64, 47)
    self._wifi_medium_txt = gui_app.texture("icons_mici/settings/network/wifi_strength_medium.png", 64, 47)
    self._wifi_full_txt = gui_app.texture("icons_mici/settings/network/wifi_strength_full.png", 64, 47)

    self._wifi_button = WifiNetworkButton("wi-fi", "not connected", self._wifi_slash_txt, scroll=True)
    self._wifi_button.set_click_callback(lambda: gui_app.push_widget(self._wifi_ui))

    # ******** Advanced settings ********
    # ******** Roaming toggle ********
    self._roaming_btn = BigParamControl("enable roaming", "GsmRoaming", toggle_callback=self._toggle_roaming)

    # ******** APN settings ********
    self._apn_btn = BigButton("apn settings", "edit")
    self._apn_btn.set_click_callback(self._edit_apn)

    # ******** Cellular metered toggle ********
    self._cellular_metered_btn = BigParamControl("cellular metered", "GsmMetered", toggle_callback=self._toggle_cellular_metered)

    # Main scroller ----------------------------------
    self._scroller = Scroller([
      self._wifi_button,
      self._network_metered_btn,
      self._tethering_toggle_btn,
      self._tethering_password_btn,
      # /* Advanced settings
      self._roaming_btn,
      self._apn_btn,
      self._cellular_metered_btn,
      # */
    ])

    # Set initial config
    roaming_enabled = ui_state.params.get_bool("GsmRoaming")
    metered = ui_state.params.get_bool("GsmMetered")
    self._wifi_manager.update_gsm_settings(roaming_enabled, ui_state.params.get("GsmApn") or "", metered)

    # Set up back navigation
    self.set_back_callback(gui_app.pop_widget)

  def _update_state(self):
    super()._update_state()

    # If not using prime SIM, show GSM settings and enable IPv4 forwarding
    show_cell_settings = ui_state.prime_state.get_type() in (PrimeType.NONE, PrimeType.LITE)
    self._wifi_manager.set_ipv4_forward(show_cell_settings)
    self._roaming_btn.set_visible(show_cell_settings)
    self._apn_btn.set_visible(show_cell_settings)
    self._cellular_metered_btn.set_visible(show_cell_settings)

    # Update wi-fi button with ssid and ip address
    # TODO: make sure we handle hidden ssids
    wifi_state = self._wifi_manager.wifi_state
    display_network = next((n for n in self._wifi_manager.networks if n.ssid == wifi_state.ssid), None)
    if wifi_state.status == ConnectStatus.CONNECTING:
      self._wifi_button.set_text(normalize_ssid(wifi_state.ssid or "wi-fi"))
      self._wifi_button.set_value("connecting...")
    elif wifi_state.status == ConnectStatus.CONNECTED:
      self._wifi_button.set_text(normalize_ssid(wifi_state.ssid or "wi-fi"))
      self._wifi_button.set_value(self._wifi_manager.ipv4_address or "obtaining IP...")
    else:
      display_network = None
      self._wifi_button.set_text("wi-fi")
      self._wifi_button.set_value("not connected")

    if display_network is not None:
      strength = WifiIcon.get_strength_icon_idx(display_network.strength)
      self._wifi_button.set_icon(self._wifi_full_txt if strength == 2 else self._wifi_medium_txt if strength == 1 else self._wifi_low_txt)
      self._wifi_button.set_draw_lock(display_network.security_type not in (SecurityType.OPEN, SecurityType.UNSUPPORTED))
    else:
      self._wifi_button.set_icon(self._wifi_slash_txt)
      self._wifi_button.set_draw_lock(False)

  def show_event(self):
    super().show_event()
    self._wifi_manager.set_active(True)
    self._scroller.show_event()

    # Process wifi callbacks while at any point in the nav stack
    gui_app.set_nav_stack_tick(self._wifi_manager.process_callbacks)

  def hide_event(self):
    super().hide_event()
    self._wifi_manager.set_active(False)

    gui_app.set_nav_stack_tick(None)

  def _toggle_roaming(self, checked: bool):
    self._wifi_manager.update_gsm_settings(checked, ui_state.params.get("GsmApn") or "", ui_state.params.get_bool("GsmMetered"))

  def _edit_apn(self):
    def update_apn(apn: str):
      apn = apn.strip()
      if apn == "":
        ui_state.params.remove("GsmApn")
      else:
        ui_state.params.put("GsmApn", apn)

      self._wifi_manager.update_gsm_settings(ui_state.params.get_bool("GsmRoaming"), apn, ui_state.params.get_bool("GsmMetered"))

    current_apn = ui_state.params.get("GsmApn") or ""
    dlg = BigInputDialog("enter APN...", current_apn, minimum_length=0, confirm_callback=update_apn)
    gui_app.push_widget(dlg)

  def _toggle_cellular_metered(self, checked: bool):
    self._wifi_manager.update_gsm_settings(ui_state.params.get_bool("GsmRoaming"), ui_state.params.get("GsmApn") or "", checked)

  def _on_network_updated(self, networks: list[Network]):
    # Update tethering state
    tethering_active = self._wifi_manager.is_tethering_active()
    # TODO: use real signals (like activated/settings changed, etc.) to speed up re-enabling buttons
    self._tethering_toggle_btn.set_enabled(True)
    self._tethering_password_btn.set_enabled(True)
    self._network_metered_btn.set_enabled(lambda: not tethering_active and bool(self._wifi_manager.ipv4_address))
    self._tethering_toggle_btn.set_checked(tethering_active)

    # Update network metered
    self._network_metered_btn.set_value(
      {
        MeteredType.UNKNOWN: 'default',
        MeteredType.YES: 'metered',
        MeteredType.NO: 'unmetered'
      }.get(self._wifi_manager.current_network_metered, 'default'))

  def _render(self, rect: rl.Rectangle):
    self._scroller.render(rect)
