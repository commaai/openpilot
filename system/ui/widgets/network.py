from enum import IntEnum
from functools import partial
from typing import cast

import pyray as rl
from openpilot.common.params import Params
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.lib.scroll_panel import GuiScrollPanel
from openpilot.system.ui.lib.wifi_manager import WifiManager, SecurityType, Network, MeteredType
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.button import ButtonStyle, Button
from openpilot.system.ui.widgets.confirm_dialog import ConfirmDialog
from openpilot.system.ui.widgets.keyboard import Keyboard
from openpilot.system.ui.widgets.label import TextAlignment, gui_label
from openpilot.system.ui.widgets.scroller import Scroller
from openpilot.system.ui.widgets.list_view import ButtonAction, ListItem, MultipleButtonAction, ToggleAction, button_item, text_item
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.selfdrive.ui.lib.prime_state import PrimeType

NM_DEVICE_STATE_NEED_AUTH = 60
MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_LENGTH = 64
ITEM_HEIGHT = 160
ICON_SIZE = 50

STRENGTH_ICONS = [
  "icons/wifi_strength_low.png",
  "icons/wifi_strength_medium.png",
  "icons/wifi_strength_high.png",
  "icons/wifi_strength_full.png",
]


class PanelType(IntEnum):
  WIFI = 0
  ADVANCED = 1


class UIState(IntEnum):
  IDLE = 0
  CONNECTING = 1
  NEEDS_AUTH = 2
  SHOW_FORGET_CONFIRM = 3
  FORGETTING = 4


class NavButton(Widget):
  def __init__(self, text: str):
    super().__init__()
    self.text = text
    self.set_rect(rl.Rectangle(0, 0, 400, 100))

  def _render(self, _):
    color = rl.Color(74, 74, 74, 255) if self.is_pressed else rl.Color(57, 57, 57, 255)
    rl.draw_rectangle_rounded(self._rect, 0.6, 10, color)
    gui_label(self.rect, self.text, font_size=60, alignment=rl.GuiTextAlignment.TEXT_ALIGN_CENTER)


class NetworkUI(Widget):
  def __init__(self, wifi_manager: WifiManager):
    super().__init__()
    self._wifi_manager = wifi_manager
    self._current_panel: PanelType = PanelType.WIFI
    self._wifi_panel = WifiManagerUI(wifi_manager)
    self._advanced_panel = AdvancedNetworkSettings(wifi_manager)
    self._nav_button = NavButton("Advanced")
    self._nav_button.set_click_callback(self._cycle_panel)

  def show_event(self):
    self._set_current_panel(PanelType.WIFI)
    self._wifi_panel.show_event()

  def hide_event(self):
    self._wifi_panel.hide_event()

  def _cycle_panel(self):
    if self._current_panel == PanelType.WIFI:
      self._set_current_panel(PanelType.ADVANCED)
    else:
      self._set_current_panel(PanelType.WIFI)

  def _render(self, _):
    # subtract button
    content_rect = rl.Rectangle(self._rect.x, self._rect.y + self._nav_button.rect.height + 20,
                                self._rect.width, self._rect.height - self._nav_button.rect.height - 20)
    if self._current_panel == PanelType.WIFI:
      self._nav_button.text = "Advanced"
      self._nav_button.set_position(self._rect.x + self._rect.width - self._nav_button.rect.width, self._rect.y + 10)
      self._wifi_panel.render(content_rect)
    else:
      self._nav_button.text = "Back"
      self._nav_button.set_position(self._rect.x, self._rect.y + 10)
      self._advanced_panel.render(content_rect)

    self._nav_button.render()

  def _set_current_panel(self, panel: PanelType):
    self._current_panel = panel


class AdvancedNetworkSettings(Widget):
  def __init__(self, wifi_manager: WifiManager):
    super().__init__()
    self._wifi_manager = wifi_manager
    self._wifi_manager.set_callbacks(networks_updated=self._on_network_updated)
    self._params = Params()

    self._keyboard = Keyboard(max_text_size=MAX_PASSWORD_LENGTH, min_text_size=MIN_PASSWORD_LENGTH, show_password_toggle=True)

    # Tethering
    self._tethering_action = ToggleAction(initial_state=False)
    tethering_btn = ListItem(title="Enable Tethering", action_item=self._tethering_action, callback=self._toggle_tethering)

    # Edit tethering password
    self._tethering_password_action = ButtonAction(text="EDIT")
    tethering_password_btn = ListItem(title="Tethering Password", action_item=self._tethering_password_action, callback=self._edit_tethering_password)

    # Roaming toggle
    roaming_enabled = self._params.get_bool("GsmRoaming")
    self._roaming_action = ToggleAction(initial_state=roaming_enabled)
    self._roaming_btn = ListItem(title="Enable Roaming", action_item=self._roaming_action, callback=self._toggle_roaming)

    # Cellular metered toggle
    cellular_metered = self._params.get_bool("GsmMetered")
    self._cellular_metered_action = ToggleAction(initial_state=cellular_metered)
    self._cellular_metered_btn = ListItem(title="Cellular Metered", description="Prevent large data uploads when on a metered cellular connection",
                                          action_item=self._cellular_metered_action, callback=self._toggle_cellular_metered)

    # APN setting
    self._apn_btn = button_item("APN Setting", "EDIT", callback=self._edit_apn)

    # Wi-Fi metered toggle
    self._wifi_metered_action = MultipleButtonAction(["default", "metered", "unmetered"], 255, 0, callback=self._toggle_wifi_metered)
    wifi_metered_btn = ListItem(title="Wi-Fi Network Metered", description="Prevent large data uploads when on a metered Wi-Fi connection",
                                action_item=self._wifi_metered_action)

    items: list[Widget] = [
      tethering_btn,
      tethering_password_btn,
      text_item("IP Address", lambda: self._wifi_manager.ipv4_address),
      self._roaming_btn,
      self._apn_btn,
      self._cellular_metered_btn,
      wifi_metered_btn,
      button_item("Hidden Network", "CONNECT", callback=self._connect_to_hidden_network),
    ]

    self._scroller = Scroller(items, line_separator=True, spacing=0)

    # Set initial config
    metered = self._params.get_bool("GsmMetered")
    self._wifi_manager.update_gsm_settings(roaming_enabled, self._params.get("GsmApn") or "", metered)

  def _on_network_updated(self, networks: list[Network]):
    self._tethering_action.set_enabled(True)
    self._tethering_action.set_state(self._wifi_manager.is_tethering_active())
    self._tethering_password_action.set_enabled(True)

    if self._wifi_manager.is_tethering_active() or self._wifi_manager.ipv4_address == "":
      self._wifi_metered_action.set_enabled(False)
      self._wifi_metered_action.selected_button = 0
    elif self._wifi_manager.ipv4_address != "":
      metered = self._wifi_manager.current_network_metered
      self._wifi_metered_action.set_enabled(True)
      self._wifi_metered_action.selected_button = int(metered) if metered in (MeteredType.UNKNOWN, MeteredType.YES, MeteredType.NO) else 0

  def _toggle_tethering(self):
    checked = self._tethering_action.state
    self._tethering_action.set_enabled(False)
    if checked:
      self._wifi_metered_action.set_enabled(False)
    self._wifi_manager.set_tethering_active(checked)

  def _toggle_roaming(self):
    roaming_state = self._roaming_action.state
    self._params.put_bool("GsmRoaming", roaming_state)
    self._wifi_manager.update_gsm_settings(roaming_state, self._params.get("GsmApn") or "", self._params.get_bool("GsmMetered"))

  def _edit_apn(self):
    def update_apn(result):
      if result != 1:
        return

      apn = self._keyboard.text.strip()
      if apn == "":
        self._params.remove("GsmApn")
      else:
        self._params.put("GsmApn", apn)

      self._wifi_manager.update_gsm_settings(self._params.get_bool("GsmRoaming"), apn, self._params.get_bool("GsmMetered"))

    current_apn = self._params.get("GsmApn") or ""
    self._keyboard.reset(min_text_size=0)
    self._keyboard.set_title("Enter APN", "leave blank for automatic configuration")
    self._keyboard.set_text(current_apn)
    gui_app.set_modal_overlay(self._keyboard, update_apn)

  def _toggle_cellular_metered(self):
    metered = self._cellular_metered_action.state
    self._params.put_bool("GsmMetered", metered)
    self._wifi_manager.update_gsm_settings(self._params.get_bool("GsmRoaming"), self._params.get("GsmApn") or "", metered)

  def _toggle_wifi_metered(self, metered):
    metered_type = {0: MeteredType.UNKNOWN, 1: MeteredType.YES, 2: MeteredType.NO}.get(metered, MeteredType.UNKNOWN)
    self._wifi_metered_action.set_enabled(False)
    self._wifi_manager.set_current_network_metered(metered_type)

  def _connect_to_hidden_network(self):
    def connect_hidden(result):
      if result != 1:
        return

      ssid = self._keyboard.text
      if not ssid:
        return

      def enter_password(result):
        password = self._keyboard.text
        if password == "":
          # connect without password
          self._wifi_manager.connect_to_network(ssid, "", hidden=True)
          return

        self._wifi_manager.connect_to_network(ssid, password, hidden=True)

      self._keyboard.reset(min_text_size=0)
      self._keyboard.set_title("Enter password", f"for \"{ssid}\"")
      gui_app.set_modal_overlay(self._keyboard, enter_password)

    self._keyboard.reset(min_text_size=1)
    self._keyboard.set_title("Enter SSID", "")
    gui_app.set_modal_overlay(self._keyboard, connect_hidden)

  def _edit_tethering_password(self):
    def update_password(result):
      if result != 1:
        return

      password = self._keyboard.text
      self._wifi_manager.set_tethering_password(password)
      self._tethering_password_action.set_enabled(False)

    self._keyboard.reset(min_text_size=MIN_PASSWORD_LENGTH)
    self._keyboard.set_title("Enter new tethering password", "")
    self._keyboard.set_text(self._wifi_manager.tethering_password)
    gui_app.set_modal_overlay(self._keyboard, update_password)

  def _update_state(self):
    print('AN process callbacks')
    self._wifi_manager.process_callbacks()

    # If not using prime SIM, show GSM settings and enable IPv4 forwarding
    show_cell_settings = ui_state.prime_state.get_type() in (PrimeType.NONE, PrimeType.LITE)
    self._wifi_manager.set_ipv4_forward(show_cell_settings)
    self._roaming_btn.set_visible(show_cell_settings)
    self._apn_btn.set_visible(show_cell_settings)
    self._cellular_metered_btn.set_visible(show_cell_settings)

  def _render(self, _):
    self._scroller.render(self._rect)


class WifiManagerUI(Widget):
  def __init__(self, wifi_manager: WifiManager):
    super().__init__()
    self._wifi_manager = wifi_manager
    self.state: UIState = UIState.IDLE
    self._state_network: Network | None = None  # for CONNECTING / NEEDS_AUTH / SHOW_FORGET_CONFIRM / FORGETTING
    self._password_retry: bool = False  # for NEEDS_AUTH
    self.btn_width: int = 200
    self.scroll_panel = GuiScrollPanel()
    self.keyboard = Keyboard(max_text_size=MAX_PASSWORD_LENGTH, min_text_size=MIN_PASSWORD_LENGTH, show_password_toggle=True)
    self._load_icons()

    self._networks: list[Network] = []
    self._networks_buttons: dict[str, Button] = {}
    self._forget_networks_buttons: dict[str, Button] = {}
    self._confirm_dialog = ConfirmDialog("", "Forget", "Cancel")

    self._wifi_manager.set_callbacks(need_auth=self._on_need_auth,
                                     activated=self._on_activated,
                                     forgotten=self._on_forgotten,
                                     networks_updated=self._on_network_updated,
                                     disconnected=self._on_disconnected)

  def show_event(self):
    # start/stop scanning when widget is visible
    self._wifi_manager.set_active(True)
    print('wifi active')

  def hide_event(self):
    self._wifi_manager.set_active(False)
    print('wifi deactive')

  def _load_icons(self):
    for icon in STRENGTH_ICONS + ["icons/checkmark.png", "icons/circled_slash.png", "icons/lock_closed.png"]:
      gui_app.texture(icon, ICON_SIZE, ICON_SIZE)

  def _update_state(self):
    print('WM process callbacks')
    self._wifi_manager.process_callbacks()

  def _render(self, rect: rl.Rectangle):
    if not self._networks:
      gui_label(rect, "Scanning Wi-Fi networks...", 72, alignment=rl.GuiTextAlignment.TEXT_ALIGN_CENTER)
      return

    if self.state == UIState.NEEDS_AUTH and self._state_network:
      self.keyboard.set_title("Wrong password" if self._password_retry else "Enter password", f"for {self._state_network.ssid}")
      self.keyboard.reset(min_text_size=MIN_PASSWORD_LENGTH)
      gui_app.set_modal_overlay(self.keyboard, lambda result: self._on_password_entered(cast(Network, self._state_network), result))
    elif self.state == UIState.SHOW_FORGET_CONFIRM and self._state_network:
      self._confirm_dialog.set_text(f'Forget Wi-Fi Network "{self._state_network.ssid}"?')
      self._confirm_dialog.reset()
      gui_app.set_modal_overlay(self._confirm_dialog, callback=lambda result: self.on_forgot_confirm_finished(self._state_network, result))
    else:
      self._draw_network_list(rect)

  def _on_password_entered(self, network: Network, result: int):
    if result == 1:
      password = self.keyboard.text
      self.keyboard.clear()

      if len(password) >= MIN_PASSWORD_LENGTH:
        self.connect_to_network(network, password)
    elif result == 0:
      self.state = UIState.IDLE

  def on_forgot_confirm_finished(self, network, result: int):
    if result == 1:
      self.forget_network(network)
    elif result == 0:
      self.state = UIState.IDLE

  def _draw_network_list(self, rect: rl.Rectangle):
    content_rect = rl.Rectangle(rect.x, rect.y, rect.width, len(self._networks) * ITEM_HEIGHT)
    offset = self.scroll_panel.update(rect, content_rect)

    rl.begin_scissor_mode(int(rect.x), int(rect.y), int(rect.width), int(rect.height))
    for i, network in enumerate(self._networks):
      y_offset = rect.y + i * ITEM_HEIGHT + offset
      item_rect = rl.Rectangle(rect.x, y_offset, rect.width, ITEM_HEIGHT)
      if not rl.check_collision_recs(item_rect, rect):
        continue

      self._draw_network_item(item_rect, network)
      if i < len(self._networks) - 1:
        line_y = int(item_rect.y + item_rect.height - 1)
        rl.draw_line(int(item_rect.x), int(line_y), int(item_rect.x + item_rect.width), line_y, rl.LIGHTGRAY)

    rl.end_scissor_mode()

  def _draw_network_item(self, rect, network: Network):
    spacing = 50
    ssid_rect = rl.Rectangle(rect.x, rect.y, rect.width - self.btn_width * 2, ITEM_HEIGHT)
    signal_icon_rect = rl.Rectangle(rect.x + rect.width - ICON_SIZE, rect.y + (ITEM_HEIGHT - ICON_SIZE) / 2, ICON_SIZE, ICON_SIZE)
    security_icon_rect = rl.Rectangle(signal_icon_rect.x - spacing - ICON_SIZE, rect.y + (ITEM_HEIGHT - ICON_SIZE) / 2, ICON_SIZE, ICON_SIZE)

    status_text = ""
    if self.state == UIState.CONNECTING and self._state_network:
      if self._state_network.ssid == network.ssid:
        self._networks_buttons[network.ssid].set_enabled(False)
        status_text = "CONNECTING..."
    elif self.state == UIState.FORGETTING and self._state_network:
      if self._state_network.ssid == network.ssid:
        self._networks_buttons[network.ssid].set_enabled(False)
        status_text = "FORGETTING..."
    elif network.security_type == SecurityType.UNSUPPORTED:
      self._networks_buttons[network.ssid].set_enabled(False)
    else:
      self._networks_buttons[network.ssid].set_enabled(True)

    self._networks_buttons[network.ssid].render(ssid_rect)

    if status_text:
      status_text_rect = rl.Rectangle(security_icon_rect.x - 410, rect.y, 410, ITEM_HEIGHT)
      gui_label(status_text_rect, status_text, font_size=48, alignment=rl.GuiTextAlignment.TEXT_ALIGN_CENTER)
    else:
      # If the network is saved, show the "Forget" button
      if network.is_saved:
        forget_btn_rect = rl.Rectangle(
          security_icon_rect.x - self.btn_width - spacing,
          rect.y + (ITEM_HEIGHT - 80) / 2,
          self.btn_width,
          80,
        )
        self._forget_networks_buttons[network.ssid].render(forget_btn_rect)

    self._draw_status_icon(security_icon_rect, network)
    self._draw_signal_strength_icon(signal_icon_rect, network)

  def _networks_buttons_callback(self, network):
    if not network.is_saved and network.security_type != SecurityType.OPEN:
      self.state = UIState.NEEDS_AUTH
      self._state_network = network
      self._password_retry = False
    elif not network.is_connected:
      self.connect_to_network(network)

  def _forget_networks_buttons_callback(self, network):
    self.state = UIState.SHOW_FORGET_CONFIRM
    self._state_network = network

  def _draw_status_icon(self, rect, network: Network):
    """Draw the status icon based on network's connection state"""
    icon_file = None
    if network.is_connected and self.state != UIState.CONNECTING:
      icon_file = "icons/checkmark.png"
    elif network.security_type == SecurityType.UNSUPPORTED:
      icon_file = "icons/circled_slash.png"
    elif network.security_type != SecurityType.OPEN:
      icon_file = "icons/lock_closed.png"

    if not icon_file:
      return

    texture = gui_app.texture(icon_file, ICON_SIZE, ICON_SIZE)
    icon_rect = rl.Vector2(rect.x, rect.y + (ICON_SIZE - texture.height) / 2)
    rl.draw_texture_v(texture, icon_rect, rl.WHITE)

  def _draw_signal_strength_icon(self, rect: rl.Rectangle, network: Network):
    """Draw the Wi-Fi signal strength icon based on network's signal strength"""
    strength_level = max(0, min(3, round(network.strength / 33.0)))
    rl.draw_texture_v(gui_app.texture(STRENGTH_ICONS[strength_level], ICON_SIZE, ICON_SIZE), rl.Vector2(rect.x, rect.y), rl.WHITE)

  def connect_to_network(self, network: Network, password=''):
    self.state = UIState.CONNECTING
    self._state_network = network
    if network.is_saved and not password:
      self._wifi_manager.activate_connection(network.ssid)
    else:
      self._wifi_manager.connect_to_network(network.ssid, password)

  def forget_network(self, network: Network):
    self.state = UIState.FORGETTING
    self._state_network = network
    self._wifi_manager.forget_connection(network.ssid)

  def _on_network_updated(self, networks: list[Network]):
    self._networks = networks
    for n in self._networks:
      self._networks_buttons[n.ssid] = Button(n.ssid, partial(self._networks_buttons_callback, n), font_size=55, text_alignment=TextAlignment.LEFT,
                                              button_style=ButtonStyle.TRANSPARENT_WHITE_TEXT)
      self._networks_buttons[n.ssid].set_touch_valid_callback(lambda: self.scroll_panel.is_touch_valid())
      self._forget_networks_buttons[n.ssid] = Button("Forget", partial(self._forget_networks_buttons_callback, n), button_style=ButtonStyle.FORGET_WIFI,
                                                     font_size=45)
      self._forget_networks_buttons[n.ssid].set_touch_valid_callback(lambda: self.scroll_panel.is_touch_valid())

  def _on_need_auth(self, ssid):
    network = next((n for n in self._networks if n.ssid == ssid), None)
    if network:
      self.state = UIState.NEEDS_AUTH
      self._state_network = network
      self._password_retry = True

  def _on_activated(self):
    if self.state == UIState.CONNECTING:
      self.state = UIState.IDLE

  def _on_forgotten(self):
    if self.state == UIState.FORGETTING:
      self.state = UIState.IDLE

  def _on_disconnected(self):
    if self.state == UIState.CONNECTING:
      self.state = UIState.IDLE


def main():
  gui_app.init_window("Wi-Fi Manager")
  wifi_ui = WifiManagerUI(WifiManager())

  for _ in gui_app.render():
    wifi_ui.render(rl.Rectangle(50, 50, gui_app.width - 100, gui_app.height - 100))

  gui_app.close()


if __name__ == "__main__":
  main()
