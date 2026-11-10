from enum import IntEnum
from functools import partial
from typing import cast

import raylib_nb as rl
from openpilot.system.ui.lib.application import gui_app, TextAlignment
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.lib.scroll_panel import GuiScrollPanel
from openpilot.system.ui.lib.wifi_manager import WifiManager, SecurityType, Network, normalize_ssid
from openpilot.system.ui.widgets import DialogResult, Widget
from openpilot.system.ui.widgets.button import ButtonStyle, Button
from openpilot.system.ui.widgets.confirm_dialog import ConfirmDialog
from openpilot.system.ui.widgets.keyboard import Keyboard
from openpilot.system.ui.widgets.label import gui_label

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
    gui_label(self.rect, self.text, font_size=60, alignment=TextAlignment.CENTER)


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

    self._wifi_manager.add_callbacks(need_auth=self._on_need_auth,
                                     activated=self._on_activated,
                                     forgotten=self._on_forgotten,
                                     networks_updated=self._on_network_updated,
                                     disconnected=self._on_disconnected)

  def show_event(self):
    super().show_event()
    # start/stop scanning when widget is visible
    self._wifi_manager.set_active(True)

  def hide_event(self):
    super().hide_event()
    self._wifi_manager.set_active(False)

  def _load_icons(self):
    for icon in STRENGTH_ICONS + ["icons/checkmark.png", "icons/circled_slash.png", "icons/lock_closed.png"]:
      gui_app.texture(icon, ICON_SIZE, ICON_SIZE)

  def _update_state(self):
    self._wifi_manager.process_callbacks()

  def _render(self, rect: rl.Rectangle):
    if not self._networks:
      gui_label(rect, tr("Scanning Wi-Fi networks..."), 72, alignment=TextAlignment.CENTER)
      return

    if self.state == UIState.NEEDS_AUTH and self._state_network:
      self.keyboard.set_title(tr("Wrong password") if self._password_retry else tr("Enter password"),
                              tr("for \"{}\"").format(normalize_ssid(self._state_network.ssid)))
      self.keyboard.reset(min_text_size=MIN_PASSWORD_LENGTH)
      self.keyboard.set_callback(lambda result: self._on_password_entered(cast(Network, self._state_network), result))
      gui_app.push_widget(self.keyboard)
    elif self.state == UIState.SHOW_FORGET_CONFIRM and self._state_network:
      confirm_dialog = ConfirmDialog("", tr("Forget"), tr("Cancel"), callback=lambda result: self.on_forgot_confirm_finished(self._state_network, result))
      confirm_dialog.set_text(tr("Forget Wi-Fi Network \"{}\"?").format(normalize_ssid(self._state_network.ssid)))
      gui_app.push_widget(confirm_dialog)
    else:
      self._draw_network_list(rect)

  def _on_password_entered(self, network: Network, result: DialogResult):
    if result == DialogResult.CONFIRM:
      password = self.keyboard.text
      self.keyboard.clear()

      if len(password) >= MIN_PASSWORD_LENGTH:
        self.connect_to_network(network, password)
    elif result == DialogResult.CANCEL:
      self.state = UIState.IDLE

  def on_forgot_confirm_finished(self, network, result: DialogResult):
    if result == DialogResult.CONFIRM:
      self.forget_network(network)
    elif result == DialogResult.CANCEL:
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
        status_text = tr("CONNECTING...")
    elif self.state == UIState.FORGETTING and self._state_network:
      if self._state_network.ssid == network.ssid:
        self._networks_buttons[network.ssid].set_enabled(False)
        status_text = tr("FORGETTING...")
    elif network.security_type == SecurityType.UNSUPPORTED:
      self._networks_buttons[network.ssid].set_enabled(False)
    else:
      self._networks_buttons[network.ssid].set_enabled(True)

    self._networks_buttons[network.ssid].render(ssid_rect)

    if status_text:
      status_text_rect = rl.Rectangle(security_icon_rect.x - 410, rect.y, 410, ITEM_HEIGHT)
      gui_label(status_text_rect, status_text, font_size=48, alignment=TextAlignment.CENTER)
    else:
      # If the network is saved, show the "Forget" button
      if self._wifi_manager.is_connection_saved(network.ssid):
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
    if not self._wifi_manager.is_connection_saved(network.ssid) and network.security_type != SecurityType.OPEN:
      self.state = UIState.NEEDS_AUTH
      self._state_network = network
      self._password_retry = False
    elif self._wifi_manager.wifi_state.ssid != network.ssid:
      self.connect_to_network(network)

  def _forget_networks_buttons_callback(self, network):
    self.state = UIState.SHOW_FORGET_CONFIRM
    self._state_network = network

  def _draw_status_icon(self, rect, network: Network):
    """Draw the status icon based on network's connection state"""
    icon_file = None
    if self._wifi_manager.connected_ssid == network.ssid and self.state != UIState.CONNECTING:
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
    if self._wifi_manager.is_connection_saved(network.ssid) and not password:
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
      self._networks_buttons[n.ssid] = Button(normalize_ssid(n.ssid), partial(self._networks_buttons_callback, n), font_size=55,
                                              text_alignment=TextAlignment.LEFT, button_style=ButtonStyle.TRANSPARENT_WHITE_TEXT)
      self._networks_buttons[n.ssid].set_touch_valid_callback(lambda: self.scroll_panel.is_touch_valid())
      self._forget_networks_buttons[n.ssid] = Button(tr("Forget"), partial(self._forget_networks_buttons_callback, n), button_style=ButtonStyle.FORGET_WIFI,
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

  def _on_forgotten(self, _):
    if self.state == UIState.FORGETTING:
      self.state = UIState.IDLE

  def _on_disconnected(self):
    if self.state == UIState.CONNECTING:
      self.state = UIState.IDLE


def main():
  gui_app.init_window("Wi-Fi Manager")
  gui_app.push_widget(WifiManagerUI(WifiManager()))

  for _ in gui_app.render():
    pass

  gui_app.close()


if __name__ == "__main__":
  main()
