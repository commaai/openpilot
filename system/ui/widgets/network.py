from enum import IntEnum
from functools import partial
from typing import cast

import pyray as rl
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.lib.scroll_panel import GuiScrollPanel
from openpilot.system.ui.lib.wifi_manager import WifiManager, SecurityType, Network
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.button import ButtonStyle, Button
from openpilot.system.ui.widgets.confirm_dialog import ConfirmDialog
from openpilot.system.ui.widgets.keyboard import Keyboard
from openpilot.system.ui.widgets.label import TextAlignment, gui_label

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


class UIState(IntEnum):
  IDLE = 0
  CONNECTING = 1
  NEEDS_AUTH = 2
  SHOW_FORGET_CONFIRM = 3
  FORGETTING = 4


class WifiManagerUI(Widget):
  def __init__(self, wifi_manager: WifiManager):
    super().__init__()
    self.wifi_manager = wifi_manager
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

    self.wifi_manager.set_callbacks(need_auth=self._on_need_auth,
                                    activated=self._on_activated,
                                    forgotten=self._on_forgotten,
                                    networks_updated=self._on_network_updated,
                                    disconnected=self._on_disconnected)

  def show_event(self):
    # start/stop scanning when widget is visible
    self.wifi_manager.set_active(True)

  def hide_event(self):
    self.wifi_manager.set_active(False)

  def _load_icons(self):
    for icon in STRENGTH_ICONS + ["icons/checkmark.png", "icons/circled_slash.png", "icons/lock_closed.png"]:
      gui_app.texture(icon, ICON_SIZE, ICON_SIZE)

  def _render(self, rect: rl.Rectangle):
    self.wifi_manager.process_callbacks()

    if not self._networks:
      gui_label(rect, "Scanning Wi-Fi networks...", 72, alignment=rl.GuiTextAlignment.TEXT_ALIGN_CENTER)
      return

    if self.state == UIState.NEEDS_AUTH and self._state_network:
      self.keyboard.set_title("Wrong password" if self._password_retry else "Enter password", f"for {self._state_network.ssid}")
      self.keyboard.reset()
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
    offset = self.scroll_panel.handle_scroll(rect, content_rect)
    clicked = self.scroll_panel.is_touch_valid() and rl.is_mouse_button_released(rl.MouseButton.MOUSE_BUTTON_LEFT)

    rl.begin_scissor_mode(int(rect.x), int(rect.y), int(rect.width), int(rect.height))
    for i, network in enumerate(self._networks):
      y_offset = rect.y + i * ITEM_HEIGHT + offset.y
      item_rect = rl.Rectangle(rect.x, y_offset, rect.width, ITEM_HEIGHT)
      if not rl.check_collision_recs(item_rect, rect):
        continue

      self._draw_network_item(item_rect, network, clicked)
      if i < len(self._networks) - 1:
        line_y = int(item_rect.y + item_rect.height - 1)
        rl.draw_line(int(item_rect.x), int(line_y), int(item_rect.x + item_rect.width), line_y, rl.LIGHTGRAY)

    rl.end_scissor_mode()

  def _draw_network_item(self, rect, network: Network, clicked: bool):
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
    if self.scroll_panel.is_touch_valid():
      if not network.is_saved and network.security_type != SecurityType.OPEN:
        self.state = UIState.NEEDS_AUTH
        self._state_network = network
        self._password_retry = False
      elif not network.is_connected:
        self.connect_to_network(network)

  def _forget_networks_buttons_callback(self, network):
    if self.scroll_panel.is_touch_valid():
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
      self.wifi_manager.activate_connection(network.ssid)
    else:
      self.wifi_manager.connect_to_network(network.ssid, password)

  def forget_network(self, network: Network):
    self.state = UIState.FORGETTING
    self._state_network = network
    self.wifi_manager.forget_connection(network.ssid)

  def _on_network_updated(self, networks: list[Network]):
    self._networks = networks
    for n in self._networks:
      self._networks_buttons[n.ssid] = Button(n.ssid, partial(self._networks_buttons_callback, n), font_size=55, text_alignment=TextAlignment.LEFT,
                                              button_style=ButtonStyle.NO_EFFECT)
      self._forget_networks_buttons[n.ssid] = Button("Forget", partial(self._forget_networks_buttons_callback, n), button_style=ButtonStyle.FORGET_WIFI,
                                                     font_size=45)

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
