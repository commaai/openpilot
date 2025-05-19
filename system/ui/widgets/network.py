from dataclasses import dataclass
from typing import Literal

import pyray as rl
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.lib.button import gui_button
from openpilot.system.ui.lib.label import gui_label
from openpilot.system.ui.lib.scroll_panel import GuiScrollPanel
from openpilot.system.ui.lib.wifi_manager import NetworkInfo, WifiManagerCallbacks, WifiManagerWrapper, SecurityType
from openpilot.system.ui.widgets.keyboard import Keyboard
from openpilot.system.ui.widgets.confirm_dialog import confirm_dialog

NM_DEVICE_STATE_NEED_AUTH = 60
MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_LENGTH = 64
ITEM_HEIGHT = 160
ICON_SIZE = 49

STRENGTH_ICONS = [
  "icons/wifi_strength_low.png",
  "icons/wifi_strength_medium.png",
  "icons/wifi_strength_high.png",
  "icons/wifi_strength_full.png",
]

@dataclass
class StateIdle:
  action: Literal["idle"] = "idle"

@dataclass
class StateConnecting:
  network: NetworkInfo
  action: Literal["connecting"] = "connecting"

@dataclass
class StateNeedsAuth:
  network: NetworkInfo
  action: Literal["needs_auth"] = "needs_auth"

@dataclass
class StateShowForgetConfirm:
  network: NetworkInfo
  action: Literal["show_forget_confirm"] = "show_forget_confirm"

@dataclass
class StateForgetting:
  network: NetworkInfo
  action: Literal["forgetting"] = "forgetting"

UIState = StateIdle | StateConnecting | StateNeedsAuth | StateShowForgetConfirm | StateForgetting


class WifiManagerUI:
  def __init__(self, wifi_manager: WifiManagerWrapper):
    self.state: UIState = StateIdle()
    self.btn_width = 200
    self.scroll_panel = GuiScrollPanel()
    self.keyboard = Keyboard(max_text_size=MAX_PASSWORD_LENGTH, min_text_size=MIN_PASSWORD_LENGTH, show_password_toggle=True)

    self._networks: list[NetworkInfo] = []

    self.wifi_manager = wifi_manager
    self.wifi_manager.set_callbacks(WifiManagerCallbacks(self._on_need_auth, self._on_activated, self._on_forgotten, self._on_network_updated))
    self.wifi_manager.start()
    self.wifi_manager.connect()

  def render(self, rect: rl.Rectangle):
    if not self._networks:
      gui_label(rect, "Scanning Wi-Fi networks...", 72, alignment=rl.GuiTextAlignment.TEXT_ALIGN_CENTER)
      return

    match self.state:
      case StateNeedsAuth(network):
        result = self.keyboard.render("Enter password", f"for {network.ssid}")
        if result == 1:
          password = self.keyboard.text
          self.keyboard.clear()

          if len(password) >= MIN_PASSWORD_LENGTH:
            self.connect_to_network(network, password)
        elif result == 0:
          self.state = StateIdle()

      case StateShowForgetConfirm(network):
        result = confirm_dialog(f'Forget Wi-Fi Network "{network.ssid}"?', "Forget")
        if result == 1:
          self.forget_network(network)
        elif result == 0:
          self.state = StateIdle()

      case _:
        self._draw_network_list(rect)

  @property
  def require_full_screen(self) -> bool:
    """Check if the WiFi UI requires exclusive full-screen rendering."""
    return isinstance(self.state, (StateNeedsAuth, StateShowForgetConfirm))

  def _draw_network_list(self, rect: rl.Rectangle):
    content_rect = rl.Rectangle(rect.x, rect.y, rect.width, len(self._networks) * ITEM_HEIGHT)
    offset = self.scroll_panel.handle_scroll(rect, content_rect)
    clicked = self.scroll_panel.is_click_valid()

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

  def _draw_network_item(self, rect, network: NetworkInfo, clicked: bool):
    spacing = 50
    ssid_rect = rl.Rectangle(rect.x, rect.y, rect.width - self.btn_width * 2, ITEM_HEIGHT)
    signal_icon_rect = rl.Rectangle(rect.x + rect.width - ICON_SIZE, rect.y + (ITEM_HEIGHT - ICON_SIZE) / 2, ICON_SIZE, ICON_SIZE)
    security_icon_rect = rl.Rectangle(signal_icon_rect.x - spacing - ICON_SIZE, rect.y + (ITEM_HEIGHT - ICON_SIZE) / 2, ICON_SIZE, ICON_SIZE)

    gui_label(ssid_rect, network.ssid, 55)

    status_text = ""
    match self.state:
      case StateConnecting(network=connecting):
        if connecting.ssid == network.ssid:
          status_text = "CONNECTING..."
      case StateForgetting(network=forgetting):
        if forgetting.ssid == network.ssid:
          status_text = "FORGETTING..."

    if status_text:
      status_text_rect = rl.Rectangle(security_icon_rect.x - 410, rect.y, 410, ITEM_HEIGHT)
      rl.gui_label(status_text_rect, status_text)
    else:
      # If the network is saved, show the "Forget" button
      if network.is_saved:
        forget_btn_rect = rl.Rectangle(security_icon_rect.x - self.btn_width - spacing,
          rect.y + (ITEM_HEIGHT - 80) / 2,
          self.btn_width,
          80,
        )
        if isinstance(self.state, StateIdle) and gui_button(forget_btn_rect, "Forget") and clicked:
          self.state = StateShowForgetConfirm(network)

    self._draw_status_icon(security_icon_rect, network)
    self._draw_signal_strength_icon(signal_icon_rect, network)

    if isinstance(self.state, StateIdle) and rl.check_collision_point_rec(rl.get_mouse_position(), ssid_rect) and clicked:
      if not network.is_saved:
        self.state = StateNeedsAuth(network)
      else:
        self.connect_to_network(network)

  def _draw_status_icon(self, rect, network: NetworkInfo):
    """Draw the status icon based on network's connection state"""
    icon_file = ""
    if network.is_connected:
      icon_file = "icons/checkmark.png"
    elif network.security_type == SecurityType.UNSUPPORTED:
      icon_file = "icons/circled_slash.png"
    else:
      icon_file = "icons/lock_closed.png"

    texture = gui_app.texture(icon_file, ICON_SIZE, ICON_SIZE)
    icon_rect = rl.Vector2(rect.x, rect.y + (ICON_SIZE - texture.height) / 2)
    rl.draw_texture_v(texture, icon_rect, rl.WHITE)

  def _draw_signal_strength_icon(self, rect: rl.Rectangle, network: NetworkInfo):
    """Draw the Wi-Fi signal strength icon based on network's signal strength"""
    strength_level = max(0, min(3, round(network.strength / 33.0)))
    rl.draw_texture_v(gui_app.texture(STRENGTH_ICONS[strength_level], ICON_SIZE, ICON_SIZE), rl.Vector2(rect.x, rect.y), rl.WHITE)

  def connect_to_network(self, network: NetworkInfo, password=''):
    self.state = StateConnecting(network)
    if network.is_saved and not password:
      self.wifi_manager.activate_connection(network.ssid)
    else:
      self.wifi_manager.connect_to_network(network.ssid, password)

  def forget_network(self, network: NetworkInfo):
    self.state = StateForgetting(network)
    network.is_saved = False
    self.wifi_manager.forget_connection(network.ssid)

  def _on_network_updated(self, networks: list[NetworkInfo]):
    self._networks = networks

  def _on_need_auth(self, ssid):
    match self.state:
      case StateConnecting(ssid):
        self.state = StateNeedsAuth(ssid)
      case _:
        # Find network by SSID
        network = next((n for n in self.wifi_manager.networks if n.ssid == ssid), None)
        if network:
          self.state = StateNeedsAuth(network)

  def _on_activated(self):
    if isinstance(self.state, StateConnecting):
      self.state = StateIdle()

  def _on_forgotten(self):
    if isinstance(self.state, StateForgetting):
      self.state = StateIdle()


def main():
  gui_app.init_window("Wi-Fi Manager")
  wifi_manager = WifiManagerWrapper()
  wifi_ui = WifiManagerUI(wifi_manager)

  for _ in gui_app.render():
    wifi_ui.render(rl.Rectangle(50, 50, gui_app.width - 100, gui_app.height - 100))

  wifi_manager.shutdown()
  gui_app.close()


if __name__ == "__main__":
  main()
