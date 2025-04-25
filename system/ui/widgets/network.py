from dataclasses import dataclass
from typing import Literal

import pyray as rl
from openpilot.system.ui.lib.wifi_manager import NetworkInfo, WifiManagerCallbacks, WifiManagerWrapper
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.lib.button import gui_button
from openpilot.system.ui.lib.label import gui_label
from openpilot.system.ui.lib.scroll_panel import GuiScrollPanel
from openpilot.system.ui.widgets.keyboard import Keyboard
from openpilot.system.ui.widgets.confirm_dialog import confirm_dialog

NM_DEVICE_STATE_NEED_AUTH = 60
ITEM_HEIGHT = 160


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
    self.keyboard = Keyboard()

    self.wifi_manager = wifi_manager
    self.wifi_manager.set_callbacks(WifiManagerCallbacks(self._on_need_auth, self._on_activated, self._on_forgotten))
    self.wifi_manager.start()
    self.wifi_manager.connect()

  def render(self, rect: rl.Rectangle):
    if not self.wifi_manager.networks:
      gui_label(rect, "Scanning Wi-Fi networks...", 72, alignment=rl.GuiTextAlignment.TEXT_ALIGN_CENTER)
      return

    match self.state:
      case StateNeedsAuth(network):
        result = self.keyboard.render(rect, "Enter password", f"for {network.ssid}")
        if result == 1:
          self.connect_to_network(network, self.keyboard.text)
        elif result == 0:
          self.state = StateIdle()

      case StateShowForgetConfirm(network):
        result = confirm_dialog(rect, f'Forget Wi-Fi Network "{network.ssid}"?', "Forget")
        if result == 1:
          self.forget_network(network)
        elif result == 0:
          self.state = StateIdle()

      case _:
        self._draw_network_list(rect)

  def _draw_network_list(self, rect: rl.Rectangle):
    content_rect = rl.Rectangle(rect.x, rect.y, rect.width, len(self.wifi_manager.networks) * ITEM_HEIGHT)
    offset = self.scroll_panel.handle_scroll(rect, content_rect)
    clicked = self.scroll_panel.is_click_valid()

    rl.begin_scissor_mode(int(rect.x), int(rect.y), int(rect.width), int(rect.height))
    for i, network in enumerate(self.wifi_manager.networks):
      y_offset = rect.y + i * ITEM_HEIGHT + offset.y
      item_rect = rl.Rectangle(rect.x, y_offset, rect.width, ITEM_HEIGHT)
      if not rl.check_collision_recs(item_rect, rect):
        continue

      self._draw_network_item(item_rect, network, clicked)
      if i < len(self.wifi_manager.networks) - 1:
        line_y = int(item_rect.y + item_rect.height - 1)
        rl.draw_line(int(item_rect.x), int(line_y), int(item_rect.x + item_rect.width), line_y, rl.LIGHTGRAY)

    rl.end_scissor_mode()

  def _draw_network_item(self, rect, network: NetworkInfo, clicked: bool):
    label_rect = rl.Rectangle(rect.x, rect.y, rect.width - self.btn_width * 2, ITEM_HEIGHT)
    state_rect = rl.Rectangle(rect.x + rect.width - self.btn_width * 2 - 150, rect.y, 300, ITEM_HEIGHT)

    gui_label(label_rect, network.ssid, 55)

    status_text = ""
    if network.is_connected:
      status_text = "Connected"
    match self.state:
      case StateConnecting(network=connecting):
        if connecting.ssid == network.ssid:
          status_text = "CONNECTING..."
      case StateForgetting(network=forgetting):
        if forgetting.ssid == network.ssid:
          status_text = "FORGETTING..."
    if status_text:
      rl.gui_label(state_rect, status_text)

    # If the network is saved, show the "Forget" button
    if self.wifi_manager.is_saved(network.ssid):
      forget_btn_rect = rl.Rectangle(
        rect.x + rect.width - self.btn_width,
        rect.y + (ITEM_HEIGHT - 80) / 2,
        self.btn_width,
        80,
      )
      if isinstance(self.state, StateIdle) and gui_button(forget_btn_rect, "Forget") and clicked:
        self.state = StateShowForgetConfirm(network)

    if isinstance(self.state, StateIdle) and rl.check_collision_point_rec(rl.get_mouse_position(), label_rect) and clicked:
      if not self.wifi_manager.is_saved(network.ssid):
        self.state = StateNeedsAuth(network)
      else:
        self.connect_to_network(network)

  def connect_to_network(self, network: NetworkInfo, password=''):
    self.state = StateConnecting(network)
    if self.wifi_manager.is_saved(network.ssid) and not password:
      self.wifi_manager.activate_connection(network.ssid)
    else:
      self.wifi_manager.connect_to_network(network.ssid, password)

  def forget_network(self, network: NetworkInfo):
    self.state = StateForgetting(network)
    self.wifi_manager.forget_connection(network.ssid)

  def _on_need_auth(self):
    match self.state:
      case StateConnecting(network):
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
