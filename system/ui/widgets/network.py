import asyncio
import pyray as rl
from enum import IntEnum
from openpilot.system.ui.lib.wifi_manager import WifiManager, NetworkInfo
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.lib.button import gui_button
from openpilot.system.ui.lib.label import gui_label
from openpilot.system.ui.lib.scroll_panel import GuiScrollPanel
from openpilot.system.ui.widgets.keyboard import Keyboard
from openpilot.system.ui.widgets.confirm_dialog import confirm_dialog

NM_DEVICE_STATE_NEED_AUTH = 60
ITEM_HEIGHT = 160

class ActionState(IntEnum):
  NONE = 0
  CONNECT = 1
  CONNECTING = 2
  FORGOT = 3
  FORGETTING = 4
  NEED_AUTH = 5
  SHOW_FORGOT_CONFIRM = 6


class WifiManagerUI:
  def __init__(self, wifi_manager):
    self.wifi_manager = wifi_manager
    self.wifi_manager.need_auth_callback = self._on_need_auth
    self.wifi_manager.activated_callback = self._on_activated
    self._selected_network = None
    self.btn_width = 200
    self.current_action: ActionState = ActionState.NONE
    self.scroll_panel = GuiScrollPanel()
    self.keyboard = Keyboard()

    asyncio.create_task(self._initialize())

  async def _initialize(self) -> None:
    try:
      await self.wifi_manager.connect()
    except Exception as e:
      print(f"Initialization error: {e}")

  def render(self, rect: rl.Rectangle):
    if not self.wifi_manager.networks:
      gui_label(rect, "Scanning Wi-Fi networks...", 40, alignment=rl.GuiTextAlignment.TEXT_ALIGN_CENTER)
      return

    if self.current_action == ActionState.SHOW_FORGOT_CONFIRM:
      result = confirm_dialog(rect, f'Forget Wi-Fi Network "{self._selected_network.ssid}"?', 'Forget')
      if result == 1:
        asyncio.create_task(self.forgot_network())
      elif result == 0:
        self.current_action = ActionState.NONE
      return

    if self.current_action == ActionState.NEED_AUTH:
      result = self.keyboard.render(rect, 'Enter password', f'for {self._selected_network.ssid}')
      if result == 1:
        asyncio.create_task(self.connect_to_network(self.keyboard.text))
      elif result == 0:
        self.current_action = ActionState.NONE
      return

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

    if network.is_connected and self.current_action == ActionState.NONE:
      rl.gui_label(state_rect, "Connected")
    elif self.current_action == ActionState.CONNECTING and self._selected_network and self._selected_network.ssid == network.ssid:
      rl.gui_label(state_rect, "CONNECTING...")

    # If the network is saved, show the "Forget" button
    if self.wifi_manager.is_saved(network.ssid):
      forget_btn_rect = rl.Rectangle(
        rect.x + rect.width - self.btn_width,
        rect.y + (ITEM_HEIGHT - 80) / 2,
        self.btn_width,
        80,
      )
      if gui_button(forget_btn_rect, "Forget") and self.current_action == ActionState.NONE and clicked:
        self._selected_network = network
        self.current_action = ActionState.SHOW_FORGOT_CONFIRM

    if (
      self.current_action == ActionState.NONE
      and rl.check_collision_point_rec(rl.get_mouse_position(), label_rect)
      and clicked
    ):
      self._selected_network = network
      if not self.wifi_manager.is_saved(self._selected_network.ssid):
        self.current_action = ActionState.NEED_AUTH
      else:
        asyncio.create_task(self.connect_to_network())

  async def forgot_network(self):
    self.current_action = ActionState.FORGETTING
    await self.wifi_manager.forget_connection(self._selected_network.ssid)
    self.current_action = ActionState.NONE

  async def connect_to_network(self, password=''):
    self.current_action = ActionState.CONNECTING
    if self.wifi_manager.is_saved(self._selected_network.ssid) and not password:
      await self.wifi_manager.activate_connection(self._selected_network.ssid)
    else:
      await self.wifi_manager.connect_to_network(self._selected_network.ssid, password)

  def _on_need_auth(self):
    if self.current_action == ActionState.CONNECTING and self._selected_network:
      self.current_action = ActionState.NEED_AUTH

  def _on_activated(self):
    if self.current_action == ActionState.CONNECTING:
      self.current_action = ActionState.NONE


async def main():
  gui_app.init_window("Wifi Manager")

  wifi_manager = WifiManager()
  wifi_ui = WifiManagerUI(wifi_manager)

  for _ in gui_app.render():
    wifi_ui.render(rl.Rectangle(50, 50, gui_app.width - 100, gui_app.height - 100))

  gui_app.close()


if __name__ == "__main__":
  asyncio.run(main())
