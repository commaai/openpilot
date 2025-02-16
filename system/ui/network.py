import asyncio
import pyray as rl
from enum import IntEnum
from dbus_next.constants import MessageType
from openpilot.system.ui.lib.wifi_manager import WifiManager, NetworkInfo
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.lib.label import gui_label
from openpilot.system.ui.lib.scroll_panel import GuiScrollPanel
from openpilot.system.ui.lib.keyboard import Keyboard

NM_DEVICE_STATE_NEED_AUTH = 60


class ActionState(IntEnum):
  NONE = 0
  CONNECT = 1
  CONNECTING = 2
  FORGOT = 3
  FORGETTING = 4
  NEED_AUTH = 5


class WifiManagerUI:
  def __init__(self, wifi_manager):
    self._wifi_manager = wifi_manager
    self._selected_network = None
    self.item_height = 160
    self.btn_width = 200
    self.current_action: ActionState = ActionState.NONE
    self._saved_networks: set[NetworkInfo] = {}
    self.scroll_panel = GuiScrollPanel()
    self.keyboard = Keyboard()

    asyncio.create_task(self.periodic_network_fetch())

  async def periodic_network_fetch(self):
    await self._wifi_manager.connect()
    self._wifi_manager.bus.add_message_handler(self.handle_signal)
    await self._wifi_manager.get_available_networks()
    while True:
      await self._wifi_manager.request_scan()
      await asyncio.sleep(60)  # Wait for 1 minute before refetching

  def draw_network_list(self, rect: rl.Rectangle):
    if not self._wifi_manager.networks:
      gui_label(rect, "Loading Wi-Fi networks...", 40, alignment=rl.GuiTextAlignment.TEXT_ALIGN_CENTER)
      return

    if self.current_action == ActionState.NEED_AUTH:
      result = self.keyboard.render(rect, 'Enter password', f'for {self._selected_network.ssid}')
      if result == 0:
        return
      else:
        self.current_action = ActionState.NONE
        asyncio.create_task(self.connect_to_network(self.keyboard.text))

    content_rect = rl.Rectangle(rect.x, rect.y, rect.width, len(self._wifi_manager.networks) * self.item_height)
    offset = self.scroll_panel.handle_scroll(rect, content_rect)
    rl.begin_scissor_mode(int(rect.x), int(rect.y), int(rect.width), int(rect.height))
    clicked = offset.y < 10 and rl.is_mouse_button_released(rl.MouseButton.MOUSE_BUTTON_LEFT)
    for i, network in enumerate(self._wifi_manager.networks):
      y_offset = i * self.item_height + offset.y
      item_rect = rl.Rectangle(rect.x, y_offset, rect.width, self.item_height)

      if rl.check_collision_recs(item_rect, rect):
        self.render_network_item(item_rect, network, clicked)
        if i != len(self._wifi_manager.networks) - 1:
          line_y = int(item_rect.y + item_rect.height - 1)
          rl.draw_line(int(item_rect.x), int(line_y), int(item_rect.x + item_rect.width), line_y, rl.LIGHTGRAY)

    rl.end_scissor_mode()

  def render_network_item(self, rect, network: NetworkInfo, clicked: bool):
    label_rect = rl.Rectangle(rect.x, rect.y, rect.width - self.btn_width * 2, self.item_height)
    state_rect = rl.Rectangle(rect.x + rect.width - self.btn_width * 2 - 30, rect.y, self.btn_width, self.item_height)

    gui_label(label_rect, network.ssid, 55)

    if network.is_connected and self.current_action == ActionState.NONE:
      rl.gui_label(state_rect, "Connected")
    elif self.current_action == "Connecting" and self._selected_network and self._selected_network.ssid == network.ssid:
      rl.gui_label(state_rect, "CONNECTING...")

    # If the network is saved, show the "Forget" button
    if network.is_saved:
      forget_btn_rect = rl.Rectangle(rect.x + rect.width - self.btn_width, rect.y + (self.item_height - 80) / 2, self.btn_width, 80)
      if rl.gui_button(forget_btn_rect, "Forget") and self.current_action == ActionState.NONE:
        self._selected_network = network
        asyncio.create_task(self.forgot_network())

    if self.current_action == ActionState.NONE and rl.check_collision_point_rec(rl.get_mouse_position(), label_rect) and clicked:
      self._selected_network = network
      asyncio.create_task(self.connect_to_network())

  async def forgot_network(self):
    self.current_action = ActionState.FORGETTING
    await self._wifi_manager.forgot_connection(self._selected_network.ssid)
    self._selected_network.is_saved = False
    self.current_action = ActionState.NONE

  async def connect_to_network(self, password=''):
    self.current_action = ActionState.CONNECTING
    if self._selected_network.is_saved and not password:
      await self._wifi_manager.activate_connection(self._selected_network.ssid)
    else:
      await self._wifi_manager.connect_to_network(self._selected_network.ssid, password)
    self.current_action = ActionState.NONE

  def handle_signal(self, message):
    if message.message_type != MessageType.SIGNAL:
      return

    if message.member == 'StateChanged':
      if len(message.body) >= 2:
        _, new_state = message.body[0], message.body[1]
        if new_state == NM_DEVICE_STATE_NEED_AUTH:
          self.current_action = ActionState.NEED_AUTH
    elif message.interface == 'org.freedesktop.DBus.Properties' and message.member == 'PropertiesChanged':
      body = message.body
      if len(body) >= 2:
        changed_properties = body[1]
        if 'LastScan' in changed_properties:
          asyncio.create_task(self._wifi_manager.get_available_networks())


async def main():
  gui_app.init_window("Wifi Manager")

  wifi_manager = WifiManager()
  wifi_ui = WifiManagerUI(wifi_manager)

  while not rl.window_should_close():
    rl.begin_drawing()
    rl.clear_background(rl.BLACK)

    wifi_ui.draw_network_list(rl.Rectangle(50, 50, gui_app.width - 100, gui_app.height - 100))

    rl.end_drawing()
    await asyncio.sleep(0)


if __name__ == "__main__":
  asyncio.run(main())
