import pyray as rl
from enum import IntEnum

from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.lib.label import gui_label
from openpilot.system.ui.lib.button import gui_button

# from openpilot.system.ui.lib.scroll_panel import GuiScrollPanel
from openpilot.system.ui.lib.toggle import Toggle
from openpilot.system.ui.lib.wifi_manager import WifiManagerWrapper
from openpilot.system.ui.widgets.network import WifiManagerUI
from openpilot.system.ui.widgets.keyboard import Keyboard


ITEM_HEIGHT = 100
ITEM_SPACING = 30
BUTTON_WIDTH = 250
BUTTON_HEIGHT = 100
LINE_COLOR = rl.GRAY


class AdvanceNetworkState(IntEnum):
  NONE = 0
  EDIT_TETHERING_PASSWORD = 1
  EDIT_APN = 2


class AdvanceNetwork:
  def __init__(self, manager: WifiManagerUI, wifi_manager: WifiManagerWrapper):
    self._state = AdvanceNetworkState.NONE
    self._manager = manager
    self._wifi_manager = wifi_manager
    self._scroll_panel = None
    self._tethering_toggle = Toggle(initial_state=True)
    self._roaming_toggle = Toggle()
    self._cellular_toggle = Toggle()
    self._keyboard = Keyboard(max_text_size=64, min_text_size=8)

    self._items = [
      {"label": "Enable Tethering", "toggle": self._tethering_toggle, "handler": self.on_tethering_toggle},
      {"label": "Tethering Password", "button": "EDIT", "handler": self.on_tethering_password_edit},
      {"label": "IP Address", "value": self._get_ip_address},
      {"label": "Enable Roaming", "toggle": self._roaming_toggle, "handler": self.on_roaming_toggle},
      {"label": "APN Setting", "button": "EDIT", "handler": self.on_apn_edit},
      {"label": "Cellular Metered", "toggle": self._cellular_toggle, "handler": self.on_cellular_toggle},
    ]

  def _get_ip_address(self):
    return self._wifi_manager.ip_address

  def on_tethering_toggle(self):
    self._wifi_manager.enable_tethering(self._tethering_toggle.get_state())

  def on_tethering_password_edit(self):
    # TODO: Show the current password in the dialog
    self._state = AdvanceNetworkState.EDIT_TETHERING_PASSWORD
    print(self._wifi_manager.get_tethering_password())
    result = self._keyboard.render("Enter new tethering password", "")
    if result == 1:
      self._wifi_manager.set_tethering_password(self._keyboard.get_text())
      self._state = AdvanceNetworkState.NONE
    elif result == 0:
      self._state = AdvanceNetworkState.NONE

  def on_roaming_toggle(self):
    pass

  def on_apn_edit(self):
    self._state = AdvanceNetworkState.EDIT_APN
    print(self._wifi_manager.get_tethering_password())
    result = self._keyboard.render("Enter APN", "leave blank for automatic configuration")
    if result == 1:
      pass
    elif result == 0:
      self._state = AdvanceNetworkState.NONE

  def on_cellular_toggle(self):
    pass

  def on_hidden_network_connect(self):
    # TODO: Implement this method
    pass

  def render(self, rect: rl.Rectangle):
    if self._state == AdvanceNetworkState.EDIT_TETHERING_PASSWORD:
      self.on_tethering_password_edit()
    elif self._state == AdvanceNetworkState.EDIT_APN:
      self.on_apn_edit()
    else:
      item_rect = rl.Rectangle(rect.x, rect.y, rect.width, ITEM_HEIGHT)
      button_rect = rl.Rectangle(
        rect.x + rect.width - BUTTON_WIDTH, rect.y + (ITEM_HEIGHT - BUTTON_HEIGHT) / 2, BUTTON_WIDTH, BUTTON_HEIGHT
      )

      for idx, item in enumerate(self._items):
        self._render_item(item, item_rect, button_rect)

        # Draw separator line between items (except after the last one)
        if idx < len(self._items) - 1:
          y_line = item_rect.y + ITEM_HEIGHT + ITEM_SPACING / 2 - 1
          rl.draw_line(int(item_rect.x), int(y_line), int(item_rect.x + item_rect.width), int(y_line), LINE_COLOR)

        item_rect.y += ITEM_HEIGHT + ITEM_SPACING
        button_rect.y += ITEM_HEIGHT + ITEM_SPACING

  def _render_item(self, item, item_rect, button_rect):
    value = item.get("value")
    value_str = value() if callable(value) else value

    gui_label(item_rect, item['label'], 50)
    if value_str is not None:
      gui_label(button_rect, str(value_str), 35, alignment=rl.GuiTextAlignment.TEXT_ALIGN_RIGHT)
    elif "toggle" in item:
      toggle = item["toggle"]
      if toggle.render(rl.Rectangle(button_rect.x + button_rect.width - 160, button_rect.y, 160, 80)):
        if "handler" in item:
          item["handler"]()
    elif "button" in item:
      button_text = item["button"]
      if gui_button(button_rect, button_text, font_size=35, border_radius=50):
        if "handler" in item:
          item["handler"]()


if __name__ == "__main__":
  gui_app.init_window("Advance Network Example")
  wifi_manager = WifiManagerWrapper()
  wifi_ui = WifiManagerUI(wifi_manager)
  advance_network = AdvanceNetwork(wifi_ui, wifi_manager)
  for _ in gui_app.render():
    # wifi_ui.render(rl.Rectangle(50, 50, gui_app.width - 100, gui_app.height - 100))
    advance_network.render(rl.Rectangle(50, 50, 1024, 768))
