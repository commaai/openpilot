import pyray as rl
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.lib.label import gui_label
from openpilot.system.ui.lib.button import gui_button

# from openpilot.system.ui.lib.scroll_panel import GuiScrollPanel
from openpilot.system.ui.lib.toggle import Toggle
from openpilot.system.ui.lib.wifi_manager import WifiManagerWrapper
from openpilot.system.ui.widgets.network import WifiManagerUI


ITEM_HEIGHT = 100
ITEM_SPACING = 30
BUTTON_WIDTH = 250
BUTTON_HEIGHT = 100
LINE_COLOR = rl.GRAY


class AdvanceNetwork:
  def __init__(self, wifi_manager: WifiManagerWrapper):
    self._wifi_manager = wifi_manager
    self._scroll_panel = None
    self._tethering_toggle = Toggle(0, 0, initial_state=True)
    self._roaming_toggle = Toggle(0, 0)
    self._cellular_toggle = Toggle(0, 0)

    self._items = [
      {"label": "Enable Tethering", "toggle": self._tethering_toggle, "handler": self.on_tethering_toggle},
      {"label": "Tethering Password", "button": "EDIT", "handler": self.on_tethering_password_edit},
      {"label": "IP Address", "value": self._get_ip_address},
      {"label": "Enable Roaming", "toggle": self._roaming_toggle, "handler": self.on_roaming_toggle},
      {"label": "APN Setting", "button": "EDIT", "handler": self.on_apn_edit},
      {"label": "Cellular Metered", "toggle": self._cellular_toggle, "handler": self.on_cellular_toggle},
      {"label": "Hidden Network", "button": "CONNECT", "handler": self.on_hidden_network_connect},
    ]

  def _get_ip_address(self):
    return self._wifi_manager.ip_address

  def on_tethering_toggle(self):
    pass

  def on_tethering_password_edit(self):
    pass

  def on_roaming_toggle(self):
    pass

  def on_apn_edit(self):
    pass

  def on_cellular_toggle(self):
    pass

  def on_hidden_network_connect(self):
    pass

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

  def render(self, rect: rl.Rectangle):
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


if __name__ == "__main__":
  gui_app.init_window("Advance Network Example")
  wifi_manager = WifiManagerWrapper()
  wifi_ui = WifiManagerUI(wifi_manager)
  advance_network = AdvanceNetwork(wifi_manager)
  for _ in gui_app.render():
    # wifi_ui.render(rl.Rectangle(50, 50, gui_app.width - 100, gui_app.height - 100))
    advance_network.render(rl.Rectangle(50, 50, 1024, 768))
