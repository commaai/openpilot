import pyray as rl
from enum import IntEnum
from functools import partial
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.lib.wifi_manager import SecurityType, Network
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.button import ButtonStyle, Button
from openpilot.system.ui.widgets.label import gui_label, Align


STRENGTH_ICONS = [
  "icons/wifi_strength_low.png",
  "icons/wifi_strength_medium.png",
  "icons/wifi_strength_high.png",
  "icons/wifi_strength_full.png",
]

BUTTON_WIDTH = 200
ITEM_HEIGHT = 160
ICON_SIZE = 50
SPACING =50


class UIState(IntEnum):
  IDLE = 0
  CONNECTING = 1
  FORGETTING = 2


class NetworkItem(Widget):
  def __init__(self, network: Network, connect_callback=None, forget_callback=None):
    super().__init__(transparent_for_input=True)
    self.state: UIState = UIState.IDLE
    self.network = network
    self.connect_btn = Button(network.ssid, partial(connect_callback, network), font_size=55,
                              text_alignment=Align.LEFT, button_style=ButtonStyle.TRANSPARENT_WHITE_TEXT)
    self.forget_btn = Button(tr("Forget"), partial(forget_callback, network), button_style=ButtonStyle.FORGET_WIFI, font_size=45)

  def set_touch_valid_callback(self, touch_callback):
    self.connect_btn.set_touch_valid_callback(touch_callback)
    self.forget_btn.set_touch_valid_callback(touch_callback)

  def _render(self, rect):
    r = self._rect
    sig_x = r.x + r.width - ICON_SIZE
    sec_x = sig_x - SPACING - ICON_SIZE
    v_mid = r.y + (ITEM_HEIGHT - ICON_SIZE) / 2

    status_text = ""
    if self.state == UIState.CONNECTING:
      status_text = tr("CONNECTING...")
    elif self.state == UIState.FORGETTING:
      status_text = tr("FORGETTING...")
    unsupported = self.network.security_type == SecurityType.UNSUPPORTED
    self.connect_btn.set_enabled(not unsupported and self.state == UIState.IDLE)
    self.connect_btn.render(rl.Rectangle(r.x, r.y, r.width - BUTTON_WIDTH * 2, ITEM_HEIGHT))

    if status_text:
      gui_label(rl.Rectangle(sec_x - 410, r.y, 410, ITEM_HEIGHT), status_text, 48, align=Align.CENTER)
    elif self.network.is_saved:
      self.forget_btn.render(rl.Rectangle(sec_x - BUTTON_WIDTH - SPACING, r.y + (ITEM_HEIGHT - 80) / 2, BUTTON_WIDTH, 80))

    self._draw_icon(sec_x, v_mid, self._get_status_icon())
    self._draw_icon(sig_x, v_mid, STRENGTH_ICONS[max(0, min(3, round(self.network.strength / 33.0)))])

  def _get_status_icon(self):
    icon_file = None
    if self.network.is_connected and self.state != UIState.CONNECTING:
      icon_file = "icons/checkmark.png"
    elif self.network.security_type == SecurityType.UNSUPPORTED:
      icon_file = "icons/circled_slash.png"
    elif self.network.security_type != SecurityType.OPEN:
      icon_file = "icons/lock_closed.png"
    return icon_file

  def _draw_icon(self, x, y, path):
    if path:
      tex = gui_app.texture(path, ICON_SIZE, ICON_SIZE)
      rl.draw_texture_v(tex, rl.Vector2(x, y + (ICON_SIZE - tex.height) / 2), rl.WHITE)
