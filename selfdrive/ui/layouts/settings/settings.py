import pyray as rl
from dataclasses import dataclass
from enum import IntEnum
from collections.abc import Callable
from openpilot.selfdrive.ui.layouts.settings.developer import DeveloperLayout
from openpilot.selfdrive.ui.layouts.settings.device import DeviceLayout
from openpilot.selfdrive.ui.layouts.settings.firehose import FirehoseLayout
from openpilot.selfdrive.ui.layouts.settings.software import SoftwareLayout
from openpilot.selfdrive.ui.layouts.settings.toggles import TogglesLayout
from openpilot.system.ui.lib.application import gui_app, FontWeight, MousePos
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.lib.wifi_manager import WifiManager
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.network import WifiManagerUI

# Settings close button
SETTINGS_CLOSE_TEXT = "×"
SETTINGS_CLOSE_TEXT_Y_OFFSET = 8  # The '×' character isn't quite vertically centered in the font so we need to offset it a bit to fully center it

# Constants
SIDEBAR_WIDTH = 500
CLOSE_BTN_SIZE = 200
NAV_BTN_HEIGHT = 110
PANEL_MARGIN = 50

# Colors
SIDEBAR_COLOR = rl.BLACK
PANEL_COLOR = rl.Color(41, 41, 41, 255)
CLOSE_BTN_COLOR = rl.Color(41, 41, 41, 255)
CLOSE_BTN_PRESSED = rl.Color(59, 59, 59, 255)
TEXT_NORMAL = rl.Color(128, 128, 128, 255)
TEXT_SELECTED = rl.WHITE


class PanelType(IntEnum):
  DEVICE = 0
  NETWORK = 1
  TOGGLES = 2
  SOFTWARE = 3
  FIREHOSE = 4
  DEVELOPER = 5


@dataclass
class PanelInfo:
  name: str
  instance: Widget
  button_rect: rl.Rectangle = rl.Rectangle(0, 0, 0, 0)


class SettingsLayout(Widget):
  def __init__(self):
    super().__init__()
    self._current_panel = PanelType.DEVICE

    # Panel configuration
    wifi_manager = WifiManager()
    wifi_manager.set_active(False)

    self._panels = {
      PanelType.DEVICE: PanelInfo("Device", DeviceLayout()),
      PanelType.NETWORK: PanelInfo("Network", WifiManagerUI(wifi_manager)),
      PanelType.TOGGLES: PanelInfo("Toggles", TogglesLayout()),
      PanelType.SOFTWARE: PanelInfo("Software", SoftwareLayout()),
      PanelType.FIREHOSE: PanelInfo("Firehose", FirehoseLayout()),
      PanelType.DEVELOPER: PanelInfo("Developer", DeveloperLayout()),
    }

    self._font_medium = gui_app.font(FontWeight.MEDIUM)

    # Callbacks
    self._close_callback: Callable | None = None

  def set_callbacks(self, on_close: Callable):
    self._close_callback = on_close

  def _render(self, rect: rl.Rectangle):
    # Calculate layout
    sidebar_rect = rl.Rectangle(rect.x, rect.y, SIDEBAR_WIDTH, rect.height)
    panel_rect = rl.Rectangle(rect.x + SIDEBAR_WIDTH, rect.y, rect.width - SIDEBAR_WIDTH, rect.height)

    # Draw components
    self._draw_sidebar(sidebar_rect)
    self._draw_current_panel(panel_rect)

  def _draw_sidebar(self, rect: rl.Rectangle):
    rl.draw_rectangle_rec(rect, SIDEBAR_COLOR)

    # Close button
    close_btn_rect = rl.Rectangle(
      rect.x + (rect.width - CLOSE_BTN_SIZE) / 2, rect.y + 60, CLOSE_BTN_SIZE, CLOSE_BTN_SIZE
    )

    pressed = (rl.is_mouse_button_down(rl.MouseButton.MOUSE_BUTTON_LEFT) and
               rl.check_collision_point_rec(rl.get_mouse_position(), close_btn_rect))
    close_color = CLOSE_BTN_PRESSED if pressed else CLOSE_BTN_COLOR
    rl.draw_rectangle_rounded(close_btn_rect, 1.0, 20, close_color)

    close_text_size = measure_text_cached(self._font_medium, SETTINGS_CLOSE_TEXT, 140)
    close_text_pos = rl.Vector2(
      close_btn_rect.x + (close_btn_rect.width - close_text_size.x) / 2,
      close_btn_rect.y + (close_btn_rect.height - close_text_size.y) / 2 - SETTINGS_CLOSE_TEXT_Y_OFFSET,
    )
    rl.draw_text_ex(self._font_medium, SETTINGS_CLOSE_TEXT, close_text_pos, 140, 0, TEXT_SELECTED)

    # Store close button rect for click detection
    self._close_btn_rect = close_btn_rect

    # Navigation buttons
    y = rect.y + 300
    for panel_type, panel_info in self._panels.items():
      button_rect = rl.Rectangle(rect.x + 50, y, rect.width - 150, NAV_BTN_HEIGHT)

      # Button styling
      is_selected = panel_type == self._current_panel
      text_color = TEXT_SELECTED if is_selected else TEXT_NORMAL
      # Draw button text (right-aligned)
      text_size = measure_text_cached(self._font_medium, panel_info.name, 65)
      text_pos = rl.Vector2(
        button_rect.x + button_rect.width - text_size.x, button_rect.y + (button_rect.height - text_size.y) / 2
      )
      rl.draw_text_ex(self._font_medium, panel_info.name, text_pos, 65, 0, text_color)

      # Store button rect for click detection
      panel_info.button_rect = button_rect

      y += NAV_BTN_HEIGHT

  def _draw_current_panel(self, rect: rl.Rectangle):
    rl.draw_rectangle_rounded(
      rl.Rectangle(rect.x + 10, rect.y + 10, rect.width - 20, rect.height - 20), 0.04, 30, PANEL_COLOR
    )
    content_rect = rl.Rectangle(rect.x + PANEL_MARGIN, rect.y + 25, rect.width - (PANEL_MARGIN * 2), rect.height - 50)
    # rl.draw_rectangle_rounded(content_rect, 0.03, 30, PANEL_COLOR)
    panel = self._panels[self._current_panel]
    if panel.instance:
      panel.instance.render(content_rect)

  def _handle_mouse_release(self, mouse_pos: MousePos) -> bool:
    # Check close button
    if rl.check_collision_point_rec(mouse_pos, self._close_btn_rect):
      if self._close_callback:
        self._close_callback()
      return True

    # Check navigation buttons
    for panel_type, panel_info in self._panels.items():
      if rl.check_collision_point_rec(mouse_pos, panel_info.button_rect):
        self.set_current_panel(panel_type)
        return True

    return False

  def set_current_panel(self, panel_type: PanelType):
    if panel_type != self._current_panel:
      self._panels[self._current_panel].instance.hide_event()
      self._current_panel = panel_type
      self._panels[self._current_panel].instance.show_event()

  def show_event(self):
    super().show_event()
    self._panels[self._current_panel].instance.show_event()

  def hide_event(self):
    super().hide_event()
    self._panels[self._current_panel].instance.hide_event()
