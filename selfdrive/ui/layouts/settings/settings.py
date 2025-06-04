import pyray as rl
from dataclasses import dataclass
from enum import IntEnum
from collections.abc import Callable
from openpilot.common.params import Params
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.label import gui_text_box

# Import individual panels

SETTINGS_CLOSE_TEXT = "X"
# Constants
SIDEBAR_WIDTH = 500
CLOSE_BTN_SIZE = 200
NAV_BTN_HEIGHT = 80
PANEL_MARGIN = 50
SCROLL_SPEED = 30

# Colors
SIDEBAR_COLOR = rl.BLACK
PANEL_COLOR = rl.Color(41, 41, 41, 255)
CLOSE_BTN_COLOR = rl.Color(41, 41, 41, 255)
CLOSE_BTN_PRESSED = rl.Color(59, 59, 59, 255)
TEXT_NORMAL = rl.Color(128, 128, 128, 255)
TEXT_SELECTED = rl.Color(255, 255, 255, 255)
TEXT_PRESSED = rl.Color(173, 173, 173, 255)


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
  instance: object
  button_rect: rl.Rectangle


class SettingsLayout:
  def __init__(self):
    self._params = Params()
    self._current_panel = PanelType.DEVICE
    self._close_btn_pressed = False
    self._scroll_offset = 0.0
    self._max_scroll = 0.0

    # Panel configuration
    self._panels = {
      PanelType.DEVICE: PanelInfo("Device", None, rl.Rectangle(0, 0, 0, 0)),
      PanelType.TOGGLES: PanelInfo("Toggles", None, rl.Rectangle(0, 0, 0, 0)),
      PanelType.SOFTWARE: PanelInfo("Software", None, rl.Rectangle(0, 0, 0, 0)),
      PanelType.FIREHOSE: PanelInfo("Firehose", None, rl.Rectangle(0, 0, 0, 0)),
      PanelType.NETWORK: PanelInfo("Network", None, rl.Rectangle(0, 0, 0, 0)),
      PanelType.DEVELOPER: PanelInfo("Developer", None, rl.Rectangle(0, 0, 0, 0)),
    }

    self._font_medium = gui_app.font(FontWeight.MEDIUM)
    self._font_bold = gui_app.font(FontWeight.SEMI_BOLD)

    # Callbacks
    self._close_callback: Callable | None = None

  def set_callbacks(self, on_close: Callable):
    self._close_callback = on_close

  def render(self, rect: rl.Rectangle):
    # Calculate layout
    sidebar_rect = rl.Rectangle(rect.x, rect.y, SIDEBAR_WIDTH, rect.height)
    panel_rect = rl.Rectangle(rect.x + SIDEBAR_WIDTH, rect.y, rect.width - SIDEBAR_WIDTH, rect.height)

    # Draw components
    self._draw_sidebar(sidebar_rect)
    self._draw_current_panel(panel_rect)

    if rl.is_mouse_button_released(rl.MouseButton.MOUSE_BUTTON_LEFT):
      self.handle_mouse_release(rl.get_mouse_position())

  def _draw_sidebar(self, rect: rl.Rectangle):
    rl.draw_rectangle_rec(rect, SIDEBAR_COLOR)

    # Close button
    close_btn_rect = rl.Rectangle(
      rect.x + (rect.width - CLOSE_BTN_SIZE) / 2, rect.y + 45, CLOSE_BTN_SIZE, CLOSE_BTN_SIZE
    )

    close_color = CLOSE_BTN_PRESSED if self._close_btn_pressed else CLOSE_BTN_COLOR
    rl.draw_rectangle_rounded(close_btn_rect, 0.5, 20, close_color)
    close_text_size = rl.measure_text_ex(self._font_bold, SETTINGS_CLOSE_TEXT, 140, 0)
    close_text_pos = rl.Vector2(
      close_btn_rect.x + (close_btn_rect.width - close_text_size.x) / 2,
      close_btn_rect.y + (close_btn_rect.height - close_text_size.y) / 2 - 20,
    )
    rl.draw_text_ex(self._font_bold, SETTINGS_CLOSE_TEXT, close_text_pos, 140, 0, TEXT_SELECTED)

    # Store close button rect for click detection
    self._close_btn_rect = close_btn_rect

    # Navigation buttons
    nav_start_y = rect.y + 300
    button_spacing = 20

    i = 0
    for panel_type, panel_info in self._panels.items():
      button_rect = rl.Rectangle(
        rect.x + 50,
        nav_start_y + i * (NAV_BTN_HEIGHT + button_spacing),
        rect.width - 150,  # Right-aligned with margin
        NAV_BTN_HEIGHT,
      )

      # Button styling
      is_selected = panel_type == self._current_panel
      text_color = TEXT_SELECTED if is_selected else TEXT_NORMAL

      # Draw button text (right-aligned)
      text_size = rl.measure_text_ex(self._font_medium, panel_info.name, 65, 0)
      text_pos = rl.Vector2(
        button_rect.x + button_rect.width - text_size.x, button_rect.y + (button_rect.height - text_size.y) / 2
      )
      rl.draw_text_ex(self._font_medium, panel_info.name, text_pos, 65, 0, text_color)

      # Store button rect for click detection
      panel_info.button_rect = button_rect
      i += 1

  def _draw_current_panel(self, rect: rl.Rectangle):
    content_rect = rl.Rectangle(rect.x + PANEL_MARGIN, rect.y + 25, rect.width - (PANEL_MARGIN * 2), rect.height - 50)
    rl.draw_rectangle_rounded(content_rect, 0.03, 30, PANEL_COLOR)
    gui_text_box(
      content_rect,
      f"Demo {self._panels[self._current_panel].name} Panel",
      font_size=170,
      color=rl.WHITE,
      alignment=rl.GuiTextAlignment.TEXT_ALIGN_CENTER,
      alignment_vertical=rl.GuiTextAlignmentVertical.TEXT_ALIGN_MIDDLE,
    )

  def handle_mouse_release(self, mouse_pos: rl.Vector2) -> bool:
    # Check close button
    if rl.check_collision_point_rec(mouse_pos, self._close_btn_rect):
      self._close_btn_pressed = True
      if self._close_callback:
        self._close_callback()
      return True

    # Check navigation buttons
    for panel_type, panel_info in self._panels.items():
      if rl.check_collision_point_rec(mouse_pos, panel_info.button_rect):
        self._switch_to_panel(panel_type)
        return True

    return False

  def _switch_to_panel(self, panel_type: PanelType):
    if panel_type != self._current_panel:
      self._current_panel = panel_type
      self._scroll_offset = 0.0  # Reset scroll when switching panels
      self._transition_progress = 0.0
      self._transitioning = True

  def set_current_panel(self, index: int, param: str = ""):
    panel_types = list(self._panels.keys())
    if 0 <= index < len(panel_types):
      self._switch_to_panel(panel_types[index])

  def close_settings(self):
    if self._close_callback:
      self._close_callback()
