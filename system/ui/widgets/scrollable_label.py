import pyray as rl
from enum import IntEnum
from openpilot.system.ui.lib.application import gui_app, FontWeight, DEFAULT_TEXT_SIZE, DEFAULT_TEXT_COLOR
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.widgets import Widget


class ScrollState(IntEnum):
  STARTING = 0
  SCROLLING = 1


class ScrollableLabel(Widget):
  def __init__(self, text: str, font_size: int = DEFAULT_TEXT_SIZE, font_weight: FontWeight = FontWeight.NORMAL,
               text_color: rl.Color = DEFAULT_TEXT_COLOR, alignment: int = rl.GuiTextAlignment.TEXT_ALIGN_LEFT,
               alignment_vertical: int = rl.GuiTextAlignmentVertical.TEXT_ALIGN_TOP):
    super().__init__()
    self._text = text
    self._font_size = font_size
    self._font_weight = font_weight
    self._text_color = text_color
    self._align = alignment
    self._align_v = alignment_vertical

    self.reset_scroll()

  def set_font_size(self, size: int):
    self._font_size = size

  def set_color(self, color: rl.Color):
    self._text_color = color

  def set_font_weight(self, font_weight: FontWeight):
    self._font_weight = font_weight

  def set_text(self, text: str):
    self._text = text

  def reset_scroll(self):
    self._scroll_offset = 0
    self._scroll_pause_t = None
    self._scroll_state = ScrollState.STARTING

  def _render(self, _):
    font = gui_app.font(self._font_weight)
    size = measure_text_cached(font, self._text, self._font_size, 0)
    scrolling = size.x > self._rect.width

    # Horizontal alignment only matters if not scrolling
    start_x = self._rect.x
    if not scrolling:
      if self._align == rl.GuiTextAlignment.TEXT_ALIGN_CENTER:
        start_x += (self._rect.width - size.x) / 2
      elif self._align == rl.GuiTextAlignment.TEXT_ALIGN_RIGHT:
        start_x += (self._rect.width - size.x)

    # Vertical alignment
    start_y = self._rect.y
    if self._align_v == rl.GuiTextAlignmentVertical.TEXT_ALIGN_MIDDLE:
      start_y += (self._rect.height - size.y) / 2
    elif self._align_v == rl.GuiTextAlignmentVertical.TEXT_ALIGN_BOTTOM:
      start_y += (self._rect.height - size.y)

    if not scrolling:
      rl.draw_text_ex(font, self._text, rl.Vector2(self._rect.x, self._rect.y), self._font_size, 0, self._text_color)
      return

    gap = self._rect.width / 3
    if self._scroll_state == ScrollState.STARTING:
      if not self._scroll_pause_t:
        self._scroll_pause_t = rl.get_time() + 2.0
      if rl.get_time() >= self._scroll_pause_t:
        self._scroll_state = ScrollState.SCROLLING
    else:
      self._scroll_offset -= 0.8 / 60. * gui_app.target_fps
      if self._scroll_offset <= -(size.x + gap):
        self.reset_scroll()

    rl.begin_scissor_mode(int(self._rect.x), int(self._rect.y - self._font_size / 2), int(self._rect.width), int(self._rect.height + self._font_size))

    for i in range(2 if self._scroll_state != ScrollState.STARTING else 1):
      x_pos = self._rect.x + self._scroll_offset + (i * (size.x + gap))
      rl.draw_text_ex(font, self._text, (x_pos, start_y), self._font_size, 0, self._text_color)

    # draw black fade on left and right
    fade_width = 20
    rl.draw_rectangle_gradient_h(int(self._rect.x + self._rect.width - fade_width), int(self._rect.y), fade_width, int(self._rect.height), rl.BLANK, rl.BLACK)
    if self._scroll_state != ScrollState.STARTING:
      rl.draw_rectangle_gradient_h(int(self._rect.x), int(self._rect.y), fade_width, int(self._rect.height), rl.BLACK, rl.BLANK)

    rl.end_scissor_mode()
