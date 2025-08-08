from enum import IntEnum
from itertools import zip_longest

import pyray as rl

from openpilot.system.ui.lib.application import gui_app, FontWeight, DEFAULT_TEXT_SIZE, DEFAULT_TEXT_COLOR
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.lib.utils import GuiStyleContext
from openpilot.system.ui.lib.emoji import find_emoji, emoji_tex
from openpilot.system.ui.lib.wrap_text import wrap_text
from openpilot.system.ui.widgets import Widget

ICON_PADDING = 15

class TextAlignment(IntEnum):
  LEFT = 0
  CENTER = 1
  RIGHT = 2

# TODO: This should be a Widget class
def gui_label(
  rect: rl.Rectangle,
  text: str,
  font_size: int = DEFAULT_TEXT_SIZE,
  color: rl.Color = DEFAULT_TEXT_COLOR,
  font_weight: FontWeight = FontWeight.NORMAL,
  alignment: int = rl.GuiTextAlignment.TEXT_ALIGN_LEFT,
  alignment_vertical: int = rl.GuiTextAlignmentVertical.TEXT_ALIGN_MIDDLE,
  elide_right: bool = True
):
  font = gui_app.font(font_weight)
  text_size = measure_text_cached(font, text, font_size)
  display_text = text

  # Elide text to fit within the rectangle
  if elide_right and text_size.x > rect.width:
    ellipsis = "..."
    left, right = 0, len(text)
    while left < right:
      mid = (left + right) // 2
      candidate = text[:mid] + ellipsis
      candidate_size = measure_text_cached(font, candidate, font_size)
      if candidate_size.x <= rect.width:
        left = mid + 1
      else:
        right = mid
    display_text = text[: left - 1] + ellipsis if left > 0 else ellipsis
    text_size = measure_text_cached(font, display_text, font_size)

  # Calculate horizontal position based on alignment
  text_x = rect.x + {
    rl.GuiTextAlignment.TEXT_ALIGN_LEFT: 0,
    rl.GuiTextAlignment.TEXT_ALIGN_CENTER: (rect.width - text_size.x) / 2,
    rl.GuiTextAlignment.TEXT_ALIGN_RIGHT: rect.width - text_size.x,
  }.get(alignment, 0)

  # Calculate vertical position based on alignment
  text_y = rect.y + {
    rl.GuiTextAlignmentVertical.TEXT_ALIGN_TOP: 0,
    rl.GuiTextAlignmentVertical.TEXT_ALIGN_MIDDLE: (rect.height - text_size.y) / 2,
    rl.GuiTextAlignmentVertical.TEXT_ALIGN_BOTTOM: rect.height - text_size.y,
  }.get(alignment_vertical, 0)

  # Draw the text in the specified rectangle
  rl.draw_text_ex(font, display_text, rl.Vector2(text_x, text_y), font_size, 0, color)


def gui_text_box(
  rect: rl.Rectangle,
  text: str,
  font_size: int = DEFAULT_TEXT_SIZE,
  color: rl.Color = DEFAULT_TEXT_COLOR,
  alignment: int = rl.GuiTextAlignment.TEXT_ALIGN_LEFT,
  alignment_vertical: int = rl.GuiTextAlignmentVertical.TEXT_ALIGN_TOP,
  font_weight: FontWeight = FontWeight.NORMAL,
):
  styles = [
    (rl.GuiControl.DEFAULT, rl.GuiControlProperty.TEXT_COLOR_NORMAL, rl.color_to_int(color)),
    (rl.GuiControl.DEFAULT, rl.GuiDefaultProperty.TEXT_SIZE, font_size),
    (rl.GuiControl.DEFAULT, rl.GuiDefaultProperty.TEXT_LINE_SPACING, font_size),
    (rl.GuiControl.DEFAULT, rl.GuiControlProperty.TEXT_ALIGNMENT, alignment),
    (rl.GuiControl.DEFAULT, rl.GuiDefaultProperty.TEXT_ALIGNMENT_VERTICAL, alignment_vertical),
    (rl.GuiControl.DEFAULT, rl.GuiDefaultProperty.TEXT_WRAP_MODE, rl.GuiTextWrapMode.TEXT_WRAP_WORD)
  ]
  if font_weight != FontWeight.NORMAL:
    rl.gui_set_font(gui_app.font(font_weight))

  with GuiStyleContext(styles):
    rl.gui_label(rect, text)

  if font_weight != FontWeight.NORMAL:
    rl.gui_set_font(gui_app.font(FontWeight.NORMAL))


# Non-interactive text area. Can render emojis and an optional specified icon.
class Label(Widget):
  def __init__(self,
               text: str,
               font_size: int = DEFAULT_TEXT_SIZE,
               font_weight: FontWeight = FontWeight.NORMAL,
               text_alignment: TextAlignment = TextAlignment.CENTER,
               text_padding: int = 20,
               text_color: rl.Color = DEFAULT_TEXT_COLOR,
               icon = None,
               ):

    super().__init__()
    self._font_weight = font_weight
    self._font = gui_app.font(self._font_weight)
    self._font_size = font_size
    self._text_alignment = text_alignment
    self._text_padding = text_padding
    self._text_color = text_color
    self._icon = icon
    self.set_text(text)

  def set_text(self, text):
    self._text_raw = text
    self._update_text(self._text_raw)

  def set_text_color(self, color):
    self._text_color = color

  def _update_layout_rects(self):
    self._update_text(self._text_raw)

  def _update_text(self, text):
    self._emojis = []
    self._text_size = []
    self._text = wrap_text(self._font, text, self._font_size, self._rect.width - (self._text_padding*2))
    for t in self._text:
      self._emojis.append(find_emoji(t))
      self._text_size.append(measure_text_cached(self._font, t, self._font_size))

  def _render(self, _):
    text = self._text[0] if self._text else None
    text_size = self._text_size[0] if self._text_size else rl.Vector2(0.0, 0.0)
    text_pos = rl.Vector2(0, (self._rect.y + (self._rect.height - (text_size.y)) // 2))

    if self._icon:
      icon_y = self._rect.y + (self._rect.height - self._icon.height) / 2
      if text:
        if self._text_alignment == TextAlignment.LEFT:
          icon_x = self._rect.x + self._text_padding
          text_pos.x = self._icon.width + ICON_PADDING
        elif self._text_alignment == TextAlignment.CENTER:
          total_width = self._icon.width + ICON_PADDING + text_size.x
          icon_x = self._rect.x + (self._rect.width - total_width) / 2
          text_pos.x = self._icon.width + ICON_PADDING
        else:
          icon_x = (self._rect.x + self._rect.width - text_size.x - self._text_padding) - ICON_PADDING - self._icon.width
      else:
        icon_x = self._rect.x + (self._rect.width - self._icon.width) / 2
      rl.draw_texture_v(self._icon, rl.Vector2(icon_x, icon_y), rl.WHITE)

    for text, text_size, emojis in zip_longest(self._text, self._text_size, self._emojis, fillvalue=[]):
      line_pos = rl.Vector2(text_pos.x, text_pos.y)
      if self._text_alignment == TextAlignment.LEFT:
        line_pos.x += self._rect.x + self._text_padding
      elif self._text_alignment == TextAlignment.CENTER:
        line_pos.x += self._rect.x + (self._rect.width - text_size.x) // 2
      elif self._text_alignment == TextAlignment.RIGHT:
        line_pos.x += self._rect.x + self._rect.width - text_size.x - self._text_padding

      prev_index = 0
      for start, end, emoji in emojis:
        text_before = text[prev_index:start]
        width_before = measure_text_cached(self._font, text_before, self._font_size)
        rl.draw_text_ex(self._font, text_before, line_pos, self._font_size, 0, self._text_color)
        line_pos.x += width_before.x

        tex = emoji_tex(emoji)
        rl.draw_texture_ex(tex, line_pos, 0.0, self._font_size / tex.height, self._text_color)
        line_pos.x += self._font_size
        prev_index = end
      rl.draw_text_ex(self._font, text[prev_index:], line_pos, self._font_size, 0, self._text_color)
      text_pos.y += text_size.y or self._font_size
