from collections.abc import Callable
from dataclasses import dataclass
from typing import Union
import pyray as rl

from openpilot.system.ui.lib.application import gui_app, FontWeight, DEFAULT_TEXT_SIZE, DEFAULT_TEXT_COLOR, FONT_SCALE
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.lib.emoji import find_emoji, emoji_tex
from openpilot.system.ui.lib.wrap_text import wrap_text


@dataclass
class RenderSegment:
  content: str
  texture: Union[rl.Texture, None]
  offset: rl.Vector2  # # Relative to the label's top-left
  size: rl.Vector2


@dataclass
class LayoutCache:
  raw_text: str
  segments: list[RenderSegment]
  content_width: float
  content_height: float


def _elide_text(font: rl.Font, text: str, font_size: int, max_width: float) -> str:
  """Binary search to truncate text that exceeds max_width."""
  ellipsis = "..."
  if measure_text_cached(font, text, font_size).x <= max_width:
    return text

  left, right = 0, len(text)
  while left < right:
    mid = (left + right) // 2
    candidate = text[:mid] + ellipsis
    if measure_text_cached(font, candidate, font_size).x <= max_width:
      left = mid + 1
    else:
      right = mid

  return text[: left - 1] + ellipsis if left > 0 else ellipsis


def _resolve_value(value, default=""):
  if callable(value):
    return value()
  return value if value is not None else default


class Label(Widget):
  def __init__(
    self,
    text: str | Callable[[], str],
    size: int = DEFAULT_TEXT_SIZE,
    weight: FontWeight = FontWeight.NORMAL,
    align: int = rl.GuiTextAlignment.TEXT_ALIGN_CENTER,
    valign: int = rl.GuiTextAlignmentVertical.TEXT_ALIGN_MIDDLE,
    padding: int = 0,
    color: rl.Color = DEFAULT_TEXT_COLOR,
    elide: bool = False,
    emojis: bool = False,
  ):
    super().__init__(transparent_for_input=True)
    self._text_source = text
    self._font_size = size
    self._font_weight = weight
    self._font = gui_app.font(self._font_weight)
    self._h_align = align
    self._v_align = valign
    self._padding = padding
    self._color = color
    self._elide = elide
    self._emojis = emojis
    self._layout_cache: LayoutCache | None = None

  def set_text(self, text: str | Callable[[], str]):
    self._text_source = text

  def set_text_color(self, color: rl.Color):
    self._color = color

  def set_font_size(self, size: int):
    if self._font_size != size:
      self._font_size = size
      self._layout_cache = None

  def _update_layout(self):
    text = _resolve_value(self._text_source)
    if (
      self._layout_cache
      and self._layout_cache.raw_text == text
      and self._layout_cache.content_width == self._rect.width
    ):
      return

    content_width = max(1, self._rect.width - (self._padding * 2))
    if self._elide:
      lines = [_elide_text(self._font, text, self._font_size, content_width)]
    else:
      lines = wrap_text(self._font, text, self._font_size, content_width)

    segments: list[RenderSegment] = []
    line_h = self._font_size * FONT_SCALE

    cursor_y = 0.0
    for line_text in lines:
      line_size = measure_text_cached(self._font, line_text, self._font_size)
      cursor_x = self._padding
      if self._h_align == rl.GuiTextAlignment.TEXT_ALIGN_CENTER:
        cursor_x += (content_width - line_size.x) // 2
      elif self._h_align == rl.GuiTextAlignment.TEXT_ALIGN_RIGHT:
        cursor_x += content_width - line_size.x

      if self._emojis:
        self._layout_emoji_line(segments, line_text, cursor_x, cursor_y, line_h)
      else:
        segments.append(RenderSegment(line_text, None, rl.Vector2(cursor_x, cursor_y), line_size))

      cursor_y += line_h + 2

    self._layout_cache = LayoutCache(
      raw_text=text, segments=segments, content_width=self.rect.width, content_height=cursor_y - 2
    )

  def _render(self, _):
    self._update_layout()
    if not self._layout_cache:
      return

    # Determine vertical block start
    vertical_offset = 0.0
    if self._v_align == rl.GuiTextAlignmentVertical.TEXT_ALIGN_MIDDLE:
      vertical_offset = (self._rect.height - self._layout_cache.content_height) / 2
    elif self._v_align == rl.GuiTextAlignmentVertical.TEXT_ALIGN_BOTTOM:
      vertical_offset = self._rect.height - self._layout_cache.content_height

    base_pos = rl.Vector2(self._rect.x, self._rect.y + vertical_offset)

    for seg in self._layout_cache.segments:
      draw_pos = rl.Vector2(base_pos.x + seg.offset.x, base_pos.y + seg.offset.y)
      if seg.texture:
        tex_scale = (self._font_size / seg.texture.height) * FONT_SCALE
        rl.draw_texture_ex(seg.texture, draw_pos, 0.0, tex_scale, self._color)
      else:
        rl.draw_text_ex(self._font, seg.content, draw_pos, self._font_size, 0, self._color)

  def _layout_emoji_line(self, segments, line_text, x, y, line_h):
    last_idx = 0
    for start, end, emoji in find_emoji(line_text):
      # Text fragment before emoji
      if start > last_idx:
        frag = line_text[last_idx:start]
        size = measure_text_cached(self._font, frag, self._font_size)
        segments.append(RenderSegment(frag, None, rl.Vector2(x, y), size))
        x += size.x

      # Emoji texture fragment
      segments.append(RenderSegment(emoji, emoji_tex(emoji), rl.Vector2(x, y), rl.Vector2(line_h, line_h)))
      x, last_idx = x + line_h, end

    # Final text fragment
    if last_idx < len(line_text):
      frag = line_text[last_idx:]
      size = measure_text_cached(self._font, frag, self._font_size)
      segments.append(RenderSegment(frag, None, rl.Vector2(x, y), size))
