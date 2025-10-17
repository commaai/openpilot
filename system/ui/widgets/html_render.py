from html.parser import HTMLParser
from dataclasses import dataclass
from enum import Enum
from typing import Any
import pyray as rl

from openpilot.system.ui.lib.application import gui_app, FontWeight, FONT_SCALE
from openpilot.system.ui.lib.scroll_panel import GuiScrollPanel
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.button import Button, ButtonStyle

LIST_INDENT_PX = 40
PADDING = 20


# Block elements only (inline tags like <b> are handled as segments within blocks)
class ElementType(Enum):
  H1 = "h1"
  H2 = "h2"
  H3 = "h3"
  H4 = "h4"
  H5 = "h5"
  H6 = "h6"
  P = "p"
  UL = "ul"
  LI = "li"
  BR = "br"


INDENT_TAGS = ['ul']  # TODO: add ol support
LIST_ITEM_TAG = 'li'
TEXT_BLOCK_TAGS = [
  'p',
  'h1',
  'h2',
  'h3',
  'h4',
  'h5',
  'h6',
]
BOLD_TAGS = ['b', 'strong']
# TODO: Add support for italic tags
# TODO: Add support for underline and strikethrough tags


@dataclass
class HtmlElement:
  type: ElementType
  # content is a list of (text, FontWeight) segments for inline styling
  content: list[tuple[str, FontWeight]]
  font_size: int
  font_weight: FontWeight
  text_color: rl.Color = rl.WHITE
  margin_top: int = 0
  margin_bottom: int = 0
  line_height: float = 0.9  # matches Qt visually, unsure why not default 1.2
  indent_level: int = 0


class _Parser(HTMLParser):
  def __init__(self, styles: dict[ElementType, dict[str, Any]], default_text_color: rl.Color = rl.WHITE):
    super().__init__(convert_charrefs=True)
    self.styles = styles
    self.default_text_color = default_text_color
    self.elements: list[HtmlElement] = []
    self._indent = 0
    self.preserve_empty_lines = True

    # Current block being built
    self._current_block: ElementType | None = None
    self._current_segments: list[tuple[str, FontWeight]] = []  # (text, font_weight)
    self._inline_weight_stack: list[FontWeight] = [FontWeight.NORMAL]

  def handle_starttag(self, tag, attrs):
    tag = tag.lower()
    attrs = dict(attrs or [])

    if tag == "br":
      self._flush_current_block()
      self._add_element(ElementType.BR, [])
      return

    if tag in INDENT_TAGS:
      self._indent += 1
      return

    if tag == LIST_ITEM_TAG:
      self._flush_current_block()
      self._current_block = ElementType.LI
      # Prepend bullet as a segment with current inline weight
      self._current_segments = [("• ", self._inline_weight_stack[-1])]
      return

    if tag in TEXT_BLOCK_TAGS:
      self._flush_current_block()
      self._current_block = ElementType(tag)
      self._current_segments = []
      return

    if tag in BOLD_TAGS:
      self._inline_weight_stack.append(FontWeight.BOLD)
      return

  def handle_endtag(self, tag):
    tag = tag.lower()
    if tag in INDENT_TAGS:
      self._indent = max(0, self._indent - 1)
      return

    if tag == LIST_ITEM_TAG:
      self._flush_current_block()
      self._current_block = None
      return

    if tag in TEXT_BLOCK_TAGS:
      self._flush_current_block()
      self._current_block = None
      return

    if tag in BOLD_TAGS:
      if len(self._inline_weight_stack) > 1:
        self._inline_weight_stack.pop()
      return

  def handle_data(self, data):
    if not data:
      return
    # Split by newlines first so each line becomes its own block
    lines = data.split('\n')
    for i, line in enumerate(lines):
      if line or self.preserve_empty_lines:
        if self._current_block is None:
          # Ignore whitespace only lines that are outside of blocks (between tags), regardless of preserve_empty_lines (fixes extra newlines between blocks)
          if not line.strip():
            continue
          # Default to <p>
          self._current_block = ElementType.P
          self._current_segments = []
        current_weight = self._inline_weight_stack[-1]
        self._current_segments.append((line, current_weight))
      # Flush after each line except the last one to create separate elements
      if i < len(lines) - 1:
        self._flush_current_block()
        self._current_block = None

  def _flush_current_block(self):
    if self._current_block is None and not self._current_segments:
      return

    if self._current_block is None:
      block = ElementType.P
    else:
      block = self._current_block

    # Copy segments
    segments = [(t, w) for (t, w) in self._current_segments if t.strip() != "" or t == " "]  # Ignore empty segments except single spaces
    self._current_segments = []

    if block == ElementType.BR:
      self._add_element(ElementType.BR, [])
      return

    self._add_element(block, segments)

  def _add_element(self, etype: ElementType, segments: list[tuple[str, FontWeight]]):
    style = self.styles.get(etype, self.styles[ElementType.P])
    # Apply any block-level weight to segments that are not already bold
    if etype in self.styles:
      weight: FontWeight = style["weight"]
      applied_segments: list[tuple[str, FontWeight]] = [(t, FontWeight.BOLD if w == FontWeight.BOLD else weight) for (t, w) in segments]
    else:
      applied_segments = segments

    element = HtmlElement(
      type=etype,
      content=applied_segments,
      font_size=style["size"],
      font_weight=style["weight"],
      text_color=style.get("color", self.default_text_color),
      margin_top=style["margin_top"],
      margin_bottom=style["margin_bottom"],
      indent_level=self._indent,
    )
    self.elements.append(element)


class HtmlRenderer(Widget):
  def __init__(
    self, file_path: str | None = None, text: str | None = None, text_size: dict | None = None, text_color: rl.Color = rl.WHITE, center_text: bool = False
  ):
    super().__init__()
    self._text_color = text_color
    self._center_text = center_text
    self._fonts: dict[FontWeight, Any] = {
      FontWeight.NORMAL: gui_app.font(FontWeight.NORMAL),
      FontWeight.BOLD: gui_app.font(FontWeight.BOLD),
    }
    self._indent_level = 0

    if text_size is None:
      text_size = {}

    # Base paragraph size (Qt stylesheet default is 48px in offroad alerts)
    base_p_size = int(text_size.get(ElementType.P, 48))

    # Block styles; Untagged text defaults to <p>
    self.styles: dict[ElementType, dict[str, Any]] = {
      ElementType.H1: {"size": round(base_p_size * 2), "weight": FontWeight.BOLD, "margin_top": 20, "margin_bottom": 16},
      ElementType.H2: {"size": round(base_p_size * 1.50), "weight": FontWeight.BOLD, "margin_top": 24, "margin_bottom": 12},
      ElementType.H3: {"size": round(base_p_size * 1.17), "weight": FontWeight.BOLD, "margin_top": 20, "margin_bottom": 10},
      ElementType.H4: {"size": round(base_p_size * 1.00), "weight": FontWeight.BOLD, "margin_top": 16, "margin_bottom": 8},
      ElementType.H5: {"size": round(base_p_size * 0.83), "weight": FontWeight.BOLD, "margin_top": 12, "margin_bottom": 6},
      ElementType.H6: {"size": round(base_p_size * 0.67), "weight": FontWeight.BOLD, "margin_top": 10, "margin_bottom": 4},
      ElementType.P: {"size": base_p_size, "weight": FontWeight.NORMAL, "margin_top": 8, "margin_bottom": 12},
      ElementType.LI: {"size": base_p_size, "weight": FontWeight.NORMAL, "margin_top": 6, "margin_bottom": 6},
      ElementType.BR: {"size": 0, "weight": FontWeight.NORMAL, "margin_top": 0, "margin_bottom": 12},
    }

    self.elements: list[HtmlElement] = []
    self._wrap_cache_width: int | None = None
    self._wrapped_elements: list[list[list[tuple[str, FontWeight]]]] = []  # Wrapped lines per element
    self._cached_total_height: float = 0.0

    if file_path is not None:
      self.parse_html_file(file_path)
    elif text is not None:
      self.parse_html_content(text)
    else:
      raise ValueError("Either file_path or text must be provided")

  def parse_html_file(self, file_path: str) -> None:
    with open(file_path, encoding='utf-8') as file:
      content = file.read()
    self.parse_html_content(content)

  def parse_html_content(self, html_content: str) -> None:
    parser = _Parser(self.styles, default_text_color=self._text_color)
    parser.feed(html_content)
    parser._flush_current_block()
    self.elements = parser.elements
    # Invalidate wrap/height cache when content changes
    self._wrap_cache_width = None
    self._wrapped_elements = []
    self._cached_total_height = 0.0

  def _get_font(self, weight: FontWeight):
    return self._fonts[weight]

  def _merge_adjacent_segments(self, line: list[tuple[str, FontWeight]]) -> list[tuple[str, FontWeight]]:
    """
    Merge adjacent segments that have the same weight into a single segment.
    This reduces draw/measure calls for pieces that can be drawn together.
    """
    if not line:
      return []
    merged: list[tuple[str, FontWeight]] = []
    cur_text, cur_weight = line[0]
    for text, weight in line[1:]:
      if weight == cur_weight:
        cur_text += text
      else:
        merged.append((cur_text, cur_weight))
        cur_text, cur_weight = text, weight
    merged.append((cur_text, cur_weight))
    return merged

  def _wrap_segments(self, segments: list[tuple[str, FontWeight]], font_size: int, content_width: int) -> list[list[tuple[str, FontWeight]]]:
    """
    Wrap segments into lines. Each line is a list of (text_piece, FontWeight).
    Splits by whitespace but preserves spacing using regex.
    """
    import re

    # Split each segment into words with trailing spaces preserved
    pieces: list[tuple[str, FontWeight]] = []
    for text, weight in segments:
      if not text:
        continue
      tokens = re.findall(r'\s*\S+\s*', text)  # Preserve leading and trailing spaces
      if not tokens and text.strip() == "":
        tokens = [text]
      for t in tokens:
        pieces.append((t, weight))

    lines: list[list[tuple[str, FontWeight]]] = []
    current_line: list[tuple[str, FontWeight]] = []
    current_width = 0.0

    for piece, weight in pieces:
      font = self._get_font(weight)
      size_vec = measure_text_cached(font, piece, font_size, 0)
      piece_w = size_vec.x

      if current_line and (current_width + piece_w) > content_width:
        lines.append(current_line)
        current_line = []
        current_width = 0.0

      current_line.append((piece, weight))
      current_width += piece_w

    if current_line:
      lines.append(current_line)

    # Merge adjacent segments on each line that share the same font style
    merged_lines = [self._merge_adjacent_segments(l) for l in lines]
    return merged_lines

  def _ensure_wrap_and_height_cache(self, usable_width: int) -> None:
    """
    Ensure wrapped lines and total height are computed for the given usable_width
    (content width after horizontal padding). If cached, do nothing.
    """
    if self._wrap_cache_width == usable_width and self._wrapped_elements:
      return  # Already cached for this width

    wrapped_per_element: list[list[list[tuple[str, FontWeight]]]] = []
    total_height = 0.0

    for element in self.elements:
      # Include top margin
      total_height += element.margin_top

      # Include wrapped lines height
      if element.content:
        wrapped_lines = self._wrap_segments(element.content, element.font_size, int(usable_width))
        wrapped_per_element.append(wrapped_lines)
        for _ in wrapped_lines:
          total_height += element.font_size * FONT_SCALE * element.line_height
      else:
        wrapped_per_element.append([])

      # Include bottom margin
      total_height += element.margin_bottom

    # Cache results
    self._wrapped_elements = wrapped_per_element
    self._cached_total_height = total_height
    self._wrap_cache_width = usable_width

  def _render(self, rect: rl.Rectangle):
    # TODO: can we speed up further by caching more calculations across renders?
    current_y = rect.y
    content_width = rect.width - (PADDING * 2)
    # Ensure wrapped lines are computed once for this usable width
    self._ensure_wrap_and_height_cache(int(content_width))

    for idx, element in enumerate(self.elements):
      # Apply top margin
      current_y += element.margin_top
      if current_y > rect.y + rect.height:
        break  # Stop if below visible area

      # Draw content lines, if any
      wrapped_lines = self._wrapped_elements[idx] if idx < len(self._wrapped_elements) else []
      for line in wrapped_lines:
        # Use FONT_SCALE from wrapped raylib text functions to match what is drawn
        if current_y < rect.y - element.font_size * FONT_SCALE:
          current_y += element.font_size * FONT_SCALE * element.line_height
          continue  # Skip lines above the visible area
        if current_y > rect.y + rect.height:
          break  # Stop if below visible area

        # Calculate starting x based on alignment and indent
        if self._center_text:
          text_width = sum(measure_text_cached(self._get_font(w), t, element.font_size, 0).x for t, w in line)  # Sum of segment widths
          text_x = rect.x + (rect.width - text_width) / 2
        else:  # left align
          text_x = rect.x + max(element.indent_level - 1, 0) * LIST_INDENT_PX  # First level has no indent
        draw_x = text_x + PADDING
        # Draw each segment in the line with the proper font style
        for seg_text, seg_weight in line:
          font = self._get_font(seg_weight)
          rl.draw_text_ex(font, seg_text, rl.Vector2(draw_x, current_y), element.font_size, 0, element.text_color)
          size_vec = measure_text_cached(font, seg_text, element.font_size, 0)
          draw_x += size_vec.x

        # Move to next line
        current_y += element.font_size * FONT_SCALE * element.line_height

      # Apply bottom margin
      current_y += element.margin_bottom

    return current_y - rect.y

  def get_total_height(self, content_width: int) -> float:
    usable_width = max(0, content_width - (PADDING * 2))
    self._ensure_wrap_and_height_cache(int(usable_width))
    return self._cached_total_height


class HtmlModal(Widget):
  def __init__(self, file_path: str | None = None, text: str | None = None):
    super().__init__()
    self._content = HtmlRenderer(file_path=file_path, text=text)
    self._scroll_panel = GuiScrollPanel()
    self._ok_button = Button("OK", click_callback=lambda: gui_app.set_modal_overlay(None), button_style=ButtonStyle.PRIMARY)

  def _render(self, rect: rl.Rectangle):
    margin = 50
    content_rect = rl.Rectangle(rect.x + margin, rect.y + margin, rect.width - (margin * 2), rect.height - (margin * 2))

    button_height = 160
    button_spacing = 20
    scrollable_height = content_rect.height - button_height - button_spacing

    scrollable_rect = rl.Rectangle(content_rect.x, content_rect.y, content_rect.width, scrollable_height)

    total_height = self._content.get_total_height(int(scrollable_rect.width))
    scroll_content_rect = rl.Rectangle(scrollable_rect.x, scrollable_rect.y, scrollable_rect.width, total_height)
    scroll_offset = self._scroll_panel.update(scrollable_rect, scroll_content_rect)
    scroll_content_rect.y += scroll_offset

    rl.begin_scissor_mode(int(scrollable_rect.x), int(scrollable_rect.y), int(scrollable_rect.width), int(scrollable_rect.height))
    self._content.render(scroll_content_rect)
    rl.end_scissor_mode()

    button_width = (rect.width - 3 * 50) // 3
    button_x = content_rect.x + content_rect.width - button_width
    button_y = content_rect.y + content_rect.height - button_height
    button_rect = rl.Rectangle(button_x, button_y, button_width, button_height)
    self._ok_button.render(button_rect)

    return -1
