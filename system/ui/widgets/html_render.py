from html.parser import HTMLParser
from dataclasses import dataclass
from enum import Enum
from typing import Any
import pyray as rl

from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.scroll_panel import GuiScrollPanel
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.button import Button, ButtonStyle

LIST_INDENT_PX = 40


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


TEXT_BLOCK_TAGS = (
  'p',
  'h1',
  'h2',
  'h3',
  'h4',
  'h5',
  'h6',
)
BOLD_TAGS = ('b', 'strong')
ITALIC_TAGS = ('i', 'em')


@dataclass
class HtmlElement:
  type: ElementType
  # content is a list of (text, FontWeight, is_italic) segments for inline styling
  content: list[tuple[str, FontWeight, bool]]
  font_size: int
  font_weight: FontWeight
  margin_top: int
  margin_bottom: int
  line_height: float = 1.2
  indent_level: int = 0


class _Parser(HTMLParser):
  def __init__(self, styles: dict[ElementType, dict[str, Any]]):
    super().__init__(convert_charrefs=True)
    self.styles = styles
    self.elements: list[HtmlElement] = []
    self._indent = 0

    # Current block being built
    self._current_block: ElementType | None = None
    self._current_segments: list[tuple[str, FontWeight, bool]] = []  # (text, font_weight, is_italized)
    self._inline_weight_stack: list[FontWeight] = [FontWeight.NORMAL]
    self._inline_italic_stack: list[bool] = [False]

  def handle_starttag(self, tag, attrs):
    tag = tag.lower()
    attrs = dict(attrs or [])

    if tag == "br":
      self._flush_current_block()
      self._add_element(ElementType.BR, [])
      return

    if tag == "ul":
      self._indent += 1
      return

    if tag == "li":
      self._flush_current_block()
      self._current_block = ElementType.LI
      # Prepend bullet as a segment with current inline weight and italic state
      self._current_segments = [("• ", self._inline_weight_stack[-1], self._inline_italic_stack[-1])]
      return

    if tag in TEXT_BLOCK_TAGS:
      self._flush_current_block()
      self._current_block = ElementType(tag)
      self._current_segments = []
      return

    if tag in BOLD_TAGS:
      self._inline_weight_stack.append(FontWeight.BOLD)
      return

    if tag in ITALIC_TAGS:
      self._inline_italic_stack.append(True)
      return

  def handle_endtag(self, tag):
    tag = tag.lower()
    if tag == "ul":
      self._indent = max(0, self._indent - 1)
      return

    if tag == "li":
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

    if tag in ITALIC_TAGS:
      if len(self._inline_italic_stack) > 1:
        self._inline_italic_stack.pop()
      return

  def handle_data(self, data):
    if not data:
      return
    # Ensure there's a block to append into; default to paragraph if none
    if self._current_block is None:
      self._current_block = ElementType.P
      self._current_segments = []

    current_weight = self._inline_weight_stack[-1]
    current_italic = self._inline_italic_stack[-1]
    # Collapse multiple spaces similarly to HTML: keep single spaces
    # But preserve trailing spaces for splitting/word boundaries
    self._current_segments.append((data, current_weight, current_italic))

  def _flush_current_block(self):
    if self._current_block is None and not self._current_segments:
      return

    if self._current_block is None:
      block = ElementType.P
    else:
      block = self._current_block

    # Copy segments
    segments = [(t, w, it) for (t, w, it) in self._current_segments if t.strip() != "" or t == " "]
    self._current_segments = []

    if block == ElementType.BR:
      self._add_element(ElementType.BR, [])
      return

    # Prepend bullet already handled in starttag for LI
    self._add_element(block, segments)

  def _add_element(self, etype: ElementType, segments: list[tuple[str, FontWeight, bool]]):
    style = self.styles.get(etype, self.styles[ElementType.P])
    # Apply block-level bold to segments that are not already bold
    if style.get("weight") == FontWeight.BOLD:
      applied_segments: list[tuple[str, FontWeight, bool]] = [(t, w if w == FontWeight.BOLD else FontWeight.BOLD, it) for (t, w, it) in segments]
    else:
      applied_segments = segments

    element = HtmlElement(
      type=etype,
      content=applied_segments,
      font_size=style["size"],
      font_weight=style["weight"],
      margin_top=style["margin_top"],
      margin_bottom=style["margin_bottom"],
      indent_level=self._indent,
    )
    self.elements.append(element)


class HtmlRenderer(Widget):
  def __init__(self, file_path: str | None = None, text: str | None = None, text_size: dict | None = None, text_color: rl.Color = rl.WHITE):
    super().__init__()
    self._text_color = text_color
    self._fonts: dict[tuple[FontWeight, bool], Any] = {
      (FontWeight.NORMAL, False): gui_app.font(FontWeight.NORMAL),
      (FontWeight.BOLD, False): gui_app.font(FontWeight.BOLD),
      (FontWeight.NORMAL, True): gui_app.font(FontWeight.NORMAL, italic=True),
      (FontWeight.BOLD, True): gui_app.font(FontWeight.BOLD, italic=True),
    }
    self._indent_level = 0

    if text_size is None:
      text_size = {}

    # Untagged text defaults to <p>
    self.styles: dict[ElementType, dict[str, Any]] = {
      ElementType.H1: {"size": 68, "weight": FontWeight.BOLD, "margin_top": 20, "margin_bottom": 16},
      ElementType.H2: {"size": 60, "weight": FontWeight.BOLD, "margin_top": 24, "margin_bottom": 12},
      ElementType.H3: {"size": 52, "weight": FontWeight.BOLD, "margin_top": 20, "margin_bottom": 10},
      ElementType.H4: {"size": 48, "weight": FontWeight.BOLD, "margin_top": 16, "margin_bottom": 8},
      ElementType.H5: {"size": 44, "weight": FontWeight.BOLD, "margin_top": 12, "margin_bottom": 6},
      ElementType.H6: {"size": 40, "weight": FontWeight.BOLD, "margin_top": 10, "margin_bottom": 4},
      ElementType.P: {"size": text_size.get(ElementType.P, 38), "weight": FontWeight.NORMAL, "margin_top": 8, "margin_bottom": 12},
      ElementType.LI: {"size": 38, "weight": FontWeight.NORMAL, "color": rl.Color(40, 40, 40, 255), "margin_top": 6, "margin_bottom": 6},
      ElementType.BR: {"size": 0, "weight": FontWeight.NORMAL, "margin_top": 0, "margin_bottom": 12},
    }

    self.elements: list[HtmlElement] = []
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
    parser = _Parser(self.styles)
    parser.feed(html_content)
    parser._flush_current_block()
    self.elements = parser.elements

  def _get_font(self, weight: FontWeight, italic: bool = False):
    return self._fonts[(weight, bool(italic))]

  def _wrap_segments(self, segments: list[tuple[str, FontWeight, bool]], font_size: int, content_width: int) -> list[list[tuple[str, FontWeight, bool]]]:
    """
    Wrap segments into lines. Each line is a list of (text_piece, FontWeight, is_italic).
    Splits by whitespace but preserves spacing using regex.
    """
    import re

    # Split each segment into words with trailing spaces preserved
    pieces: list[tuple[str, FontWeight, bool]] = []
    for text, weight, italic in segments:
      if not text:
        continue
      tokens = re.findall(r'\s*\S+\s*', text)  # Preserve leading and trailing spaces
      if not tokens and text.strip() == "":
        tokens = [text]
      for t in tokens:
        pieces.append((t, weight, italic))

    lines: list[list[tuple[str, FontWeight, bool]]] = []
    current_line: list[tuple[str, FontWeight, bool]] = []
    current_width = 0.0

    for piece, weight, italic in pieces:
      font = self._get_font(weight, italic)
      size_vec = measure_text_cached(font, piece, font_size, 0)
      piece_w = size_vec.x

      if current_line and (current_width + piece_w) > content_width:
        # commit current line
        lines.append(current_line)
        current_line = []
        current_width = 0.0

      current_line.append((piece, weight, italic))
      current_width += piece_w

    if current_line:
      lines.append(current_line)

    return lines

  def _render(self, rect: rl.Rectangle):
    # TODO: speed up by removing duplicate calculations across renders
    current_y = rect.y
    padding = 20
    content_width = rect.width - (padding * 2)

    for element in self.elements:
      if element.type == ElementType.BR:
        current_y += element.margin_bottom
        continue

      current_y += element.margin_top
      if current_y > rect.y + rect.height:
        break

      if element.content:
        wrapped_lines = self._wrap_segments(element.content, element.font_size, int(content_width))
        for line in wrapped_lines:
          if current_y < rect.y - element.font_size:
            current_y += element.font_size * element.line_height
            continue
          if current_y > rect.y + rect.height:
            break

          text_x = rect.x + (max(element.indent_level - 1, 0) * LIST_INDENT_PX)
          draw_x = text_x + padding
          # Draw each segment in the line with the proper font style
          for seg_text, seg_weight, seg_italic in line:
            font = self._get_font(seg_weight, seg_italic)
            rl.draw_text_ex(font, seg_text, rl.Vector2(draw_x, current_y), element.font_size, 0, self._text_color)
            size_vec = measure_text_cached(font, seg_text, element.font_size, 0)
            seg_w = getattr(size_vec, 'x', size_vec[0] if isinstance(size_vec, (list, tuple)) else 0)
            draw_x += seg_w

          # Move to next line
          current_y += element.font_size * element.line_height

      # Apply bottom margin
      current_y += element.margin_bottom

    return current_y - rect.y

  def get_total_height(self, content_width: int) -> float:
    total_height = 0.0
    padding = 20
    usable_width = content_width - (padding * 2)

    for element in self.elements:
      if element.type == ElementType.BR:
        total_height += element.margin_bottom
        continue

      total_height += element.margin_top

      if element.content:
        wrapped_lines = self._wrap_segments(element.content, element.font_size, int(usable_width))
        for _ in wrapped_lines:
          total_height += element.font_size * element.line_height

      total_height += element.margin_bottom

    return total_height


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
