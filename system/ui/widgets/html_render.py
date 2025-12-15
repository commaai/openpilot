import re
import pyray as rl
from dataclasses import dataclass
from enum import Enum
from typing import Any
from openpilot.system.ui.lib.application import gui_app, FontWeight, FONT_SCALE
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.lib.scroll_panel import GuiScrollPanel
from openpilot.system.ui.lib.wrap_text import wrap_text
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.button import Button, ButtonStyle
from openpilot.system.ui.lib.text_measure import measure_text_cached

LIST_INDENT_PX = 40
PADDING = 20


class ElementType(Enum):
  H1 = "h1"
  H2 = "h2"
  H3 = "h3"
  H4 = "h4"
  H5 = "h5"
  H6 = "h6"
  P = "p"
  B = "b"
  UL = "ul"
  LI = "li"
  BR = "br"


TAG_NAMES = '|'.join([t.value for t in ElementType])
START_TAG_RE = re.compile(f'<({TAG_NAMES})>')
END_TAG_RE = re.compile(f'</({TAG_NAMES})>')
COMMENT_RE = re.compile(r'<!--.*?-->', flags=re.DOTALL)
DOCTYPE_RE = re.compile(r'<!DOCTYPE[^>]*>')
HTML_BODY_TAGS_RE = re.compile(r'</?(?:html|head|body)[^>]*>')
TOKEN_RE = re.compile(r'</[^>]+>|<[^>]+>|[^<\s]+')


def is_tag(token: str) -> tuple[bool, bool, ElementType | None]:
  supported_tag = bool(START_TAG_RE.fullmatch(token))
  supported_end_tag = bool(END_TAG_RE.fullmatch(token))
  tag = ElementType(token[1:-1].strip('/')) if supported_tag or supported_end_tag else None
  return supported_tag, supported_end_tag, tag


@dataclass
class HtmlElement:
  type: ElementType
  content: str
  font_size: int
  font_weight: FontWeight
  margin_top: int
  margin_bottom: int
  line_height: float = 0.9  # matches Qt visually, unsure why not default 1.2
  indent_level: int = 0


@dataclass
class RenderLine:
  text: str
  x: float
  y: float
  height: float
  font_size: int
  font: rl.Font


@dataclass
class LayoutCache:
  content_width: int
  lines: list[RenderLine]
  total_height: float


class HtmlRenderer(Widget):
  def __init__(self, file_path: str | None = None, text: str | None = None,
               text_size: dict | None = None, text_color: rl.Color = rl.WHITE, center_text: bool = False):
    super().__init__()
    self._text_color = text_color
    self._center_text = center_text
    self._normal_font = gui_app.font(FontWeight.NORMAL)
    self._bold_font = gui_app.font(FontWeight.BOLD)
    self._indent_level = 0
    self._layout_cache: LayoutCache | None = None

    if text_size is None:
      text_size = {}

    # Base paragraph size (Qt stylesheet default is 48px in offroad alerts)
    base_p_size = int(text_size.get(ElementType.P, 48))

    # Untagged text defaults to <p>
    self.styles: dict[ElementType, dict[str, Any]] = {
      ElementType.H1: {"size": round(base_p_size * 2), "weight": FontWeight.BOLD, "margin_top": 20, "margin_bottom": 16},
      ElementType.H2: {"size": round(base_p_size * 1.50), "weight": FontWeight.BOLD, "margin_top": 24, "margin_bottom": 12},
      ElementType.H3: {"size": round(base_p_size * 1.17), "weight": FontWeight.BOLD, "margin_top": 20, "margin_bottom": 10},
      ElementType.H4: {"size": round(base_p_size * 1.00), "weight": FontWeight.BOLD, "margin_top": 16, "margin_bottom": 8},
      ElementType.H5: {"size": round(base_p_size * 0.83), "weight": FontWeight.BOLD, "margin_top": 12, "margin_bottom": 6},
      ElementType.H6: {"size": round(base_p_size * 0.67), "weight": FontWeight.BOLD, "margin_top": 10, "margin_bottom": 4},
      ElementType.P: {"size": base_p_size, "weight": FontWeight.NORMAL, "margin_top": 8, "margin_bottom": 12},
      ElementType.B: {"size": base_p_size, "weight": FontWeight.BOLD, "margin_top": 8, "margin_bottom": 12},
      ElementType.LI: {"size": base_p_size, "weight": FontWeight.NORMAL, "color": rl.Color(40, 40, 40, 255), "margin_top": 6, "margin_bottom": 6},
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
    self.elements.clear()
    self._layout_cache = None

    # Remove HTML comments
    html_content = COMMENT_RE.sub('', html_content)

    # Remove DOCTYPE, html, head, body tags but keep their content
    html_content = DOCTYPE_RE.sub('', html_content)
    html_content = HTML_BODY_TAGS_RE.sub('', html_content)

    # Parse HTML
    tokens = TOKEN_RE.findall(html_content)

    def close_tag():
      nonlocal current_content
      nonlocal current_tag

      # If no tag is set, default to paragraph so we don't lose text
      if current_tag is None:
        current_tag = ElementType.P

      text = ' '.join(current_content).strip()
      current_content = []
      if text:
        if current_tag == ElementType.LI:
          text = '• ' + text
        self._add_element(current_tag, text)

    current_content: list[str] = []
    current_tag: ElementType | None = None
    for token in tokens:
      is_start_tag, is_end_tag, tag = is_tag(token)
      if tag is not None:
        if tag == ElementType.BR:
          # Close current tag and add a line break
          close_tag()
          self._add_element(ElementType.BR, "")

        elif is_start_tag or is_end_tag:
          # Always add content regardless of opening or closing tag
          close_tag()
          current_tag = tag if is_start_tag else None

        # increment after we add the content for the current tag
        if tag == ElementType.UL:
          self._indent_level = self._indent_level + 1 if is_start_tag else max(0, self._indent_level - 1)

      else:
        current_content.append(token)

    if current_content:
      close_tag()

  def _add_element(self, element_type: ElementType, content: str) -> None:
    style = self.styles[element_type]

    element = HtmlElement(
      type=element_type,
      content=content,
      font_size=style["size"],
      font_weight=style["weight"],
      margin_top=style["margin_top"],
      margin_bottom=style["margin_bottom"],
      indent_level=self._indent_level,
    )

    self.elements.append(element)

  def _calculate_layout(self, content_width: int) -> LayoutCache:
    lines: list[RenderLine] = []
    current_y = 0.0
    usable_width = content_width - (PADDING * 2)

    for element in self.elements:
      if element.type == ElementType.BR:
        current_y += element.margin_bottom
        continue

      current_y += element.margin_top

      if element.content:
        font = self._get_font(element.font_weight)
        wrapped_lines = wrap_text(font, element.content, element.font_size, int(usable_width))
        indent = max(element.indent_level - 1, 0) * LIST_INDENT_PX

        for line_text in wrapped_lines:
          if self._center_text:
            text_width = measure_text_cached(font, line_text, element.font_size).x
            x = (content_width - text_width) // 2
          else:
            x = indent + PADDING

          height = element.font_size * FONT_SCALE * element.line_height
          lines.append(RenderLine(line_text, x, current_y, height, element.font_size, font))
          current_y += height

      current_y += element.margin_bottom

    return LayoutCache(content_width, lines, current_y)

  def _render(self, rect: rl.Rectangle) -> int:
    content_width = int(rect.width)

    if self._layout_cache is None or self._layout_cache.content_width != content_width:
      self._layout_cache = self._calculate_layout(content_width)

    viewport_rect = self._parent_rect if self._parent_rect is not None else rect
    for line in self._layout_cache.lines:
      y = rect.y + line.y
      if (y + line.height) < viewport_rect.y:
        continue  # Above viewport
      if  y > (viewport_rect.y + viewport_rect.height):
        break  # Below viewport

      pos = rl.Vector2(rect.x + line.x, y)
      rl.draw_text_ex(line.font, line.text, pos, line.font_size, 0, self._text_color)

    return int(self._layout_cache.total_height)

  def get_total_height(self, content_width: int) -> float:
    if self._layout_cache is None or self._layout_cache.content_width != content_width:
      self._layout_cache = self._calculate_layout(content_width)

    return self._layout_cache.total_height

  def _get_font(self, weight: FontWeight):
    if weight == FontWeight.BOLD:
      return self._bold_font
    return self._normal_font


class HtmlModal(Widget):
  def __init__(self, file_path: str | None = None, text: str | None = None):
    super().__init__()
    self._content = HtmlRenderer(file_path=file_path, text=text)
    self._scroll_panel = GuiScrollPanel()
    self._ok_button = Button(tr("OK"), click_callback=lambda: gui_app.set_modal_overlay(None), button_style=ButtonStyle.PRIMARY)

  def close_event(self):
    self._ok_button.set_click_callback(None)

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
