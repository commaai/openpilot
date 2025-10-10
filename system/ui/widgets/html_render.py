import re
import pyray as rl
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.scroll_panel import GuiScrollPanel
from openpilot.system.ui.lib.wrap_text import wrap_text
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.button import Button, ButtonStyle
from openpilot.system.ui.lib.text_measure import measure_text_cached

LIST_INDENT_PX = 40


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


@dataclass
class ElementInfo:
  type: ElementType
  is_start_tag: bool = False
  is_end_tag: bool = False
  style_override: dict[str, Any] = field(default_factory=dict)

  @classmethod
  def from_token(cls, token: str) -> 'ElementInfo | None':
    is_start_tag, is_end_tag, tag = cls.is_tag(token)
    # print('is_start_tag, is_end_tag, tag', is_start_tag, is_end_tag, tag)
    if tag is not None:
      return cls(tag, is_start_tag, is_end_tag, cls.parse_style(token))
    return None

  @staticmethod
  def is_tag(token: str) -> tuple[bool, bool, ElementType | None]:
    m_start = START_TAG_RE.fullmatch(token)
    m_end = END_TAG_RE.fullmatch(token)
    tag_name = m_start.group(1) if m_start else (m_end.group(1) if m_end else None)
    tag = ElementType(tag_name) if tag_name else None
    return bool(m_start), bool(m_end), tag

  @staticmethod
  def parse_style(token: str) -> dict[str, int | str]:
    """
    Parses style attributes from an HTML tag.
    Input: <h2 style="text-align: center; font-size: 50px;">
    Output: {'text-align': 'center', 'font-size': 50}
    """

    style_match = STYLE_RE.search(token)
    styles = {}
    if style_match:
      style_str = style_match.group(1)
      for item in style_str.split(';'):
        item = item.strip()
        if not len(item):
          continue

        key, value = item.split(':')
        px_match = PX_RE.fullmatch(value.strip())
        styles[key.strip()] = int(px_match.group(1)) if px_match else value.strip()
    return styles

  def __post_init__(self):
    # assert no unsupported styles
    unsupported_styles = set(self.style_override) - set(SUPPORTED_STYLES)
    assert not unsupported_styles, f"Unsupported styles: {unsupported_styles}"


SUPPORTED_STYLES = {"font-size", "text-align", "font-weight"}


TAG_NAMES = '|'.join([t.value for t in ElementType])
START_TAG_RE = re.compile(f'<({TAG_NAMES})(?:\\s+[^>]*)?>')
END_TAG_RE = re.compile(f'</({TAG_NAMES})\\s*>')

STYLE_RE = re.compile(r'style="([^"]+)"')
PX_RE = re.compile(r'(\d+)px')




@dataclass
class HtmlElement:
  type: ElementType
  content: str
  font_size: int
  font_weight: FontWeight
  text_align: str
  margin_top: int
  margin_bottom: int
  line_height: float = 1.2
  indent_level: int = 0


class HtmlRenderer(Widget):
  def __init__(self, file_path: str | None = None, text: str | None = None,
               text_size: dict | None = None, text_color: rl.Color = rl.WHITE):
    super().__init__()
    self._text_color = text_color
    self._normal_font = gui_app.font(FontWeight.NORMAL)
    self._bold_font = gui_app.font(FontWeight.BOLD)
    self._indent_level = 0

    # TODO: remove this and use new style support
    if text_size is None:
      text_size = {}

    # Untagged text defaults to <p>
    self.styles: dict[ElementType, dict[str, Any]] = {
      ElementType.H1: {"font-size": 68, "font-weight": FontWeight.BOLD, "margin_top": 20, "margin_bottom": 16},
      ElementType.H2: {"font-size": 60, "font-weight": FontWeight.BOLD, "margin_top": 24, "margin_bottom": 12},
      ElementType.H3: {"font-size": 52, "font-weight": FontWeight.BOLD, "margin_top": 20, "margin_bottom": 10},
      ElementType.H4: {"font-size": 48, "font-weight": FontWeight.BOLD, "margin_top": 16, "margin_bottom": 8},
      ElementType.H5: {"font-size": 44, "font-weight": FontWeight.BOLD, "margin_top": 12, "margin_bottom": 6},
      ElementType.H6: {"font-size": 40, "font-weight": FontWeight.BOLD, "margin_top": 10, "margin_bottom": 4},
      ElementType.P: {"font-size": text_size.get(ElementType.P, 38), "font-weight": FontWeight.NORMAL, "margin_top": 8, "margin_bottom": 12},
      ElementType.LI: {"font-size": 38, "font-weight": FontWeight.NORMAL, "color": rl.Color(40, 40, 40, 255), "margin_top": 6, "margin_bottom": 6},
      ElementType.BR: {"font-size": 0, "font-weight": FontWeight.NORMAL, "margin_top": 0, "margin_bottom": 12},
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

    # Remove HTML comments
    html_content = re.sub(r'<!--.*?-->', '', html_content, flags=re.DOTALL)

    # Remove DOCTYPE, html, head, body tags but keep their content
    html_content = re.sub(r'<!DOCTYPE[^>]*>', '', html_content)
    html_content = re.sub(r'</?(?:html|head|body)[^>]*>', '', html_content)

    # Parse HTML
    tokens = re.findall(r'</[^>]+>|<[^>]+>|[^<\s]+', html_content)

    def close_tag():
      nonlocal current_content
      nonlocal current_tag

      # If no tag is set, default to closing paragraph so we don't lose text
      if current_tag is None:
        current_tag = ElementInfo(ElementType.P)

      text = ' '.join(current_content).strip()
      current_content = []
      if text:
        if current_tag.type == ElementType.LI:
          text = '• ' + text
        self._add_element(current_tag, text)

    current_content: list[str] = []
    current_tag: ElementInfo | None = None
    for token in tokens:
      tag = ElementInfo.from_token(token)
      print('token:', token, 'tag:', tag)
      if tag is not None:
        if tag.type == ElementType.BR:
          # Close current tag and add a line break
          close_tag()
          # self._add_element(ElementInfo(ElementType.BR, False, False), "")
          self._add_element(tag, "")

        elif tag.is_start_tag or tag.is_end_tag:
          # Always add content regardless of opening or closing tag
          close_tag()

          print(token)
          if tag.is_start_tag:
            # current_tag = ElementInfo(tag, parse_style(token))
            current_tag = tag
          else:
            current_tag = None

        # increment after we add the content for the current tag
        if tag.type == ElementType.UL:
          self._indent_level = self._indent_level + 1 if tag.is_start_tag else max(0, self._indent_level - 1)

      else:
        current_content.append(token)

    if current_content:
      close_tag()

  def _add_element(self, element_info: ElementInfo, content: str) -> None:
    style = self.styles[element_info.type].copy()
    style.update(element_info.style_override)

    print('  adding element:', element_info.type, 'content:', content, 'style:', style, 'indent_level:', self._indent_level)

    element = HtmlElement(
      type=element_info.type,
      content=content,
      font_size=style["font-size"],
      font_weight=style["font-weight"],
      text_align=style.get("text-align", "left"),
      margin_top=style["margin_top"],
      margin_bottom=style["margin_bottom"],
      indent_level=self._indent_level,
    )

    assert element.text_align == "left" or element.indent_level == 0, "Indentation only supported for left-aligned text"

    self.elements.append(element)

    # print('   elements now:', self.elements)

  def _render(self, rect: rl.Rectangle):
    # TODO: speed up by removing duplicate calculations across renders
    current_y = rect.y
    padding = 20
    content_width = rect.width - (padding * 2)

    # print('rendering!')

    for element in self.elements:
      # print(' rendering element:', element.type, 'content:', element.content)
      if element.type == ElementType.BR:

        # print('  br, old y:', current_y)
        current_y += element.margin_bottom
        # print('  br, new y:', current_y)
        continue

      current_y += element.margin_top
      if current_y > rect.y + rect.height:
        break

      if element.content:
        font = self._get_font(element.font_weight)
        wrapped_lines = wrap_text(font, element.content, element.font_size, int(content_width))

        for line in wrapped_lines:
          if current_y < rect.y - element.font_size:
            current_y += element.font_size * element.line_height
            continue

          if current_y > rect.y + rect.height:
            break

          if element.text_align == "center":
            text_width = measure_text_cached(font, line, element.font_size, 0).x
            text_x = rect.x + (rect.width - text_width) / 2
          elif element.text_align == "right":
            text_width = measure_text_cached(font, line, element.font_size, 0).x
            text_x = rect.x + rect.width - text_width - padding
          else:  # left align
            text_x = rect.x + (max(element.indent_level - 1, 0) * LIST_INDENT_PX)

          rl.draw_text_ex(font, line, rl.Vector2(text_x + padding, current_y), element.font_size, 0, self._text_color)

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
        font = self._get_font(element.font_weight)
        wrapped_lines = wrap_text(font, element.content, element.font_size, int(usable_width))

        for _ in wrapped_lines:
          total_height += element.font_size * element.line_height

      total_height += element.margin_bottom

    return total_height

  def _get_font(self, weight: FontWeight):
    if weight == FontWeight.BOLD:
      return self._bold_font
    return self._normal_font


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
