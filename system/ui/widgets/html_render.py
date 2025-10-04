import re
import pyray as rl
from dataclasses import dataclass
from enum import Enum
from typing import Any
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.scroll_panel import GuiScrollPanel
from openpilot.system.ui.lib.wrap_text import wrap_text
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.button import Button, ButtonStyle

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


TAG_NAMES = '|'.join([t.value for t in ElementType])
START_TAG_RE = re.compile(f'<({TAG_NAMES})>')
END_TAG_RE = re.compile(f'</({TAG_NAMES})>')


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
  line_height: float = 1.2
  indent_level: int = 0


class HtmlRenderer(Widget):
  def __init__(self, file_path: str | None = None, text: str | None = None,
               text_size: dict | None = None, text_color: rl.Color = rl.WHITE):
    super().__init__()
    self._text_color = text_color
    self.elements: list[HtmlElement] = []
    self._normal_font = gui_app.font(FontWeight.NORMAL)
    self._bold_font = gui_app.font(FontWeight.BOLD)
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
      ElementType.BR: {"size": 0, "weight": FontWeight.NORMAL, "margin_top": 0, "margin_bottom": 12},
    }

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
          self._add_element(ElementType.BR, "")

        elif tag == ElementType.UL:
          self._indent_level = self._indent_level + 1 if is_start_tag else max(0, self._indent_level - 1)

        # elif is_start_tag:
        #   current_tag = tag

        elif is_start_tag or is_end_tag:
          # Always add content regardless of opening or closing tag
          close_tag()

          # TODO: reset to None if end tag?
          if is_start_tag:
            current_tag = tag

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
    )

    self.elements.append(element)

  def _render(self, rect: rl.Rectangle):
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
        font = self._get_font(element.font_weight)
        wrapped_lines = wrap_text(font, element.content, element.font_size, int(content_width))

        for line in wrapped_lines:
          if current_y < rect.y - element.font_size:
            current_y += element.font_size * element.line_height
            continue

          if current_y > rect.y + rect.height:
            break

          rl.draw_text_ex(font, line, rl.Vector2(rect.x + padding, current_y), element.font_size, 0, self._text_color)

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
