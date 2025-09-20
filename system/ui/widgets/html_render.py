import re
import pyray as rl
from dataclasses import dataclass
from enum import Enum
from typing import Any
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.scroll_panel import GuiScrollPanel
from openpilot.system.ui.lib.wrap_text import wrap_text
from openpilot.system.ui.widgets import Widget, DialogResult
from openpilot.system.ui.widgets.button import gui_button, ButtonStyle


class ElementType(Enum):
  H1 = "h1"
  H2 = "h2"
  H3 = "h3"
  H4 = "h4"
  H5 = "h5"
  H6 = "h6"
  P = "p"
  BR = "br"


@dataclass
class HtmlElement:
  type: ElementType
  content: str
  font_size: int
  font_weight: FontWeight
  color: rl.Color
  margin_top: int
  margin_bottom: int
  line_height: float = 1.2


class HtmlRenderer(Widget):
  def __init__(self, file_path: str):
    self.elements: list[HtmlElement] = []
    self._normal_font = gui_app.font(FontWeight.NORMAL)
    self._bold_font = gui_app.font(FontWeight.BOLD)
    self._scroll_panel = GuiScrollPanel()

    self.styles: dict[ElementType, dict[str, Any]] = {
      ElementType.H1: {"size": 68, "weight": FontWeight.BOLD, "color": rl.BLACK, "margin_top": 20, "margin_bottom": 16},
      ElementType.H2: {"size": 60, "weight": FontWeight.BOLD, "color": rl.BLACK, "margin_top": 24, "margin_bottom": 12},
      ElementType.H3: {"size": 52, "weight": FontWeight.BOLD, "color": rl.BLACK, "margin_top": 20, "margin_bottom": 10},
      ElementType.H4: {"size": 48, "weight": FontWeight.BOLD, "color": rl.BLACK, "margin_top": 16, "margin_bottom": 8},
      ElementType.H5: {"size": 44, "weight": FontWeight.BOLD, "color": rl.BLACK, "margin_top": 12, "margin_bottom": 6},
      ElementType.H6: {"size": 40, "weight": FontWeight.BOLD, "color": rl.BLACK, "margin_top": 10, "margin_bottom": 4},
      ElementType.P: {"size": 38, "weight": FontWeight.NORMAL, "color": rl.Color(40, 40, 40, 255), "margin_top": 8, "margin_bottom": 12},
      ElementType.BR: {"size": 0, "weight": FontWeight.NORMAL, "color": rl.BLACK, "margin_top": 0, "margin_bottom": 12},
    }

    self.parse_html_file(file_path)

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

    # Find all HTML elements
    pattern = r'<(h[1-6]|p)(?:[^>]*)>(.*?)</\1>|<br\s*/?>'
    matches = re.finditer(pattern, html_content, re.DOTALL | re.IGNORECASE)

    for match in matches:
      if match.group(0).lower().startswith('<br'):
        # Handle <br> tags
        self._add_element(ElementType.BR, "")
      else:
        tag = match.group(1).lower()
        content = match.group(2).strip()

        # Clean up content - remove extra whitespace
        content = re.sub(r'\s+', ' ', content)
        content = content.strip()

        if content:  # Only add non-empty elements
          element_type = ElementType(tag)
          self._add_element(element_type, content)

  def _add_element(self, element_type: ElementType, content: str) -> None:
    style = self.styles[element_type]

    element = HtmlElement(
      type=element_type,
      content=content,
      font_size=style["size"],
      font_weight=style["weight"],
      color=style["color"],
      margin_top=style["margin_top"],
      margin_bottom=style["margin_bottom"],
    )

    self.elements.append(element)

  def _render(self, rect: rl.Rectangle):
    margin = 50
    content_rect = rl.Rectangle(rect.x + margin, rect.y + margin, rect.width - (margin * 2), rect.height - (margin * 2))

    button_height = 160
    button_spacing = 20
    scrollable_height = content_rect.height - button_height - button_spacing

    scrollable_rect = rl.Rectangle(content_rect.x, content_rect.y, content_rect.width, scrollable_height)

    total_height = self.get_total_height(int(scrollable_rect.width))
    scroll_content_rect = rl.Rectangle(scrollable_rect.x, scrollable_rect.y, scrollable_rect.width, total_height)
    scroll_offset = self._scroll_panel.handle_scroll(scrollable_rect, scroll_content_rect)

    rl.begin_scissor_mode(int(scrollable_rect.x), int(scrollable_rect.y), int(scrollable_rect.width), int(scrollable_rect.height))
    self._render_content(scrollable_rect, scroll_offset.y)
    rl.end_scissor_mode()

    button_width = (rect.width - 3 * 50) // 3
    button_x = content_rect.x + (content_rect.width - button_width) / 2
    button_y = content_rect.y + content_rect.height - button_height
    button_rect = rl.Rectangle(button_x, button_y, button_width, button_height)
    if gui_button(button_rect, "OK", button_style=ButtonStyle.PRIMARY) == 1:
      return DialogResult.CONFIRM

    return DialogResult.NO_ACTION

  def _render_content(self, rect: rl.Rectangle, scroll_offset: float = 0) -> float:
    current_y = rect.y + scroll_offset
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

          rl.draw_text_ex(font, line, rl.Vector2(rect.x + padding, current_y), element.font_size, 0, rl.WHITE)

          current_y += element.font_size * element.line_height

      # Apply bottom margin
      current_y += element.margin_bottom

    return current_y - rect.y - scroll_offset  # Return total content height

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
