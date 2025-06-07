import pyray as rl
from openpilot.system.ui.lib.application import Widget, FontWeight
from openpilot.system.ui.lib.button import gui_button, ButtonStyle, TextAlignment
from openpilot.system.ui.lib.label import gui_label
from openpilot.system.ui.lib.scroll_panel import GuiScrollPanel

# Constants
MARGIN = 50
TITLE_HEIGHT = 70
ITEM_HEIGHT = 120
BUTTON_HEIGHT = 120
SPACING = 20


class MultiOptionDialog(Widget):
  def __init__(self, title, options, current=""):
    super().__init__()
    self.title = title
    self.options = options
    self.current = current
    self.selection = current
    self.scroll = GuiScrollPanel()

  def _render(self, rect):
    x, y = rect.x + MARGIN, rect.y + MARGIN
    w, h = rect.width - 2 * MARGIN, rect.height - 2 * MARGIN

    rl.draw_rectangle_rounded(rl.Rectangle(x, y, w, h), 0.02, 20, rl.Color(30, 30, 30, 255))

    # Content area
    x += MARGIN
    y += MARGIN
    w -= 2 * MARGIN
    h -= 2 * MARGIN

    # Title
    gui_label(rl.Rectangle(x, y, w, TITLE_HEIGHT), self.title, 50, font_weight=FontWeight.BOLD)

    # Options area
    options_y = y + TITLE_HEIGHT + SPACING
    options_h = h - TITLE_HEIGHT - BUTTON_HEIGHT - 2 * SPACING
    view_rect = rl.Rectangle(x, options_y, w, options_h)
    content_h = len(self.options) * (ITEM_HEIGHT + 10)
    content_rect = rl.Rectangle(x, options_y, w, content_h)

    # Scroll and render options
    offset = self.scroll.handle_scroll(view_rect, content_rect)
    valid_click = self.scroll.is_click_valid()

    rl.begin_scissor_mode(int(x), int(options_y), int(w), int(options_h))
    for i, option in enumerate(self.options):
      item_y = options_y + i * (ITEM_HEIGHT + 10) + offset.y
      item_rect = rl.Rectangle(x, item_y, w, ITEM_HEIGHT)

      if rl.check_collision_recs(item_rect, view_rect):
        selected = option == self.selection
        style = ButtonStyle.PRIMARY if selected else ButtonStyle.NORMAL

        if gui_button(item_rect, option, button_style=style, text_alignment=TextAlignment.LEFT) and valid_click:
          self.selection = option
    rl.end_scissor_mode()

    # Buttons
    button_y = y + h - BUTTON_HEIGHT
    button_w = (w - SPACING) / 2

    if gui_button(rl.Rectangle(x, button_y, button_w, BUTTON_HEIGHT), "Cancel"):
      return 0

    if gui_button(rl.Rectangle(x + button_w + SPACING, button_y, button_w, BUTTON_HEIGHT),
                 "Select", is_enabled=self.selection != self.current, button_style=ButtonStyle.PRIMARY):
      return 1

    return -1
