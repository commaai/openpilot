import pyray as rl
from openpilot.system.ui.lib.application import FontWeight
from openpilot.system.ui.lib.scroll_panel import GuiScrollPanel
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.button import gui_button, ButtonStyle, TextAlignment
from openpilot.system.ui.widgets.label import gui_label

# Constants
MARGIN = 50
TITLE_FONT_SIZE = 70
ITEM_HEIGHT = 135
BUTTON_SPACING = 50
BUTTON_HEIGHT = 160
ITEM_SPACING = 50
LIST_ITEM_SPACING = 25


class MultiOptionDialog(Widget):
  def __init__(self, title, options, current=""):
    super().__init__()
    self.title = title
    self.options = options
    self.current = current
    self.selection = current
    self.scroll = GuiScrollPanel()

  def _render(self, rect):
    dialog_rect = rl.Rectangle(rect.x + MARGIN, rect.y + MARGIN, rect.width - 2 * MARGIN, rect.height - 2 * MARGIN)
    rl.draw_rectangle_rounded(dialog_rect, 0.02, 20, rl.Color(30, 30, 30, 255))

    content_rect = rl.Rectangle(dialog_rect.x + MARGIN, dialog_rect.y + MARGIN,
                                dialog_rect.width - 2 * MARGIN, dialog_rect.height - 2 * MARGIN)

    gui_label(rl.Rectangle(content_rect.x, content_rect.y, content_rect.width, TITLE_FONT_SIZE), self.title, 70, font_weight=FontWeight.BOLD)

    # Options area
    options_y = content_rect.y + TITLE_FONT_SIZE + ITEM_SPACING
    options_h = content_rect.height - TITLE_FONT_SIZE - BUTTON_HEIGHT - 2 * ITEM_SPACING
    view_rect = rl.Rectangle(content_rect.x, options_y, content_rect.width, options_h)
    content_h = len(self.options) * (ITEM_HEIGHT + 10)
    list_content_rect = rl.Rectangle(content_rect.x, options_y, content_rect.width, content_h)

    # Scroll and render options
    offset = self.scroll.handle_scroll(view_rect, list_content_rect)
    valid_click = self.scroll.is_touch_valid() and rl.is_mouse_button_released(rl.MouseButton.MOUSE_BUTTON_LEFT)

    rl.begin_scissor_mode(int(view_rect.x), int(options_y), int(view_rect.width), int(options_h))
    for i, option in enumerate(self.options):
      item_y = options_y + i * (ITEM_HEIGHT + LIST_ITEM_SPACING) + offset.y
      item_rect = rl.Rectangle(view_rect.x, item_y, view_rect.width, ITEM_HEIGHT)

      if rl.check_collision_recs(item_rect, view_rect):
        selected = option == self.selection
        style = ButtonStyle.PRIMARY if selected else ButtonStyle.NORMAL

        if gui_button(item_rect, option, button_style=style, text_alignment=TextAlignment.LEFT) and valid_click:
          self.selection = option
    rl.end_scissor_mode()

    # Buttons
    button_y = content_rect.y + content_rect.height - BUTTON_HEIGHT
    button_w = (content_rect.width - BUTTON_SPACING) / 2

    if gui_button(rl.Rectangle(content_rect.x, button_y, button_w, BUTTON_HEIGHT), "Cancel"):
      return 0

    if gui_button(rl.Rectangle(content_rect.x + button_w + BUTTON_SPACING, button_y, button_w, BUTTON_HEIGHT),
                  "Select", is_enabled=self.selection != self.current, button_style=ButtonStyle.PRIMARY):
      return 1

    return -1
