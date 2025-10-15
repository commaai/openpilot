import pyray as rl
from openpilot.system.ui.lib.application import FontWeight
from openpilot.system.ui.widgets import Widget, DialogResult
from openpilot.system.ui.widgets.button import Button, ButtonStyle, TextAlignment
from openpilot.system.ui.widgets.label import gui_label
from openpilot.system.ui.widgets.scroller import Scroller

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
    self._result: DialogResult = DialogResult.NO_ACTION

    # Create scroller with option buttons
    self.option_buttons = [Button(option, click_callback=lambda opt=option: self._on_option_clicked(opt),
                                  text_alignment=TextAlignment.LEFT, button_style=ButtonStyle.NORMAL) for option in options]
    self.scroller = Scroller(self.option_buttons, spacing=LIST_ITEM_SPACING)

    # Buttons set a result; the application loop will clear the overlay and invoke the callback
    self.cancel_button = Button("Cancel", click_callback=lambda: self._set_result(DialogResult.CANCEL))
    self.select_button = Button("Select", click_callback=lambda: self._set_result(DialogResult.CONFIRM), button_style=ButtonStyle.PRIMARY)

  def _set_result(self, result: DialogResult):
    self._result = result

  def _on_option_clicked(self, option):
    self.selection = option

  def _render(self, rect):
    dialog_rect = rl.Rectangle(rect.x + MARGIN, rect.y + MARGIN, rect.width - 2 * MARGIN, rect.height - 2 * MARGIN)
    rl.draw_rectangle_rounded(dialog_rect, 0.02, 20, rl.Color(30, 30, 30, 255))

    content_rect = rl.Rectangle(dialog_rect.x + MARGIN, dialog_rect.y + MARGIN,
                                dialog_rect.width - 2 * MARGIN, dialog_rect.height - 2 * MARGIN)

    gui_label(rl.Rectangle(content_rect.x, content_rect.y, content_rect.width, TITLE_FONT_SIZE), self.title, 70, font_weight=FontWeight.BOLD)

    # Options area
    options_y = content_rect.y + TITLE_FONT_SIZE + ITEM_SPACING
    options_h = content_rect.height - TITLE_FONT_SIZE - BUTTON_HEIGHT - 2 * ITEM_SPACING
    options_rect = rl.Rectangle(content_rect.x, options_y, content_rect.width, options_h)

    # Update button styles and set width based on selection
    for i, option in enumerate(self.options):
      selected = option == self.selection
      button = self.option_buttons[i]
      button.set_button_style(ButtonStyle.PRIMARY if selected else ButtonStyle.NORMAL)
      button.set_rect(rl.Rectangle(0, 0, options_rect.width, ITEM_HEIGHT))

    self.scroller.render(options_rect)

    # Buttons
    button_y = content_rect.y + content_rect.height - BUTTON_HEIGHT
    button_w = (content_rect.width - BUTTON_SPACING) / 2

    cancel_rect = rl.Rectangle(content_rect.x, button_y, button_w, BUTTON_HEIGHT)
    self.cancel_button.render(cancel_rect)

    select_rect = rl.Rectangle(content_rect.x + button_w + BUTTON_SPACING, button_y, button_w, BUTTON_HEIGHT)
    self.select_button.set_enabled(self.selection != self.current)
    self.select_button.render(select_rect)

    # Keyboard shortcuts
    if rl.is_key_pressed(rl.KeyboardKey.KEY_ENTER):
      if self.selection != self.current:
        self._result = DialogResult.CONFIRM
    elif rl.is_key_pressed(rl.KeyboardKey.KEY_ESCAPE):
      self._result = DialogResult.CANCEL

    return self._result
