import pyray as rl
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.widgets import DialogResult
from openpilot.system.ui.widgets.button import ButtonStyle, Button
from openpilot.system.ui.widgets.label import Label
from openpilot.system.ui.widgets.html_render import HtmlRenderer, ElementType
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.scroller import Scroller

OUTER_MARGIN = 200
RICH_OUTER_MARGIN = 100
BUTTON_HEIGHT = 160
MARGIN = 50
TEXT_PADDING = 10
BACKGROUND_COLOR = rl.Color(27, 27, 27, 255)


class ConfirmDialog(Widget):
  def __init__(self, text: str, confirm_text: str, cancel_text: str = "Cancel", rich: bool = False):
    super().__init__()
    self._label = Label(text, 70, FontWeight.BOLD, text_color=rl.Color(201, 201, 201, 255))
    self._html_renderer = HtmlRenderer(text=text, text_size={ElementType.P: 50})
    self._cancel_button = Button(cancel_text, self._cancel_button_callback)
    self._confirm_button = Button(confirm_text, self._confirm_button_callback, button_style=ButtonStyle.PRIMARY)
    self._rich = rich
    self._dialog_result = DialogResult.NO_ACTION
    self._cancel_text = cancel_text
    self._scroller = Scroller([self._html_renderer], line_separator=False, spacing=0)

  def set_text(self, text):
    if not self._rich:
      self._label.set_text(text)
    else:
      self._html_renderer.parse_html_content(text)

  def reset(self):
    self._dialog_result = DialogResult.NO_ACTION

  def _cancel_button_callback(self):
    self._dialog_result = DialogResult.CANCEL

  def _confirm_button_callback(self):
    self._dialog_result = DialogResult.CONFIRM

  def _render(self, rect: rl.Rectangle):
    dialog_x = OUTER_MARGIN if not self._rich else RICH_OUTER_MARGIN
    dialog_y = OUTER_MARGIN if not self._rich else RICH_OUTER_MARGIN
    dialog_width = gui_app.width - 2 * dialog_x
    dialog_height = gui_app.height - 2 * dialog_y
    dialog_rect = rl.Rectangle(dialog_x, dialog_y, dialog_width, dialog_height)

    bottom = dialog_rect.y + dialog_rect.height
    button_width = (dialog_rect.width - 3 * MARGIN) // 2
    cancel_button_x = dialog_rect.x + MARGIN
    confirm_button_x = dialog_rect.x + dialog_rect.width - button_width - MARGIN
    button_y = bottom - BUTTON_HEIGHT - MARGIN
    cancel_button = rl.Rectangle(cancel_button_x, button_y, button_width, BUTTON_HEIGHT)
    confirm_button = rl.Rectangle(confirm_button_x, button_y, button_width, BUTTON_HEIGHT)

    rl.draw_rectangle_rec(dialog_rect, BACKGROUND_COLOR)

    text_rect = rl.Rectangle(dialog_rect.x + MARGIN, dialog_rect.y + TEXT_PADDING,
                             dialog_rect.width - 2 * MARGIN, dialog_rect.height - BUTTON_HEIGHT - MARGIN - TEXT_PADDING * 2)
    if not self._rich:
      self._label.render(text_rect)
    else:
      html_rect = rl.Rectangle(text_rect.x, text_rect.y, text_rect.width, self._html_renderer.get_total_height(int(text_rect.width)))
      self._html_renderer.set_rect(html_rect)
      self._scroller.render(text_rect)

    if rl.is_key_pressed(rl.KeyboardKey.KEY_ENTER):
      self._dialog_result = DialogResult.CONFIRM
    elif rl.is_key_pressed(rl.KeyboardKey.KEY_ESCAPE):
      self._dialog_result = DialogResult.CANCEL

    if self._cancel_text:
      self._confirm_button.render(confirm_button)
      self._cancel_button.render(cancel_button)
    else:
      centered_button_x = dialog_rect.x + (dialog_rect.width - button_width) / 2
      centered_confirm_button = rl.Rectangle(centered_button_x, button_y, button_width, BUTTON_HEIGHT)
      self._confirm_button.render(centered_confirm_button)

    return self._dialog_result


def alert_dialog(message: str, button_text: str = "OK"):
  return ConfirmDialog(message, button_text, cancel_text="")
