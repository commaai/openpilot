import pyray as rl
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.widgets import DialogResult
from openpilot.system.ui.widgets.button import ButtonStyle, Button
from openpilot.system.ui.widgets.label import Label
from openpilot.system.ui.widgets import Widget

DIALOG_WIDTH = 1520 * 1.15
DIALOG_HEIGHT = 600 * 1.15
BUTTON_HEIGHT = 160
MARGIN = 50
TEXT_AREA_HEIGHT_REDUCTION = 200
BACKGROUND_COLOR = rl.Color(27, 27, 27, 255)


class ConfirmDialog(Widget):
  def __init__(self, text: str, confirm_text: str, cancel_text: str = "Cancel"):
    super().__init__()
    self._label = Label(text, 70, FontWeight.BOLD)
    self._cancel_button = Button(cancel_text, self._cancel_button_callback)
    self._confirm_button = Button(confirm_text, self._confirm_button_callback, button_style=ButtonStyle.PRIMARY)
    self._dialog_result = DialogResult.NO_ACTION
    self._cancel_text = cancel_text

  def set_text(self, text):
    self._label.set_text(text)

  def reset(self):
    self._dialog_result = DialogResult.NO_ACTION

  def _cancel_button_callback(self):
    self._dialog_result = DialogResult.CANCEL

  def _confirm_button_callback(self):
    self._dialog_result = DialogResult.CONFIRM

  def _render(self, rect: rl.Rectangle):
    dialog_x = (gui_app.width - DIALOG_WIDTH) / 2
    dialog_y = (gui_app.height - DIALOG_HEIGHT) / 2
    dialog_rect = rl.Rectangle(dialog_x, dialog_y, DIALOG_WIDTH, DIALOG_HEIGHT)

    bottom = dialog_rect.y + dialog_rect.height
    button_width = (dialog_rect.width - 3 * MARGIN) // 2
    cancel_button_x = dialog_rect.x + MARGIN
    confirm_button_x = dialog_rect.x + dialog_rect.width - button_width - MARGIN
    button_y = bottom - BUTTON_HEIGHT - MARGIN
    cancel_button = rl.Rectangle(cancel_button_x, button_y, button_width, BUTTON_HEIGHT)
    confirm_button = rl.Rectangle(confirm_button_x, button_y, button_width, BUTTON_HEIGHT)

    rl.draw_rectangle_rec(dialog_rect, BACKGROUND_COLOR)

    text_rect = rl.Rectangle(dialog_rect.x + MARGIN, dialog_rect.y, dialog_rect.width - 2 * MARGIN, dialog_rect.height - TEXT_AREA_HEIGHT_REDUCTION)
    self._label.render(text_rect)

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
