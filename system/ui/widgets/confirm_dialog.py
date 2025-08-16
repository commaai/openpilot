import pyray as rl
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.widgets import DialogResult
from openpilot.system.ui.widgets.button import gui_button, ButtonStyle, Button
from openpilot.system.ui.widgets.label import gui_text_box, Label
from openpilot.system.ui.widgets import Widget

DIALOG_WIDTH = 1520
DIALOG_HEIGHT = 600
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

def confirm_dialog(message: str, confirm_text: str, cancel_text: str = "Cancel") -> DialogResult:
  dialog_x = (gui_app.width - DIALOG_WIDTH) / 2
  dialog_y = (gui_app.height - DIALOG_HEIGHT) / 2
  dialog_rect = rl.Rectangle(dialog_x, dialog_y, DIALOG_WIDTH, DIALOG_HEIGHT)

  # Calculate button positions at the bottom of the dialog
  bottom = dialog_rect.y + dialog_rect.height
  button_width = (dialog_rect.width - 3 * MARGIN) // 2
  no_button_x = dialog_rect.x + MARGIN
  yes_button_x = dialog_rect.x + dialog_rect.width - button_width - MARGIN
  button_y = bottom - BUTTON_HEIGHT - MARGIN
  no_button = rl.Rectangle(no_button_x, button_y, button_width, BUTTON_HEIGHT)
  yes_button = rl.Rectangle(yes_button_x, button_y, button_width, BUTTON_HEIGHT)

  # Draw the dialog background
  rl.draw_rectangle_rec(dialog_rect, BACKGROUND_COLOR)

  # Draw the message in the dialog, centered
  text_rect = rl.Rectangle(dialog_rect.x + MARGIN, dialog_rect.y, dialog_rect.width - 2 * MARGIN, dialog_rect.height - TEXT_AREA_HEIGHT_REDUCTION)
  gui_text_box(
    text_rect,
    message,
    font_size=70,
    alignment=rl.GuiTextAlignment.TEXT_ALIGN_CENTER,
    alignment_vertical=rl.GuiTextAlignmentVertical.TEXT_ALIGN_MIDDLE,
    font_weight=FontWeight.BOLD,
  )

  # Initialize result; -1 means no action taken yet
  result = DialogResult.NO_ACTION

  # Check for keyboard input for accessibility
  if rl.is_key_pressed(rl.KeyboardKey.KEY_ENTER):
    result = DialogResult.CONFIRM
  elif rl.is_key_pressed(rl.KeyboardKey.KEY_ESCAPE):
    result = DialogResult.CANCEL

  # Check for button clicks
  if cancel_text:
    if gui_button(yes_button, confirm_text, button_style=ButtonStyle.PRIMARY):
      result = DialogResult.CONFIRM
    if gui_button(no_button, cancel_text):
      result = DialogResult.CANCEL
  else:
    centered_button_x = dialog_rect.x + (dialog_rect.width - button_width) / 2
    centered_yes_button = rl.Rectangle(centered_button_x, button_y, button_width, BUTTON_HEIGHT)
    if gui_button(centered_yes_button, confirm_text, button_style=ButtonStyle.PRIMARY):
      result = DialogResult.CONFIRM

  return result


def alert_dialog(message: str, button_text: str = "OK") -> DialogResult:
  return confirm_dialog(message, button_text, cancel_text="")
