import pyray as rl
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.widgets import DialogResult
from openpilot.system.ui.widgets.button import gui_button, ButtonStyle
from openpilot.system.ui.widgets.label import gui_text_box

DIALOG_WIDTH = 1520
DIALOG_HEIGHT = 600
BUTTON_HEIGHT = 160
MARGIN = 50
TEXT_AREA_HEIGHT_REDUCTION = 200
BACKGROUND_COLOR = rl.Color(27, 27, 27, 255)


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
