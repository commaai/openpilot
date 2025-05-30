import pyray as rl
from enum import IntEnum
from openpilot.system.ui.lib.application import gui_app, FontWeight


class ButtonStyle(IntEnum):
  NORMAL = 0  # Most common, neutral buttons
  PRIMARY = 1  # For main actions
  DANGER = 2  # For critical actions, like reboot or delete
  TRANSPARENT = 3  # For buttons with transparent background and border


class TextAlignment(IntEnum):
  LEFT = 0
  CENTER = 1
  RIGHT = 2


DEFAULT_BUTTON_FONT_SIZE = 60
BUTTON_ENABLED_TEXT_COLOR = rl.Color(228, 228, 228, 255)
BUTTON_DISABLED_TEXT_COLOR = rl.Color(228, 228, 228, 51)


BUTTON_BACKGROUND_COLORS = {
  ButtonStyle.NORMAL: rl.Color(51, 51, 51, 255),
  ButtonStyle.PRIMARY: rl.Color(70, 91, 234, 255),
  ButtonStyle.DANGER: rl.Color(255, 36, 36, 255),
  ButtonStyle.TRANSPARENT: rl.BLACK,
}

BUTTON_PRESSED_BACKGROUND_COLORS = {
  ButtonStyle.NORMAL: rl.Color(74, 74, 74, 255),
  ButtonStyle.PRIMARY: rl.Color(48, 73, 244, 255),
  ButtonStyle.DANGER: rl.Color(255, 36, 36, 255),
  ButtonStyle.TRANSPARENT: rl.BLACK,
}


def gui_button(
  rect: rl.Rectangle,
  text: str,
  font_size: int = DEFAULT_BUTTON_FONT_SIZE,
  font_weight: FontWeight = FontWeight.MEDIUM,
  button_style: ButtonStyle = ButtonStyle.NORMAL,
  is_enabled: bool = True,
  border_radius: int = 10,  # Corner rounding in pixels
  text_alignment: TextAlignment = TextAlignment.CENTER,
  text_padding: int = 20,  # Padding for left/right alignment
) -> int:
  result = 0

  # Set background color based on button type
  bg_color = BUTTON_BACKGROUND_COLORS[button_style]
  if is_enabled and rl.check_collision_point_rec(rl.get_mouse_position(), rect):
    if rl.is_mouse_button_down(rl.MouseButton.MOUSE_BUTTON_LEFT):
      bg_color = BUTTON_PRESSED_BACKGROUND_COLORS[button_style]
    elif rl.is_mouse_button_released(rl.MouseButton.MOUSE_BUTTON_LEFT):
      result = 1

  # Draw the button with rounded corners
  roundness = border_radius / (min(rect.width, rect.height) / 2)
  if button_style != ButtonStyle.TRANSPARENT:
    rl.draw_rectangle_rounded(rect, roundness, 20, bg_color)
  else:
    rl.draw_rectangle_rounded(rect, roundness, 20, rl.BLACK)
    rl.draw_rectangle_rounded_lines_ex(rect, roundness, 20, 2, rl.WHITE)

  font = gui_app.font(font_weight)
  text_size = rl.measure_text_ex(font, text, font_size, 0)
  text_pos = rl.Vector2(0, rect.y + (rect.height - text_size.y) // 2)  # Vertical centering

  # Horizontal alignment
  if text_alignment == TextAlignment.LEFT:
    text_pos.x = rect.x + text_padding
  elif text_alignment == TextAlignment.CENTER:
    text_pos.x = rect.x + (rect.width - text_size.x) // 2
  elif text_alignment == TextAlignment.RIGHT:
    text_pos.x = rect.x + rect.width - text_size.x - text_padding

  # Draw the button text
  text_color = BUTTON_ENABLED_TEXT_COLOR if is_enabled else BUTTON_DISABLED_TEXT_COLOR
  rl.draw_text_ex(font, text, text_pos, font_size, 0, text_color)

  return result
