import pyray as rl
from openpilot.system.ui.lib.application import gui_app


class InputBox:
  def __init__(self, max_text_size=255, password_mode=False):
    self._max_text_size = max_text_size
    self._input_text = ""
    self._cursor_position = 0
    self._password_mode = password_mode
    self._blink_counter = 0
    self._show_cursor = False
    self._last_key_pressed = 0
    self._key_press_time = 0
    self._repeat_delay = 30
    self._repeat_rate = 5

  @property
  def text(self):
    return self._input_text

  @text.setter
  def text(self, value):
    self._input_text = value[: self._max_text_size]
    self._cursor_position = len(self._input_text)

  def set_password_mode(self, password_mode):
    self._password_mode = password_mode

  def clear(self):
    self._input_text = ''
    self._cursor_position = 0

  def set_cursor_position(self, position):
    """Set the cursor position and reset the blink counter."""
    if 0 <= position <= len(self._input_text):
      self._cursor_position = position
      self._blink_counter = 0
      self._show_cursor = True

  def add_char_at_cursor(self, char):
    """Add a character at the current cursor position."""
    if len(self._input_text) < self._max_text_size:
      self._input_text = self._input_text[: self._cursor_position] + char + self._input_text[self._cursor_position :]
      self.set_cursor_position(self._cursor_position + 1)
      return True
    return False

  def delete_char_before_cursor(self):
    """Delete the character before the cursor position (backspace)."""
    if self._cursor_position > 0:
      self._input_text = self._input_text[: self._cursor_position - 1] + self._input_text[self._cursor_position :]
      self.set_cursor_position(self._cursor_position - 1)
      return True
    return False

  def delete_char_at_cursor(self):
    """Delete the character at the cursor position (delete)."""
    if self._cursor_position < len(self._input_text):
      self._input_text = self._input_text[: self._cursor_position] + self._input_text[self._cursor_position + 1 :]
      self.set_cursor_position(self._cursor_position)
      return True
    return False

  def render(self, rect, color=rl.LIGHTGRAY, border_color=rl.DARKGRAY, text_color=rl.BLACK, font_size=80):
    # Handle mouse input
    self._handle_mouse_input(rect, font_size)

    # Draw input box
    rl.draw_rectangle_rec(rect, color)
    rl.draw_rectangle_lines_ex(rect, 1, border_color)

    # Process keyboard input
    self._handle_keyboard_input()

    # Update cursor blink
    self._blink_counter += 1
    if self._blink_counter >= 30:
      self._show_cursor = not self._show_cursor
      self._blink_counter = 0

    # Display text
    font = gui_app.font()
    display_text = "•" * len(self._input_text) if self._password_mode else self._input_text
    padding = 10
    rl.draw_text_ex(
      font,
      display_text,
      rl.Vector2(int(rect.x + padding), int(rect.y + rect.height / 2 - font_size / 2)),
      font_size,
      0,
      text_color,
    )

    # Draw cursor
    if self._show_cursor:
      cursor_x = rect.x + padding
      if len(display_text) > 0 and self._cursor_position > 0:
        cursor_x += rl.measure_text_ex(font, display_text[: self._cursor_position], font_size, 0).x

      cursor_height = font_size + 4
      cursor_y = rect.y + rect.height / 2 - cursor_height / 2
      rl.draw_line(int(cursor_x), int(cursor_y), int(cursor_x), int(cursor_y + cursor_height), rl.BLACK)

  def _handle_mouse_input(self, rect, font_size):
    """Handle mouse clicks to position cursor."""
    mouse_pos = rl.get_mouse_position()
    if rl.is_mouse_button_pressed(rl.MOUSE_LEFT_BUTTON) and rl.check_collision_point_rec(mouse_pos, rect):
      # Calculate cursor position from click
      if len(self._input_text) > 0:
        text_width = rl.measure_text_ex(gui_app.font(), self._input_text, font_size, 0).x
        text_pos_x = rect.x + 10

        if mouse_pos.x - text_pos_x > text_width:
          self.set_cursor_position(len(self._input_text))
        else:
          click_ratio = (mouse_pos.x - text_pos_x) / text_width
          self.set_cursor_position(int(len(self._input_text) * click_ratio))
      else:
        self.set_cursor_position(0)

  def _handle_keyboard_input(self):
    """Process keyboard input."""
    key = rl.get_key_pressed()

    # Handle key repeats
    if key == self._last_key_pressed and key != 0:
      self._key_press_time += 1
      if self._key_press_time > self._repeat_delay and self._key_press_time % self._repeat_rate == 0:
        # Process repeated key
        pass
      else:
        return  # Skip processing until repeat triggers
    else:
      self._last_key_pressed = key
      self._key_press_time = 0

    # Handle navigation keys
    if key == rl.KEY_LEFT:
      if self._cursor_position > 0:
        self.set_cursor_position(self._cursor_position - 1)
    elif key == rl.KEY_RIGHT:
      if self._cursor_position < len(self._input_text):
        self.set_cursor_position(self._cursor_position + 1)
    elif key == rl.KEY_BACKSPACE:
      self.delete_char_before_cursor()
    elif key == rl.KEY_DELETE:
      self.delete_char_at_cursor()
    elif key == rl.KEY_HOME:
      self.set_cursor_position(0)
    elif key == rl.KEY_END:
      self.set_cursor_position(len(self._input_text))

    # Handle text input
    char = rl.get_char_pressed()
    if char != 0 and char >= 32:  # Filter out control characters
      self.add_char_at_cursor(chr(char))
