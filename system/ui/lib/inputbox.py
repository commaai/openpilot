import pyray as rl
import time
from openpilot.system.ui.lib.application import gui_app


PASSWORD_MASK_CHAR = "•"
PASSWORD_MASK_DELAY = 1.5  # Seconds to show character before masking


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
    self._repeat_rate = 4
    self._text_offset = 0
    self._visible_width = 0
    self._last_char_time = 0  # Track when last character was added
    self._masked_length = 0   # How many characters are currently masked

  @property
  def text(self):
    return self._input_text

  @text.setter
  def text(self, value):
    self._input_text = value[: self._max_text_size]
    self._cursor_position = len(self._input_text)
    self._update_text_offset()

  def set_password_mode(self, password_mode):
    self._password_mode = password_mode

  def clear(self):
    self._input_text = ''
    self._cursor_position = 0
    self._text_offset = 0

  def set_cursor_position(self, position):
    """Set the cursor position and reset the blink counter."""
    if 0 <= position <= len(self._input_text):
      self._cursor_position = position
      self._blink_counter = 0
      self._show_cursor = True
      self._update_text_offset()

  def _update_text_offset(self):
    """Ensure the cursor is visible by adjusting text offset."""
    if self._visible_width == 0:
      return

    font = gui_app.font()
    display_text = self._get_display_text()
    padding = 10

    if self._cursor_position > 0:
      cursor_x = rl.measure_text_ex(font, display_text[: self._cursor_position], self._font_size, 0).x
    else:
      cursor_x = 0

    visible_width = self._visible_width - (padding * 2)

    # Adjust offset if cursor would be outside visible area
    if cursor_x < self._text_offset:
      self._text_offset = max(0, cursor_x - padding)
    elif cursor_x > self._text_offset + visible_width:
      self._text_offset = cursor_x - visible_width + padding

  def add_char_at_cursor(self, char):
    """Add a character at the current cursor position."""
    if len(self._input_text) < self._max_text_size:
      self._input_text = self._input_text[: self._cursor_position] + char + self._input_text[self._cursor_position :]
      self.set_cursor_position(self._cursor_position + 1)

      if self._password_mode:
        self._last_char_time = time.time()

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

  def render(self, rect, color=rl.BLACK, border_color=rl.DARKGRAY, text_color=rl.WHITE, font_size=80):
    # Store dimensions for text offset calculations
    self._visible_width = rect.width
    self._font_size = font_size

    # Handle mouse input
    self._handle_mouse_input(rect, font_size)

    # Draw input box
    rl.draw_rectangle_rec(rect, color)

    # Process keyboard input
    self._handle_keyboard_input()

    # Update cursor blink
    self._blink_counter += 1
    if self._blink_counter >= 30:
      self._show_cursor = not self._show_cursor
      self._blink_counter = 0

    # Display text
    font = gui_app.font()
    display_text = self._get_display_text()
    padding = 10

    # Clip text within input box bounds
    buffer = 2
    rl.begin_scissor_mode(int(rect.x + padding - buffer), int(rect.y), int(rect.width - padding * 2 + buffer * 2), int(rect.height))
    rl.draw_text_ex(
      font,
      display_text,
      rl.Vector2(int(rect.x + padding - self._text_offset), int(rect.y + rect.height / 2 - font_size / 2)),
      font_size,
      0,
      text_color,
    )

    # Draw cursor
    if self._show_cursor:
      cursor_x = rect.x + padding
      if len(display_text) > 0 and self._cursor_position > 0:
        cursor_x += rl.measure_text_ex(font, display_text[: self._cursor_position], font_size, 0).x

      # Apply text offset to cursor position
      cursor_x -= self._text_offset

      cursor_height = font_size + 4
      cursor_y = rect.y + rect.height / 2 - cursor_height / 2
      rl.draw_line(int(cursor_x), int(cursor_y), int(cursor_x), int(cursor_y + cursor_height), rl.WHITE)

    rl.end_scissor_mode()

  def _get_display_text(self):
    """Get text to display, applying password masking with delay if needed."""
    if not self._password_mode:
      return self._input_text

    # Show character at last edited position if within delay window
    masked_text = PASSWORD_MASK_CHAR * len(self._input_text)
    recent_edit = time.time() - self._last_char_time < PASSWORD_MASK_DELAY
    if recent_edit and self._input_text:
      last_pos = max(0, self._cursor_position - 1)
      if last_pos < len(self._input_text):
        return masked_text[:last_pos] + self._input_text[last_pos] + masked_text[last_pos + 1 :]

    return masked_text

  def _handle_mouse_input(self, rect, font_size):
    """Handle mouse clicks to position cursor."""
    mouse_pos = rl.get_mouse_position()
    if rl.is_mouse_button_pressed(rl.MOUSE_LEFT_BUTTON) and rl.check_collision_point_rec(mouse_pos, rect):
      # Calculate cursor position from click
      if len(self._input_text) > 0:
        font = gui_app.font()
        display_text = self._get_display_text()

        # Find the closest character position to the click
        relative_x = mouse_pos.x - (rect.x + 10) + self._text_offset
        best_pos = 0
        min_distance = float('inf')

        for i in range(len(self._input_text) + 1):
          char_width = rl.measure_text_ex(font, display_text[:i], font_size, 0).x
          distance = abs(relative_x - char_width)
          if distance < min_distance:
            min_distance = distance
            best_pos = i

        self.set_cursor_position(best_pos)
      else:
        self.set_cursor_position(0)

  def _handle_keyboard_input(self):
    # Handle navigation keys
    key = rl.get_key_pressed()
    if key != 0:
      self._process_key(key)
      if key in (rl.KEY_LEFT, rl.KEY_RIGHT, rl.KEY_BACKSPACE, rl.KEY_DELETE):
        self._last_key_pressed = key
        self._key_press_time = 0

    # Handle repeats for held keys
    elif self._last_key_pressed != 0:
      if rl.is_key_down(self._last_key_pressed):
        self._key_press_time += 1
        if self._key_press_time > self._repeat_delay and self._key_press_time % self._repeat_rate == 0:
          self._process_key(self._last_key_pressed)
      else:
        self._last_key_pressed = 0

    # Handle text input
    char = rl.get_char_pressed()
    if char != 0 and char >= 32:  # Filter out control characters
      self.add_char_at_cursor(chr(char))

  def _process_key(self, key):
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
