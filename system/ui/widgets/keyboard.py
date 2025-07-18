import time
from typing import Literal
import pyray as rl
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.button import ButtonStyle, gui_button
from openpilot.system.ui.widgets.inputbox import InputBox
from openpilot.system.ui.widgets.label import gui_label

KEY_FONT_SIZE = 96
DOUBLE_CLICK_THRESHOLD = 0.5  # seconds
DELETE_REPEAT_DELAY = 0.5
DELETE_REPEAT_INTERVAL = 0.07

# Constants for special keys
CONTENT_MARGIN = 50
BACKSPACE_KEY = "<-"
ENTER_KEY = "->"
SPACE_KEY = "  "
SHIFT_INACTIVE_KEY = "SHIFT_OFF"
SHIFT_ACTIVE_KEY = "SHIFT_ON"
CAPS_LOCK_KEY = "CAPS"
NUMERIC_KEY = "123"
SYMBOL_KEY = "#+="
ABC_KEY = "ABC"

# Define keyboard layouts as a dictionary for easier access
KEYBOARD_LAYOUTS = {
  "lowercase": [
    ["q", "w", "e", "r", "t", "y", "u", "i", "o", "p"],
    ["a", "s", "d", "f", "g", "h", "j", "k", "l"],
    [SHIFT_INACTIVE_KEY, "z", "x", "c", "v", "b", "n", "m", BACKSPACE_KEY],
    [NUMERIC_KEY, "/", "-", SPACE_KEY, ".", ENTER_KEY],
  ],
  "uppercase": [
    ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
    ["A", "S", "D", "F", "G", "H", "J", "K", "L"],
    [SHIFT_ACTIVE_KEY, "Z", "X", "C", "V", "B", "N", "M", BACKSPACE_KEY],
    [NUMERIC_KEY, "/", "-", SPACE_KEY, ".", ENTER_KEY],
  ],
  "numbers": [
    ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"],
    ["-", "/", ":", ";", "(", ")", "$", "&", "@", "\""],
    [SYMBOL_KEY, ".", ",", "?", "!", "`", BACKSPACE_KEY],
    [ABC_KEY, SPACE_KEY, ".", ENTER_KEY],
  ],
  "specials": [
    ["[", "]", "{", "}", "#", "%", "^", "*", "+", "="],
    ["_", "\\", "|", "~", "<", ">", "€", "£", "¥", "•"],
    [NUMERIC_KEY, ".", ",", "?", "!", "'", BACKSPACE_KEY],
    [ABC_KEY, SPACE_KEY, ".", ENTER_KEY],
  ],
}


class Keyboard(Widget):
  def __init__(self, max_text_size: int = 255, min_text_size: int = 0, password_mode: bool = False, show_password_toggle: bool = False):
    super().__init__()
    self._layout_name: Literal["lowercase", "uppercase", "numbers", "specials"] = "lowercase"
    self._caps_lock = False
    self._last_shift_press_time = 0
    self._title = ""
    self._sub_title = ""

    self._max_text_size = max_text_size
    self._min_text_size = min_text_size
    self._input_box = InputBox(max_text_size)
    self._password_mode = password_mode
    self._show_password_toggle = show_password_toggle

    # Backspace key repeat tracking
    self._backspace_pressed: bool = False
    self._backspace_press_time: float = 0.0
    self._backspace_last_repeat: float = 0.0

    self._eye_open_texture = gui_app.texture("icons/eye_open.png", 81, 54)
    self._eye_closed_texture = gui_app.texture("icons/eye_closed.png", 81, 54)
    self._key_icons = {
      BACKSPACE_KEY: gui_app.texture("icons/backspace.png", 80, 80),
      SHIFT_INACTIVE_KEY: gui_app.texture("icons/shift.png", 80, 80),
      SHIFT_ACTIVE_KEY: gui_app.texture("icons/shift-fill.png", 80, 80),
      CAPS_LOCK_KEY: gui_app.texture("icons/capslock-fill.png", 80, 80),
      ENTER_KEY: gui_app.texture("icons/arrow-right.png", 80, 80),
    }

  @property
  def text(self):
    return self._input_box.text

  def clear(self):
    self._layout_name = "lowercase"
    self._caps_lock = False
    self._input_box.clear()
    self._backspace_pressed = False

  def set_title(self, title: str, sub_title: str = ""):
    self._title = title
    self._sub_title = sub_title

  def _render(self, rect: rl.Rectangle):
    rect = rl.Rectangle(rect.x + CONTENT_MARGIN, rect.y + CONTENT_MARGIN, rect.width - 2 * CONTENT_MARGIN, rect.height - 2 * CONTENT_MARGIN)
    gui_label(rl.Rectangle(rect.x, rect.y, rect.width, 95), self._title, 90, font_weight=FontWeight.BOLD)
    gui_label(rl.Rectangle(rect.x, rect.y + 95, rect.width, 60), self._sub_title, 55, font_weight=FontWeight.NORMAL)
    if gui_button(rl.Rectangle(rect.x + rect.width - 386, rect.y, 386, 125), "Cancel"):
      self.clear()
      return 0

    # Draw input box and password toggle
    input_margin = 25
    input_box_rect = rl.Rectangle(rect.x + input_margin, rect.y + 160, rect.width - input_margin, 100)
    self._render_input_area(input_box_rect)

    # Process backspace key repeat if it's held down
    if not rl.is_mouse_button_down(rl.MouseButton.MOUSE_BUTTON_LEFT):
      self._backspace_pressed = False

    if self._backspace_pressed:
      current_time = time.monotonic()
      time_since_press = current_time - self._backspace_press_time

      # After initial delay, start repeating with shorter intervals
      if time_since_press > DELETE_REPEAT_DELAY:
        time_since_last_repeat = current_time - self._backspace_last_repeat
        if time_since_last_repeat > DELETE_REPEAT_INTERVAL:
          self._input_box.delete_char_before_cursor()
          self._backspace_last_repeat = current_time

    layout = KEYBOARD_LAYOUTS[self._layout_name]

    h_space, v_space = 15, 15
    row_y_start = rect.y + 300  # Starting Y position for the first row
    key_height = (rect.height - 300 - 3 * v_space) / 4
    key_max_width = (rect.width - (len(layout[2]) - 1) * h_space) / len(layout[2])

    # Iterate over the rows of keys in the current layout
    for row, keys in enumerate(layout):
      key_width = min((rect.width - (180 if row == 1 else 0) - h_space * (len(keys) - 1)) / len(keys), key_max_width)
      start_x = rect.x + (90 if row == 1 else 0)

      for i, key in enumerate(keys):
        if i > 0:
          start_x += h_space

        new_width = (key_width * 3 + h_space * 2) if key == SPACE_KEY else (key_width * 2 + h_space if key == ENTER_KEY else key_width)
        key_rect = rl.Rectangle(start_x, row_y_start + row * (key_height + v_space), new_width, key_height)
        start_x += new_width

        is_enabled = key != ENTER_KEY or len(self._input_box.text) >= self._min_text_size
        result = -1

        # Check for backspace key press-and-hold
        mouse_pos = rl.get_mouse_position()
        mouse_over_key = rl.check_collision_point_rec(mouse_pos, key_rect)

        if key == BACKSPACE_KEY and mouse_over_key:
          if rl.is_mouse_button_pressed(rl.MouseButton.MOUSE_BUTTON_LEFT):
            self._backspace_pressed = True
            self._backspace_press_time = time.monotonic()
            self._backspace_last_repeat = time.monotonic()

        if key in self._key_icons:
          if key == SHIFT_ACTIVE_KEY and self._caps_lock:
            key = CAPS_LOCK_KEY
          texture = self._key_icons[key]
          result = gui_button(key_rect, "", icon=texture, button_style=ButtonStyle.PRIMARY if key == ENTER_KEY else ButtonStyle.NORMAL, is_enabled=is_enabled)
        else:
          result = gui_button(key_rect, key, KEY_FONT_SIZE, is_enabled=is_enabled)

        if result:
          if key == ENTER_KEY:
            return 1
          else:
            self.handle_key_press(key)

    return -1

  def _render_input_area(self, input_rect: rl.Rectangle):
    if self._show_password_toggle:
      self._input_box.set_password_mode(self._password_mode)
      self._input_box.render(rl.Rectangle(input_rect.x, input_rect.y, input_rect.width - 100, input_rect.height))

      # render eye icon
      eye_texture = self._eye_closed_texture if self._password_mode else self._eye_open_texture

      eye_rect = rl.Rectangle(input_rect.x + input_rect.width - 90, input_rect.y, 80, input_rect.height)
      eye_x = eye_rect.x + (eye_rect.width - eye_texture.width) / 2
      eye_y = eye_rect.y + (eye_rect.height - eye_texture.height) / 2

      rl.draw_texture_v(eye_texture, rl.Vector2(eye_x, eye_y), rl.WHITE)

      # Handle click on eye icon
      if rl.is_mouse_button_pressed(rl.MouseButton.MOUSE_BUTTON_LEFT) and rl.check_collision_point_rec(
        rl.get_mouse_position(), eye_rect
      ):
        self._password_mode = not self._password_mode
    else:
      self._input_box.render(input_rect)

    rl.draw_line_ex(
      rl.Vector2(input_rect.x, input_rect.y + input_rect.height - 2),
      rl.Vector2(input_rect.x + input_rect.width, input_rect.y + input_rect.height - 2),
      3.0,  # 3 pixel thickness
      rl.Color(189, 189, 189, 255),
    )

  def handle_key_press(self, key):
    if key in (CAPS_LOCK_KEY, ABC_KEY):
      self._caps_lock = False
      self._layout_name = "lowercase"
    elif key == SHIFT_INACTIVE_KEY:
      self._last_shift_press_time = time.monotonic()
      self._layout_name = "uppercase"
    elif key == SHIFT_ACTIVE_KEY:
      if time.monotonic() - self._last_shift_press_time < DOUBLE_CLICK_THRESHOLD:
        self._caps_lock = True
      else:
        self._layout_name = "lowercase"
    elif key == NUMERIC_KEY:
      self._layout_name = "numbers"
    elif key == SYMBOL_KEY:
      self._layout_name = "specials"
    elif key == BACKSPACE_KEY:
      self._input_box.delete_char_before_cursor()
    else:
      self._input_box.add_char_at_cursor(key)
      if not self._caps_lock and self._layout_name == "uppercase":
        self._layout_name = "lowercase"


if __name__ == "__main__":
  gui_app.init_window("Keyboard")
  keyboard = Keyboard(min_text_size=8, show_password_toggle=True)
  for _ in gui_app.render():
    keyboard.set_title("Keyboard Input", "Type your text below")
    result = keyboard.render(rl.Rectangle(0, 0, gui_app.width, gui_app.height))
    if result == 1:
      print(f"You typed: {keyboard.text}")
      gui_app.request_close()
    elif result == 0:
      print("Canceled")
      gui_app.request_close()
  gui_app.close()
