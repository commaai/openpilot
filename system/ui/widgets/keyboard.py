import pyray as rl
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.lib.button import gui_button
from openpilot.system.ui.lib.inputbox import InputBox
from openpilot.system.ui.lib.label import gui_label

# Constants for special keys
CONTENT_MARGIN = 50
BACKSPACE_KEY = "<-"
ENTER_KEY = "Enter"
SPACE_KEY = "  "
SHIFT_KEY = "↑"
SHIFT_DOWN_KEY = "↓"
NUMERIC_KEY = "123"
SYMBOL_KEY = "#+="
ABC_KEY = "ABC"

# Define keyboard layouts as a dictionary for easier access
keyboard_layouts = {
  "lowercase": [
    ["q", "w", "e", "r", "t", "y", "u", "i", "o", "p"],
    ["a", "s", "d", "f", "g", "h", "j", "k", "l"],
    [SHIFT_KEY, "z", "x", "c", "v", "b", "n", "m", BACKSPACE_KEY],
    [NUMERIC_KEY, "/", "-", SPACE_KEY, ".", ENTER_KEY],
  ],
  "uppercase": [
    ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
    ["A", "S", "D", "F", "G", "H", "J", "K", "L"],
    [SHIFT_DOWN_KEY, "Z", "X", "C", "V", "B", "N", "M", BACKSPACE_KEY],
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


class Keyboard:
  def __init__(self, max_text_size: int = 255, min_text_size: int = 0):
    self._layout = keyboard_layouts["lowercase"]
    self._max_text_size = max_text_size
    self._min_text_size = min_text_size
    self._input_box = InputBox(max_text_size)

  @property
  def text(self):
    return self._input_box.text

  def clear(self):
    self._input_box.clear()

  def render(self, title: str, sub_title: str):
    rect = rl.Rectangle(CONTENT_MARGIN, CONTENT_MARGIN, gui_app.width - 2 * CONTENT_MARGIN, gui_app.height - 2 * CONTENT_MARGIN)
    gui_label(rl.Rectangle(rect.x, rect.y, rect.width, 95), title, 90)
    gui_label(rl.Rectangle(rect.x, rect.y + 95, rect.width, 60), sub_title, 55, rl.GRAY)
    if gui_button(rl.Rectangle(rect.x + rect.width - 300, rect.y, 300, 100), "Cancel"):
      self.clear()
      return 0

    # Text box for input
    self._input_box.render(rl.Rectangle(rect.x, rect.y + 160, rect.width, 100))
    h_space, v_space = 15, 15
    row_y_start = rect.y + 300  # Starting Y position for the first row
    key_height = (rect.height - 300 - 3 * v_space) / 4
    key_max_width = (rect.width - (len(self._layout[2]) - 1) * h_space) / len(self._layout[2])

    # Iterate over the rows of keys in the current layout
    for row, keys in enumerate(self._layout):
      key_width = min((rect.width - (180 if row == 1 else 0) - h_space * (len(keys) - 1)) / len(keys), key_max_width)
      start_x = rect.x + (90 if row == 1 else 0)

      for i, key in enumerate(keys):
        if i > 0:
          start_x += h_space

        new_width = (key_width * 3 + h_space * 2) if key == SPACE_KEY else (key_width * 2 + h_space if key == ENTER_KEY else key_width)
        key_rect = rl.Rectangle(start_x, row_y_start + row * (key_height + v_space), new_width, key_height)
        start_x += new_width

        is_enabled = key != ENTER_KEY or len(self._input_box.text) >= self._min_text_size
        if gui_button(key_rect, key, is_enabled=is_enabled):
          if key == ENTER_KEY:
            return 1
          else:
            self.handle_key_press(key)

    return -1

  def handle_key_press(self, key):
    if key in (SHIFT_DOWN_KEY, ABC_KEY):
      self._layout = keyboard_layouts["lowercase"]
    elif key == SHIFT_KEY:
      self._layout = keyboard_layouts["uppercase"]
    elif key == NUMERIC_KEY:
      self._layout = keyboard_layouts["numbers"]
    elif key == SYMBOL_KEY:
      self._layout = keyboard_layouts["specials"]
    elif key == BACKSPACE_KEY:
      self._input_box.delete_char_before_cursor()
    else:
      self._input_box.add_char_at_cursor(key)


if __name__ == "__main__":
  gui_app.init_window("Keyboard")
  keyboard = Keyboard(min_text_size=8)
  for _ in gui_app.render():
    result = keyboard.render("Keyboard", "Type here")
    if result == 1:
      print(f"You typed: {keyboard.text}")
      gui_app.request_close()
    elif result == 0:
      print("Canceled")
      gui_app.request_close()
  gui_app.close()
