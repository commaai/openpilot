import pyray as rl

BACKSPACE_KEY = "⌫"
ENTER_KEY = "→"
SPACE_KEY = "  "

# Define keyboard layouts
lowercase = [
  ["q", "w", "e", "r", "t", "y", "u", "i", "o", "p"],
  ["a", "s", "d", "f", "g", "h", "j", "k", "l"],
  ["↑", "z", "x", "c", "v", "b", "n", "m", BACKSPACE_KEY],
  ["123", "/", "-", SPACE_KEY, ".", ENTER_KEY]
]

uppercase = [
  ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
  ["A", "S", "D", "F", "G", "H", "J", "K", "L"],
  ["↓", "Z", "X", "C", "V", "B", "N", "M", BACKSPACE_KEY],
  ["123", "/", "-", SPACE_KEY, ".", ENTER_KEY]
]

numbers = [
  ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"],
  ["-", "/", ":", ";", "(", ")", "$", "&", "@", "\""],
  ["#+=", ".", ",", "?", "!", "`", BACKSPACE_KEY],
  ["ABC", SPACE_KEY, ".", ENTER_KEY]
]

specials = [
  ["[", "]", "{", "}", "#", "%", "^", "*", "+", "="],
  ["_", "\\", "|", "~", "<", ">", "€", "£", "¥", "•"],
  ["123", ".", ",", "?", "!", "'", BACKSPACE_KEY],
  ["ABC", SPACE_KEY, ".", ENTER_KEY]
]


class Keyboard:
  def __init__(self):
    self.layout = lowercase
    self.input_text = ""
    # rl.gui_set_style(rl.DEFAULT, rl.TEXT_SIZE, 40)

  def render(self, rect, title, sub_title):
    rl.draw_text(title, int(rect.x), int(rect.y), 40, rl.WHITE)
    rl.draw_text(sub_title, int(rect.x), int(rect.y) + 45, 20, rl.GRAY)

    rl.gui_text_box(rl.Rectangle(rect.x, rect.y + 75, rect.width, 30), self.input_text, 256, True)

    key_height = 155
    v_space = 3
    key_max_width = rect.width / len(self.layout[2])

    for row, keys in enumerate(self.layout):
      key_width = min((rect.width - (180 if row == 1 else 0)) / len(keys), key_max_width)
      start_x = rect.x + (90 if row == 1 else 0)

      for key in keys:
        new_width = key_width * 3 if key == "  " else (key_width * 2 if key == ENTER_KEY else key_width)
        key_rect = rl.Rectangle(start_x, rect.y + 115 + row * (key_height + v_space), new_width, key_height)
        start_x += new_width

        if rl.gui_button(key_rect, key):
          self.handle_key_press(key)

  def handle_key_press(self, key):
    if key == "↓" or key == "ABC":
      self.layout = lowercase
    elif key == "↑":
      self.layout = uppercase
    elif key == "123":
      self.layout = numbers
    elif key == "#+=":
      self.layout = specials
    elif key == BACKSPACE_KEY and len(self.input_text) > 0:
      self.input_text = self.input_text[:-1]
    elif key != BACKSPACE_KEY and len(self.input_text) < 255:
      self.input_text += key
