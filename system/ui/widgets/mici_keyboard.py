from enum import IntEnum
import pyray as rl
import numpy as np
from openpilot.system.ui.lib.application import gui_app, FontWeight, MousePos, MouseEvent
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.widgets import Widget
from openpilot.common.filter_simple import BounceFilter, FirstOrderFilter

CHAR_FONT_SIZE = 42
CHAR_NEAR_FONT_SIZE = CHAR_FONT_SIZE * 2
SELECTED_CHAR_FONT_SIZE = 128
CHAR_CAPS_FONT_SIZE = 38  # TODO: implement this
NUMBER_LAYER_SWITCH_FONT_SIZE = 24
KEYBOARD_COLUMN_PADDING = 33
KEYBOARD_ROW_PADDING = {0: 44, 1: 33, 2: 44}  # TODO: 2 should be 116 with extra control keys added in

KEY_TOUCH_AREA_OFFSET = 10  # px
KEY_DRAG_HYSTERESIS = 5  # px
KEY_MIN_ANIMATION_TIME = 0.075  # s

DEBUG = False
ANIMATION_SCALE = 0.65


def zip_repeat(a, b):
  la, lb = len(a), len(b)
  for i in range(max(la, lb)):
    yield (a[i] if i < la else a[-1],
           b[i] if i < lb else b[-1])


def fast_euclidean_distance(dx, dy):
  # https://en.wikibooks.org/wiki/Algorithms/Distance_approximations
  max_d, min_d = abs(dx), abs(dy)
  if max_d < min_d:
    max_d, min_d = min_d, max_d
  return 0.941246 * max_d + 0.41 * min_d


class Key(Widget):
  def __init__(self, char: str, font_weight: FontWeight = FontWeight.SEMI_BOLD):
    super().__init__()
    self.char = char
    self._font = gui_app.font(font_weight)
    self._x_filter = BounceFilter(0.0, 0.1 * ANIMATION_SCALE, 1 / gui_app.target_fps)
    self._y_filter = BounceFilter(0.0, 0.1 * ANIMATION_SCALE, 1 / gui_app.target_fps)
    self._size_filter = BounceFilter(CHAR_FONT_SIZE, 0.1 * ANIMATION_SCALE, 1 / gui_app.target_fps)
    self._alpha_filter = BounceFilter(1.0, 0.075 * ANIMATION_SCALE, 1 / gui_app.target_fps)

    self._color = rl.Color(255, 255, 255, 255)

    self._position_initialized = False
    self.original_position = rl.Vector2(0, 0)

  def set_position(self, x: float, y: float, smooth: bool = True):
    # Smooth keys within parent rect
    base_y = self._parent_rect.y if self._parent_rect else 0.0
    local_y = y - base_y

    if not self._position_initialized:
      self._x_filter.x = x
      self._y_filter.x = local_y
      # keep track of original position so dragging around feels consistent. also move touch area down a bit
      self.original_position = rl.Vector2(x, y + KEY_TOUCH_AREA_OFFSET)
      self._position_initialized = True

    if not smooth:
      self._x_filter.x = x
      self._y_filter.x = local_y

    self._rect.x = self._x_filter.update(x)
    self._rect.y = base_y + self._y_filter.update(local_y)

  def set_alpha(self, alpha: float):
    self._alpha_filter.update(alpha)

  def get_position(self) -> tuple[float, float]:
    return self._rect.x, self._rect.y

  def _update_state(self):
    self._color.a = min(int(255 * self._alpha_filter.x), 255)

  def _render(self, _):
    # center char at rect position
    text_size = measure_text_cached(self._font, self.char, self._get_font_size())
    x = self._rect.x + self._rect.width / 2 - text_size.x / 2
    y = self._rect.y + self._rect.height / 2 - text_size.y / 2
    rl.draw_text_ex(self._font, self.char, (x, y), self._get_font_size(), 0, self._color)

    if DEBUG:
      rl.draw_circle(int(self._rect.x), int(self._rect.y), 5, rl.RED)  # Debug: draw circle around key
      rl.draw_rectangle_lines_ex(self._rect, 2, rl.RED)

  def set_font_size(self, size: float):
    self._size_filter.update(size)

  def _get_font_size(self) -> int:
    return int(round(self._size_filter.x))


class SmallKey(Key):
  def __init__(self, chars: str):
    super().__init__(chars, FontWeight.BOLD)
    self._size_filter.x = NUMBER_LAYER_SWITCH_FONT_SIZE

  def set_font_size(self, size: float):
    self._size_filter.update(size * (NUMBER_LAYER_SWITCH_FONT_SIZE / CHAR_FONT_SIZE))


class IconKey(Key):
  def __init__(self, icon: str, vertical_align: str = "center", char: str = "", icon_size: tuple[int, int] = (38, 38)):
    super().__init__(char)
    self._icon_size = icon_size
    self._icon = gui_app.texture(icon, *icon_size)
    self._vertical_align = vertical_align

  def set_icon(self, icon: str, icon_size: tuple[int, int] | None = None):
    size = icon_size if icon_size is not None else self._icon_size
    self._icon = gui_app.texture(icon, *size)

  def _render(self, _):
    scale = np.interp(self._size_filter.x, [CHAR_FONT_SIZE, CHAR_NEAR_FONT_SIZE], [1, 1.5])

    if self._vertical_align == "center":
      dest_rec = rl.Rectangle(self._rect.x + (self._rect.width - self._icon.width * scale) / 2,
                              self._rect.y + (self._rect.height - self._icon.height * scale) / 2,
                              self._icon.width * scale, self._icon.height * scale)
      src_rec = rl.Rectangle(0, 0, self._icon.width, self._icon.height)
      rl.draw_texture_pro(self._icon, src_rec, dest_rec, rl.Vector2(0, 0), 0, self._color)

    elif self._vertical_align == "bottom":
      dest_rec = rl.Rectangle(self._rect.x + (self._rect.width - self._icon.width * scale) / 2, self._rect.y,
                              self._icon.width * scale, self._icon.height * scale)
      src_rec = rl.Rectangle(0, 0, self._icon.width, self._icon.height)
      rl.draw_texture_pro(self._icon, src_rec, dest_rec, rl.Vector2(0, 0), 0, self._color)

    if DEBUG:
      rl.draw_circle(int(self._rect.x), int(self._rect.y), 5, rl.RED)  # Debug: draw circle around key
      rl.draw_rectangle_lines_ex(self._rect, 2, rl.RED)


class CapsState(IntEnum):
  LOWER = 0
  UPPER = 1
  LOCK = 2


class MiciKeyboard(Widget):
  def __init__(self):
    super().__init__()

    lower_chars = [
      "qwertyuiop",
      "asdfghjkl",
      "zxcvbnm",
    ]
    upper_chars = ["".join([char.upper() for char in row]) for row in lower_chars]
    special_chars = [
      "1234567890",
      "-/:;()$&@\"",
      "~.,?!'#%",
    ]
    super_special_chars = [
      "1234567890",
      "`[]{}^*+=_",
      "\\|<>¥€£•",
    ]

    self._lower_keys = [[Key(char) for char in row] for row in lower_chars]
    self._upper_keys = [[Key(char) for char in row] for row in upper_chars]
    self._special_keys = [[Key(char) for char in row] for row in special_chars]
    self._super_special_keys = [[Key(char) for char in row] for row in super_special_chars]

    # control keys
    self._space_key = IconKey("icons_mici/settings/keyboard/space.png", char=" ", vertical_align="bottom", icon_size=(43, 14))
    self._caps_key = IconKey("icons_mici/settings/keyboard/caps_lower.png", icon_size=(38, 33))
    # these two are in different places on some layouts
    self._123_key, self._123_key2 = SmallKey("123"), SmallKey("123")
    self._abc_key = SmallKey("abc")
    self._super_special_key = SmallKey("#+=")

    # insert control keys
    for keys in (self._lower_keys, self._upper_keys):
      keys[2].insert(0, self._caps_key)
      keys[2].append(self._123_key)

    for keys in (self._lower_keys, self._upper_keys, self._special_keys, self._super_special_keys):
      keys[1].append(self._space_key)

    for keys in (self._special_keys, self._super_special_keys):
      keys[2].append(self._abc_key)

    self._special_keys[2].insert(0, self._super_special_key)
    self._super_special_keys[2].insert(0, self._123_key2)

    # set initial keys
    self._current_keys: list[list[Key]] = []
    self._set_keys(self._lower_keys)
    self._caps_state = CapsState.LOWER
    self._initialized = False

    self._load_images()

    self._closest_key: tuple[Key | None, float] = None, float('inf')
    self._selected_key_t: float | None = None  # time key was initially selected
    self._unselect_key_t: float | None = None  # time to unselect key after release
    self._dragging_on_keyboard = False

    self._text: str = ""

    self._bg_scale_filter = BounceFilter(1.0, 0.1 * ANIMATION_SCALE, 1 / gui_app.target_fps)
    self._selected_key_filter = FirstOrderFilter(0.0, 0.075 * ANIMATION_SCALE, 1 / gui_app.target_fps)

  def get_candidate_character(self) -> str:
    # return str of character about to be added to text
    key = self._closest_key[0]
    return key.char if key is not None and key.__class__ is Key and self._dragging_on_keyboard else ""

  def get_keyboard_height(self) -> int:
    return int(self._txt_bg.height)

  def _load_images(self):
    self._txt_bg = gui_app.texture("icons_mici/settings/keyboard/keyboard_background.png", 520, 170, keep_aspect_ratio=False)

  def _set_keys(self, keys: list[list[Key]]):
    # inherit previous keys' positions to fix switching animation
    for current_row, row in zip(self._current_keys, keys, strict=False):
      # not all layouts have the same number of keys
      for current_key, key in zip_repeat(current_row, row):
        current_pos = current_key.get_position()
        key.set_position(current_pos[0], current_pos[1], smooth=False)

    self._current_keys = keys

  def set_text(self, text: str):
    self._text = text

  def text(self) -> str:
    return self._text

  def _handle_mouse_event(self, mouse_event: MouseEvent) -> None:
    keyboard_pos_y = self._rect.y + self._rect.height - self._txt_bg.height
    if mouse_event.left_pressed:
      if mouse_event.pos.y > keyboard_pos_y:
        self._dragging_on_keyboard = True
    elif mouse_event.left_released:
      self._dragging_on_keyboard = False

    if mouse_event.left_down and self._dragging_on_keyboard:
      self._closest_key = self._get_closest_key()
      if self._selected_key_t is None:
        self._selected_key_t = rl.get_time()

      # unselect key temporarily if mouse goes above keyboard
      if mouse_event.pos.y <= keyboard_pos_y:
        self._closest_key = (None, float('inf'))

    if DEBUG:
      print('HANDLE MOUSE EVENT', mouse_event, self._closest_key[0].char if self._closest_key[0] else 'None')

  def _get_closest_key(self) -> tuple[Key | None, float]:
    closest_key: tuple[Key | None, float] = (None, float('inf'))
    for row in self._current_keys:
      for key in row:
        mouse_pos = gui_app.last_mouse_event.pos
        # approximate distance for comparison is accurate enough
        dist = abs(key.original_position.x - mouse_pos.x) + abs(key.original_position.y - mouse_pos.y)
        if dist < closest_key[1]:
          if self._closest_key[0] is None or key is self._closest_key[0] or dist < self._closest_key[1] - KEY_DRAG_HYSTERESIS:
            closest_key = (key, dist)
    return closest_key

  def _set_uppercase(self, cycle: bool):
    self._set_keys(self._upper_keys if cycle else self._lower_keys)
    if not cycle:
      self._caps_state = CapsState.LOWER
      self._caps_key.set_icon("icons_mici/settings/keyboard/caps_lower.png", icon_size=(38, 33))
    else:
      if self._caps_state == CapsState.LOWER:
        self._caps_state = CapsState.UPPER
        self._caps_key.set_icon("icons_mici/settings/keyboard/caps_upper.png", icon_size=(38, 33))
      elif self._caps_state == CapsState.UPPER:
        self._caps_state = CapsState.LOCK
        self._caps_key.set_icon("icons_mici/settings/keyboard/caps_lock.png", icon_size=(39, 38))
      else:
        self._set_uppercase(False)

  def _handle_mouse_release(self, mouse_pos: MousePos):
    if self._closest_key[0] is not None:
      if self._closest_key[0] == self._caps_key:
        self._set_uppercase(True)
      elif self._closest_key[0] in (self._123_key, self._123_key2):
        self._set_keys(self._special_keys)
      elif self._closest_key[0] == self._abc_key:
        self._set_uppercase(False)
      elif self._closest_key[0] == self._super_special_key:
        self._set_keys(self._super_special_keys)
      else:
        self._text += self._closest_key[0].char

        # Reset caps state
        if self._caps_state == CapsState.UPPER:
          self._set_uppercase(False)

    # ensure minimum selected animation time
    key_selected_dt = rl.get_time() - (self._selected_key_t or 0)
    cur_t = rl.get_time()
    self._unselect_key_t = cur_t + KEY_MIN_ANIMATION_TIME if (key_selected_dt < KEY_MIN_ANIMATION_TIME) else cur_t

  def backspace(self):
    if self._text:
      self._text = self._text[:-1]

  def space(self):
    self._text += ' '

  def _update_state(self):
    # update selected key filter
    self._selected_key_filter.update(self._closest_key[0] is not None)

    # unselect key after animation plays
    if self._unselect_key_t is not None and rl.get_time() > self._unselect_key_t:
      self._closest_key = (None, float('inf'))
      self._unselect_key_t = None
      self._selected_key_t = None

  def _lay_out_keys(self, bg_x, bg_y, keys: list[list[Key]]):
    key_rect = rl.Rectangle(bg_x, bg_y, self._txt_bg.width, self._txt_bg.height)
    for row_idx, row in enumerate(keys):
      padding = KEYBOARD_ROW_PADDING[row_idx]
      step_y = (key_rect.height - 2 * KEYBOARD_COLUMN_PADDING) / (len(keys) - 1)
      for key_idx, key in enumerate(row):
        key_x = key_rect.x + padding + key_idx * ((key_rect.width - 2 * padding) / (len(row) - 1))
        key_y = key_rect.y + KEYBOARD_COLUMN_PADDING + row_idx * step_y

        if self._closest_key[0] is None:
          key.set_alpha(1.0)
          key.set_font_size(CHAR_FONT_SIZE)
        elif key == self._closest_key[0]:
          # push key up with a max and inward so user can see key easier
          key_y = max(key_y - 120, 40)
          key_x += np.interp(key_x, [self._rect.x, self._rect.x + self._rect.width], [100, -100])
          key.set_alpha(1.0)
          key.set_font_size(SELECTED_CHAR_FONT_SIZE)

          # draw black circle behind selected key
          circle_alpha = int(self._selected_key_filter.x * 225)
          rl.draw_circle_gradient(int(key_x + key.rect.width / 2), int(key_y + key.rect.height / 2),
                                  SELECTED_CHAR_FONT_SIZE, rl.Color(0, 0, 0, circle_alpha), rl.BLANK)
        else:
          # move other keys away from selected key a bit
          dx = key.original_position.x - self._closest_key[0].original_position.x
          dy = key.original_position.y - self._closest_key[0].original_position.y
          distance_from_selected_key = fast_euclidean_distance(dx, dy)

          inv = 1 / (distance_from_selected_key or 1.0)
          ux = dx * inv
          uy = dy * inv

          # NOTE: hardcode to 20 to get entire keyboard to move
          push_pixels = np.interp(distance_from_selected_key, [0, 250], [20, 0])
          key_x += ux * push_pixels
          key_y += uy * push_pixels

          # TODO: slow enough to use an approximation or nah? also caching might work
          font_size = np.interp(distance_from_selected_key, [0, 150], [CHAR_NEAR_FONT_SIZE, CHAR_FONT_SIZE])

          key_alpha = np.interp(distance_from_selected_key, [0, 100], [1.0, 0.35])
          key.set_alpha(key_alpha)
          key.set_font_size(font_size)

        # TODO: I like the push amount, so we should clip the pos inside the keyboard rect
        key.set_parent_rect(self._rect)
        key.set_position(key_x, key_y)

  def _render(self, _):
    # draw bg
    bg_x = self._rect.x + (self._rect.width - self._txt_bg.width) / 2
    bg_y = self._rect.y + self._rect.height - self._txt_bg.height

    scale = self._bg_scale_filter.update(1.0307692307692307 if self._closest_key[0] is not None else 1.0)
    src_rec = rl.Rectangle(0, 0, self._txt_bg.width, self._txt_bg.height)
    dest_rec = rl.Rectangle(self._rect.x + self._rect.width / 2 - self._txt_bg.width * scale / 2, bg_y,
                            self._txt_bg.width * scale, self._txt_bg.height)

    rl.draw_texture_pro(self._txt_bg, src_rec, dest_rec, rl.Vector2(0, 0), 0.0, rl.WHITE)

    # draw keys
    if not self._initialized:
      for keys in (self._lower_keys, self._upper_keys, self._special_keys, self._super_special_keys):
        self._lay_out_keys(bg_x, bg_y, keys)
      self._initialized = True

    self._lay_out_keys(bg_x, bg_y, self._current_keys)
    for row in self._current_keys:
      for key in row:
        key.render()
