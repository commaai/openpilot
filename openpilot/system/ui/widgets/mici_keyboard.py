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
KEY_MIN_ANIMATION_TIME = 0.075  # s

CAPS_DOUBLE_TAP_WINDOW = 0.3
CAPS_RETURN_DELAY = 0.7
# Frozen touch offsets from the nominal glyph centers, in native screen pixels.
KEY_TOUCH_OFFSETS = {
  'q': (-36.5, 8.0), 'w': (-14.0, 14.5), 'e': (-4.0, 6.0), 'r': (3.0, 11.0), 't': (-4.0, 12.0),
  'y': (0.5, 16.0), 'u': (-1.0, 11.0), 'i': (1.0, 6.0), 'o': (6.0, 8.0), 'p': (38.0, 12.0),
  'a': (-34.0, 4.0), 's': (11.056, 15.5), 'd': (3.111, 18.0), 'f': (9.667, 15.0), 'g': (4.222, 20.0),
  'h': (8.778, 15.0), 'j': (-8.167, 12.5), 'k': (6.889, 20.0), 'l': (3.444, 15.0),
  'z': (11.0, 26.5), 'x': (0.0, 22.5), 'c': (2.0, 23.0), 'v': (-1.0, 18.0), 'b': (2.0, 22.0), 'n': (2.5, 20.5), 'm': (-7.0, 15.0),
}
ROW_TOUCH_OFFSETS = ((-1.0, 9.0), (8.111, 18.0), (0.0, 22.0))

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
      self.original_position = rl.Vector2(x, local_y + KEY_TOUCH_AREA_OFFSET)
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
    return round(self._size_filter.x)


class SmallKey(Key):
  def __init__(self, chars: str):
    super().__init__(chars, FontWeight.BOLD)
    self._size_filter.x = NUMBER_LAYER_SWITCH_FONT_SIZE

  def set_font_size(self, size: float):
    self._size_filter.update(size * (NUMBER_LAYER_SWITCH_FONT_SIZE / CHAR_FONT_SIZE))


class NumberKey(SmallKey):
  def __init__(self):
    super().__init__('123')
    self._caps_icon = gui_app.texture('icons_mici/settings/keyboard/caps_lower.png', 38, 33)

  def _render(self, rect):
    size = self._get_font_size()
    text_width = measure_text_cached(self._font, self.char, size).x
    icon_scale = size * 0.75 / self._caps_icon.height
    icon_width = self._caps_icon.width * icon_scale
    gap = size / 8
    center_x = rect.x + rect.width / 2
    original_x = self._rect.x
    self._rect.x -= (gap + icon_width) / 2
    super()._render(rect)
    self._rect.x = original_x
    rl.draw_texture_ex(self._caps_icon, (center_x + (text_width + gap - icon_width) / 2,
                                       rect.y + (rect.height - self._caps_icon.height * icon_scale) / 2),
                       0, icon_scale, self._color)


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
  def __init__(self, auto_return_to_letters: str = ""):
    super().__init__()
    self._auto_return_to_letters = auto_return_to_letters

    lower_chars = [
      "qwertyuiop",
      "asdfghjkl",
      "zxcvbnm",
    ]
    upper_chars = ["".join([char.upper() for char in row]) for row in lower_chars]
    special_chars = [
      "1234567890",
      "-/:;()$&@\"",
      ".,?!'",
    ]
    super_special_chars = [
      "[]{}#%^*+=",
      "_\\|~<>€£¥•",
      ".,?!'",
    ]

    self._lower_keys = [[Key(char) for char in row] for row in lower_chars]
    self._upper_keys = [[Key(char) for char in row] for row in upper_chars]
    self._special_keys = [[Key(char) for char in row] for row in special_chars]
    self._super_special_keys = [[Key(char) for char in row] for row in super_special_chars]

    # control keys
    self._space_key = IconKey("icons_mici/settings/keyboard/space.png", char=" ", vertical_align="bottom", icon_size=(43, 14))
    self._caps_key = IconKey("icons_mici/settings/keyboard/caps_lower.png", icon_size=(38, 33))
    # these two are in different places on some layouts
    self._123_key, self._123_key2 = NumberKey(), SmallKey("123")
    self._abc_key = SmallKey("abc")
    self._super_special_key = SmallKey("#+=")

    for keys in (self._lower_keys, self._upper_keys):
      keys[2].insert(0, self._123_key)
      keys[2].append(self._space_key)

    self._symbol_keys = {key for keys in (self._special_keys, self._super_special_keys) for row in keys for key in row}
    for keys, switch in ((self._special_keys, self._super_special_key), (self._super_special_keys, self._123_key2)):
      keys[2][0:0] = [self._abc_key, self._caps_key, switch]

    self._layer_targets = {self._123_key: self._special_keys, self._123_key2: self._special_keys,
                           self._abc_key: self._lower_keys, self._super_special_key: self._super_special_keys}
    self._slide_origin = None
    self._slide_return_control = None
    self._selection_pos = MousePos(0, 0)
    self._caps_return_at = None
    self._caps_last_tap_at = None
    self._caps_quick_tap = False

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
        # reset parent rect for new keys
        key.set_parent_rect(self._rect)
        current_pos = current_key.get_position()
        key.set_position(current_pos[0], current_pos[1], smooth=False)

    self._current_keys = keys

  def set_text(self, text: str):
    self._text = text

  def text(self) -> str:
    return self._text

  @property
  def _hit_rect(self):
    return rl.Rectangle(self.rect.x, self.rect.y + self.rect.height - self.get_keyboard_height(),
                        self.rect.width, self.get_keyboard_height())

  def _handle_mouse_event(self, mouse_event: MouseEvent) -> None:
    super()._handle_mouse_event(mouse_event)
    if mouse_event.left_pressed:
      self._restore_slide_layer()
      self._closest_key = (None, float('inf'))
      self._selected_key_t = self._unselect_key_t = None
      self._dragging_on_keyboard = True
    elif mouse_event.left_released:
      self._dragging_on_keyboard = False

    if mouse_event.left_down and self._dragging_on_keyboard:
      self._selection_pos = mouse_event.pos
      self._closest_key = self._get_closest_key()
      if self._selected_key_t is None:
        self._selected_key_t = rl.get_time()
      if mouse_event.pos.y < self._hit_rect.y:
        self._closest_key = (None, float('inf'))

    if mouse_event.left_pressed:
      key = self._closest_key[0]
      self._caps_quick_tap = (key is self._caps_key and self._caps_last_tap_at is not None and
                              rl.get_time() - self._caps_last_tap_at <= CAPS_DOUBLE_TAP_WINDOW)
      if key is self._caps_key and self._caps_return_at is not None:
        self._caps_return_at = rl.get_time() + CAPS_RETURN_DELAY
      elif key is not self._caps_key:
        self._caps_return_at = self._caps_last_tap_at = None
      if key in self._layer_targets:
        self._slide_origin = (self._current_keys, self._caps_state)
        self._activate_layer_control(key)
        self._closest_key = self._get_closest_key()
        self._slide_return_control = self._closest_key[0]
    if mouse_event.left_released and (not self.is_pressed or not rl.check_collision_point_rec(mouse_event.pos, self._hit_rect)):
      self._restore_slide_layer()

  def touch_center(self, key, row):
    offset = (0.0, KEY_TOUCH_AREA_OFFSET)
    if len(key.char) == 1:
      offset = ROW_TOUCH_OFFSETS[row] if key in self._symbol_keys or key is self._space_key else KEY_TOUCH_OFFSETS[key.char.lower()]
    return (key.original_position.x + offset[0],
            self.rect.y + key.original_position.y - KEY_TOUCH_AREA_OFFSET + offset[1])

  def _get_closest_key(self) -> tuple[Key | None, float]:
    closest = (None, float('inf'))
    for row_index, row in enumerate(self._current_keys):
      for key in row:
        center_x, center_y = self.touch_center(key, row_index)
        distance = (center_x - self._selection_pos.x) ** 2 + (center_y - self._selection_pos.y) ** 2
        if distance < closest[1]:
          closest = (key, distance)
    return closest

  def _set_caps_state(self, state):
    self._caps_state = state
    size = (39, 38) if state == CapsState.LOCK else (38, 33)
    self._caps_key.set_icon(f'icons_mici/settings/keyboard/caps_{state.name.lower()}.png', icon_size=size)

  def _show_letters(self):
    self._set_keys(self._lower_keys if self._caps_state == CapsState.LOWER else self._upper_keys)

  def _activate_layer_control(self, key):
    if key is self._abc_key:
      self._show_letters()
    else:
      self._set_keys(self._layer_targets[key])

  def _handle_mouse_release(self, mouse_pos: MousePos):
    super()._handle_mouse_release(mouse_pos)
    key = self._closest_key[0]
    if key is None:
      self._restore_slide_layer()
      return
    if key is self._caps_key:
      state = CapsState.LOCK if self._caps_state == CapsState.UPPER and self._caps_quick_tap else (
        CapsState.UPPER if self._caps_state == CapsState.LOWER else CapsState.LOWER)
      self._set_caps_state(state)
      if self._slide_origin is not None:
        if self._slide_origin[0] in (self._lower_keys, self._upper_keys):
          self._show_letters()
          self._closest_key = (None, float('inf'))
        self._slide_origin = self._slide_return_control = None
      else:
        self._caps_last_tap_at = rl.get_time()
        self._caps_return_at = rl.get_time() + CAPS_RETURN_DELAY
    elif key in self._layer_targets:
      # Touch-down already opened the initial page; other controls navigate on release.
      if self._slide_origin is not None and key is not self._slide_return_control:
        self._activate_layer_control(key)
        self._closest_key = self._get_closest_key()
      self._slide_origin = self._slide_return_control = None
    else:
      self._text += key.char
      if self._caps_state == CapsState.UPPER:
        self._set_caps_state(CapsState.LOWER)
        self._show_letters()
      if key.char in self._auto_return_to_letters and self._current_keys in (self._special_keys, self._super_special_keys):
        self._set_caps_state(CapsState.LOWER)
        self._show_letters()
      self._restore_slide_layer()

    now = rl.get_time()
    self._unselect_key_t = now + KEY_MIN_ANIMATION_TIME if now - (self._selected_key_t or now) < KEY_MIN_ANIMATION_TIME else now

  def _restore_slide_layer(self):
    if self._slide_origin is not None:
      keys, caps_state = self._slide_origin
      self._slide_origin = self._slide_return_control = None
      self._set_caps_state(caps_state)
      self._set_keys(keys)
      self._closest_key = (None, float('inf'))
      self._selected_key_t = self._unselect_key_t = None

  def hide_event(self):
    super().hide_event()
    self._caps_return_at = self._caps_last_tap_at = None
    self._restore_slide_layer()

  def backspace(self):
    if self._text:
      self._text = self._text[:-1]

  def space(self):
    self._text += ' '

  def _update_state(self):
    super()._update_state()
    if not self.enabled or not self.is_visible:
      self._caps_return_at = self._caps_last_tap_at = None
      self._restore_slide_layer()
    elif (self._caps_return_at is not None and not self.is_pressed and rl.get_time() >= self._caps_return_at and
          not any(event.slot == 0 and event.left_pressed for event in gui_app.mouse_events)):
      self._caps_return_at = None
      self._show_letters()
      self._closest_key = (None, float('inf'))
      self._selected_key_t = self._unselect_key_t = None

    # update selected key filter
    self._selected_key_filter.update(self._closest_key[0] is not None)

    # unselect key after animation plays
    if (self._unselect_key_t is not None and rl.get_time() > self._unselect_key_t) or not self.enabled:
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
          rl.draw_circle_gradient(rl.Vector2(key_x + key.rect.width / 2, key_y + key.rect.height / 2),
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
