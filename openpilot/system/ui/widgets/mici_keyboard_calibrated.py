"""Frozen spatial calibration for the typing study; rendering stays with MiciKeyboard."""
from functools import cache
import hashlib
import json
from pathlib import Path

import pyray as rl

from openpilot.common.filter_simple import BounceFilter
from openpilot.system.ui.lib.application import MousePos, gui_app
from openpilot.system.ui.widgets.mici_keyboard import ANIMATION_SCALE, CapsState, KEY_TOUCH_AREA_OFFSET, Key, MiciKeyboard, SmallKey

MODEL_PATH = Path(__file__).with_name('mici_keyboard_calibration.json')
SPACE_KEY_VARIANTS = ('space_right_letters_only', 'space_right_caps_above', 'space_right_symbols_above')
KEYBOARD_VARIANTS = ('reference', 'caps_original', 'space_right_letters_only', 'swipe_space_caps_middle', *SPACE_KEY_VARIANTS[1:])
SPACE_FLICK_MIN = 32
SPACE_FLICK_MAX = 110
SPACE_FLICK_VERTICAL = 24
CAPS_HINT_HOLD = 1.2
CAPS_HINT_IDLE = 4.0
CAPS_DOUBLE_TAP_WINDOW = 0.3
CAPS_RETURN_DELAY = 0.7
CURRENT_LAYOUT_TARGET_VERSION = 'conservative_l_v2'
CURRENT_LAYOUT_TARGET_ADJUSTMENTS = {'l': (8.0, 0.0)}


@cache
def calibration():
  return json.loads(MODEL_PATH.read_text())


class CapsHintKey(SmallKey):
  def __init__(self):
    super().__init__('123')
    self._hint_phase = None
    self._hint_started = None
    self._hint_finished_at = None
    self._hint_scale = BounceFilter(0.0, 0.1 * ANIMATION_SCALE, 1 / gui_app.target_fps)
    self._hint_icon = gui_app.texture('icons_mici/settings/keyboard/caps_lower.png', 38, 33)

  def start_hint(self, reminder=False):
    self._hint_phase = 'label_out' if reminder else 'caps_in'
    self._hint_finished_at = None
    self._hint_started = rl.get_time()
    self._hint_scale.x = 1.0
    self._hint_scale.velocity.x = 0.0

  def _update_state(self):
    super()._update_state()
    if self._hint_phase is None:
      return
    scale = self._hint_scale.update(0.0 if self._hint_phase in ('label_out', 'caps_out') else 1.0)
    # Hold Caps briefly, then let the stock spring drive the visual swap.
    if self._hint_phase == 'label_out' and scale <= 0.0:
      self._hint_phase = 'caps_rise'
      self._hint_scale.x = self._hint_scale.velocity.x = 0.0
    elif self._hint_phase == 'caps_rise' and scale >= 1.0:
      self._hint_phase = 'caps_in'
      self._hint_started = rl.get_time()
    elif self._hint_phase == 'caps_in' and rl.get_time() - self._hint_started >= CAPS_HINT_HOLD:
      self._hint_phase = 'caps_out'
    elif self._hint_phase == 'caps_out' and scale <= 0.0:
      self._hint_phase = 'label_in'
      self._hint_scale.x = self._hint_scale.velocity.x = 0.0
    elif self._hint_phase == 'label_in' and abs(scale - 1.0) < 0.001 and self._hint_scale.velocity.x == 0:
      self._hint_phase = None
      self._hint_finished_at = rl.get_time()

  def _get_font_size(self):
    size = super()._get_font_size()
    return max(1, round(size * self._hint_scale.x)) if self._hint_phase is not None else size

  def _render(self, rect):
    if self._hint_phase is None:
      super()._render(rect)
      return
    scale = max(0.0, self._hint_scale.x)
    color = self._color
    tint = rl.Color(color.r, color.g, color.b, round(color.a * min(1.0, scale)))
    if self._hint_phase in ('label_out', 'label_in'):
      self._color = tint
      super()._render(rect)
      self._color = color
    elif scale > 0.0:
      position = (rect.x + (rect.width - self._hint_icon.width * scale) / 2,
                  rect.y + (rect.height - self._hint_icon.height * scale) / 2)
      rl.draw_texture_ex(self._hint_icon, position, 0, scale, tint)


class CalibratedKeyboard(MiciKeyboard):
  def __init__(self, calibrated=True, variant='reference'):
    super().__init__()
    self.calibrated = calibrated
    if variant not in KEYBOARD_VARIANTS:
      raise ValueError(f'Unknown keyboard variant: {variant}')
    self.variant = variant
    self._space_flick_origin = None
    self._caps_return_at = None
    self._caps_last_tap_at = None
    self._caps_quick_tap = False
    self._last_keyboard_activity = rl.get_time()
    self._idle_hint_shown = False
    self._symbol_keys = set()
    if calibrated and variant in SPACE_KEY_VARIANTS:
      original_layer_key = self._123_key
      self._123_key = CapsHintKey()
      for keys in (self._lower_keys, self._upper_keys):
        keys[2][keys[2].index(original_layer_key)] = self._123_key
    if calibrated:
      left_layer_button = variant != 'caps_original'
      for keys in (self._lower_keys, self._upper_keys):
        keys[1].remove(self._space_key)
        if left_layer_button:
          keys[2].remove(self._caps_key)
          if variant not in SPACE_KEY_VARIANTS:
            keys[1].insert(0, self._caps_key)
          keys[2].insert(0, keys[2].pop())
        if variant in SPACE_KEY_VARIANTS:
          keys[2].append(self._space_key)
      # iPhone English symbol rows, retaining the floating controls.
      self._special_keys = [[Key(char) for char in row] for row in ("1234567890", '-/:;()$&@"', ".,?!'")]
      self._super_special_keys = [[Key(char) for char in row] for row in ("[]{}#%^*+=", "_\\|~<>€£¥•", ".,?!'")]
      for keys, switch in ((self._special_keys, self._super_special_key), (self._super_special_keys, self._123_key2)):
        self._symbol_keys.update(key for row in keys for key in row)
        if variant in SPACE_KEY_VARIANTS:
          keys[2].insert(0, self._abc_key)
          if variant == 'space_right_letters_only':
            keys[2][1:1] = [self._caps_key, switch]
          else:
            above, beside = (self._caps_key, switch) if variant == 'space_right_caps_above' else (switch, self._caps_key)
            keys[1].insert(0, above)
            keys[2].insert(1, beside)
        else:
          keys[2].insert(2, self._space_key)
          keys[2].insert(0, self._abc_key if left_layer_button else switch)
          keys[2].append(switch if left_layer_button else self._abc_key)
    self._slide_origin = None
    self._slide_return_control = None
    self._layer_targets = {self._123_key: self._special_keys, self._123_key2: self._special_keys,
                           self._abc_key: self._lower_keys, self._super_special_key: self._super_special_keys}
    self._selection_pos = MousePos(0, 0)
    self._calibration = calibration() if calibrated else None

  def touch_center(self, key, row):
    x, y = key.original_position.x, self.rect.y + key.original_position.y
    if self.calibrated and len(key.char) == 1:
      calibration_char = key.char.lower()
      offset = self._calibration['keys'].get(calibration_char, self._calibration['rows'][str(row)])
      if key is self._space_key or key in self._symbol_keys:
        # Old per-symbol horizontal biases do not transfer to the rearranged pages.
        offset = self._calibration['rows'][str(row)]
      dx, dy = offset['offset']
      if self.variant in SPACE_KEY_VARIANTS and key not in self._symbol_keys:
        extra_dx, extra_dy = CURRENT_LAYOUT_TARGET_ADJUSTMENTS.get(calibration_char, (0.0, 0.0))
        dx += extra_dx
        dy += extra_dy
      return x + dx, y - KEY_TOUCH_AREA_OFFSET + dy
    return x, y

  def configuration(self):
    configuration = {
      'condition': ('calibrated_static_v1_space_before_punctuation' if self.variant == 'reference' else
                    f'calibrated_static_v1_variant_{self.variant}') if self.calibrated else 'stock_master', 'prediction': False,
      'variant': self.variant if self.calibrated else 'stock',
      'layout': ('space_before_punctuation_v1' if self.variant == 'reference' else f'{self.variant}_v1') if self.calibrated else 'stock_master',
      'layer_slide': self.calibrated,
      'layer_switch_on': 'press' if self.calibrated else 'release',
      'chained_layer_switch_on': 'release' if self.calibrated else None,
      'layer_drag_return': 'letters_except_return_to_start_control' if self.calibrated else None,
      'distance': 'squared_euclidean' if self.calibrated else 'stock_manhattan_with_inherited_hysteresis',
      'touch_offset': 'per_key_median' if self.calibrated else KEY_TOUCH_AREA_OFFSET,
      'control_penalty': 0,
      'keyboard_sha256': hashlib.sha256(Path(__file__).with_name('mici_keyboard.py').read_bytes()).hexdigest(),
    }
    if self.calibrated:
      configuration.update(calibration=self._calibration, calibration_sha256=hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
                           selector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), selection_point='last_down_event',
                           symbol_calibration='row_median', space_calibration='bottom_row_median')
    if self.calibrated and self.variant in SPACE_KEY_VARIANTS:
      configuration['condition'] += '_' + CURRENT_LAYOUT_TARGET_VERSION
      configuration.update(target_version=CURRENT_LAYOUT_TARGET_VERSION, target_adjustments=CURRENT_LAYOUT_TARGET_ADJUSTMENTS)
      configuration.update(caps_placement='symbols_middle_left' if self.variant == 'space_right_caps_above' else 'symbols_bottom',
                           caps_slide='drag_returns_letters_tap_waits_for_double_tap',
                           caps_double_tap_seconds=CAPS_DOUBLE_TAP_WINDOW, caps_return_delay_seconds=CAPS_RETURN_DELAY,
                           symbol_switch_placement=('middle_left' if self.variant == 'space_right_symbols_above' else
                                                    'bottom_beside_abc' if self.variant == 'space_right_caps_above' else 'bottom_left_after_caps'),
                           space_placement='letters_bottom_right',
                           caps_hint={'animation': 'stock_bounce_scale_and_opacity', 'transition': 'hold_then_spring_zero_crossing',
                                      'idle_seconds': CAPS_HINT_IDLE, 'idle_repeats': 'once_per_pause', 'idle_timer': 'after_animation_or_last_touch',
                                      'idle_entry': 'label_out_then_caps_rise',
                                      'hold_seconds': CAPS_HINT_HOLD, 'initial_scale': 1.0, 'initial_opacity': 1.0,
                                      'dismiss_on_touch': True, 'activation': '123'})
    return configuration

  def _handle_mouse_event(self, mouse_event):
    self._finish_caps_tap()
    if self.calibrated:
      if mouse_event.left_down or mouse_event.left_released:
        self._last_keyboard_activity = rl.get_time()
        self._idle_hint_shown = False
      if mouse_event.left_pressed:
        if isinstance(self._123_key, CapsHintKey):
          self._123_key._hint_phase = None
        self._restore_slide_layer()
        self._space_flick_origin = None
        # A previous tap's preview timer must not clear the next tap's selection.
        self._closest_key = (None, float('inf'))
        self._selected_key_t = None
        self._unselect_key_t = None
      if mouse_event.left_down:
        # Every event in a frame has its own position. Lift coordinates do not retarget a key.
        self._selection_pos = mouse_event.pos
    super()._handle_mouse_event(mouse_event)
    if not self.calibrated:
      return
    if mouse_event.left_pressed:
      self._caps_quick_tap = (self._closest_key[0] is self._caps_key and self._caps_last_tap_at is not None and
                              rl.get_time() - self._caps_last_tap_at <= CAPS_DOUBLE_TAP_WINDOW)
      if self._closest_key[0] is self._caps_key and self._caps_return_at is not None:
        self._caps_return_at = rl.get_time() + CAPS_RETURN_DELAY
      elif self._closest_key[0] is not self._caps_key:
        self._caps_return_at = None
        self._caps_last_tap_at = None
    if mouse_event.left_pressed and self.variant == 'swipe_space_caps_middle' and self._closest_key[0] is self._123_key:
      self._space_flick_origin = mouse_event.pos
    if mouse_event.left_pressed and self._closest_key[0] in self._layer_targets:
      self._slide_origin = (self._current_keys, self._caps_state)
      self._activate_layer_control(self._closest_key[0])
      self._closest_key = self._get_closest_key()
      # Returning to the control now under the starting finger keeps this page.
      self._slide_return_control = self._closest_key[0]
    if mouse_event.left_down and self._space_flick_origin is not None:
      dx = mouse_event.pos.x - self._space_flick_origin.x
      dy = mouse_event.pos.y - self._space_flick_origin.y
      if dx < -SPACE_FLICK_VERTICAL or dx > SPACE_FLICK_MAX or abs(dy) > SPACE_FLICK_VERTICAL:
        self._space_flick_origin = None
    if mouse_event.left_released:
      # Widget skips the release callback for cancelled/outside contacts.
      if not self.is_pressed or not rl.check_collision_point_rec(mouse_event.pos, self._hit_rect):
        self._restore_slide_layer()

  def _activate_layer_control(self, key):
    target = self._layer_targets[key]
    if target is self._lower_keys and self.variant in SPACE_KEY_VARIANTS:
      self._set_keys(self._lower_keys if self._caps_state == CapsState.LOWER else self._upper_keys)
    elif target is self._lower_keys:
      self._set_uppercase(False)
    else:
      self._set_keys(target)

  def is_space_flick(self):
    if self._space_flick_origin is None or self._slide_origin is None or self._current_keys is not self._special_keys:
      return False
    dx = self._selection_pos.x - self._space_flick_origin.x
    dy = self._selection_pos.y - self._space_flick_origin.y
    return SPACE_FLICK_MIN <= dx <= SPACE_FLICK_MAX and abs(dy) <= SPACE_FLICK_VERTICAL

  def _set_uppercase(self, cycle):
    if self.calibrated and self.variant in SPACE_KEY_VARIANTS and cycle and self._current_keys in (self._special_keys, self._super_special_keys):
      # Changing case on symbols must not run a page transition: that copies
      # letter-key animation positions onto the longer symbol rows.
      if self._caps_state == CapsState.UPPER and self._caps_quick_tap:
        self._caps_state = CapsState.LOCK
      else:
        self._caps_state = CapsState.UPPER if self._caps_state == CapsState.LOWER else CapsState.LOWER
      icon = self._caps_state.name.lower()
      size = (39, 38) if self._caps_state == CapsState.LOCK else (38, 33)
      self._caps_key.set_icon(f'icons_mici/settings/keyboard/caps_{icon}.png', icon_size=size)
    else:
      super()._set_uppercase(cycle)

  def _handle_mouse_release(self, mouse_pos):
    if self.is_space_flick():
      self._closest_key = (self._space_key, 0)
    self._space_flick_origin = None
    latched_control = None
    return_after_caps = False
    tapped_caps = False
    if self.calibrated and self.variant in SPACE_KEY_VARIANTS and self._closest_key[0] is self._caps_key:
      # Separate taps leave a short window for Caps Lock before returning.
      tapped_caps = self._slide_origin is None
      return_after_caps = self._slide_origin is not None and self._slide_origin[0] in (self._lower_keys, self._upper_keys)
      self._slide_origin = self._slide_return_control = None
    if self._slide_origin is not None:
      key = self._closest_key[0]
      if key in self._layer_targets:
        if key is not self._slide_return_control:
          self._activate_layer_control(key)
          self._closest_key = self._get_closest_key()
        latched_control = self._closest_key
        self._slide_origin = None
        self._slide_return_control = None
      if key is None or len(key.char) != 1:
        self._closest_key = (None, float('inf'))
    super()._handle_mouse_release(mouse_pos)
    if tapped_caps:
      self._caps_last_tap_at = rl.get_time()
      self._caps_return_at = rl.get_time() + CAPS_RETURN_DELAY
    if return_after_caps:
      self._caps_return_at = None
      self._set_keys(self._lower_keys if self._caps_state == CapsState.LOWER else self._upper_keys)
      self._closest_key = (None, float('inf'))
      self._selected_key_t = self._unselect_key_t = None
    if latched_control is not None:
      # Suppress a second navigation action, but retain the stock minimum-time
      # selection animation on the control now displayed in this location.
      self._closest_key = latched_control
    self._restore_slide_layer()

  def _finish_caps_tap(self):
    if self._caps_return_at is None or self.is_pressed or rl.get_time() < self._caps_return_at:
      return
    # Process queued presses before deciding that a pause has elapsed.
    if any(event.slot == 0 and event.left_pressed for event in gui_app.mouse_events):
      return
    self._caps_return_at = None
    if self._current_keys in (self._special_keys, self._super_special_keys):
      self._set_keys(self._lower_keys if self._caps_state == CapsState.LOWER else self._upper_keys)
      self._closest_key = (None, float('inf'))
      self._selected_key_t = self._unselect_key_t = None

  def _restore_slide_layer(self):
    if self._slide_origin is None:
      return
    self._space_flick_origin = None
    keys, caps_state = self._slide_origin
    self._slide_return_control = None
    self._slide_origin = None
    self._closest_key = (None, float('inf'))
    self._selected_key_t = self._unselect_key_t = None
    self._set_uppercase(False)
    if caps_state != CapsState.LOWER:
      self._set_uppercase(True)
    if caps_state == CapsState.LOCK:
      self._set_uppercase(True)
    self._set_keys(keys)

  def _update_state(self):
    super()._update_state()
    if not self.enabled or not self.is_visible:
      self._caps_return_at = None
      self._restore_slide_layer()
    elif (isinstance(self._123_key, CapsHintKey) and not self.is_pressed and not self._idle_hint_shown and
          self._current_keys in (self._lower_keys, self._upper_keys) and self._123_key._hint_phase is None and
          rl.get_time() - max(self._last_keyboard_activity, self._123_key._hint_finished_at or 0) >= CAPS_HINT_IDLE):
      self._123_key.start_hint(reminder=True)
      self._idle_hint_shown = True

    self._finish_caps_tap()

  def start_caps_hint(self):
    if isinstance(self._123_key, CapsHintKey):
      self._123_key.start_hint()
      self._last_keyboard_activity = rl.get_time()
      self._idle_hint_shown = False

  def show_event(self):
    super().show_event()
    self.start_caps_hint()

  def hide_event(self):
    self._caps_return_at = None
    super().hide_event()
    self._restore_slide_layer()

  def _get_closest_key(self):
    if not self.calibrated:
      return super()._get_closest_key()
    closest = (None, float('inf'))
    for row_index, row in enumerate(self._current_keys):
      for key in row:
        x, y = self.touch_center(key, row_index)
        distance = (x - self._selection_pos.x) ** 2 + (y - self._selection_pos.y) ** 2
        if distance < closest[1]:
          closest = (key, distance)
    return closest
