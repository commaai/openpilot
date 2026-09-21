import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pyray as rl

from openpilot.system.ui.lib.application import MouseEvent, MousePos, gui_app
from openpilot.system.ui.widgets.mici_keyboard import CapsState, KEY_MIN_ANIMATION_TIME, KEY_TOUCH_AREA_OFFSET
from openpilot.tools.ui.mici_keyboard_study import KeyboardStudy
from openpilot.system.ui.widgets.mici_keyboard_calibrated import CAPS_HINT_HOLD, CAPS_HINT_IDLE, KEYBOARD_VARIANTS, SPACE_KEY_VARIANTS
from openpilot.tools.ui.keyboard_study_report import make_report


class TestCalibratedKeyboard(unittest.TestCase):
  def setUp(self):
    self.enterContext(patch.object(gui_app, 'font', return_value=rl.Font()))
    self.enterContext(patch.object(gui_app, 'texture', side_effect=lambda name, width, height, **kwargs: rl.Texture(0, width, height, 1, 0)))
    self.enterContext(patch('pyray.get_time', return_value=1.0))
    self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
    self.study = KeyboardStudy(self.directory, synthetic=True)
    self.addCleanup(self.study.hide_event)
    self.study._finger = 'index'
    self.study.set_rect(rl.Rectangle(0, 0, 536, 240))
    self.keyboard = self.study._keyboard

  def events(self, char, slot=0, layer=None):
    layer = self.keyboard._current_keys if layer is None else layer
    row_index, key = next((index, key) for index, row in enumerate(layer) for key in row if key.char == char)
    position = MousePos(*self.keyboard.touch_center(key, row_index))
    return [MouseEvent(position, slot, True, False, True, 1.0), MouseEvent(position, slot, False, True, False, 1.01)]

  def dispatch(self, events):
    with patch.object(gui_app, '_mouse_events', events), patch.object(gui_app, '_last_mouse_event', events[-1]):
      self.keyboard._process_mouse_events()

  def test_fast_batch_uses_each_taps_position_and_preserves_repeated_letters(self):
    self.study.start()
    events = []
    for char in 'hello comma':
      if char == ' ':
        press = self.events('123')[0]
        move, release = self.events(char, layer=self.keyboard._special_keys)
        events.extend([press, move._replace(left_pressed=False), release])
      else:
        events.extend(self.events(char))
    self.dispatch(events)
    self.assertEqual(self.keyboard.text(), 'hello comma')
    self.assertEqual(len([record for record in self.study._records if record['type'] == 'gesture']), 11)

  def test_space_is_only_before_question_and_exclamation_on_symbols(self):
    for layer in (self.keyboard._lower_keys, self.keyboard._upper_keys):
      self.assertNotIn(self.keyboard._space_key, [key for row in layer for key in row])
      self.assertEqual(len(layer[2]), 8)
      self.assertIs(layer[2][0], self.keyboard._123_key)
      self.assertIs(layer[1][0], self.keyboard._caps_key)
      self.assertEqual(len(layer[1]), 10)
      self.assertLess(self.keyboard._caps_key.original_position.y, self.keyboard._123_key.original_position.y)
    for layer in (self.keyboard._special_keys, self.keyboard._super_special_keys):
      self.assertEqual([key.char for key in layer[2][3:6]], [' ', '?', '!'])
      self.assertIs(layer[2][0], self.keyboard._abc_key)
      self.assertEqual(len(layer[1]), 10)
      self.assertEqual(len(layer[2]), 8)
      self.keyboard._set_keys(layer)
      self.dispatch(self.events(' '))
      self.assertIs(self.keyboard._current_keys, layer)
    self.assertEqual(self.keyboard.text(), '  ')

  def test_iphone_symbol_order_and_shared_punctuation(self):
    first, second = self.keyboard._special_keys, self.keyboard._super_special_keys
    self.assertEqual(''.join(key.char for key in first[0]), '1234567890')
    self.assertEqual(''.join(key.char for key in first[1]), '-/:;()$&@"')
    self.assertEqual(''.join(key.char for key in second[0]), '[]{}#%^*+=')
    self.assertEqual(''.join(key.char for key in second[1]), '_\\|~<>€£¥•')
    self.assertNotIn('_', [key.char for row in first for key in row])
    for layer in (first, second):
      self.assertEqual(''.join(key.char for key in layer[2][1:-1]), "., ?!'")

  def test_drag_to_extra_symbols_switches_only_on_release(self):
    for variant in KEYBOARD_VARIANTS:
      for release_on_symbols in (False, True):
        self.select_variant(variant)
        press = self.events('123')[0]
        symbols = self.events('#+=', layer=self.keyboard._special_keys)[0]._replace(left_pressed=False)
        back = press._replace(left_pressed=False)
        self.dispatch([press])
        for _ in range(4):
          self.dispatch([symbols])
          self.assertIs(self.keyboard._current_keys, self.keyboard._special_keys)
          self.dispatch([back])
          self.assertIs(self.keyboard._current_keys, self.keyboard._special_keys)
        destination = symbols if release_on_symbols else back
        self.dispatch([destination, destination._replace(left_down=False, left_released=True)])
        expected = self.keyboard._super_special_keys if release_on_symbols else self.keyboard._special_keys
        self.assertIs(self.keyboard._current_keys, expected)
        self.assertEqual(self.keyboard.text(), '')
        if release_on_symbols:
          self.dispatch(self.events('_'))
          self.assertEqual(self.keyboard.text(), '_')

  def test_all_layer_control_drags_wait_for_release_and_can_return_to_start(self):
    for variant in KEYBOARD_VARIANTS:
      self.select_variant(variant)
      keyboard = self.keyboard
      for origin in (keyboard._lower_keys, keyboard._special_keys, keyboard._super_special_keys):
        starts = [key.char for row in origin for key in row if key in keyboard._layer_targets]
        for start in starts:
          keyboard._set_uppercase(False)
          keyboard._set_keys(origin)
          self.dispatch(self.events(start))
          destination_page = keyboard._current_keys
          destinations = [key.char for row in destination_page for key in row if key in keyboard._layer_targets]
          for destination in destinations:
            with self.subTest(variant=variant, start=start, destination=destination):
              keyboard._set_uppercase(False)
              keyboard._set_keys(origin)
              press = self.events(start)[0]
              self.dispatch([press])
              opened_page = keyboard._current_keys
              return_control = keyboard._slide_return_control
              destination_key = next(key for row in opened_page for key in row if key.char == destination)
              move = self.events(destination)[0]._replace(left_pressed=False)
              back = press._replace(left_pressed=False)
              for _ in range(3):
                self.dispatch([move])
                self.assertIs(keyboard._current_keys, opened_page)
                self.dispatch([back])
                self.assertIs(keyboard._current_keys, opened_page)
              self.dispatch([move, move._replace(left_down=False, left_released=True)])
              expected = opened_page if destination_key is return_control else keyboard._layer_targets[destination_key]
              self.assertIs(keyboard._current_keys, expected)
              self.assertEqual(keyboard.text(), '')

  def test_old_preview_timer_cannot_clear_the_next_held_key(self):
    self.dispatch(self.events('h'))
    self.assertIsNotNone(self.keyboard._unselect_key_t)
    press, release = self.events('e')
    self.dispatch([press])
    with patch('pyray.get_time', return_value=2.0):
      self.keyboard._update_state()
    self.dispatch([release])
    self.assertEqual(self.keyboard.text(), 'he')

  def test_last_down_selects_but_lift_does_not_retarget(self):
    press, _ = self.events('h')
    move = self.events('j')[0]._replace(left_pressed=False)
    release = self.events('k')[1]
    self.dispatch([press, move, release])
    self.assertEqual(self.keyboard.text(), 'j')

  def test_secondary_contact_does_not_type(self):
    self.dispatch(self.events('q', slot=1))
    self.assertEqual(self.keyboard.text(), '')

  def test_outside_release_cancels_and_next_press_works(self):
    press, release = self.events('h')
    self.dispatch([press, release._replace(pos=MousePos(-1, -1))])
    self.assertEqual(self.keyboard.text(), '')
    self.dispatch(self.events('e'))
    self.assertEqual(self.keyboard.text(), 'e')

  def test_all_layer_glyph_anchors_still_select_their_own_key(self):
    for layer in (self.keyboard._lower_keys, self.keyboard._upper_keys, self.keyboard._special_keys, self.keyboard._super_special_keys):
      self.keyboard._set_keys(layer)
      for row in layer:
        for key in row:
          self.keyboard._selection_pos = MousePos(key.original_position.x, key.original_position.y - KEY_TOUCH_AREA_OFFSET)
          self.assertIs(self.keyboard._get_closest_key()[0], key, key.char)

  def test_layer_tap_stays_on_page_but_drag_selects_once_and_returns(self):
    self.dispatch(self.events('123'))
    self.assertIs(self.keyboard._current_keys, self.keyboard._special_keys)
    self.dispatch(self.events('abc'))
    self.assertIs(self.keyboard._current_keys, self.keyboard._lower_keys)
    press = self.events('123')[0]
    self.dispatch([press])
    self.assertIs(self.keyboard._current_keys, self.keyboard._special_keys)
    self.dispatch([press._replace(left_pressed=False, pos=MousePos(press.pos.x - 1, press.pos.y))])
    self.assertIs(self.keyboard._current_keys, self.keyboard._special_keys)
    move, release = self.events('7', layer=self.keyboard._special_keys)
    self.dispatch([move._replace(left_pressed=False)])
    self.assertIs(self.keyboard._current_keys, self.keyboard._special_keys)
    self.assertEqual(self.keyboard.text(), '')
    self.dispatch([release])
    self.assertEqual(self.keyboard.text(), '7')
    self.assertIs(self.keyboard._current_keys, self.keyboard._lower_keys)

  def test_layer_slide_can_be_batched_and_does_not_contaminate_tap_labels(self):
    self.study.start()
    self.study._trial = 4
    self.keyboard.set_text('driver')
    press = self.events('123')[0]
    move, release = self.events('7', layer=self.keyboard._special_keys)
    self.dispatch([press, move._replace(left_pressed=False), release])
    self.assertEqual(self.keyboard.text(), 'driver7')
    self.assertIs(self.keyboard._current_keys, self.keyboard._lower_keys)
    gesture = next(record for record in self.study._records if record['type'] == 'gesture')
    self.assertEqual(gesture['kind'], 'layer_slide')
    self.assertEqual(gesture['committed'], '7')
    self.assertIsNone(gesture['target'])
    self.assertIsNone(gesture['selection_target'])
    self.assertEqual(gesture['layer_transitions'][-1]['geometry'], gesture['selection_geometry'])

  def test_all_page_buttons_switch_on_press_and_release_does_not_reverse(self):
    self.study.start()
    for char, target in (('123', self.keyboard._special_keys), ('#+=', self.keyboard._super_special_keys),
                         ('123', self.keyboard._special_keys), ('abc', self.keyboard._lower_keys)):
      press, release = self.events(char)
      self.dispatch([press])
      self.assertIs(self.keyboard._current_keys, target)
      self.dispatch([release])
      self.assertIs(self.keyboard._current_keys, target)
      self.assertEqual(self.keyboard.text(), '')
    gestures = [record for record in self.study._records if record['type'] == 'gesture']
    self.assertTrue(all(record['kind'] == 'layer_switch' and record['target'] is None for record in gestures))

  def test_quick_page_taps_retain_minimum_animation_without_switching_back(self):
    for char, target in (('123', self.keyboard._special_keys), ('#+=', self.keyboard._super_special_keys),
                         ('123', self.keyboard._special_keys), ('abc', self.keyboard._lower_keys)):
      self.dispatch(self.events(char))
      selected = self.keyboard._closest_key[0]
      self.assertIn(selected, target[2])
      self.assertIs(self.keyboard._current_keys, target)
      with patch('pyray.get_time', return_value=1.0 + KEY_MIN_ANIMATION_TIME / 2):
        self.keyboard._update_state()
      self.assertIs(self.keyboard._closest_key[0], selected)
      with patch('pyray.get_time', return_value=1.0 + KEY_MIN_ANIMATION_TIME + 0.001):
        self.keyboard._update_state()
      self.assertIsNone(self.keyboard._closest_key[0])
      self.assertIs(self.keyboard._current_keys, target)

  def test_abc_slide_returns_to_the_actual_origin_page(self):
    self.keyboard._set_keys(self.keyboard._super_special_keys)
    press = self.events('abc')[0]
    move, release = self.events('a', layer=self.keyboard._lower_keys)
    self.dispatch([press, move._replace(left_pressed=False), release])
    self.assertEqual(self.keyboard.text(), 'a')
    self.assertIs(self.keyboard._current_keys, self.keyboard._super_special_keys)

  def test_layer_slide_preserves_pending_capital_and_caps_lock(self):
    for caps_cycles in (1, 2):
      self.keyboard._set_uppercase(False)
      for _ in range(caps_cycles):
        self.keyboard._set_uppercase(True)
      original_caps = self.keyboard._caps_state
      press = self.events('123')[0]
      move, release = self.events('2', layer=self.keyboard._special_keys)
      self.dispatch([press, move._replace(left_pressed=False), release])
      self.assertIs(self.keyboard._current_keys, self.keyboard._upper_keys)
      self.assertEqual(self.keyboard._caps_state, original_caps)

  def test_cancelled_release_restores_original_page(self):
    for position in (MousePos(-1, -1), MousePos(600, 217)):
      press = self.events('123')[0]
      move, release = self.events('1', layer=self.keyboard._special_keys)
      self.dispatch([press, move._replace(left_pressed=False)])
      self.dispatch([move._replace(pos=position, left_pressed=False), release._replace(pos=position)])
      self.assertEqual(self.keyboard.text(), '')
      self.assertIs(self.keyboard._current_keys, self.keyboard._lower_keys)

  def test_disabling_mid_slide_restores_page(self):
    press = self.events('123')[0]
    move = self.events('1', layer=self.keyboard._special_keys)[0]._replace(left_pressed=False)
    self.dispatch([press, move])
    self.keyboard.set_enabled(False)
    self.keyboard._update_state()
    self.assertIs(self.keyboard._current_keys, self.keyboard._lower_keys)

  def select_variant(self, variant):
    self.study._set_variant(variant)
    self.study.start()
    self.keyboard = self.study._keyboard

  def test_all_variants_keep_letters_symbols_and_visible_centers_selectable(self):
    for variant in KEYBOARD_VARIANTS:
      self.select_variant(variant)
      for layer in (self.keyboard._lower_keys, self.keyboard._upper_keys):
        self.assertEqual(''.join(key.char for key in layer[0]).lower(), 'qwertyuiop')
        self.assertEqual(len(layer[0]), 10)
        self.assertEqual(''.join(key.char for key in layer[2] if len(key.char) == 1 and key.char.isalpha()).lower(), 'zxcvbnm')
        self.assertLessEqual(len(layer[2]) - len('zxcvbnm'), 2, variant)
      for layer in (self.keyboard._lower_keys, self.keyboard._upper_keys, self.keyboard._special_keys, self.keyboard._super_special_keys):
        self.keyboard._set_keys(layer)
        for row in layer:
          for key in row:
            self.keyboard._selection_pos = MousePos(key.original_position.x, key.original_position.y - KEY_TOUCH_AREA_OFFSET)
            self.assertIs(self.keyboard._get_closest_key()[0], key, (variant, key.char))
      self.keyboard._set_uppercase(False)
      self.dispatch([event for char in 'hello' for event in self.events(char)])
      self.assertEqual(self.keyboard.text(), 'hello', variant)
      press = self.events('123')[0]
      move, release = self.events('7', layer=self.keyboard._special_keys)
      self.dispatch([press, move._replace(left_pressed=False), release])
      self.assertEqual(self.keyboard.text(), 'hello7', variant)
      self.assertIs(self.keyboard._current_keys, self.keyboard._lower_keys)

  def test_returning_layer_drag_to_start_keeps_symbols_open(self):
    for variant in KEYBOARD_VARIANTS:
      self.select_variant(variant)
      for uppercase in (False, True):
        self.keyboard._set_uppercase(False)
        if uppercase:
          self.keyboard._set_uppercase(True)
        press = self.events('123')[0]
        move = self.events('7', layer=self.keyboard._special_keys)[0]._replace(left_pressed=False)
        back = press._replace(left_pressed=False)
        self.dispatch([press, move, back])
        self.assertIs(self.keyboard._current_keys, self.keyboard._special_keys)
        self.dispatch([back._replace(left_down=False, left_released=True)])
        self.assertIs(self.keyboard._current_keys, self.keyboard._special_keys)
        self.dispatch(self.events('abc'))
        expected = self.keyboard._upper_keys if uppercase and variant in SPACE_KEY_VARIANTS else self.keyboard._lower_keys
        self.assertIs(self.keyboard._current_keys, expected)

  def test_returning_abc_drag_to_start_keeps_letters_open(self):
    for variant in KEYBOARD_VARIANTS:
      self.select_variant(variant)
      for second_page in (False, True):
        for caps_cycles in (0, 1, 2):
          with self.subTest(variant=variant, second_page=second_page, caps_cycles=caps_cycles):
            self.keyboard._set_uppercase(False)
            for _ in range(caps_cycles):
              self.keyboard._set_uppercase(True)
            origin = self.keyboard._super_special_keys if second_page else self.keyboard._special_keys
            self.keyboard._set_keys(origin)
            letters = self.keyboard._upper_keys if caps_cycles and variant in SPACE_KEY_VARIANTS else self.keyboard._lower_keys
            press = self.events('abc')[0]
            move = self.events('H' if letters is self.keyboard._upper_keys else 'h', layer=letters)[0]._replace(left_pressed=False)
            back = press._replace(left_pressed=False)
            self.dispatch([press])
            self.assertIs(self.keyboard._current_keys, letters)
            for _ in range(3):
              self.dispatch([move, back])
              self.assertIs(self.keyboard._current_keys, letters)
            self.dispatch([back._replace(left_down=False, left_released=True)])
            self.assertIs(self.keyboard._current_keys, letters)
            self.assertEqual(self.keyboard.text(), '')
            self.dispatch(self.events('123'))
            self.assertIs(self.keyboard._current_keys, self.keyboard._special_keys)

  def test_url_delimiter_returns_to_letters_only_for_url_trial(self):
    self.select_variant('space_right_letters_only')
    for _ in range(3):
      self.study.finish_trial()
    self.dispatch(self.events('123'))
    self.dispatch(self.events('/'))
    self.assertIs(self.keyboard._current_keys, self.keyboard._special_keys)
    self.dispatch(self.events('.'))
    self.assertIs(self.keyboard._current_keys, self.keyboard._lower_keys)
    self.assertEqual(self.keyboard.text(), '/.')
    self.study.finish_trial()
    self.dispatch(self.events('123'))
    self.dispatch(self.events('.'))
    self.assertIs(self.keyboard._current_keys, self.keyboard._special_keys)

  def test_space_key_only_on_letter_pages_allows_repeated_spaces(self):
    self.select_variant('space_right_letters_only')
    self.assertEqual(''.join(key.char for key in self.keyboard._lower_keys[1]), 'asdfghjkl')
    self.assertIs(self.keyboard._lower_keys[2][0], self.keyboard._123_key)
    for layer in (self.keyboard._lower_keys, self.keyboard._upper_keys):
      self.keyboard._set_keys(layer)
      self.keyboard.set_text('')
      self.assertIs(layer[2][-1], self.keyboard._space_key)
      self.dispatch([event for _ in range(3) for event in self.events(' ')])
      self.assertEqual(self.keyboard.text(), '   ')

    for layer in (self.keyboard._special_keys, self.keyboard._super_special_keys):
      self.assertNotIn(self.keyboard._space_key, [key for row in layer for key in row])

  def test_letters_only_layout_keeps_caps_on_symbols_and_can_slide_to_uppercase(self):
    self.select_variant('space_right_letters_only')
    for layer in (self.keyboard._lower_keys, self.keyboard._upper_keys):
      self.assertNotIn(self.keyboard._caps_key, [key for row in layer for key in row])
      self.assertEqual(''.join(key.char for key in layer[1]).lower(), 'asdfghjkl')
      self.assertEqual([key.char for key in layer[2]], ['123'] + list('zxcvbnm' if layer is self.keyboard._lower_keys else 'ZXCVBNM') + [' '])
    for symbols in (self.keyboard._special_keys, self.keyboard._super_special_keys):
      self.assertIs(symbols[2][0], self.keyboard._abc_key)
      self.assertIs(symbols[2][1], self.keyboard._caps_key)
      self.assertIn(symbols[2][2], (self.keyboard._super_special_key, self.keyboard._123_key2))
    press = self.events('123')[0]
    move, release = self.events('', layer=self.keyboard._special_keys)
    self.dispatch([press, move._replace(left_pressed=False), release])
    self.assertIs(self.keyboard._current_keys, self.keyboard._upper_keys)
    self.assertEqual(self.keyboard._caps_state, CapsState.UPPER)
    self.assertIs(self.keyboard._current_keys, self.keyboard._upper_keys)
    self.dispatch(self.events('A'))
    self.assertIs(self.keyboard._current_keys, self.keyboard._lower_keys)
    self.dispatch([event for char in 'asdfghjklzxcvbnm   ' for event in self.events(char)])
    self.assertEqual(self.keyboard.text(), 'Aasdfghjklzxcvbnm   ')
    self.dispatch(self.events('123'))
    self.dispatch(self.events('#+='))
    self.dispatch(self.events(''))
    self.assertIs(self.keyboard._current_keys, self.keyboard._super_special_keys)
    self.dispatch(self.events('abc'))
    self.assertIs(self.keyboard._current_keys, self.keyboard._upper_keys)
    self.dispatch(self.events('B'))
    self.assertTrue(self.keyboard.text().endswith('B'))
    self.dispatch(self.events('123'))
    self.dispatch(self.events('') + self.events(''))
    self.assertIs(self.keyboard._current_keys, self.keyboard._special_keys)
    self.assertEqual(self.keyboard._caps_state, CapsState.LOCK)
    self.dispatch(self.events('abc'))
    self.dispatch(self.events('C'))
    self.dispatch(self.events('C'))
    self.assertIs(self.keyboard._current_keys, self.keyboard._upper_keys)
    self.dispatch(self.events('123'))
    self.dispatch(self.events(''))
    self.assertIs(self.keyboard._current_keys, self.keyboard._special_keys)
    self.dispatch(self.events('abc'))
    self.assertIs(self.keyboard._current_keys, self.keyboard._lower_keys)

  def test_caps_tap_returns_after_window_and_double_tap_locks(self):
    for double_tap in (False, True):
      self.select_variant('space_right_caps_above')
      self.dispatch(self.events('123'))
      self.dispatch(self.events(''))
      self.assertEqual(self.keyboard._caps_state, CapsState.UPPER)
      with patch('pyray.get_time', return_value=1.2):
        self.keyboard._finish_caps_tap()
        self.assertIs(self.keyboard._current_keys, self.keyboard._special_keys)
        if double_tap:
          self.dispatch(self.events(''))
          self.assertEqual(self.keyboard._caps_state, CapsState.LOCK)
      with patch('pyray.get_time', return_value=1.91):
        self.keyboard._finish_caps_tap()
      self.assertIs(self.keyboard._current_keys, self.keyboard._upper_keys)
      self.dispatch(self.events('A'))
      self.assertEqual(self.keyboard._caps_state, CapsState.LOCK if double_tap else CapsState.LOWER)

  def test_repeated_caps_presses_restart_the_full_timeout(self):
    self.select_variant('space_right_caps_above')
    self.dispatch(self.events('123'))
    with patch('openpilot.system.ui.widgets.mici_keyboard_calibrated.CAPS_RETURN_DELAY', 0.7):
      for tap_index in range(8):
        press_time = 1.0 + tap_index * 0.6
        press, release = self.events('')
        with patch('pyray.get_time', return_value=press_time):
          self.dispatch([press])
        with patch('pyray.get_time', return_value=press_time + 0.1):
          self.dispatch([release])
          self.assertAlmostEqual(self.keyboard._caps_return_at, press_time + 0.8)
        with patch('pyray.get_time', return_value=press_time + 0.59):
          self.keyboard._finish_caps_tap()
          self.assertIs(self.keyboard._current_keys, self.keyboard._special_keys)
      with patch('pyray.get_time', return_value=6.01):
        self.keyboard._finish_caps_tap()
      self.assertIs(self.keyboard._current_keys, self.keyboard._lower_keys)

  def test_caps_taps_500ms_apart_toggle_off_and_extend_return(self):
    self.select_variant('space_right_caps_above')
    self.dispatch(self.events('123'))
    self.dispatch(self.events(''))
    with patch('pyray.get_time', return_value=1.5):
      self.dispatch(self.events(''))
      self.assertEqual(self.keyboard._caps_state, CapsState.LOWER)
      self.assertAlmostEqual(self.keyboard._caps_return_at, 2.2)
    with patch('pyray.get_time', return_value=2.0):
      self.keyboard._finish_caps_tap()
      self.assertIs(self.keyboard._current_keys, self.keyboard._special_keys)
    with patch('pyray.get_time', return_value=2.21):
      self.keyboard._finish_caps_tap()
      self.assertIs(self.keyboard._current_keys, self.keyboard._lower_keys)

  def test_slow_caps_tap_toggles_shift_off_instead_of_locking(self):
    self.select_variant('space_right_caps_above')
    self.dispatch(self.events('123'))
    self.dispatch(self.events(''))
    with patch('pyray.get_time', return_value=1.8):
      self.keyboard._finish_caps_tap()
      self.dispatch(self.events('123'))
      self.dispatch(self.events(''))
      self.assertEqual(self.keyboard._caps_state, CapsState.LOWER)
    with patch('pyray.get_time', return_value=2.6):
      self.keyboard._finish_caps_tap()
    self.assertIs(self.keyboard._current_keys, self.keyboard._lower_keys)

  def test_symbol_input_cancels_pending_caps_page_return(self):
    self.select_variant('space_right_caps_above')
    self.dispatch(self.events('123'))
    self.dispatch(self.events(''))
    self.dispatch(self.events('!'))
    with patch('pyray.get_time', return_value=2.0):
      self.keyboard._finish_caps_tap()
    self.assertIsNone(self.keyboard._caps_return_at)

  def test_opening_caps_hint_bounces_and_does_not_delay_layer_press(self):
    for variant in SPACE_KEY_VARIANTS:
      self.select_variant(variant)
      key = self.keyboard._123_key
      caps_scales, label_scales = [], []
      for frame in range(240):
        with patch('pyray.get_time', return_value=1.0 + frame / 60):
          key._update_state()
        if frame / 60 < CAPS_HINT_HOLD:
          self.assertEqual(key._hint_phase, 'caps_in')
          self.assertEqual(key._hint_scale.x, 1.0)
        (caps_scales if key._hint_phase in ('caps_in', 'caps_out') else label_scales).append(key._hint_scale.x)
      self.assertEqual(max(caps_scales), 1.0)
      self.assertGreater(max(label_scales), 1.0)
      self.assertIsNone(key._hint_phase)
      self.keyboard.start_caps_hint()
      self.dispatch(self.events('123'))
      self.assertIs(self.keyboard._current_keys, self.keyboard._special_keys)
      self.assertIsNone(key._hint_phase)
      self.assertEqual(self.keyboard._caps_state, CapsState.LOWER)

  def test_caps_slide_returns_to_letters_but_separate_tap_stays_on_symbols(self):
    for variant in SPACE_KEY_VARIANTS:
      for second_page in (False, True):
        self.select_variant(variant)
        self.keyboard._set_uppercase(False)
        press = self.events('123')[0]
        self.dispatch([press])
        if second_page:
          self.dispatch([self.events('#+=')[0]._replace(left_pressed=False)])
          self.assertIs(self.keyboard._current_keys, self.keyboard._special_keys)
        move, release = self.events('')
        self.dispatch([move._replace(left_pressed=False), release])
        self.assertIs(self.keyboard._current_keys, self.keyboard._upper_keys)
        self.assertEqual(self.keyboard._caps_state, CapsState.UPPER)
        self.dispatch(self.events('A'))
        self.assertTrue(self.keyboard.text().endswith('A'))
        self.dispatch(self.events('123'))
        self.dispatch(self.events(''))
        self.assertIs(self.keyboard._current_keys, self.keyboard._special_keys)
        self.assertEqual(self.keyboard._caps_state, CapsState.UPPER)
        self.dispatch(self.events(''))
        self.assertIs(self.keyboard._current_keys, self.keyboard._special_keys)
        self.assertEqual(self.keyboard._caps_state, CapsState.LOCK)

  def test_idle_hint_is_once_per_pause_and_never_interrupts_symbols_or_a_touch(self):
    self.select_variant('space_right_caps_above')
    self.dispatch(self.events('h'))
    key = self.keyboard._123_key
    with patch('pyray.get_time', return_value=1.0 + CAPS_HINT_IDLE - 0.01):
      self.keyboard._update_state()
      self.assertIsNone(key._hint_phase)
    with patch('pyray.get_time', return_value=1.0 + CAPS_HINT_IDLE):
      self.keyboard._update_state()
      self.assertEqual(key._hint_phase, 'label_out')
      key._hint_phase = None
    with patch('pyray.get_time', return_value=20.0):
      self.keyboard._update_state()
      self.assertIsNone(key._hint_phase)
      self.dispatch(self.events('123'))
    with patch('pyray.get_time', return_value=30.0):
      self.keyboard._update_state()
      self.assertIsNone(key._hint_phase)
      self.dispatch(self.events('abc'))
      self.dispatch([self.events('h')[0]])
    with patch('pyray.get_time', return_value=40.0):
      self.keyboard._update_state()
      self.assertIsNone(key._hint_phase)
      self.dispatch([self.events('h')[1]])
    with patch('pyray.get_time', return_value=40.0 + CAPS_HINT_IDLE):
      self.keyboard._update_state()
      self.assertEqual(key._hint_phase, 'label_out')
      self.dispatch(self.events('e'))
      self.assertIsNone(key._hint_phase)

  def test_idle_wait_starts_after_animation_and_reminder_scales_label_out_first(self):
    self.select_variant('space_right_caps_above')
    key = self.keyboard._123_key
    finished_at = None
    for frame in range(240):
      now = 1.0 + frame / 60
      with patch('pyray.get_time', return_value=now):
        key._update_state()
        self.keyboard._update_state()
      if key._hint_phase is None:
        finished_at = now
        break
    self.assertIsNotNone(finished_at)
    self.assertEqual(key._hint_finished_at, finished_at)
    with patch('pyray.get_time', return_value=finished_at + CAPS_HINT_IDLE - 0.01):
      self.keyboard._update_state()
      self.assertIsNone(key._hint_phase)
    with patch('pyray.get_time', return_value=finished_at + CAPS_HINT_IDLE):
      self.keyboard._update_state()
      self.assertEqual(key._hint_phase, 'label_out')
      self.assertEqual(key._hint_scale.x, 1.0)
      key._update_state()
      self.assertEqual(key._hint_phase, 'label_out')
      self.assertLess(key._hint_scale.x, 1.0)
    phases = set()
    for frame in range(240):
      with patch('pyray.get_time', return_value=finished_at + CAPS_HINT_IDLE + (frame + 1) / 60):
        key._update_state()
        self.keyboard._update_state()
        phases.add(key._hint_phase)
    self.assertTrue({'caps_rise', 'caps_in', 'caps_out', 'label_in', None}.issubset(phases))

  def test_conservative_targets_fix_recorded_l_misses_in_both_cases(self):
    for variant in SPACE_KEY_VARIANTS:
      self.select_variant(variant)
      for uppercase in (False, True):
        self.keyboard._set_uppercase(False)
        if uppercase:
          self.keyboard._set_uppercase(True)
        for position in (MousePos(532, 152), MousePos(535, 159)):
          self.keyboard._selection_pos = position
          self.assertEqual(self.keyboard._get_closest_key()[0].char, 'L' if uppercase else 'l')
      configuration = self.keyboard.configuration()
      self.assertTrue(configuration['condition'].endswith('_conservative_l_v2'))
      self.assertEqual(configuration['target_adjustments'], {'l': (8.0, 0.0)})

  def test_symbol_caps_does_not_reposition_keys(self):
    for variant in SPACE_KEY_VARIANTS:
      self.select_variant(variant)
      for symbols in (self.keyboard._special_keys, self.keyboard._super_special_keys):
        self.keyboard._set_uppercase(False)
        self.keyboard._set_keys(symbols)
        # Give every symbol its settled position, including those beyond the
        # shorter letter row that previously inherited its rightmost position.
        for row in symbols:
          for key in row:
            key.set_position(key.original_position.x, key.original_position.y - KEY_TOUCH_AREA_OFFSET, smooth=False)
        before = {key: key.get_position() for row in symbols for key in row}
        for expected in (CapsState.UPPER, CapsState.LOCK, CapsState.LOWER):
          self.dispatch(self.events(''))
          self.assertEqual(self.keyboard._caps_state, expected)
          self.assertIs(self.keyboard._current_keys, symbols)
          for key, position in before.items():
            self.assertEqual(key.get_position(), position, (variant, key.char, expected))
          self.assertIs(self.keyboard._closest_key[0], self.keyboard._caps_key)
          self.assertIsNotNone(self.keyboard._unselect_key_t)

  def test_corner_modifiers_allow_release_on_symbols_and_caps_lock(self):
    for variant in SPACE_KEY_VARIANTS[1:]:
      self.select_variant(variant)
      keyboard = self.keyboard
      for symbols, switch in ((keyboard._special_keys, keyboard._super_special_key), (keyboard._super_special_keys, keyboard._123_key2)):
        above, beside = (keyboard._caps_key, switch) if variant == 'space_right_caps_above' else (switch, keyboard._caps_key)
        self.assertIs(symbols[1][0], above)
        self.assertEqual(symbols[2][:2], [keyboard._abc_key, beside])
        self.assertLess(above.original_position.y, keyboard._abc_key.original_position.y)
        self.assertLess(abs(above.original_position.x - keyboard._abc_key.original_position.x), 12)
        self.assertNotIn(keyboard._space_key, [key for row in symbols for key in row])
      press = self.events('123')[0]
      second = self.events('#+=', layer=keyboard._special_keys)[0]._replace(left_pressed=False)
      character, release = self.events('_', layer=keyboard._super_special_keys)
      self.dispatch([press, second, second._replace(left_down=False, left_released=True)])
      self.assertIs(keyboard._current_keys, keyboard._super_special_keys)
      self.dispatch([character, release])
      self.assertEqual(keyboard.text(), '_')
      self.dispatch(self.events('abc'))
      self.assertIs(keyboard._current_keys, keyboard._lower_keys)
      self.dispatch(self.events('123'))
      self.dispatch(self.events('') + self.events(''))
      self.assertIs(keyboard._current_keys, keyboard._special_keys)
      self.assertEqual(keyboard._caps_state, CapsState.LOCK)
      self.dispatch(self.events('abc'))
      self.dispatch(self.events('A'))
      self.assertEqual(keyboard.text(), '_A')
      self.assertIs(keyboard._current_keys, keyboard._upper_keys)

  def test_short_right_swipe_inserts_space_repeatedly_and_logs_shortcut(self):
    self.select_variant('swipe_space_caps_middle')
    self.keyboard.set_text('hello')
    for _ in range(3):
      press = self.events('123')[0]
      move = press._replace(left_pressed=False, pos=MousePos(press.pos.x + 60, press.pos.y))
      release = move._replace(left_down=False, left_released=True)
      self.dispatch([press, move, release])
      self.assertIs(self.keyboard._current_keys, self.keyboard._lower_keys)
    self.assertEqual(self.keyboard.text(), 'hello   ')
    gestures = [record for record in self.study._records if record['type'] == 'gesture']
    self.assertEqual(len(gestures), 3)
    self.assertTrue(all(record['shortcut'] == 'space_flick' and record['target'] is None for record in gestures))
    self.dispatch(self.events('123'))
    self.assertIs(self.keyboard._current_keys, self.keyboard._special_keys)
    self.dispatch(self.events('.'))
    self.assertEqual(self.keyboard.text(), 'hello   .')

  def test_swipe_shortcut_cannot_override_a_drag_that_visited_another_row(self):
    self.select_variant('swipe_space_caps_middle')
    press = self.events('123')[0]
    upper = self.events('1', layer=self.keyboard._special_keys)[0]._replace(left_pressed=False)
    move, release = self.events('.', layer=self.keyboard._special_keys)
    self.dispatch([press, upper, move._replace(left_pressed=False), release])
    self.assertEqual(self.keyboard.text(), '.')
    self.assertIs(self.keyboard._current_keys, self.keyboard._lower_keys)

  def test_saved_model_and_geometry_identify_the_actual_condition(self):
    self.study.start()
    self.study.writer.close()
    metadata = json.loads(self.study.writer.path.read_text().splitlines()[0])
    self.assertEqual(metadata['condition'], 'calibrated_static_v1_space_before_punctuation')
    self.assertEqual(metadata['calibration']['training_taps'], 1395)
    self.assertEqual(metadata['selection_point'], 'last_down_event')
    self.assertFalse(metadata['prediction'])
    self.assertEqual(len(metadata['calibration_sha256']), 64)
    key = next(key for key in metadata['geometry'] if key['char'] == 'q')
    self.assertAlmostEqual(key['touch_center'][0] - key['center'][0], -36.5)
    stock = KeyboardStudy(self.directory, synthetic=True, condition='stock')
    self.addCleanup(stock.hide_event)
    stock.set_rect(self.study.rect)
    stock._finger = 'index'
    stock.start()
    stock.writer.close()
    with self.assertRaisesRegex(ValueError, 'Mixed keyboard conditions'):
      make_report(self.directory, self.directory / 'report', include_synthetic=True)


if __name__ == '__main__':
  unittest.main()
