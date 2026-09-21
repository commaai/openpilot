import unittest
from unittest.mock import patch

import pyray as rl

from openpilot.system.ui.lib.application import MouseEvent, MousePos, gui_app
from openpilot.selfdrive.ui.mici.widgets.dialog import BigInputDialog
from openpilot.system.ui.widgets.mici_keyboard import CapsState, KEY_MIN_ANIMATION_TIME, MiciKeyboard, NumberKey


class TestMiciKeyboard(unittest.TestCase):
  def setUp(self):
    self.enterContext(patch.object(gui_app, 'font', return_value=rl.Font()))
    self.enterContext(patch.object(gui_app, 'texture', side_effect=lambda name, width, height, **kwargs: rl.Texture(0, width, height, 1, 0)))
    self.clock = self.enterContext(patch('pyray.get_time', return_value=1.0))
    self.keyboard = MiciKeyboard()
    self.keyboard.set_rect(rl.Rectangle(0, 0, 536, 240))
    for keys in (self.keyboard._lower_keys, self.keyboard._upper_keys, self.keyboard._special_keys, self.keyboard._super_special_keys):
      self.keyboard._lay_out_keys(8, 70, keys)
    self.keyboard._initialized = True

  def events(self, char, layer=None, slot=0):
    keys = self.keyboard._current_keys if layer is None else layer
    row, key = next((row, key) for row, keys in enumerate(keys) for key in keys if key.char == char)
    point = MousePos(*self.keyboard.touch_center(key, row))
    return [MouseEvent(point, slot, True, False, True, self.clock.return_value),
            MouseEvent(point, slot, False, True, False, self.clock.return_value + 0.01)]

  def dispatch(self, events):
    with patch.object(gui_app, '_mouse_events', events), patch.object(gui_app, '_last_mouse_event', events[-1]):
      self.keyboard._process_mouse_events()

  def test_final_layout_and_symbol_pages(self):
    self.assertEqual([''.join(key.char for key in row) for row in self.keyboard._lower_keys],
                     ['qwertyuiop', 'asdfghjkl', '123zxcvbnm '])
    self.assertIsInstance(self.keyboard._123_key, NumberKey)
    for keys, switch in ((self.keyboard._special_keys, self.keyboard._super_special_key),
                         (self.keyboard._super_special_keys, self.keyboard._123_key2)):
      self.assertEqual(keys[2][:3], [self.keyboard._abc_key, self.keyboard._caps_key, switch])
      self.assertNotIn(self.keyboard._space_key, [key for row in keys for key in row])
      self.assertEqual(''.join(key.char for key in keys[2][3:]), ".,?!'")
    self.assertEqual(''.join(key.char for key in self.keyboard._super_special_keys[0]), '[]{}#%^*+=')
    self.assertEqual(''.join(key.char for key in self.keyboard._super_special_keys[1]), '_\\|~<>€£¥•')

  def test_fast_taps_and_spaces_use_each_events_position(self):
    self.dispatch([event for char in 'hello  comma' for event in self.events(char)])
    self.assertEqual(self.keyboard.text(), 'hello  comma')
    self.assertGreaterEqual(self.keyboard._unselect_key_t, 1.0 + KEY_MIN_ANIMATION_TIME)

  def test_secondary_contact_does_not_type(self):
    self.dispatch(self.events('q', slot=1) + self.events('w'))
    self.assertEqual(self.keyboard.text(), 'w')

  def test_release_coordinate_does_not_retarget(self):
    self.dispatch([self.events('a')[0], self.events('z')[1]])
    self.assertEqual(self.keyboard.text(), 'a')

  def test_layer_opens_on_down_and_drag_returns(self):
    self.dispatch([self.events('123')[0]])
    self.assertIs(self.keyboard._current_keys, self.keyboard._special_keys)
    move, release = self.events('7')
    self.dispatch([move._replace(left_pressed=False), release])
    self.assertEqual(self.keyboard.text(), '7')
    self.assertIs(self.keyboard._current_keys, self.keyboard._lower_keys)

  def test_every_page_control_switches_once_on_release(self):
    keyboard = self.keyboard
    origins = [(keyboard._lower_keys, '123'), (keyboard._special_keys, 'abc'),
               (keyboard._special_keys, '#+='), (keyboard._super_special_keys, 'abc'), (keyboard._super_special_keys, '123')]
    for origin, control in origins:
      for finish_on_start in (False, True):
        with self.subTest(control=control, return_to_start=finish_on_start):
          keyboard._set_keys(origin)
          press = self.events(control)[0]
          self.dispatch([press])
          opened = keyboard._current_keys
          return_control = keyboard._slide_return_control
          destination = next((key for row in opened for key in row if key in keyboard._layer_targets and key is not return_control), return_control)
          destination_event = self.events(destination.char)[0]._replace(left_pressed=False)
          for _ in range(3):
            self.dispatch([destination_event, press._replace(left_pressed=False)])
            self.assertIs(keyboard._current_keys, opened)
          ending = press if finish_on_start else destination_event
          self.dispatch([ending._replace(left_pressed=False), ending._replace(left_pressed=False, left_down=False, left_released=True)])
          expected = opened if finish_on_start or destination is return_control else keyboard._layer_targets[destination]
          self.assertIs(keyboard._current_keys, expected)

  def test_drag_from_abc_types_letter_and_returns_to_symbols(self):
    self.keyboard._set_keys(self.keyboard._super_special_keys)
    self.dispatch([self.events('abc')[0]])
    move, release = self.events('a')
    self.dispatch([move._replace(left_pressed=False), release])
    self.assertEqual(self.keyboard.text(), 'a')
    self.assertIs(self.keyboard._current_keys, self.keyboard._super_special_keys)

  def test_backspace_does_not_wait_for_icon_fade(self):
    dialog = BigInputDialog('test', default_text='a')
    dialog._top_right_button_rect = rl.Rectangle(430, 0, 106, 70)
    dialog._backspace_img_alpha.x = 0
    dialog._handle_mouse_press(MousePos(480, 30))
    self.assertEqual(dialog._keyboard.text(), '')

  def test_parent_vertical_offset_is_included_in_touch_centers(self):
    key = self.keyboard._lower_keys[0][0]
    before = self.keyboard.touch_center(key, 0)
    self.keyboard.set_rect(rl.Rectangle(0, 25, 536, 240))
    after = self.keyboard.touch_center(key, 0)
    self.assertEqual(after, (before[0], before[1] + 25))

  def test_caps_quick_double_and_slow_taps_have_separate_windows(self):
    self.dispatch(self.events('123'))
    self.dispatch(self.events(''))
    self.assertEqual(self.keyboard._caps_state, CapsState.UPPER)
    self.clock.return_value = 1.2
    self.dispatch(self.events(''))
    self.assertEqual(self.keyboard._caps_state, CapsState.LOCK)
    self.clock.return_value = 1.6
    self.keyboard._update_state()
    self.assertIs(self.keyboard._current_keys, self.keyboard._special_keys)
    self.clock.return_value = 1.91
    self.keyboard._update_state()
    self.assertIs(self.keyboard._current_keys, self.keyboard._upper_keys)
    self.dispatch(self.events('A') + self.events('B'))
    self.assertEqual(self.keyboard.text(), 'AB')
    self.assertEqual(self.keyboard._caps_state, CapsState.LOCK)

  def test_caps_taps_500ms_apart_toggle_off(self):
    self.dispatch(self.events('123'))
    self.dispatch(self.events(''))
    self.clock.return_value = 1.5
    self.dispatch(self.events(''))
    self.assertEqual(self.keyboard._caps_state, CapsState.LOWER)
    self.assertAlmostEqual(self.keyboard._caps_return_at, 2.2)
    self.clock.return_value = 2.0
    self.keyboard._update_state()
    self.assertIs(self.keyboard._current_keys, self.keyboard._special_keys)
    self.clock.return_value = 2.21
    self.keyboard._update_state()
    self.assertIs(self.keyboard._current_keys, self.keyboard._lower_keys)

  def test_repeated_caps_presses_postpone_return(self):
    self.dispatch(self.events('123'))
    for index in range(8):
      self.clock.return_value = 1.0 + index * 0.5
      self.dispatch(self.events(''))
      self.clock.return_value += 0.4
      self.keyboard._update_state()
      self.assertIs(self.keyboard._current_keys, self.keyboard._special_keys)
    self.clock.return_value += 0.31
    self.keyboard._update_state()
    self.assertIs(self.keyboard._current_keys, self.keyboard._lower_keys)

  def test_caps_timer_waits_for_a_held_press(self):
    self.dispatch(self.events('123'))
    self.dispatch(self.events(''))
    self.clock.return_value = 1.2
    press, release = self.events('')
    self.dispatch([press])
    self.clock.return_value = 3.0
    self.keyboard._update_state()
    self.assertIs(self.keyboard._current_keys, self.keyboard._special_keys)
    self.dispatch([release])
    self.assertAlmostEqual(self.keyboard._caps_return_at, 3.7)

  def test_drag_caps_returns_immediately_and_shift_resets_after_letter(self):
    self.dispatch([self.events('123')[0]])
    move, release = self.events('')
    self.dispatch([move._replace(left_pressed=False), release])
    self.assertIs(self.keyboard._current_keys, self.keyboard._upper_keys)
    self.dispatch(self.events('A'))
    self.assertEqual(self.keyboard.text(), 'A')
    self.assertIs(self.keyboard._current_keys, self.keyboard._lower_keys)

  def test_drag_from_symbol_switch_to_caps_keeps_symbol_page(self):
    self.keyboard._set_keys(self.keyboard._special_keys)
    self.dispatch([self.events('#+=')[0]])
    move, release = self.events('')
    self.dispatch([move._replace(left_pressed=False), release])
    self.assertIs(self.keyboard._current_keys, self.keyboard._super_special_keys)
    self.assertEqual(self.keyboard._caps_state, CapsState.UPPER)

  def test_url_period_returns_but_slash_and_generic_period_stay(self):
    for auto_return, char, return_to_letters in (('.', '.', True), ('.', '/', False), ('', '.', False)):
      self.keyboard._auto_return_to_letters = auto_return
      self.keyboard._set_keys(self.keyboard._lower_keys)
      self.dispatch(self.events('123'))
      self.dispatch(self.events(char))
      self.assertIs(self.keyboard._current_keys, self.keyboard._lower_keys if return_to_letters else self.keyboard._special_keys)

  def test_aborted_slide_restores_page_and_case(self):
    self.keyboard._set_caps_state(CapsState.LOCK)
    self.keyboard._show_letters()
    self.dispatch([self.events('123')[0]])
    outside = MousePos(-10, -10)
    self.dispatch([MouseEvent(outside, 0, False, True, False, 1.2)])
    self.assertIs(self.keyboard._current_keys, self.keyboard._upper_keys)
    self.assertEqual(self.keyboard._caps_state, CapsState.LOCK)
    self.assertEqual(self.keyboard.text(), '')

  def test_hide_and_disable_cancel_pending_layer_and_caps(self):
    self.dispatch([self.events('123')[0]])
    self.keyboard.hide_event()
    self.assertIs(self.keyboard._current_keys, self.keyboard._lower_keys)
    self.assertIsNone(self.keyboard._slide_origin)
    self.dispatch(self.events('123'))
    self.dispatch(self.events(''))
    self.keyboard.set_enabled(False)
    self.keyboard._update_state()
    self.assertIsNone(self.keyboard._caps_return_at)

  def test_caps_does_not_move_symbol_positions(self):
    self.dispatch(self.events('123'))
    before = {key: key.get_position() for row in self.keyboard._special_keys for key in row}
    self.dispatch(self.events(''))
    self.assertEqual(before, {key: key.get_position() for row in self.keyboard._special_keys for key in row})


if __name__ == '__main__':
  unittest.main()
