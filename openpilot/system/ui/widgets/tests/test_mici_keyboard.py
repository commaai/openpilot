import unittest
from unittest.mock import patch

import pyray as rl

from openpilot.system.ui.lib.application import MouseEvent, MousePos, gui_app
from openpilot.selfdrive.ui.mici.widgets.dialog import BigInputDialog
from openpilot.system.ui.widgets.mici_keyboard import CapsState, MiciKeyboard


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
    key = next(key for row in keys for key in row if key.char == char)
    point = MousePos(key.original_position.x, key.original_position.y + self.keyboard.rect.y)
    return [MouseEvent(point, slot, True, False, True, self.clock.return_value),
            MouseEvent(point, slot, False, True, False, self.clock.return_value + 0.01)]

  def dispatch(self, events):
    for event in events:
      with patch.object(gui_app, '_mouse_events', [event]), patch.object(gui_app, '_last_mouse_event', event):
        self.keyboard._process_mouse_events()
      if event.left_released:
        # Let the stock minimum-press animation finish before the next tap.
        self.clock.return_value += 0.08
        self.keyboard._update_state()

  def test_letter_layout_unchanged_and_iphone_symbol_pages(self):
    keyboard = self.keyboard
    self.assertEqual([''.join(key.char for key in row) for row in keyboard._lower_keys],
                     ['qwertyuiop', 'asdfghjkl ', 'zxcvbnm123'])
    self.assertIs(keyboard._lower_keys[2][0], keyboard._caps_key)
    self.assertEqual([''.join(key.char for key in row) for row in keyboard._special_keys],
                     ['1234567890', '-/:;()$&@"', "#+=.,?!'abc"])
    self.assertEqual([''.join(key.char for key in row) for row in keyboard._super_special_keys],
                     ['[]{}#%^*+=', '_\\|~<>€£¥•', "123.,?!'abc"])

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
          keyboard._closest_key = (None, float('inf'))
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

  def test_tapping_page_control_stays_on_opened_page(self):
    self.dispatch(self.events('123'))
    self.assertIs(self.keyboard._current_keys, self.keyboard._special_keys)
    self.dispatch(self.events('7'))
    self.assertEqual(self.keyboard.text(), '7')
    self.assertIs(self.keyboard._current_keys, self.keyboard._special_keys)

  def test_caps_double_tap_locks_and_slow_tap_toggles_off(self):
    for delay, expected in ((0.2, CapsState.LOCK), (0.5, CapsState.LOWER)):
      with self.subTest(delay=delay):
        self.keyboard._set_caps_state(CapsState.LOWER)
        self.keyboard._show_letters()
        self.clock.return_value = 1.0
        self.dispatch(self.events(''))
        self.assertEqual(self.keyboard._caps_state, CapsState.UPPER)
        self.assertIs(self.keyboard._current_keys, self.keyboard._upper_keys)
        self.clock.return_value += delay
        self.dispatch(self.events(''))
        self.assertEqual(self.keyboard._caps_state, expected)

  def test_letter_between_caps_taps_does_not_lock(self):
    self.dispatch(self.events(''))
    self.dispatch(self.events('A'))
    self.dispatch(self.events(''))
    self.assertEqual(self.keyboard._caps_state, CapsState.UPPER)

  def test_caps_reset_does_not_move_symbols_through_letter_layout(self):
    self.dispatch(self.events(''))
    self.dispatch(self.events('123'))
    before = {key: key.get_position() for row in self.keyboard._special_keys for key in row}
    self.dispatch(self.events('&'))
    self.assertEqual(self.keyboard._caps_state, CapsState.LOWER)
    self.assertIs(self.keyboard._current_keys, self.keyboard._special_keys)
    self.assertEqual(before, {key: key.get_position() for row in self.keyboard._special_keys for key in row})

  def test_backspace_does_not_wait_for_icon_fade(self):
    dialog = BigInputDialog('test', default_text='a')
    dialog._top_right_button_rect = rl.Rectangle(430, 0, 106, 70)
    dialog._backspace_img_alpha.x = 0
    dialog._handle_mouse_press(MousePos(480, 30))
    self.assertEqual(dialog._keyboard.text(), '')

  def test_backspace_hold_starts_before_icon_fades_in(self):
    dialog = BigInputDialog('test', default_text='abc')
    dialog._top_right_button_rect = rl.Rectangle(430, 0, 106, 70)
    dialog._backspace_img_alpha.x = 0
    held = MouseEvent(MousePos(480, 30), 0, False, False, True, 1.0)
    with patch.object(gui_app, '_last_mouse_event', held), patch.object(gui_app, '_frame', 0):
      dialog._update_state()
      self.assertEqual(dialog._backspace_held_time, 1.0)
      self.clock.return_value = 1.6
      dialog._update_state()
    self.assertEqual(dialog._keyboard.text(), 'ab')

  def test_existing_url_delimiters_still_return_to_letters(self):
    self.keyboard._auto_return_to_letters = './'
    for char in './':
      self.dispatch(self.events('123'))
      self.dispatch(self.events(char))
      self.assertIs(self.keyboard._current_keys, self.keyboard._lower_keys)


if __name__ == '__main__':
  unittest.main()
