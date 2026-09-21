import json
from collections import deque
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pyray as rl

from openpilot.system.ui.lib.application import MouseEvent, MousePos, MouseState, gui_app
from openpilot.system.ui.widgets.mici_keyboard import MiciKeyboard
from openpilot.tools.ui.keyboard_study_capture import INPUT_EVENT, TouchCapture
from openpilot.tools.ui.keyboard_study_data import edit_distance, summarize
from openpilot.tools.ui.mici_keyboard_study import ABORT_HOLD_SECONDS, KeyboardStudy


class TestKeyboardStudy(unittest.TestCase):
  def setUp(self):
    self.enterContext(patch.object(gui_app, 'font', return_value=rl.Font()))
    self.enterContext(patch.object(gui_app, 'texture', side_effect=lambda name, width, height, **kwargs: rl.Texture(0, width, height, 1, 0)))
    self.enterContext(patch('pyray.get_time', return_value=1.0))
    self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
    self.study = KeyboardStudy(self.directory, synthetic=True, condition='stock')
    self.study._finger = 'index'
    self.study.set_rect(rl.Rectangle(0, 0, 536, 240))
    self.addCleanup(self.study.hide_event)
    self.time = 100.0

  def events(self, char):
    keyboard = self.study._keyboard
    key = next(key for row in keyboard._current_keys for key in row if key.char == char)
    position = MousePos(keyboard.rect.x + key.original_position.x, keyboard.rect.y + key.original_position.y)
    self.time += 0.1
    return [MouseEvent(position, 0, True, False, True, self.time), MouseEvent(position, 0, False, True, False, self.time + 0.03)]

  def dispatch(self, events):
    with (patch.object(gui_app, '_mouse_events', events), patch.object(gui_app, '_last_mouse_event', events[-1]),
          patch('pyray.get_time', return_value=self.time)):
      self.study._keyboard._update_state()
      self.study._keyboard._process_mouse_events()

  def header_events(self, events):
    self.study._buttons = [(self.study._backspace_rect, 'backspace')]
    with patch.object(gui_app, '_mouse_events', events):
      self.study._process_mouse_events()

  def backspace(self):
    position = MousePos(500, 50)
    self.header_events([MouseEvent(position, 0, True, False, True, self.time + 0.01),
                        MouseEvent(position, 0, False, True, False, self.time + 0.05)])

  def test_backspace_accepts_screen_edge_and_deletes_on_press_even_while_fading(self):
    self.study.start()
    for position in (MousePos(535, 32), MousePos(534, 18), MousePos(470, 45), MousePos(500, 0)):
      self.study._keyboard.set_text('ab')
      self.study._backspace_alpha.x = 0
      press = MouseEvent(position, 0, True, False, True, self.time)
      self.header_events([press])
      self.assertEqual(self.study._keyboard.text(), 'a')
      self.header_events([press._replace(pos=MousePos(450, 100), left_pressed=False, left_down=False, left_released=True)])
      self.assertEqual(self.study._keyboard.text(), 'a')
    self.assertEqual(len([r for r in self.study._records if r['type'] == 'backspace']), 4)

  def test_backspace_hold_repeats_and_moving_away_cancels(self):
    self.study.start()
    self.study._keyboard.set_text('abcdef')
    press = MouseEvent(MousePos(500, 30), 0, True, False, True, self.time)
    self.header_events([press])
    self.study._update_backspace(self.time + 0.49)
    self.assertEqual(self.study._keyboard.text(), 'abcde')
    self.study._update_backspace(self.time + 0.5)
    self.assertEqual(self.study._keyboard.text(), 'abcd')
    self.study._update_backspace(self.time + 0.54)
    self.assertEqual(self.study._keyboard.text(), 'abc')
    self.header_events([press._replace(pos=MousePos(300, 30), left_pressed=False)])
    self.study._update_backspace(self.time + 1)
    self.assertEqual(self.study._keyboard.text(), 'abc')
    self.header_events([press._replace(left_pressed=False, left_down=False, left_released=True)])
    self.assertEqual(self.study._keyboard.text(), 'abc')

  def test_backspace_repeat_rate_does_not_round_down_to_render_frames(self):
    self.study.start()
    self.study._keyboard.set_text('a' * 100)
    self.header_events([MouseEvent(MousePos(500, 30), 0, True, False, True, self.time)])
    for frame in range(1, 91):
      self.study._update_backspace(self.time + frame / 60)
    repeated = [record for record in self.study._records if record.get('trigger') == 'repeat']
    self.assertGreaterEqual(len(repeated), 25)
    self.assertLessEqual(len(repeated), 26)

  def test_backspace_rejects_secondary_contacts_and_dragging_into_header(self):
    self.study.start()
    self.study._keyboard.set_text('abc')
    press = MouseEvent(MousePos(500, 30), 1, True, False, True, self.time)
    self.header_events([press, press._replace(left_pressed=False, left_released=True, left_down=False)])
    self.assertEqual(self.study._keyboard.text(), 'abc')
    self.header_events([press._replace(slot=0, pos=MousePos(300, 150)),
                        press._replace(slot=0, left_pressed=False),
                        press._replace(slot=0, left_pressed=False, left_released=True, left_down=False)])
    self.assertEqual(self.study._keyboard.text(), 'abc')

  def test_no_recording_before_start_and_separate_people(self):
    self.assertEqual(list(self.directory.iterdir()), [])
    self.study.start()
    first = self.study.writer.path
    self.study._finger = 'thumb'
    self.study.start()
    second = self.study.writer.path
    self.assertNotEqual(first, second)
    self.study.writer.close()
    self.assertEqual(json.loads(first.read_text().splitlines()[0])['finger'], 'index')
    self.assertEqual(json.loads(second.read_text().splitlines()[0])['finger'], 'thumb')
    self.assertEqual(json.loads(first.read_text().splitlines()[0])['technique'], 'one_index_finger')
    self.assertEqual(json.loads(second.read_text().splitlines()[0])['technique'], 'two_thumbs')
    self.assertEqual(json.loads(second.read_text().splitlines()[0])['intended_contacts'], 2)

  def test_wrong_taps_are_labelled_but_ambiguous_followups_are_not(self):
    self.study.start()
    self.dispatch(self.events('q'))
    self.dispatch(self.events('w'))
    gestures = [record for record in self.study._records if record['type'] == 'gesture']
    self.assertEqual(gestures[0]['expected'], 'h')
    self.assertEqual(gestures[0]['committed'], 'q')
    self.assertEqual(gestures[0]['target']['char'], 'h')
    self.assertIsNone(gestures[1]['target'])
    self.backspace()
    self.backspace()
    self.dispatch(self.events('h'))
    summary = summarize(self.study._records)
    self.assertEqual(summary['labelled_taps'], 2)
    self.assertEqual(summary['labelled_mistaps'], 1)
    self.assertEqual(summary['backspaces'], 2)

  def test_full_gesture_and_lift_are_preserved(self):
    self.study.start()
    press, release = self.events('h')
    move = MouseEvent(MousePos(press.pos.x + 2, press.pos.y + 1), 0, False, False, True, press.t + 0.01)
    release = release._replace(pos=MousePos(press.pos.x + 25, press.pos.y + 15))
    self.dispatch([press, move, release])
    record = self.study._records[-1]
    self.assertEqual([sample[3] for sample in record['samples']], ['press', 'move', 'release'])
    self.assertEqual(record['samples'][0][1:3], [press.pos.x, press.pos.y])
    self.assertEqual(record['samples'][-1][1:3], [release.pos.x, release.pos.y])
    self.assertEqual(record['expected'], 'h')
    self.assertEqual(record['after'], self.study._keyboard.text())

  def test_batch_records_every_gesture_without_fixing_stock_behavior(self):
    self.study.start()
    events = [event for char in 'hello' for event in self.events(char)]
    self.dispatch(events)
    gestures = [record for record in self.study._records if record['type'] == 'gesture']
    self.assertEqual(len(gestures), 5)
    self.assertEqual(sum(len(record['samples']) for record in gestures), len(events))
    original = MiciKeyboard()
    original.set_rect(self.study.rect)
    for rows in (original._lower_keys, original._upper_keys, original._special_keys, original._super_special_keys):
      original._lay_out_keys(8, 70, rows)
    with patch.object(gui_app, '_mouse_events', events), patch.object(gui_app, '_last_mouse_event', events[-1]):
      original._process_mouse_events()
    self.assertEqual(self.study._keyboard.text(), original.text())

  def test_all_poll_samples_and_overflow_count_are_recorded(self):
    self.study.start()
    first, last = self.events('h')
    samples = [first, first._replace(slot=1), last, last._replace(slot=1)]
    self.study._observe_samples(self.study.writer.path, samples, 2)
    self.study.writer.close()
    records = [json.loads(line) for line in self.study.writer.path.read_text().splitlines()]
    recorded = next(record for record in records if record['type'] == 'python_samples')
    self.assertEqual(len(recorded['samples']), 4)
    self.assertEqual([sample[1] for sample in recorded['samples']], [0, 1, 0, 1])
    self.assertEqual(recorded['ui_queue_dropped'], 2)

  def test_poll_observer_sees_unchanged_slots_before_bounded_queue(self):
    mouse = MouseState(scale=2)
    batches = []
    mouse.set_event_observer(lambda samples, dropped: batches.append((samples, dropped)))
    with (patch('pyray.get_touch_position', return_value=rl.Vector2(40, 60)),
          patch('pyray.is_mouse_button_pressed', return_value=False),
          patch('pyray.is_mouse_button_released', return_value=False),
          patch('pyray.is_mouse_button_down', return_value=False)):
      mouse._handle_mouse_event()
      mouse._handle_mouse_event()
      self.assertEqual(len(mouse.get_events()), 2)
      self.assertEqual([len(samples) for samples, _ in batches], [2, 2])
      self.assertEqual([event.slot for event in batches[0][0]], [0, 1])
      self.assertEqual(batches[0][0][0].pos, MousePos(20, 30))
      mouse._events = deque(batches[0][0], maxlen=2)
      with patch('pyray.is_mouse_button_down', return_value=True):
        mouse._handle_mouse_event()
      self.assertEqual(batches[-1][1], 2)
      mouse.set_event_observer(None)
      mouse._handle_mouse_event()
      self.assertEqual(len(batches), 3)

  def test_cancelled_gesture_is_recorded_once(self):
    self.study.start()
    press, release = self.events('h')
    self.dispatch([press, release._replace(pos=MousePos(-10, -10))])
    gestures = [record for record in self.study._records if record['type'] == 'gesture']
    self.assertEqual(len(gestures), 1)
    self.assertTrue(gestures[0]['cancelled'])
    self.assertEqual(gestures[0]['committed'], '')

  def test_sliding_from_keyboard_or_another_control_cannot_advance(self):
    self.study.start()
    self.study._keyboard.set_text('h')
    self.study._buttons = [(rl.Rectangle(2, 1, 54, 46), 'next'), (rl.Rectangle(476, 1, 58, 66), 'backspace')]
    self.study._handle_mouse_press(MousePos(200, 150))
    self.study._handle_mouse_release(MousePos(25, 20))
    self.assertEqual(self.study._trial, 0)
    self.study._handle_mouse_press(MousePos(500, 30))
    self.study._handle_mouse_release(MousePos(25, 20))
    self.assertEqual(self.study._trial, 0)
    self.study._handle_mouse_press(MousePos(25, 20))
    self.study._handle_mouse_release(MousePos(25, 20))
    self.assertEqual(self.study._trial, 1)

  def test_native_capture_keeps_all_slots_codes_and_sync_events(self):
    records = []
    capture = TouchCapture(records.append)
    read_fd, write_fd = os.pipe2(os.O_NONBLOCK)
    self.addCleanup(os.close, read_fd)
    self.addCleanup(os.close, write_fd)
    capture._fd = read_fd
    capture._stop.set()
    events = [(100, 0, 3, 47, 0), (100, 1, 3, 57, 12), (100, 2, 3, 53, 60), (100, 3, 3, 58, 42),
              (100, 4, 3, 47, 1), (100, 5, 3, 57, 13), (100, 6, 3, 54, 200), (100, 7, 0, 0, 0),
              (100, 8, 3, 47, 9), (100, 9, 3, 57, -1), (100, 10, 0, 3, 0), (100, 11, 0, 0, 0)]
    os.write(write_fd, b''.join(INPUT_EVENT.pack(*event) for event in events))
    capture._run()
    self.assertIsNone(capture.error)
    self.assertEqual([event for record in records for event in record['events']], events)

  def test_technique_must_be_chosen_before_start(self):
    self.study._finger = None
    self.study.start()
    self.assertIsNone(self.study.writer.path)
    self.study._buttons = [(rl.Rectangle(275, 104, 245, 64), 'thumb')]
    self.study._handle_mouse_press(MousePos(380, 130))
    self.study._handle_mouse_release(MousePos(380, 130))
    self.assertEqual(self.study._state, 'ready')
    self.assertEqual(self.study._finger, 'thumb')
    self.assertIsNone(self.study.writer.path)

  def test_hold_next_aborts_without_advancing_or_deleting_input(self):
    self.study.start()
    self.dispatch(self.events('h'))
    self.study._buttons = [(rl.Rectangle(2, 1, 54, 46), 'next')]
    position = MousePos(25, 20)
    self.study._handle_mouse_press(position)
    self.study._handle_mouse_event(MouseEvent(position, 0, True, False, True, 100.0))
    self.study._update_hold(100 + ABORT_HOLD_SECONDS - 0.01)
    self.assertEqual(self.study._state, 'typing')
    self.study._update_hold(100 + ABORT_HOLD_SECONDS)
    self.study._handle_mouse_release(position)
    self.assertEqual(self.study._state, 'intro')
    self.assertEqual(self.study._trial, 0)
    self.assertIsNone(self.study._finger)
    self.study.writer.close()
    records = [json.loads(line) for line in self.study.writer.path.read_text().splitlines()]
    self.assertTrue(any(record['type'] == 'gesture' for record in records))
    abort = next(record for record in records if record['type'] == 'session_abort')
    self.assertEqual(abort['text'], 'h')
    self.assertFalse(any(record['type'] == 'session_end' for record in records))

  def test_large_next_and_start_include_edges_and_abort_tolerates_motion(self):
    self.study._state = 'ready'
    self.study._buttons = [(self.study._start_rect, 'start')]
    position = MousePos(535, 239)
    self.study._handle_mouse_press(position)
    self.study._handle_mouse_release(position)
    self.assertEqual(self.study._state, 'typing')
    self.study._keyboard.set_text('h')
    self.study._buttons = [(self.study._next_rect, 'next')]
    with patch.object(gui_app, '_mouse_events', [MouseEvent(MousePos(0, 0), 0, True, False, True, 100),
                                              MouseEvent(MousePos(90, 60), 0, False, False, True, 102)]):
      self.study._process_mouse_events()
    self.study._update_hold(100 + ABORT_HOLD_SECONDS)
    self.assertEqual(self.study._state, 'intro')
    self.assertEqual(self.study._keyboard.text(), 'h')
    self.assertEqual(self.study._trial, 0)

  def test_high_score_persists_and_requires_improvement_without_errors(self):
    for speed, errors, best, is_record in ((20.0, 0, None, True), (19.0, 0, 20.0, False),
                                         (30.0, 1, 20.0, False), (20.0, 0, 20.0, False), (21.0, 0, 20.0, True)):
      self.study._finish_score({'output_wpm': speed, 'final_edit_errors': errors, 'trials': 6})
      self.assertEqual(self.study._best_score, best)
      self.assertEqual(self.study._new_high_score, is_record)
    restarted = KeyboardStudy(self.directory, synthetic=True, condition='stock')
    self.addCleanup(restarted.hide_event)
    restarted._finish_score({'output_wpm': 20.0, 'final_edit_errors': 0, 'trials': 6})
    self.assertEqual(restarted._best_score, 21.0)

  def test_variant_persists_and_cannot_change_mid_session(self):
    self.study._set_variant('space_right_letters_only')
    self.study._set_assignment_mode('calibrated')
    self.study.start()
    self.assertEqual(self.study._keyboard.variant, 'space_right_letters_only')
    self.study._set_variant('caps_original')
    self.assertEqual(self.study._keyboard.variant, 'space_right_letters_only')
    restarted = KeyboardStudy(self.directory, synthetic=True)
    self.addCleanup(restarted.hide_event)
    self.assertEqual(restarted._variant, 'caps_original')
    self.study.writer.close()
    metadata = json.loads(self.study.writer.path.read_text().splitlines()[0])
    self.assertEqual(metadata['variant'], 'space_right_letters_only')
    self.assertEqual(metadata['abort_hold_seconds'], ABORT_HOLD_SECONDS)

  def test_moving_off_next_cancels_abort_hold(self):
    self.study.start()
    self.study._buttons = [(rl.Rectangle(2, 1, 54, 46), 'next')]
    self.study._handle_mouse_press(MousePos(25, 20))
    self.study._handle_mouse_event(MouseEvent(MousePos(25, 20), 0, True, False, True, 100.0))
    self.study._handle_mouse_event(MouseEvent(MousePos(65, 20), 0, False, False, True, 102.0))
    self.study._update_hold(106.0)
    self.assertEqual(self.study._state, 'typing')

  def test_settings_hold_consumes_release_and_mode_survives_restart(self):
    self.study._finger = None
    self.study._buttons = [(rl.Rectangle(275, 104, 245, 64), 'thumb')]
    position = MousePos(380, 130)
    self.study._handle_mouse_press(position)
    self.study._handle_mouse_event(MouseEvent(position, 0, True, False, True, 100.0))
    self.study._handle_mouse_event(MouseEvent(position, 0, False, True, False, 105.0))
    self.study._handle_mouse_release(position)
    self.assertEqual(self.study._state, 'settings')
    self.assertIsNone(self.study._finger)
    self.assertIsNone(self.study.writer.path)
    self.study._set_assignment_mode('random')
    restarted = KeyboardStudy(self.directory, synthetic=True)
    self.addCleanup(restarted.hide_event)
    self.assertEqual(restarted._assignment_mode, 'random')
    self.assertIsNone(restarted._finger)

  def test_random_assignment_is_once_per_session_and_logged(self):
    self.study._set_assignment_mode('random')
    with patch('openpilot.tools.ui.mici_keyboard_study.random.choice', side_effect=['stock', 'calibrated']) as choose:
      self.study.start()
      first = self.study.writer.path
      self.assertFalse(self.study._keyboard.calibrated)
      self.study.finish_trial()
      self.assertEqual(choose.call_count, 1)
      self.study.abort_session()
      self.study._finger = 'thumb'
      self.study.start()
      second = self.study.writer.path
      self.assertTrue(self.study._keyboard.calibrated)
      self.assertEqual(choose.call_count, 2)
    self.study.writer.close()
    first_metadata = json.loads(first.read_text().splitlines()[0])
    second_metadata = json.loads(second.read_text().splitlines()[0])
    self.assertEqual(first_metadata['condition'], 'stock_master')
    self.assertEqual(second_metadata['condition'], 'calibrated_static_v1_space_before_punctuation')
    self.assertEqual(second_metadata['assignment_mode'], 'random')
    self.assertEqual(second_metadata['assignment_unit'], 'session')
    self.assertEqual(second_metadata['random_probability'], 0.5)
    self.assertEqual(second_metadata['technique'], 'two_thumbs')

  def test_summary_roundtrip_and_edit_distance(self):
    self.assertEqual(edit_distance('hello', 'helo'), 1)
    self.assertEqual(edit_distance('hello', 'jello'), 1)
    self.study.start()
    self.dispatch(self.events('h'))
    self.study.finish_trial()
    self.study.writer.close()
    records = [json.loads(line) for line in self.study.writer.path.read_text().splitlines()]
    summary = summarize(records)
    self.assertEqual(summary['trials'], 1)
    self.assertEqual(summary['final_edit_errors'], len('hello comma') - 1)
    self.assertAlmostEqual(summary['duration_seconds'], 0.03)
    self.assertEqual(records[0]['condition'], 'stock_master')


if __name__ == '__main__':
  unittest.main()
