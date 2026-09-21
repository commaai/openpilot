"""Guided typing study using the original floating layout. Saves only after Start."""
import argparse
import math
from functools import partial
import os
import json
import random
import time
from pathlib import Path

import pyray as rl

from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.hardware import COMMA_HARDWARE
from openpilot.common.realtime import Priority, config_realtime_process, set_core_affinity
from openpilot.system.ui.lib.application import FontWeight, MousePos, gui_app
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.selfdrive.ui.mici.widgets.dialog import BigInputDialog
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.mici_keyboard import KEY_TOUCH_AREA_OFFSET
from openpilot.system.ui.widgets.mici_keyboard_calibrated import CalibratedKeyboard, KEYBOARD_VARIANTS
from openpilot.tools.ui.keyboard_study_capture import TouchCapture, encode_mouse_event
from openpilot.tools.ui.keyboard_study_data import PHRASES, URL_PHRASE, StudyWriter, summarize


HOLD_SECONDS = 5.0
ABORT_HOLD_SECONDS = 3.0
NEXT_TOUCH_WIDTH = 106
HOLD_MOVEMENT_PIXELS = 20
ASSIGNMENT_MODES = ('random', 'stock', 'calibrated')
BACKSPACE_TOUCH_WIDTH = 106
BACKSPACE_REPEAT_DELAY = 0.5
VARIANT_INFO = {
  'reference': ('1 Current', 'Caps middle-left; slide to space before ?!'),
  'caps_original': ('2 Caps original', 'Caps bottom-left; slide to space before ?!'),
  'space_right_letters_only': ('3 Space key', '123 left, space right; symbol controls left'),
  'swipe_space_caps_middle': ('4 Swipe space', 'Caps middle-left; short swipe right from 123 = space'),
  'space_right_caps_above': ('5 Caps above', 'Caps above abc; #+= to its right'),
  'space_right_symbols_above': ('6 #+= above', '#+= above abc; Caps to its right'),
}


class StudyKeyboard(CalibratedKeyboard):
  def __init__(self, record, expected, calibrated=True, variant='reference'):
    super().__init__(calibrated=calibrated, variant=variant)
    self.record = record
    self.expected = expected
    self._gesture = None

  def initialize_layout(self):
    # Stock initializes these on its first draw; the study also needs the same
    # geometry for session metadata before showing the first phrase.
    if not self._initialized:
      bg_x = self.rect.x + (self.rect.width - self._txt_bg.width) / 2
      bg_y = self.rect.y + self.rect.height - self._txt_bg.height
      for rows in (self._lower_keys, self._upper_keys, self._special_keys, self._super_special_keys):
        self._lay_out_keys(bg_x, bg_y, rows)
      self._initialized = True

  @property
  def _hit_rect(self):
    rect = super()._hit_rect
    top = self.rect.y + self.rect.height - self.get_keyboard_height()
    return rl.get_collision_rec(rect, rl.Rectangle(self.rect.x, top, self.rect.width, self.get_keyboard_height()))

  def geometry(self):
    result = []
    for row_index, row in enumerate(self._current_keys):
      for key in row:
        result.append({'char': key.char, 'row': row_index,
                       'center': [key.original_position.x, self.rect.y + key.original_position.y - KEY_TOUCH_AREA_OFFSET],
                       'touch_center': list(self.touch_center(key, row_index)),
                       'displayed_center': [key.rect.x + key.rect.width / 2, key.rect.y + key.rect.height / 2]})
    return result

  def _handle_mouse_event(self, event):
    if event.left_pressed and rl.check_collision_point_rec(event.pos, self._hit_rect):
      before = self.text()
      prompt = self.expected()
      expected = prompt[len(before)] if prompt.startswith(before) and len(before) < len(prompt) else None
      geometry = self.geometry()
      target = next((key for key in geometry if key['char'] == expected), None)
      self._gesture = {'type': 'gesture', 'slot': event.slot, 'before': before, 'expected': expected, 'target': target,
                       'geometry': geometry, 'samples': [], 'press_time': event.t}
    if self._gesture is not None:
      phase = 'press' if event.left_pressed else 'release' if event.left_released else 'move'
      self._gesture['samples'].append([event.t, event.pos.x, event.pos.y, phase])
    previous_keys = self._current_keys
    super()._handle_mouse_event(event)
    if self._gesture is not None and previous_keys is not self._current_keys:
      self._gesture.setdefault('layer_transitions', []).append({'time': event.t, 'geometry': self.geometry()})
      if self._slide_origin is not None:
        # The press was aimed at a page control, not the eventual character's position.
        self._gesture['kind'] = 'layer_switch' if event.left_pressed else 'layer_slide'
        self._gesture.setdefault('press_target', self._gesture['target'])
        self._gesture['target'] = None
    if self._gesture is not None and self._slide_origin is not None and event.left_down and not event.left_pressed:
      if self._closest_key[0] not in self._layer_targets:
        self._gesture['kind'] = 'layer_slide'
    if event.left_released and self._gesture is not None and not rl.check_collision_point_rec(event.pos, self._hit_rect):
      self._finish_gesture('', cancelled=True)

  def _finish_gesture(self, committed, cancelled=False):
    if self._gesture is not None:
      self._gesture.update(committed=committed, after=self.text(), cancelled=cancelled)
      self.record(self._gesture)
      self._gesture = None

  def _handle_mouse_release(self, mouse_pos):
    if self._gesture is not None and self.is_space_flick():
      self._gesture.update(kind='layer_slide', shortcut='space_flick', target=None)
    if self._gesture is not None and self._slide_origin is not None:
      geometry = self.geometry()
      self._gesture['selection_geometry'] = geometry
      self._gesture['selection_target'] = next((key for key in geometry if key['char'] == self._gesture['expected']), None)
    before = self.text()
    super()._handle_mouse_release(mouse_pos)
    committed = self.text()[len(before):] if self.text().startswith(before) else ''
    self._finish_gesture(committed)


class KeyboardStudy(Widget):
  def __init__(self, output_dir: Path, synthetic=False, phrase_count=6, condition=None):
    super().__init__()
    self._settings_path = output_dir / 'settings.json'
    self._settings_error = None
    self._scores_path = output_dir / ('synthetic-high-scores.json' if synthetic else 'high-scores.json')
    self._score = None
    self._best_score = None
    self._new_high_score = False
    saved_mode = 'calibrated'
    self._variant = 'reference'
    try:
      settings = json.loads(self._settings_path.read_text())
      if isinstance(settings, dict) and settings.get('assignment_mode') in ASSIGNMENT_MODES:
        saved_mode = settings['assignment_mode']
      if isinstance(settings, dict):
        saved_variant = settings.get('variant')
        if saved_variant in ('space_key', 'space_key_caps_top', 'space_key_caps_middle', 'space_key_small_caps', 'space_key_small_caps_slots',
                             'space_key_letters_only', 'space_key_main_only', 'space_left_letters_only'):
          saved_variant = 'space_right_letters_only'
        elif saved_variant == 'caps_top':
          saved_variant = 'reference'
        elif saved_variant == 'swipe_space':
          saved_variant = 'swipe_space_caps_middle'
        if saved_variant in KEYBOARD_VARIANTS:
          self._variant = saved_variant
    except (OSError, ValueError):
      pass
    self._assignment_mode = saved_mode if condition is None else condition
    if self._assignment_mode not in ASSIGNMENT_MODES:
      raise ValueError(f'Unknown keyboard condition: {self._assignment_mode}')
    self.condition = 'calibrated' if self._assignment_mode == 'random' else self._assignment_mode
    self.writer = StudyWriter(output_dir, synthetic)
    if phrase_count < 1:
      raise ValueError('Phrase count must be positive')
    self.phrases = tuple(PHRASES[index % len(PHRASES)] for index in range(phrase_count))
    self.capture = TouchCapture(self.writer.write) if COMMA_HARDWARE and not synthetic else None
    self._frame = 0
    self._keyboard = self._child(StudyKeyboard(self._record, lambda: self.phrases[self._trial],
                                              calibrated=self.condition == 'calibrated', variant=self._variant))
    self._font = gui_app.font(FontWeight.NORMAL)
    self._state = 'intro'
    self._finger = None
    self._trial = 0
    self._records = []
    self._buttons = []
    self._pressed_action = None
    self._backspace_img = gui_app.texture('icons_mici/settings/keyboard/backspace.png', 42, 36)
    self._backspace_alpha = FirstOrderFilter(0, 0.05, 1 / gui_app.target_fps)
    self._backspace_repeat_at = None
    self._backspace_pos = MousePos(0, 0)
    self._event_time = 0
    self._hold_started = None
    self._hold_origin = MousePos(0, 0)
    self._hold_kind = None

  def _record(self, record):
    if self._state == 'typing':
      record = dict(record, trial=self._trial, recorded_monotonic=time.monotonic())
      self._records.append(record)
      self.writer.write(record)

  def start(self):
    if self._finger not in ('index', 'thumb'):
      return
    self._stop_capture()
    condition = random.choice(('stock', 'calibrated')) if self._assignment_mode == 'random' else self._assignment_mode
    if condition != self.condition or self._keyboard.variant != self._variant:
      self._keyboard.hide_event()
      self._children.remove(self._keyboard)
      self.condition = condition
      self._keyboard = self._child(StudyKeyboard(self._record, lambda: self.phrases[self._trial], calibrated=condition == 'calibrated', variant=self._variant))
      self._update_layout_rects()
    self._records = []
    self._backspace_repeat_at = None
    self._trial = 0
    self._keyboard.set_text('')
    self._keyboard._set_uppercase(False)
    try:
      configuration = dict(self._keyboard.configuration(), study_controls='large_edges_abort_v2', abort_hold_seconds=ABORT_HOLD_SECONDS,
                           assignment_mode=self._assignment_mode, assignment_unit='session', url_auto_return_to_letters='.',
                           random_probability=0.5 if self._assignment_mode == 'random' else None,
                           backspace_control={'version': 'big_input_dialog_v1', 'activation': 'press',
                                              'hit_rect': self._backspace_geometry, 'repeat_delay': BACKSPACE_REPEAT_DELAY,
                                              'repeat_hz': BigInputDialog.BACKSPACE_RATE})
      self.writer.start(self._finger, self._keyboard.geometry(), self.phrases, configuration)
      if self.capture is not None:
        self.capture.start()
      gui_app._mouse.set_event_observer(partial(self._observe_samples, self.writer.path))
    except OSError as error:
      self._stop_capture()
      self.writer.error = str(error)
      return
    self._state = 'typing'
    self._keyboard.start_caps_hint()
    self._start_trial()

  def _start_trial(self):
    prompt = self.phrases[self._trial]
    self._keyboard._auto_return_to_letters = '.' if prompt == URL_PHRASE else ''
    self._record({'type': 'trial_start', 'prompt': prompt, 'auto_return_to_letters': self._keyboard._auto_return_to_letters})

  def finish_trial(self):
    gestures = [record for record in self._records if record['type'] == 'gesture' and record['trial'] == self._trial]
    edits = [record for record in self._records if record['type'] == 'backspace' and record['trial'] == self._trial]
    last_time = max([record['samples'][-1][0] for record in gestures] + [record['time'] for record in edits], default=0)
    duration = max(0, last_time - gestures[0]['press_time']) if gestures else 0
    self._record({'type': 'trial_end', 'prompt': self.phrases[self._trial], 'text': self._keyboard.text(), 'duration': duration})
    if self._trial == len(self.phrases) - 1:
      self._stop_capture()
      summary = summarize(self._records)
      self._finish_score(summary)
      self.writer.write({'type': 'session_end', 'summary': summary, 'score_wpm': self._score,
                         'previous_best_wpm': self._best_score, 'new_high_score': self._new_high_score})
      self._state = 'done'
    else:
      self._trial += 1
      self._keyboard.set_text('')
      self._keyboard._set_uppercase(False)
      self._keyboard._closest_key = (None, float('inf'))
      self._keyboard._unselect_key_t = None
      self._start_trial()

  def _finish_score(self, summary):
    self._score = round(summary['output_wpm'], 1) if summary['output_wpm'] is not None else None
    key = json.dumps([self._keyboard.configuration()['condition'], self.phrases])
    try:
      scores = json.loads(self._scores_path.read_text())
      if not isinstance(scores, dict):
        scores = {}
    except (OSError, ValueError):
      scores = {}
    self._best_score = scores.get(key)
    self._new_high_score = (self._score is not None and summary['trials'] == len(self.phrases) and summary['final_edit_errors'] == 0 and
                            (self._best_score is None or self._score > self._best_score))
    if self._new_high_score:
      scores[key] = self._score
      try:
        temporary = self._scores_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(scores) + '\n')
        temporary.replace(self._scores_path)
      except OSError:
        self._new_high_score = False

  def abort_session(self):
    if self._state != 'typing':
      return
    self._record({'type': 'session_abort', 'reason': 'held_next', 'prompt': self.phrases[self._trial],
                  'text': self._keyboard.text(), 'summary': summarize(self._records)})
    self._stop_capture()
    self._keyboard.hide_event()
    self._state = 'intro'
    self._finger = None

  def _set_assignment_mode(self, mode):
    if mode not in ASSIGNMENT_MODES:
      raise ValueError(f'Unknown assignment mode: {mode}')
    self._assignment_mode = mode
    self._save_settings()

  def _set_variant(self, variant):
    if variant not in KEYBOARD_VARIANTS:
      raise ValueError(f'Unknown keyboard variant: {variant}')
    self._variant = variant
    self._save_settings()

  def _save_settings(self):
    try:
      self._settings_path.parent.mkdir(parents=True, exist_ok=True)
      temporary = self._settings_path.with_suffix('.tmp')
      temporary.write_text(json.dumps({'assignment_mode': self._assignment_mode, 'variant': self._variant}) + '\n')
      temporary.replace(self._settings_path)
      self._settings_error = None
    except OSError as error:
      self._settings_error = str(error)

  def _update_hold(self, now):
    if self._hold_started is None or now - self._hold_started < (ABORT_HOLD_SECONDS if self._hold_kind == 'abort' else HOLD_SECONDS):
      return
    kind = self._hold_kind
    self._hold_started = self._hold_kind = None
    self._pressed_action = None
    if kind == 'settings' and self._state in ('intro', 'ready'):
      self._state = 'settings'
    elif kind == 'abort' and self._state == 'typing':
      self.abort_session()

  @property
  def _next_rect(self):
    return rl.Rectangle(self.rect.x, self.rect.y, NEXT_TOUCH_WIDTH, self.rect.height - self._keyboard.get_keyboard_height())

  @property
  def _start_rect(self):
    return rl.Rectangle(self.rect.x, self.rect.y + self.rect.height - 82, self.rect.width, 82)

  @property
  def _backspace_rect(self):
    return rl.Rectangle(self.rect.x + self.rect.width - BACKSPACE_TOUCH_WIDTH, self.rect.y,
                        BACKSPACE_TOUCH_WIDTH, self.rect.height - self._keyboard.get_keyboard_height())

  @property
  def _backspace_geometry(self):
    rect = self._backspace_rect
    return [rect.x, rect.y, rect.width, rect.height]

  def _backspace(self, timestamp, trigger):
    before = self._keyboard.text()
    self._keyboard.backspace()
    self._record({'type': 'backspace', 'before': before, 'after': self._keyboard.text(), 'time': timestamp,
                  'trigger': trigger, 'position': list(self._backspace_pos), 'hit_rect': self._backspace_geometry})

  def _update_backspace(self, now):
    if self._state != 'typing' or not self.enabled or not self.is_visible:
      self._backspace_repeat_at = None
    if self._backspace_repeat_at is not None and now >= self._backspace_repeat_at:
      self._backspace(now, 'repeat')
      self._backspace_repeat_at += 1 / BigInputDialog.BACKSPACE_RATE
      if self._backspace_repeat_at < now:
        self._backspace_repeat_at = now + 1 / BigInputDialog.BACKSPACE_RATE

  def _update_state(self):
    super()._update_state()
    now = time.monotonic()
    self._update_hold(now)
    # A queued lift/move must cancel the hold before a repeat can fire.
    if not any(event.slot == 0 and (event.left_released or not rl.check_collision_point_rec(event.pos, self._backspace_rect))
               for event in gui_app.mouse_events):
      self._update_backspace(now)

  def _update_layout_rects(self):
    super()._update_layout_rects()
    self._keyboard.set_rect(self.rect)
    self._keyboard.initialize_layout()

  def _button(self, rect, label, action, selected=False):
    self._buttons.append((rect, action))
    rl.draw_rectangle_rec(rect, rl.Color(55, 90, 120, 255) if selected else rl.DARKGRAY)
    size = 22
    width = measure_text_cached(self._font, label, size).x
    rl.draw_text_ex(self._font, label, (rect.x + (rect.width - width) / 2, rect.y + (rect.height - size) / 2), size, 0, rl.WHITE)

  def _label(self, text, x, y, size=22, color=rl.WHITE):
    rl.draw_text_ex(self._font, text, (self.rect.x + x, self.rect.y + y), size, 0, color)

  def _render(self, rect):
    super()._render(rect)
    self._buttons = []
    if self.writer.error or (self.capture is not None and self.capture.error):
      self._label('Could not save. Please stop this session.', 12, 70)
      return
    if self._state == 'intro':
      self._label('Keyboard typing study', 16, 12, 30)
      self._label('How will you type?', 16, 57, 25)
      self._button(rl.Rectangle(rect.x + 16, rect.y + 104, 245, 64), 'one index finger', 'index')
      self._button(rl.Rectangle(rect.x + 275, rect.y + 104, 245, 64), 'two thumbs', 'thumb')
      self._button(rl.Rectangle(rect.x, rect.y + 180, rect.width, 60), 'Layouts: ' + VARIANT_INFO[self._variant][0], 'layouts')
    elif self._state == 'ready':
      self._label('Two thumbs' if self._finger == 'thumb' else 'One index finger', 16, 12, 30)
      description = VARIANT_INFO[self._variant][1] if self._assignment_mode == 'calibrated' else \
                    'Original master keyboard' if self._assignment_mode == 'stock' else 'One keyboard is chosen when you start'
      self._label(description, 16, 52, 19)
      self._label(f'Copy {len(self.phrases)} examples; taps are saved locally.', 16, 78, 18)
      self._button(rl.Rectangle(rect.x + 16, rect.y + 106, 245, 40), 'change fingers', 'change_fingers')
      self._button(rl.Rectangle(rect.x + 275, rect.y + 106, 245, 40), 'layouts', 'layouts')
      self._button(self._start_rect, 'start', 'start')
    elif self._state == 'layouts':
      self._label('Choose a layout', 16, 8, 30)
      for index, variant in enumerate(KEYBOARD_VARIANTS):
        self._button(rl.Rectangle(rect.x + 8 + (index % 2) * 268, rect.y + 48 + (index // 2) * 42, 252, 38),
                     VARIANT_INFO[variant][0], 'variant_' + variant, self._variant == variant)
      self._button(rl.Rectangle(rect.x, rect.y + 176, rect.width, 64), 'done', 'close_layouts')
    elif self._state == 'settings':
      self._label('Study settings', 16, 12, 30)
      for index, (mode, label) in enumerate((('random', 'Random'), ('stock', 'Stock'), ('calibrated', 'Latest'))):
        self._button(rl.Rectangle(rect.x + 16 + index * 172, rect.y + 62, 160, 44), label,
                     'mode_' + mode, self._assignment_mode == mode)
      description = 'Equal chance of either keyboard per person.' if self._assignment_mode == 'random' else 'Every person gets the selected keyboard.'
      self._label(description, 16, 120, 20)
      self._label('Could not save setting.' if self._settings_error else 'Saved for the next person and after restart.', 16, 151, 18, rl.LIGHTGRAY)
      self._button(rl.Rectangle(rect.x + 16, rect.y + 194, 504, 40), 'done', 'close_settings')
    elif self._state == 'done':
      summary = summarize(self._records)
      self._label('NEW HIGH SCORE!' if self._new_high_score else 'Finished - thank you', 16, 12, 30,
                  rl.YELLOW if self._new_high_score else rl.WHITE)
      score = f'{self._score:.1f}' if self._score is not None else '--'
      best = f'{self._best_score:.1f}' if self._best_score is not None else '--'
      self._label(f'Your score: {score} WPM', 16, 56, 28)
      self._label(f'Previous best: {best} WPM', 16, 94, 25)
      self._label(f"Backspaces: {summary['backspaces']}   Time: {summary['duration_seconds']:.1f}s", 16, 133, 20)
      detail = 'Fix all errors to qualify.' if summary['final_edit_errors'] else 'Saved. Thanks for testing!'
      self._label(detail, 16, 161, 18, rl.LIGHTGRAY)
      self._button(rl.Rectangle(rect.x + 16, rect.y + 192, 504, 42), 'next person', 'new')
    else:
      before = self._keyboard.text()
      # Capture the entire frame batch before Widget drops non-primary slots.
      events = [encode_mouse_event(event) for event in gui_app.mouse_events]
      rl.draw_rectangle_rec(rl.Rectangle(rect.x, rect.y, rect.width, 70), rl.BLACK)
      prompt, entered = self.phrases[self._trial], self._keyboard.text()
      text_x, text_width = rect.x + NEXT_TOUCH_WIDTH + 6, rect.width - NEXT_TOUCH_WIDTH - BACKSPACE_TOUCH_WIDTH - 12
      progress = min(len(entered), len(prompt))
      preceding_width = measure_text_cached(self._font, prompt[:progress], 22).x
      scroll = min(0, text_width - preceding_width - 60)
      rl.begin_scissor_mode(int(text_x), int(rect.y), int(text_width), 70)
      rl.draw_text_ex(self._font, prompt, (text_x + scroll, rect.y + 4), 22, 0, rl.LIGHTGRAY)
      width = measure_text_cached(self._font, entered, 24).x
      rl.draw_text_ex(self._font, entered or 'type here', (text_x + min(0, text_width - width - 6), rect.y + 36),
                      24, 0, rl.WHITE if prompt.startswith(entered) else rl.ORANGE)
      rl.end_scissor_mode()
      self._button(self._next_rect, 'next', 'next', entered == prompt)
      self._label(f'{self._trial + 1}/{len(self.phrases)}  hold: reset', 5, 52, 12, rl.LIGHTGRAY)
      self._buttons.append((self._backspace_rect, 'backspace'))
      self._backspace_alpha.update(255 * bool(entered))
      if self._backspace_alpha.x > 1:
        color = rl.Color(255, 255, 255, int(self._backspace_alpha.x))
        rl.draw_texture_ex(self._backspace_img, (rect.x + rect.width - self._backspace_img.width - 27, rect.y + 14), 0, 1, color)
      # Match the stock dialog: enlarged key previews draw over the header.
      self._keyboard.render(rect)
      self.writer.write({'type': 'frame', 'frame': self._frame, 'time': time.monotonic(), 'trial': self._trial,
                         'events': events, 'before': before, 'after': self._keyboard.text(),
                         'last_mouse_event': encode_mouse_event(gui_app.last_mouse_event),
                         'geometry': self._keyboard.geometry()})
      self._frame += 1
    if self._hold_started is not None:
      progress = min(1, max(0, (time.monotonic() - self._hold_started) / (ABORT_HOLD_SECONDS if self._hold_kind == 'abort' else HOLD_SECONDS)))
      if progress > 0.06:
        bar = rl.Rectangle(rect.x, rect.y + 67, NEXT_TOUCH_WIDTH * progress, 3) if self._hold_kind == 'abort' else \
              rl.Rectangle(rect.x + 16, rect.y + rect.height - 4, 504 * progress, 3)
        rl.draw_rectangle_rec(bar, rl.LIGHTGRAY)
        if self._hold_kind == 'abort':
          remaining = max(1, math.ceil(ABORT_HOLD_SECONDS * (1 - progress)))
          rl.draw_rectangle_rec(self._next_rect, rl.DARKGRAY)
          self._label(f'reset {remaining}', 12, 20, 21)
          rl.draw_rectangle_rec(bar, rl.LIGHTGRAY)

  def _handle_mouse_event(self, event):
    super()._handle_mouse_event(event)
    self._event_time = event.t
    if event.left_pressed:
      self._backspace_repeat_at = None
      if self._state == 'typing' and self._pressed_action == 'backspace':
        self._backspace_pos = event.pos
        self._backspace(event.t, 'press')
        self._backspace_repeat_at = event.t + BACKSPACE_REPEAT_DELAY
      self._hold_kind = 'settings' if self._state in ('intro', 'ready') else 'abort' if self._state == 'typing' and self._pressed_action == 'next' else None
      self._hold_started = event.t if self._hold_kind is not None else None
      self._hold_origin = event.pos
    elif event.left_down and self._hold_started is not None:
      moved = (event.pos.x - self._hold_origin.x) ** 2 + (event.pos.y - self._hold_origin.y) ** 2 > HOLD_MOVEMENT_PIXELS ** 2
      next_rect = next((rect for rect, action in self._buttons if action == 'next'), None)
      left_next = self._hold_kind == 'abort' and (next_rect is None or not rl.check_collision_point_rec(event.pos, next_rect))
      if (moved and self._hold_kind != 'abort') or left_next or not rl.check_collision_point_rec(event.pos, self._hit_rect):
        self._hold_started = self._hold_kind = None
    elif event.left_released:
      self._update_hold(event.t)
      self._hold_started = self._hold_kind = None
    if event.left_released or not rl.check_collision_point_rec(event.pos, self._backspace_rect):
      self._backspace_repeat_at = None
    elif event.left_down:
      self._backspace_pos = event.pos

  def _handle_mouse_press(self, mouse_pos: MousePos):
    super()._handle_mouse_press(mouse_pos)
    self._pressed_action = next((action for rect, action in self._buttons if rl.check_collision_point_rec(mouse_pos, rect)), None)

  def _handle_mouse_release(self, mouse_pos: MousePos):
    super()._handle_mouse_release(mouse_pos)
    action = next((action for rect, action in self._buttons if rl.check_collision_point_rec(mouse_pos, rect)), None)
    pressed_action, self._pressed_action = self._pressed_action, None
    if action is None or action != pressed_action:
      return
    if action in ('index', 'thumb'):
      self._finger = action
      self._state = 'ready'
    elif action == 'start':
      self.start()
    elif action == 'next' and self._keyboard.text():
      self.finish_trial()
    elif action in ('new', 'change_fingers'):
      self._state = 'intro'
      self._finger = None
    elif action == 'layouts':
      self._state = 'layouts'
    elif action.startswith('variant_'):
      self._assignment_mode = 'calibrated'
      self._set_variant(action.removeprefix('variant_'))
    elif action == 'close_layouts':
      self._state = 'ready' if self._finger is not None else 'intro'
    elif action.startswith('mode_'):
      self._set_assignment_mode(action.removeprefix('mode_'))
    elif action == 'close_settings':
      self._state = 'ready' if self._finger is not None else 'intro'

  def _observe_samples(self, path, samples, dropped):
    self.writer.write({'type': 'python_samples', 'trial_at_poll': self._trial,
                       'samples': [encode_mouse_event(event) for event in samples], 'ui_queue_dropped': dropped}, path=path)

  def _stop_capture(self):
    gui_app._mouse.set_event_observer(None)
    if self.capture is not None:
      self.capture.stop()

  def hide_event(self):
    super().hide_event()
    self._stop_capture()
    self._hold_started = self._hold_kind = None
    self._backspace_repeat_at = None
    self.writer.close()


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output-dir', type=Path, default=Path('/data/keyboard-study') if COMMA_HARDWARE else Path.home() / 'tmp/keyboard-study')
  parser.add_argument('--phrases', type=int, default=6)
  parser.add_argument('--condition', choices=ASSIGNMENT_MODES, help='Override the saved assignment mode')
  args = parser.parse_args()
  config_realtime_process(0, Priority.CTRL_HIGH)
  gui_app.init_window('Keyboard typing study')
  study = KeyboardStudy(args.output_dir, phrase_count=args.phrases, condition=args.condition)
  gui_app.push_widget(study)
  try:
    for _ in gui_app.render():
      if COMMA_HARDWARE and os.sched_getaffinity(0) != {5}:
        set_core_affinity([5])
  finally:
    study.hide_event()
    gui_app.close()
