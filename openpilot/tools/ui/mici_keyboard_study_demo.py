"""Reversible combined Caps label demo; run the normal study with a visual override."""
import json
from pathlib import Path
import runpy

import pyray as rl

from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.widgets import mici_keyboard_calibrated
from openpilot.system.ui.widgets.mici_keyboard_calibrated import CalibratedKeyboard, CapsHintKey, SPACE_KEY_VARIANTS


def draw_combined_label(self, rect):
  font_size = self._get_font_size()
  text_size = measure_text_cached(self._font, self.char, font_size)
  icon_height = font_size * 0.75
  icon_scale = icon_height / self._hint_icon.height
  icon_width = self._hint_icon.width * icon_scale
  gap = font_size / 8
  group_width = text_size.x + gap + icon_width
  left = rect.x + (rect.width - group_width) / 2
  center_y = rect.y + rect.height / 2
  rl.draw_text_ex(self._font, self.char, (left, center_y - text_size.y / 2), font_size, 0, self._color)
  rl.draw_texture_ex(self._hint_icon, (left + text_size.x + gap, center_y - icon_height / 2), 0, icon_scale, self._color)


def disable_animated_hint(self, reminder=False):
  self._hint_phase = None


CapsHintKey._render = draw_combined_label
CapsHintKey.start_hint = disable_animated_hint

original_configuration = CalibratedKeyboard.configuration


def configuration_with_caps_label(self):
  configuration = original_configuration(self)
  if self.calibrated and self.variant in SPACE_KEY_VARIANTS:
    configuration['condition'] += '_permanent_caps_label_v1'
    configuration['caps_hint'] = {'display': 'permanent_combined_123_caps', 'activation': '123', 'idle_reminder': False}
  return configuration


CalibratedKeyboard.configuration = configuration_with_caps_label

model = json.loads(Path(__file__).with_name('keyboard_study_targets_v4.json').read_text())
assert model['accepted']
mici_keyboard_calibrated.CURRENT_LAYOUT_TARGET_ADJUSTMENTS = model['target_adjustments']
mici_keyboard_calibrated.CURRENT_LAYOUT_TARGET_VERSION = model['version']

if __name__ == '__main__':
  runpy.run_module('openpilot.tools.ui.mici_keyboard_study', run_name='__main__')
