import unittest

import numpy as np

from openpilot.tools.ui.keyboard_study_optimize import KEYS, MAX_SHIFT, correction_confirmed, fit, layout_matches, metrics, stable_fit


class TestKeyboardStudyOptimize(unittest.TestCase):
  def samples(self, sessions=4, confidence=1.0):
    centers = [[1000 + index * 100, 1000] for index in range(len(KEYS))]
    centers[0], centers[1] = [0, 0], [40, 0]
    result = []
    for session in range(sessions):
      for target, point, count in ((0, [0, 0], 20), (1, [40, 0], 20), (0, [22, 0], 1)):
        for _ in range(count):
          result.append(
            {
              'session': str(session),
              'point': point,
              'centers': centers,
              'target': target,
              'confidence': confidence if point == [22, 0] else 1.0,
              'far': confidence == 0 and point == [22, 0],
            }
          )
    return result

  def test_supported_shift_fixes_misses_without_regressing_correct_taps(self):
    taps = self.samples()
    shifts = stable_fit(taps)
    result = metrics(taps, shifts)
    self.assertEqual(result['fixes'], 4)
    self.assertEqual(result['regressions'], 0)
    self.assertLessEqual(np.max(np.linalg.norm(shifts, axis=1)), MAX_SHIFT)
    self.assertTrue(np.all(shifts[KEYS.index('123')] == 0))
    self.assertTrue(np.all(shifts[KEYS.index(' ')] == 0))

  def test_single_session_cannot_move_targets(self):
    self.assertTrue(np.all(fit(self.samples(sessions=1)) == 0))

  def test_excluded_training_taps_still_count_in_evaluation(self):
    taps = self.samples(confidence=0.0)
    shifts = fit(taps)
    self.assertTrue(np.all(shifts == 0))
    self.assertEqual(metrics(taps, shifts)['after_errors'], 4)
    self.assertEqual(metrics(taps, shifts)['far_taps'], 4)

  def test_extra_empty_label_control_is_not_the_same_layout(self):
    geometry = [{'char': key, 'row': 0 if index < 10 else 1 if index < 19 else 2} for index, key in enumerate(KEYS)]
    self.assertTrue(layout_matches(geometry))
    geometry.append({'char': '', 'row': 1})
    self.assertFalse(layout_matches(geometry))

  def test_correction_requires_backspace_and_correct_replacement(self):
    wrong = {'type': 'gesture', 'trial': 0, 'before': 'he', 'expected': 'l', 'committed': 'k'}
    replacement = {'type': 'gesture', 'trial': 0, 'before': 'he', 'committed': 'l'}
    backspace = {'type': 'backspace', 'trial': 0, 'after': 'he'}
    self.assertFalse(correction_confirmed([wrong, replacement], 0, wrong))
    self.assertTrue(correction_confirmed([wrong, backspace, replacement], 0, wrong))
    replacement['committed'] = 'k'
    self.assertFalse(correction_confirmed([wrong, backspace, replacement], 0, wrong))


if __name__ == '__main__':
  unittest.main()
