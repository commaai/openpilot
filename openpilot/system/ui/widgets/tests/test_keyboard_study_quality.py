import unittest

from openpilot.tools.ui.keyboard_study_quality import assess_quality


def gesture(before, char, timestamp):
  return {'type': 'gesture', 'trial': 0, 'press_time': timestamp, 'committed': char, 'before': before,
          'samples': [[timestamp, 10, 10, 'press'], [timestamp + 0.01, 10, 10, 'release']]}


class TestKeyboardStudyQuality(unittest.TestCase):
  def test_fast_correct_password_is_not_flagged(self):
    prompt = 'Blue7!river'
    records = [gesture(prompt[:index], char, index * 0.08) for index, char in enumerate(prompt)]
    self.assertEqual(assess_quality(records, [prompt])['flags'], [])

  def test_isolated_wrong_character_and_backspace_are_not_flagged(self):
    records = [gesture('', 'h', 0), gesture('h', '!', 0.08), {'type': 'backspace'}, gesture('h', 'e', 0.2)]
    self.assertEqual(assess_quality(records, ['hello'])['flags'], [])

  def test_rapid_out_of_prompt_run_is_flagged_without_excluding_prefix(self):
    records = [gesture('', 'h', 0), gesture('h', 'x', 0.2)]
    records += [gesture('hx' + '22,!$'[:index], char, 0.4 + index * 0.12) for index, char in enumerate('22,!$')]
    quality = assess_quality(records, ['hello'])
    self.assertEqual(quality['flagged_gestures'], 5)
    self.assertEqual(quality['flags'][0]['characters'], '22,!$')
    self.assertEqual(quality['flags'][0]['start'], 0.4)

  def test_slow_mistakes_are_not_called_a_burst(self):
    records = [gesture('x', char, index) for index, char in enumerate('22,!$')]
    self.assertEqual(assess_quality(records, ['hello'])['flags'], [])


if __name__ == '__main__':
  unittest.main()
