import unittest

from openpilot.selfdrive.controls.controlsd import lane_flags_from_probs


class TestLaneFlagsFromProbs(unittest.TestCase):
  def test_flags_off_stock(self):
    left_v, right_v = lane_flags_from_probs([], False)
    self.assertTrue(left_v and right_v)

  def test_missing_probs_grey(self):
    left_v, right_v = lane_flags_from_probs([], True)
    self.assertFalse(left_v or right_v)

  def test_unsure_side_grey(self):
    left_v, right_v = lane_flags_from_probs([1.0, 0.9, 0.4], True)
    self.assertTrue(left_v)
    self.assertFalse(right_v)


if __name__ == "__main__":
  unittest.main()
