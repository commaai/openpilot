import unittest
from types import SimpleNamespace

from openpilot.selfdrive.controls.controlsd import lane_flags_from_probs, lead_follow_status


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


class TestLeadFollowStatus(unittest.TestCase):
  def _plan(self, source, a_target):
    return SimpleNamespace(longitudinalPlanSource=source, aTarget=a_target)

  def test_no_lead(self):
    lead = SimpleNamespace(present=False, dRel=0.0)
    self.assertEqual(lead_follow_status(lead, None, 1, 10.0), 1)

  def test_low_speed(self):
    lead = SimpleNamespace(present=True, dRel=1.0)
    self.assertEqual(lead_follow_status(lead, None, 0, 1.9), 1)
    self.assertEqual(lead_follow_status(lead, None, 2, 0.0), 1)

  def test_gap_boundaries(self):
    lead = SimpleNamespace(present=True)
    for personality, target in ((0, 1.25), (1, 1.45), (2, 1.75)):
      d_target = target * 10.0 + 6.0
      lead.dRel = d_target
      self.assertEqual(lead_follow_status(lead, None, personality, 10.0), 1)
      lead.dRel = 0.8 * d_target
      self.assertEqual(lead_follow_status(lead, None, personality, 10.0), 1)
      lead.dRel = 0.7 * d_target
      self.assertEqual(lead_follow_status(lead, None, personality, 10.0), 2)
      lead.dRel = 0.65 * d_target
      self.assertEqual(lead_follow_status(lead, None, personality, 10.0), 2)
      lead.dRel = 0.55 * d_target
      self.assertEqual(lead_follow_status(lead, None, personality, 10.0), 3)

  def test_unknown_personality_is_standard(self):
    lead = SimpleNamespace(present=True, dRel=14.0)
    self.assertEqual(lead_follow_status(lead, None, 7, 10.0), 2)

  def test_negative_distance_clamped(self):
    lead = SimpleNamespace(present=True, dRel=-5.0)
    self.assertEqual(lead_follow_status(lead, None, 1, 10.0), 3)

  def test_lead_braking_warn(self):
    lead = SimpleNamespace(present=True, dRel=40.0)
    self.assertEqual(lead_follow_status(lead, self._plan(1, -0.6), 1, 10.0), 2)
    self.assertEqual(lead_follow_status(lead, self._plan(2, -0.6), 1, 10.0), 2)

  def test_lead_hard_braking_too_close(self):
    lead = SimpleNamespace(present=True, dRel=40.0)
    self.assertEqual(lead_follow_status(lead, self._plan(1, -1.6), 1, 10.0), 3)

  def test_cruise_or_e2e_braking_ignored(self):
    lead = SimpleNamespace(present=True, dRel=40.0)
    self.assertEqual(lead_follow_status(lead, self._plan(0, -2.0), 1, 10.0), 1)
    self.assertEqual(lead_follow_status(lead, self._plan(4, -2.0), 1, 10.0), 1)


if __name__ == "__main__":
  unittest.main()
