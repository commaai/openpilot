from types import SimpleNamespace
import unittest
from unittest.mock import patch

from openpilot.selfdrive.controls.lib.desire_helper import DesireHelper, log


class TestTurnDesire(unittest.TestCase):
  def setUp(self):
    self.helper = DesireHelper()
    self.cs = SimpleNamespace(leftBlinker=False, rightBlinker=False, leftBlindspot=False, rightBlindspot=False,
                              steeringPressed=False, steeringTorque=0.0, vEgo=0.0)
    clock = patch('openpilot.selfdrive.controls.lib.desire_helper.time.monotonic', return_value=0.0)
    self.clock = clock.start()
    self.addCleanup(clock.stop)

  def update(self, now, active=True):
    self.clock.return_value = now
    self.helper.update(self.cs, active)
    return self.helper.desire

  def test_periodic_left_and_right_without_nudge_or_speed_requirement(self):
    for side, desire in [('leftBlinker', log.Desire.turnLeft), ('rightBlinker', log.Desire.turnRight)]:
      self.helper = DesireHelper()
      self.cs.leftBlinker = self.cs.rightBlinker = False
      setattr(self.cs, side, True)
      self.assertEqual(self.update(0.0), desire)
      self.assertEqual(self.update(0.05), log.Desire.none)
      self.assertEqual(self.update(0.99), log.Desire.none)
      self.assertEqual(self.update(1.0), desire)
      self.assertEqual(self.update(1.05), log.Desire.none)
      self.assertEqual(self.update(2.0), desire)
      self.assertEqual(self.helper.lane_change_state, log.LaneChangeState.off)

  def test_off_stops_pulses_and_on_restarts_immediately(self):
    self.cs.leftBlinker = True
    self.assertEqual(self.update(0), log.Desire.turnLeft)
    self.cs.leftBlinker = False
    self.assertEqual(self.update(0.1), log.Desire.none)
    self.assertEqual(self.update(2), log.Desire.none)
    self.cs.leftBlinker = True
    self.assertEqual(self.update(2.1), log.Desire.turnLeft)

  def test_blindspot_suppresses_pulses_until_clear(self):
    for side, blindspot, desire in [('leftBlinker', 'leftBlindspot', log.Desire.turnLeft),
                                   ('rightBlinker', 'rightBlindspot', log.Desire.turnRight)]:
      self.helper = DesireHelper()
      self.cs.leftBlinker = self.cs.rightBlinker = False
      setattr(self.cs, side, True)
      self.assertEqual(self.update(0), desire)
      setattr(self.cs, blindspot, True)
      self.assertEqual(self.update(1), log.Desire.none)
      self.assertEqual(self.update(2), log.Desire.none)
      setattr(self.cs, blindspot, False)
      self.assertEqual(self.update(2.1), desire)

  def test_lateral_inactive_suppresses_pulses(self):
    self.cs.leftBlinker = True
    self.assertEqual(self.update(0, active=False), log.Desire.none)
    self.assertEqual(self.update(1), log.Desire.turnLeft)
    self.assertEqual(self.update(2, active=False), log.Desire.none)
    self.assertEqual(self.update(3, active=False), log.Desire.none)
    self.assertEqual(self.update(3.1), log.Desire.turnLeft)

  def test_hazards_and_direction_change(self):
    self.cs.leftBlinker = self.cs.rightBlinker = True
    self.assertEqual(self.update(0), log.Desire.none)
    self.assertEqual(self.update(2), log.Desire.none)
    self.cs.rightBlinker = False
    self.assertEqual(self.update(2.1), log.Desire.turnLeft)
    self.cs.leftBlinker, self.cs.rightBlinker = False, True
    self.assertEqual(self.update(2.2), log.Desire.turnRight)
    self.assertEqual(self.update(2.25), log.Desire.none)

  def test_delayed_update_does_not_burst(self):
    self.cs.leftBlinker = True
    self.assertEqual(self.update(0), log.Desire.turnLeft)
    self.assertEqual(self.update(5), log.Desire.none)
    self.assertEqual(self.update(5.05), log.Desire.turnLeft)
    self.assertEqual(self.update(5.1), log.Desire.none)
    self.assertEqual(self.update(6.1), log.Desire.turnLeft)
