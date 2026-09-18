from openpilot.common.test import OpenpilotTestCase
from types import SimpleNamespace
from openpilot.selfdrive.controls.lib.longcontrol import LongControl, LongCtrlState, long_control_state_trans


class TestBrakingUtilization(OpenpilotTestCase):
  def test_vehicle_limit_and_inactive_control(self):
    cp = SimpleNamespace(longitudinalTuning=SimpleNamespace(kiBP=[0.0], kiV=[0.0]))
    cs = SimpleNamespace(vEgo=20.0, aEgo=0.0, brakePressed=False, cruiseState=SimpleNamespace(standstill=False))
    for limit in (-2.0, -3.5, -4.0):
      controller = LongControl(cp)
      for demand, expected in ((1.0, 0.0), (0.0, 0.0), (limit * 0.75, 0.75), (limit, 1.0), (limit * 2, 1.0)):
        with self.subTest(limit=limit, demand=demand):
          output = controller.update(True, cs, demand, False, (limit, 2.0))
          self.assertAlmostEqual(controller.braking_utilization, expected)
          self.assertAlmostEqual(output, max(limit, min(2.0, demand)))
      controller.update(False, cs, limit, False, (limit, 2.0))
      self.assertEqual(controller.braking_utilization, 0.0)


class TestLongControlStateTransition(OpenpilotTestCase):

  def test_stay_stopped(self):
    active = True
    current_state = LongCtrlState.stopping
    next_state = long_control_state_trans(active, current_state,
                             should_stop=True, brake_pressed=False, cruise_standstill=False)
    assert next_state == LongCtrlState.stopping
    next_state = long_control_state_trans(active, current_state,
                             should_stop=False, brake_pressed=True, cruise_standstill=False)
    assert next_state == LongCtrlState.stopping
    next_state = long_control_state_trans(active, current_state,
                             should_stop=False, brake_pressed=False, cruise_standstill=True)
    assert next_state == LongCtrlState.stopping
    next_state = long_control_state_trans(active, current_state,
                             should_stop=False, brake_pressed=False, cruise_standstill=False)
    assert next_state == LongCtrlState.pid
    active = False
    next_state = long_control_state_trans(active, current_state,
                             should_stop=False, brake_pressed=False, cruise_standstill=False)
    assert next_state == LongCtrlState.off

  def test_engage(self):
    active = True
    current_state = LongCtrlState.off
    next_state = long_control_state_trans(active, current_state,
                             should_stop=True, brake_pressed=False, cruise_standstill=False)
    assert next_state == LongCtrlState.stopping
    next_state = long_control_state_trans(active, current_state,
                             should_stop=False, brake_pressed=True, cruise_standstill=False)
    assert next_state == LongCtrlState.stopping
    next_state = long_control_state_trans(active, current_state,
                             should_stop=False, brake_pressed=False, cruise_standstill=True)
    assert next_state == LongCtrlState.stopping
    next_state = long_control_state_trans(active, current_state,
                             should_stop=False, brake_pressed=False, cruise_standstill=False)
    assert next_state == LongCtrlState.pid
