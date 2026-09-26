from opendbc.car.structs import car
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.controls.lib.longcontrol import LongControl, LongCtrlState, long_control_state_trans


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

  def test_gas_override(self):
    for current_state in LongCtrlState.schema.enumerants.values():
      next_state = long_control_state_trans(True, current_state, should_stop=True, brake_pressed=False,
                                            cruise_standstill=False, override=True)
      assert next_state == LongCtrlState.overriding
    # disengaging always wins
    next_state = long_control_state_trans(False, LongCtrlState.overriding, should_stop=False, brake_pressed=False,
                                          cruise_standstill=False, override=True)
    assert next_state == LongCtrlState.off
    # releasing the gas resumes control
    next_state = long_control_state_trans(True, LongCtrlState.overriding, should_stop=False, brake_pressed=False, cruise_standstill=False)
    assert next_state == LongCtrlState.pid
    next_state = long_control_state_trans(True, LongCtrlState.overriding, should_stop=True, brake_pressed=False, cruise_standstill=False)
    assert next_state == LongCtrlState.stopping


class TestLongControlOverride(OpenpilotTestCase):

  def test_override_accel(self):
    LoC = LongControl(car.CarParams())
    CS = car.CarState(vEgo=10.)
    accel_limits = (-3.5, 2.0)

    assert LoC.update(True, CS, 1.0, False, accel_limits, override=True) == 1.0
    assert LoC.long_control_state == LongCtrlState.overriding
    assert LoC.update(True, CS, -1.0, False, accel_limits, override=True) == 0.0
