import pytest

from opendbc.car.structs import car
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.longcontrol import TAKEOVER_ACCEL_JERK, LongControl, LongCtrlState, long_control_state_trans


class TestLongControlStateTransition:

  def test_stay_stopped(self):
    CP = car.CarParams.new_message()
    active = True
    current_state = LongCtrlState.stopping
    next_state = long_control_state_trans(CP, active, current_state, v_ego=0.1,
                             should_stop=True, brake_pressed=False, cruise_standstill=False)
    assert next_state == LongCtrlState.stopping
    next_state = long_control_state_trans(CP, active, current_state, v_ego=0.1,
                             should_stop=False, brake_pressed=True, cruise_standstill=False)
    assert next_state == LongCtrlState.stopping
    next_state = long_control_state_trans(CP, active, current_state, v_ego=0.1,
                             should_stop=False, brake_pressed=False, cruise_standstill=True)
    assert next_state == LongCtrlState.stopping
    next_state = long_control_state_trans(CP, active, current_state, v_ego=1.0,
                             should_stop=False, brake_pressed=False, cruise_standstill=False)
    assert next_state == LongCtrlState.pid
    active = False
    next_state = long_control_state_trans(CP, active, current_state, v_ego=1.0,
                             should_stop=False, brake_pressed=False, cruise_standstill=False)
    assert next_state == LongCtrlState.off

def test_engage():
  CP = car.CarParams.new_message()
  active = True
  current_state = LongCtrlState.off
  next_state = long_control_state_trans(CP, active, current_state, v_ego=0.1,
                             should_stop=True, brake_pressed=False, cruise_standstill=False)
  assert next_state == LongCtrlState.stopping
  next_state = long_control_state_trans(CP, active, current_state, v_ego=0.1,
                             should_stop=False, brake_pressed=True, cruise_standstill=False)
  assert next_state == LongCtrlState.stopping
  next_state = long_control_state_trans(CP, active, current_state, v_ego=0.1,
                             should_stop=False, brake_pressed=False, cruise_standstill=True)
  assert next_state == LongCtrlState.stopping
  next_state = long_control_state_trans(CP, active, current_state, v_ego=0.1,
                             should_stop=False, brake_pressed=False, cruise_standstill=False)
  assert next_state == LongCtrlState.pid

def test_starting():
  CP = car.CarParams.new_message(startingState=True, vEgoStarting=0.5)
  active = True
  current_state = LongCtrlState.starting
  next_state = long_control_state_trans(CP, active, current_state, v_ego=0.1,
                             should_stop=False, brake_pressed=False, cruise_standstill=False)
  assert next_state == LongCtrlState.starting
  next_state = long_control_state_trans(CP, active, current_state, v_ego=1.0,
                             should_stop=False, brake_pressed=False, cruise_standstill=False)
  assert next_state == LongCtrlState.pid

  current_state = LongCtrlState.off
  next_state = long_control_state_trans(CP, active, current_state, v_ego=0.1,
                             should_stop=False, brake_pressed=False, cruise_standstill=False)
  assert next_state == LongCtrlState.starting
  next_state = long_control_state_trans(CP, active, current_state, v_ego=1.0,
                             should_stop=False, brake_pressed=False, cruise_standstill=False)
  assert next_state == LongCtrlState.pid


def long_control_params(ki=0.0):
  CP = car.CarParams.new_message()
  CP.longitudinalTuning.kpBP = [0.]
  CP.longitudinalTuning.kpV = [0.]
  CP.longitudinalTuning.kiBP = [0.]
  CP.longitudinalTuning.kiV = [ki]
  return CP


def run_control(long_control, CS, a_target, active=True):
  return long_control.update(active, CS, a_target, should_stop=False, accel_limits=(-3.5, 2.0))


def test_positive_takeover_is_smooth():
  long_control = LongControl(long_control_params(ki=1.0))
  CS = car.CarState.new_message(vEgo=10.0, aEgo=0.2)

  assert run_control(long_control, CS, 1.0, active=False) == 0.0
  assert run_control(long_control, CS, 1.0) == pytest.approx(CS.aEgo)
  assert run_control(long_control, CS, 1.0) == pytest.approx(CS.aEgo + TAKEOVER_ACCEL_JERK * DT_CTRL)
  assert long_control.pid.i == 0.0


def test_takeover_does_not_delay_decel():
  long_control = LongControl(long_control_params())
  CS = car.CarState.new_message(vEgo=10.0, aEgo=0.2)

  assert run_control(long_control, CS, -1.0) == pytest.approx(-1.0)
  assert not long_control.takeover_active


def test_decel_during_takeover_is_immediate():
  long_control = LongControl(long_control_params())
  CS = car.CarState.new_message(vEgo=10.0, aEgo=0.0)

  assert run_control(long_control, CS, 1.0) == 0.0
  assert run_control(long_control, CS, 1.0) == pytest.approx(TAKEOVER_ACCEL_JERK * DT_CTRL)
  assert run_control(long_control, CS, -1.0) == pytest.approx(-1.0)
  assert not long_control.takeover_active


def test_standstill_starting_kick_is_unchanged():
  CP = long_control_params()
  CP.startingState = True
  CP.startAccel = 1.0
  CP.vEgoStarting = 0.1
  long_control = LongControl(CP)
  CS = car.CarState.new_message(vEgo=0.0, aEgo=0.0)

  assert run_control(long_control, CS, 1.0) == CP.startAccel
  assert long_control.long_control_state == LongCtrlState.starting


def test_short_override_restarts_takeover():
  long_control = LongControl(long_control_params())
  CS = car.CarState.new_message(vEgo=10.0, aEgo=0.0)

  assert run_control(long_control, CS, 0.5) == 0.0
  assert run_control(long_control, CS, 0.5) == pytest.approx(TAKEOVER_ACCEL_JERK * DT_CTRL)
  assert run_control(long_control, CS, 0.5, active=False) == 0.0
  assert run_control(long_control, CS, 0.5) == 0.0
  assert run_control(long_control, CS, 0.5) == pytest.approx(TAKEOVER_ACCEL_JERK * DT_CTRL)
