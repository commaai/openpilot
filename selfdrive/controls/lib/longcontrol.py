import numpy as np
from cereal import car
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N
from openpilot.common.pid import PIDController
from openpilot.selfdrive.modeld.constants import ModelConstants

CONTROL_N_T_IDX = ModelConstants.T_IDXS[:CONTROL_N]

LongCtrlState = car.CarControl.Actuators.LongControlState


def long_control_state_trans(CP, active, long_control_state, v_ego, should_stop, brake_pressed, cruise_standstill):
  """Determines the next longitudinal control state based on current conditions and vehicle state.
  
  Args:
      CP: Car parameters
      active: Whether the control system is active
      long_control_state: Current longitudinal control state
      v_ego: Current vehicle speed in m/s
      should_stop: Whether the system should stop
      brake_pressed: Whether brake is pressed
      cruise_standstill: Whether cruise control is in standstill mode
      
  Returns:
      The next longitudinal control state
  """
  if not active:
    return LongCtrlState.off

  # Determine conditions for different states
  is_stopping = should_stop
  is_starting = (not should_stop and not cruise_standstill and not brake_pressed)
  is_started = v_ego > CP.vEgoStarting

  # State transition logic
  if long_control_state == LongCtrlState.off:
    if not is_starting:
      return LongCtrlState.stopping
    elif is_starting and CP.startingState:
      return LongCtrlState.starting
    else:
      return LongCtrlState.pid

  elif long_control_state == LongCtrlState.stopping:
    if is_starting and CP.startingState:
      return LongCtrlState.starting
    elif is_starting:
      return LongCtrlState.pid
    else:
      return long_control_state  # Stay in stopping state

  elif long_control_state in [LongCtrlState.starting, LongCtrlState.pid]:
    if is_stopping:
      return LongCtrlState.stopping
    elif is_started:
      return LongCtrlState.pid
    else:
      return long_control_state  # Maintain current state

  # Fallback - should not reach here with proper state machine
  return long_control_state

class LongControl:
  def __init__(self, CP):
    self.CP = CP
    self.long_control_state = LongCtrlState.off
    self.pid = PIDController(proportional_gain=(CP.longitudinalTuning.kpBP, CP.longitudinalTuning.kpV),
                             integral_gain=(CP.longitudinalTuning.kiBP, CP.longitudinalTuning.kiV),
                             rate=1 / DT_CTRL)
    self.last_output_accel = 0.0

  def reset(self):
    self.pid.reset()

  def update(self, active, CS, a_target, should_stop, accel_limits):
    """Update longitudinal control. This updates the state machine and runs a PID loop"""
    self.pid.neg_limit = accel_limits[0]
    self.pid.pos_limit = accel_limits[1]

    self.long_control_state = long_control_state_trans(self.CP, active, self.long_control_state, CS.vEgo,
                                                       should_stop, CS.brakePressed,
                                                       CS.cruiseState.standstill)
    if self.long_control_state == LongCtrlState.off:
      self.reset()
      output_accel = 0.

    elif self.long_control_state == LongCtrlState.stopping:
      output_accel = self.last_output_accel
      if output_accel > self.CP.stopAccel:
        output_accel = min(output_accel, 0.0)
        output_accel -= self.CP.stoppingDecelRate * DT_CTRL
      self.reset()

    elif self.long_control_state == LongCtrlState.starting:
      output_accel = self.CP.startAccel
      self.reset()

    else:  # LongCtrlState.pid
      error = a_target - CS.aEgo
      output_accel = self.pid.update(error, speed=CS.vEgo,
                                     feedforward=a_target)

    self.last_output_accel = np.clip(output_accel, accel_limits[0], accel_limits[1])
    return self.last_output_accel
