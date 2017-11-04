import numpy as np
from common.numpy_fast import clip, interp
from selfdrive.config import Conversions as CV
from selfdrive.controls.lib.pid import PIController

STOPPING_EGO_SPEED = 0.5
STOPPING_TARGET_SPEED = 0.3
STARTING_TARGET_SPEED = 0.5
BRAKE_THRESHOLD_TO_PID = 0.2


class LongCtrlState:
  #*** this function handles the long control state transitions
  # long_control_state labels (see capnp enum):
  off = 'off'  # Off
  pid = 'pid'  # moving and tracking targets, with PID control running
  stopping = 'stopping'  # stopping and changing controls to almost open loop as PID does not fit well at such a low speed
  starting = 'starting'  # starting and releasing brakes in open loop before giving back to PID


def long_control_state_trans(active, long_control_state, v_ego, v_target, v_pid,
                             output_gb, brake_pressed):

  stopping_condition = (v_ego < STOPPING_EGO_SPEED) and \
                       (((v_pid < STOPPING_TARGET_SPEED) and (v_target < STOPPING_TARGET_SPEED)) or
                        (brake_pressed))

  if not active:
    long_control_state = LongCtrlState.off

  else:
    if long_control_state == LongCtrlState.off:
      if active:
        long_control_state = LongCtrlState.pid

    elif long_control_state == LongCtrlState.pid:
      if stopping_condition:
        long_control_state = LongCtrlState.stopping

    elif long_control_state == LongCtrlState.stopping:
      if (v_target > STARTING_TARGET_SPEED):
        long_control_state = LongCtrlState.starting

    elif long_control_state == LongCtrlState.starting:
      if stopping_condition:
        long_control_state = LongCtrlState.stopping
      elif output_gb >= -BRAKE_THRESHOLD_TO_PID:
        long_control_state = LongCtrlState.pid

  return long_control_state


_KP_BP = [0., 5., 35.]
_KP_V = [1.2, 0.8, 0.5]

_KI_BP = [0., 35.]
_KI_V = [0.18, 0.12]

stopping_brake_rate = 0.2  # brake_travel/s while trying to stop
starting_brake_rate = 0.8  # brake_travel/s while releasing on restart
starting_Ui = 0.5  # Since we don't have much info about acceleration at this point, be conservative
brake_stopping_target = 0.5  # apply at least this amount of brake to maintain the vehicle stationary

_MAX_SPEED_ERROR_BP = [0., 30.]  # speed breakpoints
_MAX_SPEED_ERROR_V = [1.5, .8]  # max positive v_pid error VS actual speed; this avoids controls windup due to slow pedal resp


class LongControl(object):
  def __init__(self, compute_gb):
    self.long_control_state = LongCtrlState.off  # initialized to off
    self.pid = PIController((_KP_BP, _KP_V),
                            (_KI_BP, _KI_V),
                            rate=100.0,
                            sat_limit=0.8,
                            convert=compute_gb)
    self.v_pid = 0.0
    self.last_output_gb = 0.0

  def reset(self, v_pid):
    self.pid.reset()
    self.v_pid = v_pid

  def update(self, active, v_ego, brake_pressed, standstill, v_cruise, v_target_lead, a_target,
             jerk_factor, CP):

    # actuation limits
    gas_max = interp(v_ego, CP.gasMaxBP, CP.gasMaxV)
    brake_max = interp(v_ego, CP.brakeMaxBP, CP.brakeMaxV)

    overshoot_allowance = 2.0  # overshoot allowed when changing accel sign

    output_gb = self.last_output_gb

    # limit max target speed based on cruise setting
    v_target = min(v_target_lead, v_cruise * CV.KPH_TO_MS)
    rate = 100.0
    max_speed_delta_up = a_target[1] * 1.0 / rate
    max_speed_delta_down = a_target[0] * 1.0 / rate

    self.long_control_state = long_control_state_trans(active, self.long_control_state, v_ego,\
                                                       v_target, self.v_pid, output_gb, brake_pressed)

    v_ego_pid = max(v_ego, 0.3)  # Without this we get jumps, CAN bus reports 0 when speed < 0.3

    # *** long control behavior based on state
    if self.long_control_state == LongCtrlState.off:
      self.v_pid = v_ego_pid  # do nothing
      output_gb = 0.
      self.pid.reset()

    # tracking objects and driving
    elif self.long_control_state == LongCtrlState.pid:
      #reset v_pid close to v_ego if it was too far and new v_target is closer to v_ego
      if ((self.v_pid > v_ego + overshoot_allowance) and (v_target < self.v_pid)):
        self.v_pid = max(v_target, v_ego + overshoot_allowance)
      elif ((self.v_pid < v_ego - overshoot_allowance) and (v_target > self.v_pid)):
        self.v_pid = min(v_target, v_ego - overshoot_allowance)

      # move v_pid no faster than allowed accel limits
      if (v_target > self.v_pid + max_speed_delta_up):
        self.v_pid += max_speed_delta_up
      elif (v_target < self.v_pid + max_speed_delta_down):
        self.v_pid += max_speed_delta_down
      else:
        self.v_pid = v_target

      # to avoid too much wind up on acceleration, limit positive speed error
      if CP.enableGas:
        max_speed_error = interp(v_ego, _MAX_SPEED_ERROR_BP, _MAX_SPEED_ERROR_V)
        self.v_pid = min(self.v_pid, v_ego + max_speed_error)

      self.pid.pos_limit = gas_max
      self.pid.neg_limit = - brake_max
      deadzone = interp(v_ego_pid, CP.longPidDeadzoneBP, CP.longPidDeadzoneV)
      output_gb = self.pid.update(self.v_pid, v_ego_pid, speed=v_ego_pid, jerk_factor=jerk_factor, deadzone=deadzone)

    # intention is to stop, switch to a different brake control until we stop
    elif self.long_control_state == LongCtrlState.stopping:
      # TODO: use the standstill bit from CAN to detect movements, usually more accurate than looking at v_ego
      if not standstill or output_gb > -brake_stopping_target:
        output_gb -= stopping_brake_rate / rate
      output_gb = clip(output_gb, -brake_max, gas_max)

      self.v_pid = v_ego
      self.pid.reset()

    # intention is to move again, release brake fast before handling control to PID
    elif self.long_control_state == LongCtrlState.starting:
      if output_gb < -0.2:
        output_gb += starting_brake_rate / rate
      self.v_pid = v_ego
      self.pid.reset()
      self.pid.i = starting_Ui

    self.last_output_gb = output_gb
    final_gas = clip(output_gb, 0., gas_max)
    final_brake = -clip(output_gb, -brake_max, 0.)

    return final_gas, final_brake
