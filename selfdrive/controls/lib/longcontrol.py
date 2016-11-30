import numpy as np
from selfdrive.config import Conversions as CV

class LongCtrlState:
  #*** this function handles the long control state transitions
  # long_control_state labels:
  off =      0  # Off
  pid =      1  # moving and tracking targets, with PID control running
  stopping = 2  # stopping and changing controls to almost open loop as PID does not fit well at such a low speed
  starting = 3  # starting and releasing brakes in open loop before giving back to PID

def long_control_state_trans(enabled, long_control_state, v_ego, v_target, v_pid, output_gb):

  stopping_speed = 0.5
  stopping_target_speed = 0.3
  starting_target_speed = 0.5
  brake_threshold_to_pid = 0.2

  stopping_condition = ((v_ego < stopping_speed) and (v_pid < stopping_target_speed) and (v_target < stopping_target_speed))

  if not enabled:
    long_control_state = LongCtrlState.off
  else:
    if long_control_state == LongCtrlState.off:
      if enabled:
        long_control_state = LongCtrlState.pid
    elif long_control_state == LongCtrlState.pid:
      if stopping_condition:
        long_control_state = LongCtrlState.stopping
    elif long_control_state == LongCtrlState.stopping:
      if (v_target > starting_target_speed):
        long_control_state = LongCtrlState.starting
    elif long_control_state == LongCtrlState.starting:
      if stopping_condition:
        long_control_state = LongCtrlState.stopping
      elif output_gb >= -brake_threshold_to_pid:
        long_control_state = LongCtrlState.pid

  return long_control_state

def get_compute_gb():
  # see debug/dump_accel_from_fiber.py
  w0 = np.array([[ 1.22056961, -0.39625418,  0.67952657],
                 [ 1.03691769,  0.78210306, -0.41343188]])
  b0 = np.array([ 0.01536703, -0.14335321, -0.26932889])
  w2 = np.array([[-0.59124422,  0.42899439,  0.38660881],
                 [ 0.79973811,  0.13178682,  0.08550351],
                 [-0.15651935, -0.44360259,  0.76910877]])
  b2 = np.array([ 0.15624429,  0.02294923, -0.0341086 ])
  w4 = np.array([[-0.31521443],
                 [-0.38626176],
                 [ 0.52667892]])
  b4 = np.array([-0.02922216])

  def compute_output(dat, w0, b0, w2, b2, w4, b4):
    m0 = np.dot(dat, w0) + b0
    m0 = leakyrelu(m0, 0.1)
    m2 = np.dot(m0, w2) + b2
    m2 = leakyrelu(m2, 0.1)
    m4 = np.dot(m2, w4) + b4
    return m4

  def leakyrelu(x, alpha):
    return np.maximum(x, alpha * x)

  def _compute_gb(dat):
    #linearly extrap below v1 using v1 and v2 data
    v1 = 5.
    v2 = 10.
    vx = dat[1]
    if vx > 5.:
      m4 = compute_output(dat, w0, b0, w2, b2, w4, b4)
    else:
      dat[1] = v1
      m4v1 = compute_output(dat, w0, b0, w2, b2, w4, b4)
      dat[1] = v2
      m4v2 = compute_output(dat, w0, b0, w2, b2, w4, b4)
      m4 = (vx - v1) * (m4v2 - m4v1) / (v2 - v1) + m4v1
    return m4
  return _compute_gb

# takes in [desired_accel, current_speed] -> [-1.0, 1.0] where -1.0 is max brake and 1.0 is max gas
compute_gb = get_compute_gb()

def pid_long_control(v_ego, v_pid, Ui_accel_cmd, gas_max, brake_max, jerk_factor, gear, rate):
  #*** This function compute the gb pedal positions in order to track the desired speed
  # proportional and integral terms. More precision at low speed
  Kp_v =  [1.2, 0.8, 0.5]
  Kp_bp = [0., 5., 35.]
  Kp = np.interp(v_ego, Kp_bp, Kp_v)
  Ki_v =  [0.18, 0.12]
  Ki_bp = [0., 35.]
  Ki = np.interp(v_ego, Ki_bp, Ki_v)

  # scle Kp and Ki by jerk factor drom drive_thread
  Kp = (1. + jerk_factor)*Kp
  Ki = (1. + jerk_factor)*Ki

  # this is ugly but can speed reports 0 when speed<0.3m/s and we can't have that jump  
  v_ego_min = 0.3
  v_ego = np.maximum(v_ego, v_ego_min)

  v_error = v_pid - v_ego

  Up_accel_cmd = v_error*Kp
  Ui_accel_cmd_new = Ui_accel_cmd + v_error*Ki*1.0/rate
  accel_cmd_new = Ui_accel_cmd_new + Up_accel_cmd
  output_gb_new = compute_gb([accel_cmd_new, v_ego])

  # Anti-wind up for integrator: only update integrator if we not against the thottle and brake limits
  # do not wind up if we are changing gear and we are on the gas pedal
  if (((v_error >= 0. and (output_gb_new < gas_max or Ui_accel_cmd < 0)) or
       (v_error <= 0. and (output_gb_new > - brake_max or Ui_accel_cmd > 0))) and
       not (v_error >= 0. and gear == 11 and output_gb_new > 0)):
    #update integrator
    Ui_accel_cmd = Ui_accel_cmd_new

  accel_cmd = Ui_accel_cmd + Up_accel_cmd

  # go from accel to pedals
  output_gb = compute_gb([accel_cmd, v_ego])
  output_gb = output_gb[0]

  # useful to know if control is against the limit
  long_control_sat = False
  if output_gb > gas_max or output_gb < -brake_max:
    long_control_sat = True

  output_gb = np.clip(output_gb, -brake_max, gas_max)

  return output_gb, Up_accel_cmd, Ui_accel_cmd, long_control_sat


stopping_brake_rate = 0.2    # brake_travel/s while trying to stop
starting_brake_rate = 0.6    # brake_travel/s while releasing on restart
starting_Ui = 0.5            # Since we don't have much info about acceleration at this point, be conservative
brake_stopping_target = 0.5  # apply at least this amount of brake to maintain the vehicle stationary

max_speed_error_v  = [1.5, .8]  # max positive v_pid error VS actual speed; this avoids controls windup due to slow pedal resp
max_speed_error_bp = [0., 30.]  # speed breakpoints

class LongControl(object):
  def __init__(self):
    self.long_control_state = LongCtrlState.off # initialized to off
    self.long_control_sat = False
    self.Up_accel_cmd = 0.
    self.last_output_gb = 0.
    self.reset(0.)

  def reset(self, v_pid):
    self.Ui_accel_cmd = 0.
    self.v_pid = v_pid

  def update(self, enabled, CS, v_cruise, v_target_lead, a_target, jerk_factor):
    # TODO: not every time
    if CS.brake_only:
      gas_max_v = [0, 0]                # values
    else:
      gas_max_v = [0.6, 0.6]            # values
    gas_max_bp = [0., 100.]             # speeds
    brake_max_v = [1.0, 1.0, 0.8, 0.8]  # values
    brake_max_bp = [0., 5., 20., 100.]  # speeds     

    # brake and gas limits
    brake_max = np.interp(CS.v_ego, brake_max_bp, brake_max_v)
    gas_max = np.interp(CS.v_ego, gas_max_bp, gas_max_v)

    overshoot_allowance = 2.0    # overshoot allowed when changing accel sign

    output_gb = self.last_output_gb
    rate = 100

    # limit max target speed based on cruise setting:
    v_cruise_mph = round(v_cruise * CV.KPH_TO_MPH)   # what's displayed in mph on the IC
    v_target = np.minimum(v_target_lead, v_cruise_mph * CV.MPH_TO_MS / CS.ui_speed_fudge)

    max_speed_delta_up = a_target[1]*1.0/rate
    max_speed_delta_down = a_target[0]*1.0/rate

    # *** long control substate transitions
    self.long_control_state = long_control_state_trans(enabled, self.long_control_state, CS.v_ego, v_target, self.v_pid, output_gb)

    # *** long control behavior based on state
    # TODO: move this to drive_helpers
    # disabled
    if self.long_control_state == LongCtrlState.off:
      self.v_pid = CS.v_ego # do nothing
      output_gb = 0.
      self.Ui_accel_cmd = 0.
    # tracking objects and driving
    elif self.long_control_state == LongCtrlState.pid:
      #reset v_pid close to v_ego if it was too far and new v_target is closer to v_ego
      if ((self.v_pid > CS.v_ego + overshoot_allowance) and
          (v_target < self.v_pid)):
        self.v_pid = np.maximum(v_target, CS.v_ego + overshoot_allowance)
      elif ((self.v_pid < CS.v_ego - overshoot_allowance) and
          (v_target > self.v_pid)):
        self.v_pid = np.minimum(v_target, CS.v_ego - overshoot_allowance)

      # move v_pid no faster than allowed accel limits
      if (v_target > self.v_pid + max_speed_delta_up):
        self.v_pid += max_speed_delta_up
      elif (v_target < self.v_pid + max_speed_delta_down):
        self.v_pid += max_speed_delta_down
      else:
        self.v_pid = v_target

      # to avoid too much wind up on acceleration, limit positive speed error
      if not CS.brake_only:
        max_speed_error = np.interp(CS.v_ego, max_speed_error_bp, max_speed_error_v)
        self.v_pid = np.minimum(self.v_pid, CS.v_ego + max_speed_error)

      output_gb, self.Up_accel_cmd, self.Ui_accel_cmd, self.long_control_sat = pid_long_control(CS.v_ego, self.v_pid, \
                                  self.Ui_accel_cmd, gas_max, brake_max, jerk_factor, CS.gear, rate)
    # intention is to stop, switch to a different brake control until we stop
    elif self.long_control_state == LongCtrlState.stopping:
      if CS.v_ego > 0. or output_gb > -brake_stopping_target or not CS.standstill:
        output_gb -= stopping_brake_rate/rate
      output_gb = np.clip(output_gb, -brake_max, gas_max)
      self.v_pid = CS.v_ego
      self.Ui_accel_cmd = 0.
    # intention is to move again, release brake fast before handling control to PID
    elif self.long_control_state == LongCtrlState.starting:
      if output_gb < -0.2:
        output_gb += starting_brake_rate/rate
      self.v_pid = CS.v_ego
      self.Ui_accel_cmd = starting_Ui

    self.last_output_gb = output_gb
    final_gas = np.clip(output_gb, 0., gas_max)
    final_brake = -np.clip(output_gb, -brake_max, 0.)
    return final_gas, final_brake
