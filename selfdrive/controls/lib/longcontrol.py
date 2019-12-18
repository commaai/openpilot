from cereal import log
from common.numpy_fast import clip, interp
from selfdrive.controls.lib.pid import PIController
from common.travis_checker import travis

LongCtrlState = log.ControlsState.LongControlState

STOPPING_EGO_SPEED = 0.5
MIN_CAN_SPEED = 0.3  # TODO: parametrize this in car interface
STOPPING_TARGET_SPEED = MIN_CAN_SPEED + 0.01
STARTING_TARGET_SPEED = 0.5
BRAKE_THRESHOLD_TO_PID = 0.2

STOPPING_BRAKE_RATE = 0.2  # brake_travel/s while trying to stop
STARTING_BRAKE_RATE = 0.8  # brake_travel/s while releasing on restart
BRAKE_STOPPING_TARGET = 0.5  # apply at least this amount of brake to maintain the vehicle stationary

_MAX_SPEED_ERROR_BP = [0., 30.]  # speed breakpoints
_MAX_SPEED_ERROR_V = [1.5, .8]  # max positive v_pid error VS actual speed; this avoids controls windup due to slow pedal resp

RATE = 100.0


def long_control_state_trans(active, long_control_state, v_ego, v_target, v_pid,
                             output_gb, brake_pressed, cruise_standstill):
  """Update longitudinal control state machine"""
  stopping_condition = (v_ego < 2.0 and cruise_standstill) or \
                       (v_ego < STOPPING_EGO_SPEED and \
                        ((v_pid < STOPPING_TARGET_SPEED and v_target < STOPPING_TARGET_SPEED) or
                        brake_pressed))

  starting_condition = v_target > STARTING_TARGET_SPEED and not cruise_standstill

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
      if starting_condition:
        long_control_state = LongCtrlState.starting

    elif long_control_state == LongCtrlState.starting:
      if stopping_condition:
        long_control_state = LongCtrlState.stopping
      elif output_gb >= -BRAKE_THRESHOLD_TO_PID:
        long_control_state = LongCtrlState.pid

  return long_control_state


class LongControl():
  def __init__(self, CP, compute_gb):
    self.long_control_state = LongCtrlState.off  # initialized to off
    self.pid = PIController((CP.longitudinalTuning.kpBP, CP.longitudinalTuning.kpV),
                            (CP.longitudinalTuning.kiBP, CP.longitudinalTuning.kiV),
                            rate=RATE,
                            sat_limit=0.8,
                            convert=compute_gb)
    self.v_pid = 0.0
    self.last_output_gb = 0.0
    self.lead_data = {'v_rel': None, 'a_lead': None, 'x_lead': None, 'status': False}
    self.v_ego = 0.0
    self.gas_pressed = False

  def reset(self, v_pid):
    """Reset PID controller and change setpoint"""
    self.pid.reset()
    self.v_pid = v_pid

  def dynamic_gas(self, CP):
    if True:  # CP.enableGasInterceptor:  # if user has pedal (probably Toyota) #todo: make different profiles for different toyotas, and check if toyota or not
      x = [0.0, 1.4082, 2.80311, 4.22661, 5.38271, 6.16561, 7.24781, 8.28308, 10.24465, 12.96402, 15.42303, 18.11903, 20.11703, 24.46614, 29.05805, 32.71015, 35.76326]
      # y = [0.2, 0.20443, 0.21592, 0.23334, 0.25734, 0.27916, 0.3229, 0.34784, 0.36765, 0.38, 0.396, 0.409, 0.425, 0.478, 0.55, 0.621, 0.7]
      # y = [0.175, 0.178, 0.185, 0.195, 0.209, 0.222, 0.249, 0.264, 0.276, 0.283, 0.293, 0.3, 0.31, 0.342, 0.385, 0.428, 0.475]  # todo: elvaluate if this is better
      y = [0.2, 0.20443, 0.21592, 0.23334, 0.25734, 0.27916, 0.3229, 0.35, 0.368, 0.377, 0.389, 0.399, 0.411, 0.45, 0.504, 0.558, 0.617]  # todo: this is the average of the above, only above the 8th index (about .75 reduction)
    # else:
    #   x, y = CP.gasMaxBP, CP.gasMaxV  # if user doesn't have pedal, use stock params

    gas = interp(self.v_ego, x, y)

    with open('/data/lead_data', 'a') as f:
      f.write(str(self.lead_data) + '\n')

    if self.lead_data['status']:  # if lead
      if self.v_ego <= 8.9408:  # if under 20 mph
        # TR = 1.8  # desired TR, might need to switch this to hardcoded distance values
        # current_TR = self.lead_data['x_lead'] / self.v_ego if self.v_ego > 0 else TR

        x = [0.0, 0.24588812499999999, 0.432818589, 0.593044697, 0.730381365, 1.050833588, 1.3965, 1.714627481]  # relative velocity mod
        y = [-(gas / 1.01), -(gas / 1.105), -(gas / 1.243), -(gas / 1.6), -(gas / 2.32), -(gas / 4.8), -(gas / 15), 0]
        gas_mod = interp(self.lead_data['v_rel'], x, y)

        # x = [0.0, 0.22, 0.44518483, 0.675, 1.0, 1.76361684]  # lead accel mod  # todo: this
        # y = [0.0, (gas * 0.08), (gas * 0.20), (gas * 0.4), (gas * 0.52), (gas * 0.6)]
        # gas_mod += interp(a_lead, x, y)

        # x = [TR * 0.5, TR, TR * 1.5]  # as lead gets further from car, lessen gas mod  # todo: this
        # y = [gas_mod * 1.5, gas_mod, gas_mod * 0.5]
        # gas_mod += (interp(current_TR, x, y))
        new_gas = gas + gas_mod  # (interp(current_TR, x, y))

        x = [1.78816, 6.0, 8.9408]  # slowly ramp mods down as we approach 20 mph
        y = [new_gas, (new_gas * 0.8 + gas * 0.2), gas]
        gas = interp(self.v_ego, x, y)
      else:
        x = [-0.89408, 0, 2.0]  # need to tune this
        y = [-.17, -.08, .01]
        gas += interp(self.lead_data['v_rel'], x, y)

    return clip(gas, 0.0, 1.0)

  def handle_passable(self, passable):
    self.gas_pressed = passable['gas_pressed']
    self.lead_data['v_rel'] = passable['lead_one'].vRel
    self.lead_data['a_lead'] = passable['lead_one'].aLeadK
    self.lead_data['x_lead'] = passable['lead_one'].dRel
    self.lead_data['status'] = passable['has_lead']  # this fixes radarstate always reporting a lead, thanks to arne

  def update(self, active, v_ego, brake_pressed, standstill, cruise_standstill, v_cruise, v_target, v_target_future, a_target, CP, passable):
    """Update longitudinal control. This updates the state machine and runs a PID loop"""
    self.v_ego = v_ego

    # Actuation limits
    if not travis and CP.enableGasInterceptor:
      self.handle_passable(passable)  # so travis doesn't call vRel... on None
      gas_max = self.dynamic_gas(CP)
    else:
      gas_max = interp(v_ego, CP.gasMaxBP, CP.gasMaxV)
    brake_max = interp(v_ego, CP.brakeMaxBP, CP.brakeMaxV)

    # Update state machine
    output_gb = self.last_output_gb
    self.long_control_state = long_control_state_trans(active, self.long_control_state, v_ego,
                                                       v_target_future, self.v_pid, output_gb,
                                                       brake_pressed, cruise_standstill)

    v_ego_pid = max(v_ego, MIN_CAN_SPEED)  # Without this we get jumps, CAN bus reports 0 when speed < 0.3

    if self.long_control_state == LongCtrlState.off:
      self.v_pid = v_ego_pid
      self.pid.reset()
      output_gb = 0.

    # tracking objects and driving
    elif self.long_control_state == LongCtrlState.pid:
      self.v_pid = v_target
      self.pid.pos_limit = gas_max
      self.pid.neg_limit = - brake_max

      # Toyota starts braking more when it thinks you want to stop
      # Freeze the integrator so we don't accelerate to compensate, and don't allow positive acceleration
      prevent_overshoot = not CP.stoppingControl and v_ego < 1.5 and v_target_future < 0.7
      deadzone = interp(v_ego_pid, CP.longitudinalTuning.deadzoneBP, CP.longitudinalTuning.deadzoneV)

      output_gb = self.pid.update(self.v_pid, v_ego_pid, speed=v_ego_pid, deadzone=deadzone, feedforward=a_target, freeze_integrator=prevent_overshoot)

      if prevent_overshoot:
        output_gb = min(output_gb, 0.0)

    # Intention is to stop, switch to a different brake control until we stop
    elif self.long_control_state == LongCtrlState.stopping:
      # Keep applying brakes until the car is stopped
      if not standstill or output_gb > -BRAKE_STOPPING_TARGET:
        output_gb -= STOPPING_BRAKE_RATE / RATE
      output_gb = clip(output_gb, -brake_max, gas_max)

      self.v_pid = v_ego
      self.pid.reset()

    # Intention is to move again, release brake fast before handing control to PID
    elif self.long_control_state == LongCtrlState.starting:
      if output_gb < -0.2:
        output_gb += STARTING_BRAKE_RATE / RATE
      self.v_pid = v_ego
      self.pid.reset()

    self.last_output_gb = output_gb
    final_gas = clip(output_gb, 0., gas_max)
    final_brake = -clip(output_gb, -brake_max, 0.)

    return final_gas, final_brake
