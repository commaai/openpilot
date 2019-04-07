import zmq
from cereal import log
from common.numpy_fast import clip, interp
from selfdrive.controls.lib.pid import PIController
from selfdrive.kegman_conf import kegman_conf
import selfdrive.messaging as messaging
from selfdrive.services import service_list

kegman = kegman_conf()
LongCtrlState = log.Live100Data.LongControlState

STOPPING_EGO_SPEED = 0.5
MIN_CAN_SPEED = 0.3  # TODO: parametrize this in car interface
STOPPING_TARGET_SPEED = MIN_CAN_SPEED + 0.01
STARTING_TARGET_SPEED = 0.5
BRAKE_THRESHOLD_TO_PID = 0.2

STOPPING_BRAKE_RATE = 0.2  # brake_travel/s while trying to stop
STARTING_BRAKE_RATE = 0.8  # brake_travel/s while releasing on restart
BRAKE_STOPPING_TARGET = float(kegman.conf['brakeStoppingTarget'])  # apply at least this amount of brake to maintain the vehicle stationary

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


class LongControl(object):
  def __init__(self, CP, compute_gb):
    self.long_control_state = LongCtrlState.off  # initialized to off
    self.pid = PIController((CP.longitudinalKpBP, CP.longitudinalKpV),
                            (CP.longitudinalKiBP, CP.longitudinalKiV),
                            rate=RATE,
                            sat_limit=0.8,
                            convert=compute_gb)
    self.v_pid = 0.0
    self.last_output_gb = 0.0

    context = zmq.Context()
    self.poller = zmq.Poller()
    self.live20 = messaging.sub_sock(context, service_list['live20'].port, conflate=True, poller=self.poller)

  def reset(self, v_pid):
    """Reset PID controller and change setpoint"""
    self.pid.reset()
    self.v_pid = v_pid

  def dynamic_gas(self, v_ego, v_rel, d_rel, gasinterceptor, gasbuttonstatus):
    x = []
    dynamic = False
    if gasinterceptor:
      if gasbuttonstatus == 0:
        x = [0.0, 1.1176, 2.2352, 4.4704, 6.7056, 9.3878, 18.7757, 29.0576, 35.7632]  # velocity/gasMaxBP
        y = [0.125, .14, 0.15, 0.19, .25, .3375, .425, .55, .7]  # accel values/gasMaxV
        dynamic = True
      elif gasbuttonstatus == 1:
        y = [0.25, 0.9, 0.9]
      elif gasbuttonstatus == 2:
        y = [0.2, 0.2, 0.2]
    else:
      if gasbuttonstatus == 0:
        y = [0.5, 0.7, 0.9]
      elif gasbuttonstatus == 1:
        y = [0.7, 0.9, 0.9]
      elif gasbuttonstatus == 2:
        y = [0.2, 0.2, 0.2]

    if x == []:
      x = [0., 9., 35.]  # default BP values

    accel = interp(v_ego, x, y)

    if dynamic:  # dynamic gas profile specific operations
      if v_rel is not None:  # if lead
        if v_ego <= 6.7056:  # if under 15 mph
          x = [0, 2.2352, 6.7056]
          y = [accel, (accel + .055), accel]
          accel = interp(v_ego, x, y)

          x = [0, 0.89408, 1.78816]
          y = [accel, (accel + .015), (accel + .025)]
          accel = interp(v_rel, x, y)
        else:
          x = [-0.89408, 0, 0.89408, 3.12928]
          y = [(accel - .05), accel, (accel + .025), (accel + .045)]
          accel = interp(v_rel, x, y)
      else:
        if v_ego <= 6.7056:  # if under 15 mph with no lead, give a little boost ;)
          x = [0, 2.2352, 6.7056]
          y = [accel, (accel + .075), accel]
          accel = interp(v_ego, x, y)

    min_return = 0.1
    max_return = 1.0
    return max(min(accel, max_return), min_return)  # ensure we return a value between .1 and 1

  def update(self, active, v_ego, brake_pressed, standstill, cruise_standstill, v_cruise, v_target, v_target_future,
             a_target, CP, gasinterceptor, gasbuttonstatus):
    """Update longitudinal control. This updates the state machine and runs a PID loop"""
    # Actuation limits
    l20 = None

    for socket, event in self.poller.poll(0):
      if socket is self.live20:
        l20 = messaging.recv_one(socket)

    if l20 is not None:
      self.lead_1 = l20.live20.leadOne
      vRel = self.lead_1.vRel
      dRel = self.lead_1.dRel
    else:
      vRel = None
      dRel = None

    #gas_max = interp(v_ego, CP.gasMaxBP, CP.gasMaxV)
    gas_max = self.dynamic_gas(v_ego, vRel, dRel, gasinterceptor, gasbuttonstatus)
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
      deadzone = interp(v_ego_pid, CP.longPidDeadzoneBP, CP.longPidDeadzoneV)

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
