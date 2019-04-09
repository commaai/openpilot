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
        dynamic = True
        #x = [0.0, 0.5588, 1.1176, 1.9223, 2.5481, 3.3975, 4.0234, 5.1921, 6.0797, 7.1526, 9.388, 12.964, 15.423, 18.119, 20.117, 24.4661, 29.058, 32.7101, 35.7632]  # velocity/gasMaxBP
        #y = [0.12, 0.1275, 0.135, 0.149, 0.1635, 0.185, 0.206, 0.257, 0.2858, 0.31, 0.343, 0.38, 0.396, 0.409, 0.425, 0.478, 0.55, 0.621, 0.7]  # accel values/gasMaxV
        x = [0.0, 0.5588, 1.1176, 1.9223, 2.5481, 3.3975, 4.0234, 5.1921, 6.0797, 7.1526, 9.388, 12.964, 15.423, 18.119, 20.117, 24.4661, 29.058, 32.7101, 35.7632]
        y = [0.12, 0.1275, 0.135, 0.149, 0.1635, 0.1845, 0.203, 0.243, 0.282, 0.31, 0.343, 0.38, 0.396, 0.409, 0.425, 0.478, 0.55, 0.621, 0.7]
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

    if not dynamic:
      x = [0., 9., 35.]  # default BP values

    accel = interp(v_ego, x, y)
    if dynamic:  # dynamic gas profile specific operations
      if v_rel is not None:  # if lead
        if (v_ego) < 6.7056:  # if under 15 mph
          x = [0.0, 0.2235, 0.447, 0.8941, 1.3411, 1.7882, 2.2352, 2.6822]
          y = [-.0225, -.017, -.0075, 0, .003, .007, .0125, .02]
          accel = accel + interp(v_rel, x, y)
        else:
          x = [-0.89408, 0, 0.89408, 4.4704]
          y = [-.05, 0, .005, .02]
          accel = accel + interp(v_rel, x, y)


    min_return = 0.025
    max_return = 1.0
    return round(max(min(accel, max_return), min_return), 4)  # ensure we return a value between range

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
      try:
        vRel = self.lead_1.vRel
        dRel = self.lead_1.dRel
      except:
        vRel = None
        dRel = None
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
