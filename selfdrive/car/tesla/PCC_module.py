from selfdrive.car.tesla import teslacan
from selfdrive.car.tesla.longcontrol_tesla import LongControl, LongCtrlState, STARTING_TARGET_SPEED
from selfdrive.car.tesla import teslacan
from common.numpy_fast import clip, interp
from selfdrive.services import service_list
from selfdrive.car.tesla.values import AH,CruiseState, CruiseButtons, CAR
from selfdrive.boardd.boardd import can_list_to_can_capnp
from selfdrive.config import Conversions as CV
from selfdrive.controls.lib.speed_smoother import speed_smoother
from common.realtime import sec_since_boot
import selfdrive.messaging as messaging
import os
import subprocess
import time
import zmq
import math
import numpy as np
from collections import OrderedDict


# TODO: these should end up in values.py at some point, probably variable by trim
# Accel limits
ACCEL_HYST_GAP = 0.5  # don't change accel command for small oscilalitons within this value

PEDAL_MAX_UP = 4.
PEDAL_MAX_DOWN = 150.
#BB
# min safe distance in meters. Roughly 2 car lengths.
MIN_SAFE_DIST_M = 10.

MAX_PEDAL_VALUE = 112.

#BBTODO: move the vehicle variables; maybe make them speed variable
TORQUE_LEVEL_ACC = 0.
TORQUE_LEVEL_DECEL = -30.
FOLLOW_TIME_S = 1.5  # time in seconds to follow car in front
MIN_PCC_V_KPH = 0. #
MAX_PCC_V_KPH = 170.

MIN_CAN_SPEED = 0.3  #TODO: parametrize this in car interface

# Map of speed to max allowed decel.
# Make sure these accelerations are smaller than mpc limits.
_A_CRUISE_MIN = OrderedDict([
  # (speed in m/s, allowed deceleration)
  (0.0, 2),
  (40., 2)])

# Map of speed to max allowed acceleration.
# Need higher accel at very low speed for stop and go.
# make sure these accelerations are smaller than mpc limits.
_A_CRUISE_MAX = OrderedDict([
  # (speed in m/s, allowed acceleration)
  (0.0, 0.50),
  (5.0, 0.28),
  (10., 0.18),
  (20., 0.14),
  (40., 0.10)])
  
# Lookup table for turns
_A_TOTAL_MAX = OrderedDict([
  (0.0, 1.5),
  (20., 1.5),
  (40., 1.5)])

_DT = 0.05    # 20Hz in our case, since we don't want to process more than once the same live20 message
_DT_MPC = 0.05  # 20Hz

class Mode(object):
  label = None

class OpMode(Mode):
  label = 'OP'

class FollowMode(Mode):
  label = 'FOLLOW'
  
class PCCModes(object):
  _all_modes = [OpMode(), FollowMode()]
  _mode_map = {mode.label : mode for mode in _all_modes}
  BUTTON_NAME = 'pedal'
  BUTTON_ABREVIATION = 'PDL'
  
  @classmethod
  def from_label(cls, label):
    return cls._mode_map.get(label, OpMode())
    
  @classmethod
  def from_buttons(cls, cstm_btns):
    return cls.from_label(cstm_btns.get_button_label2(cls.BUTTON_NAME))
    
  @classmethod
  def is_selected(cls, mode, cstm_butns):
    """Tell if the UI buttons are set to the given mode"""
    return type(mode) == type(cls.from_buttons(cstm_butns))
    
  @classmethod
  def labels(cls):
    return [mode.label for mode in cls._all_modes]
    

def tesla_compute_gb(accel, speed):
  return float(accel)  / 3.

def calc_cruise_accel_limits(CS, lead):
  a_cruise_min = _interp_map(CS.v_ego, _A_CRUISE_MIN)
  a_cruise_max = _interp_map(CS.v_ego, _A_CRUISE_MAX)
    
  a_while_turning_max = max_accel_in_turns(CS.v_ego, CS.angle_steers, CS.CP)
  a_cruise_max = min(a_cruise_max, a_while_turning_max)
  # Reduce accel if lead car is close
  a_cruise_max *= _accel_limit_multiplier(CS.v_ego, lead)
  # Reduce decel if lead car is distant
  a_cruise_min *= _decel_limit_multiplier(CS.v_ego, lead)
  
  return float(a_cruise_min), float(a_cruise_max)


def max_accel_in_turns(v_ego, angle_steers, CP):
  """
  This function returns a limited long acceleration allowed, depending on the existing lateral acceleration
  this should avoid accelerating when losing the target in turns
  """

  a_total_max = _interp_map(v_ego, _A_TOTAL_MAX)
  a_y = v_ego**2 * angle_steers * CV.DEG_TO_RAD / (CP.steerRatio * CP.wheelbase)
  a_x_allowed = math.sqrt(max(a_total_max**2 - a_y**2, 0.))
  return a_x_allowed


class PCCState(object):
  # Possible state of the ACC system, following the DI_cruiseState naming
  # scheme.
  OFF = 0         # Disabled by UI.
  STANDBY = 1     # Ready to be enaged.
  ENABLED = 2     # Engaged.
  NOT_READY = 9   # Not ready to be engaged due to the state of the car.



def _current_time_millis():
  return int(round(time.time() * 1000))

def accel_hysteresis(accel, accel_steady, enabled):

  # for small accel oscillations within ACCEL_HYST_GAP, don't change the accel command
  if not enabled:
    # send 0 when disabled, otherwise acc faults
    accel_steady = 0.
  elif accel > accel_steady + ACCEL_HYST_GAP:
    accel_steady = accel - ACCEL_HYST_GAP
  elif accel < accel_steady - ACCEL_HYST_GAP:
    accel_steady = accel + ACCEL_HYST_GAP
  accel = accel_steady
  return accel, accel_steady


#this is for the pedal cruise control
class PCCController(object):
  def __init__(self,carcontroller):
    self.CC = carcontroller
    self.human_cruise_action_time = 0
    self.automated_cruise_action_time = 0
    self.last_angle = 0.
    context = zmq.Context()
    self.poller = zmq.Poller()
    self.live20 = messaging.sub_sock(context, service_list['live20'].port, conflate=True, poller=self.poller)
    self.lead_1 = None
    self.last_update_time = 0
    self.enable_pedal_cruise = False
    self.last_cruise_stalk_pull_time = 0
    self.prev_pcm_acc_status = 0
    self.prev_cruise_buttons = CruiseButtons.IDLE
    self.pedal_speed_kph = 0.
    self.pedal_idx = 0
    self.accel_steady = 0.
    self.prev_tesla_accel = 0.
    self.prev_tesla_pedal = 0.
    self.pedal_interceptor_state = 0
    self.torqueLevel_last = 0.
    self.prev_v_ego = 0.
    self.PedalForZeroTorque = 18. #starting number, works on my S85
    self.lastTorqueForPedalForZeroTorque = TORQUE_LEVEL_DECEL
    self.v_pid = 0.
    self.a_pid = 0.
    self.last_output_gb = 0.
    self.last_speed_kph = 0.
    #for smoothing the changes in speed
    self.v_acc_start = 0.0
    self.a_acc_start = 0.0
    self.acc_start_time = sec_since_boot()
    self.v_acc = 0.0
    self.v_acc_sol = 0.0
    self.v_acc_future = 0.0
    self.a_acc = 0.0
    self.a_acc_sol = 0.0
    self.v_cruise = 0.0
    self.a_cruise = 0.0
    #Long Control
    self.LoC = None
    #when was radar data last updated?
    self.last_md_ts = None
    self.last_l100_ts = None
    self.md_ts = None
    self.l100_ts = None
    self.lead_last_seen_time_ms = 0
    self.continuous_lead_sightings = 0

  def reset(self, v_pid):
    if self.LoC:
      self.LoC.reset(v_pid)

  def update_stat(self, CS, enabled, sendcan):
    if not self.LoC:
      self.LoC = LongControl(CS.CP, tesla_compute_gb)

    can_sends = []
    if CS.pedal_interceptor_available and not CS.cstm_btns.get_button_status("pedal"):
      # pedal hardware, enable button
      CS.cstm_btns.set_button_status("pedal", 1)
      print "enabling pedal"
    elif not CS.pedal_interceptor_available:
      if CS.cstm_btns.get_button_status("pedal"):
        # no pedal hardware, disable button
        CS.cstm_btns.set_button_status("pedal", 0)
        print "disabling pedal"
      print "Pedal unavailable."
      return
    
    # check if we had error before
    if self.pedal_interceptor_state != CS.pedal_interceptor_state:
      self.pedal_interceptor_state = CS.pedal_interceptor_state
      CS.cstm_btns.set_button_status("pedal", 1 if self.pedal_interceptor_state > 0 else 0)
      if self.pedal_interceptor_state > 0:
        # send reset command
        idx = self.pedal_idx
        self.pedal_idx = (self.pedal_idx + 1) % 16
        can_sends.append(teslacan.create_pedal_command_msg(0, 0, idx))
        sendcan.send(can_list_to_can_capnp(can_sends, msgtype='sendcan').to_bytes())
        CS.UE.custom_alert_message(3, "Pedal Interceptor Off (state %s)" % self.pedal_interceptor_state, 150, 3)
      else:
        CS.UE.custom_alert_message(3, "Pedal Interceptor On", 150, 3)
    # disable on brake
    if CS.brake_pressed and self.enable_pedal_cruise:
      self.enable_pedal_cruise = False
      self.reset(0.)
      CS.UE.custom_alert_message(3, "PDL Disabled", 150, 4)
      CS.cstm_btns.set_button_status("pedal", 1)
      print "brake pressed"

    prev_enable_pedal_cruise = self.enable_pedal_cruise
    # process any stalk movement
    curr_time_ms = _current_time_millis()
    speed_uom_kph = 1.
    if CS.imperial_speed_units:
      speed_uom_kph = CV.MPH_TO_KPH
    if (CS.cruise_buttons == CruiseButtons.MAIN and
        self.prev_cruise_buttons != CruiseButtons.MAIN):
      double_pull = curr_time_ms - self.last_cruise_stalk_pull_time < 750
      self.last_cruise_stalk_pull_time = curr_time_ms
      ready = (CS.cstm_btns.get_button_status("pedal") > PCCState.OFF
               and enabled
               and CruiseState.is_off(CS.pcm_acc_status))
      if ready and double_pull:
        # A double pull enables ACC. updating the max ACC speed if necessary.
        self.enable_pedal_cruise = True
        self.LoC.reset(CS.v_ego)
        # Increase PCC speed to match current, if applicable.
        self.pedal_speed_kph = max(CS.v_ego * CV.MS_TO_KPH, self.pedal_speed_kph)
      else:
        # A single pull disables PCC (falling back to just steering).
        self.enable_pedal_cruise = False
    # Handle pressing the cancel button.
    elif CS.cruise_buttons == CruiseButtons.CANCEL:
      self.enable_pedal_cruise = False
      self.pedal_speed_kph = 0. 
      self.last_cruise_stalk_pull_time = 0
    # Handle pressing up and down buttons.
    elif (self.enable_pedal_cruise 
          and CS.cruise_buttons != self.prev_cruise_buttons):
      # Real stalk command while PCC is already enabled. Adjust the max PCC
      # speed if necessary. 
      actual_speed_kph = CS.v_ego * CV.MS_TO_KPH
      if CS.cruise_buttons == CruiseButtons.RES_ACCEL:
        self.pedal_speed_kph = max(self.pedal_speed_kph, actual_speed_kph) + speed_uom_kph
      elif CS.cruise_buttons == CruiseButtons.RES_ACCEL_2ND:
        self.pedal_speed_kph = max(self.pedal_speed_kph, actual_speed_kph) + 5 * speed_uom_kph
      elif CS.cruise_buttons == CruiseButtons.DECEL_SET:
        self.pedal_speed_kph = min(self.pedal_speed_kph, actual_speed_kph) - speed_uom_kph
      elif CS.cruise_buttons == CruiseButtons.DECEL_2ND:
        self.pedal_speed_kph = min(self.pedal_speed_kph, actual_speed_kph) - 5 * speed_uom_kph
      # Clip PCC speed between 0 and 170 KPH.
      self.pedal_speed_kph = clip(self.pedal_speed_kph, MIN_PCC_V_KPH, MAX_PCC_V_KPH)
    # If something disabled cruise control, disable PCC too
    elif self.enable_pedal_cruise and CS.pcm_acc_status:
      self.enable_pedal_cruise = False
    
    # Notify if PCC was toggled
    if prev_enable_pedal_cruise and not self.enable_pedal_cruise:
      CS.UE.custom_alert_message(3, "PCC Disabled", 150, 4)
      CS.cstm_btns.set_button_status("pedal", PCCState.STANDBY)
    elif self.enable_pedal_cruise and not prev_enable_pedal_cruise:
      CS.UE.custom_alert_message(2, "PCC Enabled", 150)
      CS.cstm_btns.set_button_status("pedal", PCCState.ENABLED)

    # Update the UI to show whether the current car state allows PCC.
    if CS.cstm_btns.get_button_status("pedal") in [PCCState.STANDBY, PCCState.NOT_READY]:
      if enabled and CruiseState.is_off(CS.pcm_acc_status):
        CS.cstm_btns.set_button_status("pedal", PCCState.STANDBY)
      else:
        CS.cstm_btns.set_button_status("pedal", PCCState.NOT_READY)
          
    # Update prev state after all other actions.
    self.prev_cruise_buttons = CS.cruise_buttons
    self.prev_pcm_acc_status = CS.pcm_acc_status
    
  def update_pdl(self, enabled, CS, frame, actuators, pcm_speed):
    cur_time = sec_since_boot()
    idx = self.pedal_idx
    self.pedal_idx = (self.pedal_idx + 1) % 16
    if not CS.pedal_interceptor_available or not enabled:
      return 0., 0, idx
    # Alternative speed decision logic that uses the lead car's distance
    # and speed more directly.
    # Bring in the lead car distance from the Live20 feed
    l20 = None
    if enabled:
      for socket, _ in self.poller.poll(0):
        if socket is self.live20:
          l20 = messaging.recv_one(socket)
          break
    if l20 is not None:
      self.lead_1 = l20.live20.leadOne
      if _is_present(self.lead_1):
        self.lead_last_seen_time_ms = _current_time_millis()
        self.continuous_lead_sightings += 1
      else:
        self.continuous_lead_sightings = 0
      self.md_ts = l20.live20.mdMonoTime
      self.l100_ts = l20.live20.l100MonoTime

    brake_max, accel_max = calc_cruise_accel_limits(CS, self.lead_1)
    output_gb = 0
    ####################################################################
    # this mode (Follow) uses the Follow logic created by JJ for ACC
    #
    # once the speed is detected we have to use our own PID to determine
    # how much accel and break we have to do
    ####################################################################
    if PCCModes.is_selected(FollowMode(), CS.cstm_btns):
      self.v_pid = self.calc_follow_speed_ms(CS)
      # cruise speed can't be negative even is user is distracted
      self.v_pid = max(self.v_pid, 0.)

      enabled = self.enable_pedal_cruise and self.LoC.long_control_state in [LongCtrlState.pid, LongCtrlState.stopping]

      if self.enable_pedal_cruise:
        jerk_min, jerk_max = _jerk_limits(CS.v_ego, self.lead_1, self.pedal_speed_kph, self.lead_last_seen_time_ms)
        self.v_cruise, self.a_cruise = speed_smoother(self.v_acc_start, self.a_acc_start,
                                                      self.v_pid,
                                                      accel_max, brake_max,
                                                      jerk_max, jerk_min,
                                                      _DT_MPC)
        
        # cruise speed can't be negative even is user is distracted
        self.v_cruise = max(self.v_cruise, 0.)
        self.v_acc = self.v_cruise
        self.a_acc = self.a_cruise

        self.v_acc_future = self.v_pid

        self.v_acc_start = self.v_acc_sol
        self.a_acc_start = self.a_acc_sol
        self.acc_start_time = cur_time

        # Interpolation of trajectory
        dt = min(cur_time - self.acc_start_time, _DT_MPC + _DT) + _DT  # no greater than dt mpc + dt, to prevent too high extraps
        self.a_acc_sol = self.a_acc_start + (dt / _DT_MPC) * (self.a_acc - self.a_acc_start)
        self.v_acc_sol = self.v_acc_start + dt * (self.a_acc_sol + self.a_acc_start) / 2.0

        # we will try to feed forward the pedal position.... we might want to feed the last output_gb....
        # it's all about testing now.
        vTarget = clip(self.v_acc_sol, 0, self.v_pid)
        self.vTargetFuture = clip(self.v_acc_future, 0, self.v_pid)

        t_go, t_brake = self.LoC.update(self.enable_pedal_cruise, CS.v_ego, CS.brake_pressed != 0, CS.standstill, False, 
                  self.v_pid , vTarget, self.vTargetFuture, self.a_acc_sol, CS.CP, None)
        output_gb = t_go - t_brake
        #print "Output GB Follow:", output_gb
      else:
        self.LoC.reset(CS.v_ego)
        #print "PID reset"
        output_gb = 0.
        starting = self.LoC.long_control_state == LongCtrlState.starting
        a_ego = min(CS.a_ego, 0.0)
        reset_speed = MIN_CAN_SPEED if starting else CS.v_ego
        reset_accel = CS.CP.startAccel if starting else a_ego
        self.v_acc = reset_speed
        self.a_acc = reset_accel
        self.v_acc_start = reset_speed
        self.a_acc_start = reset_accel
        self.v_cruise = reset_speed
        self.a_cruise = reset_accel
        self.v_acc_sol = reset_speed
        self.a_acc_sol = reset_accel
        self.v_pid = reset_speed

    ##############################################################
    # This mode uses the longitudinal MPC built in OP
    #
    # we use the values from actuator.accel and actuator.brake
    ##############################################################
    elif PCCModes.is_selected(OpMode(), CS.cstm_btns):
      output_gb = actuators.gas - actuators.brake

    ######################################################################################
    # Determine pedal "zero"
    #
    #save position for cruising (zero acc, zero brake, no torque) when we are above 10 MPH
    ######################################################################################
    if (CS.torqueLevel < TORQUE_LEVEL_ACC
        and CS.torqueLevel > TORQUE_LEVEL_DECEL
        and CS.v_ego >= 10.* CV.MPH_TO_MS
        and abs(CS.torqueLevel) < abs(self.lastTorqueForPedalForZeroTorque)):
      self.PedalForZeroTorque = self.prev_tesla_accel
      self.lastTorqueForPedalForZeroTorque = CS.torqueLevel
      #print "Detected new Pedal For Zero Torque at %s" % (self.PedalForZeroTorque)
      #print "Torque level at detection %s" % (CS.torqueLevel)
      #print "Speed level at detection %s" % (CS.v_ego * CV.MS_TO_MPH)

    self.last_output_gb = output_gb
    # accel and brake
    apply_accel = clip(output_gb, 0., accel_max)
    MPC_BRAKE_MULTIPLIER = 6.
    apply_brake = -clip(output_gb * MPC_BRAKE_MULTIPLIER, -brake_max, 0.)

    # if speed is over 5mpg, the "zero" is at PedalForZeroTorque; otherwise it is zero
    pedal_zero = 0.
    if CS.v_ego >= 5.* CV.MPH_TO_MS:
      pedal_zero = self.PedalForZeroTorque
    tesla_brake = clip((1. - apply_brake) * pedal_zero, 0, pedal_zero)
    tesla_accel = clip(apply_accel * MAX_PEDAL_VALUE, 0, MAX_PEDAL_VALUE - tesla_brake)
    tesla_pedal = tesla_brake + tesla_accel

    tesla_pedal, self.accel_steady = accel_hysteresis(tesla_pedal, self.accel_steady, enabled)
    
    tesla_pedal = clip(tesla_pedal, self.prev_tesla_pedal - PEDAL_MAX_DOWN, self.prev_tesla_pedal + PEDAL_MAX_UP)
    tesla_pedal = clip(tesla_pedal, 0., MAX_PEDAL_VALUE) if self.enable_pedal_cruise else 0.
    enable_pedal = 1. if self.enable_pedal_cruise else 0.
    
    self.torqueLevel_last = CS.torqueLevel
    self.prev_tesla_pedal = tesla_pedal * enable_pedal
    self.prev_tesla_accel = apply_accel * enable_pedal
    self.prev_v_ego = CS.v_ego

    self.last_md_ts = self.md_ts
    self.last_l100_ts = self.l100_ts

    return self.prev_tesla_pedal, enable_pedal, idx

  # function to calculate the cruise speed based on a safe follow distance
  def calc_follow_speed_ms(self, CS):
     # Make sure we were able to populate lead_1.
    if self.lead_1 is None:
      return None, None, None
    # dRel is in meters.
    lead_dist_m = _visual_radar_adjusted_dist_m(self.lead_1.dRel)
    # Grab the relative speed.
    rel_speed_kph = self.lead_1.vRel * CV.MS_TO_KPH
    # v_ego is in m/s, so safe_distance is in meters.
    safe_dist_m = _safe_distance_m(CS.v_ego)
    # Current speed in kph
    actual_speed_kph = CS.v_ego * CV.MS_TO_KPH
    # speed and brake to issue
    new_speed_kph = self.last_speed_kph
    ###   Logic to determine best cruise speed ###
    if self.enable_pedal_cruise:
      # If no lead is present, accel up to max speed
      if lead_dist_m == 0:
        new_speed_kph = self.pedal_speed_kph
      elif lead_dist_m > 0:
        lead_absolute_speed_kph = actual_speed_kph + rel_speed_kph
        if lead_dist_m < MIN_SAFE_DIST_M:
          new_speed_kph = MIN_PCC_V_KPH
        else:
          # Force speed into a band that is generally slower than lead if too
          # close, and faster than lead if too far. Allow a range of speeds at
          # any given distance, to prevent continuous jerky adjustments.
          min_vrel_kph_map = OrderedDict([
            # (distance in m, min allowed relative kph)
            (0.5 * safe_dist_m, 2),
            (1.0 * safe_dist_m, -5),
            (1.5 * safe_dist_m, -8),
            (3.0 * safe_dist_m, -20)])
          min_vrel_kph = _interp_map(lead_dist_m, min_vrel_kph_map)
          max_vrel_kph_map = OrderedDict([
            # (distance in m, max allowed relative kph)
            (0.5 * safe_dist_m, 10),
            (1.0 * safe_dist_m, 3),
            (1.5 * safe_dist_m, 0),
            # With visual radar the relative velocity is 0 until the confidence
            # gets high. So even a small negative number here gives constant
            # accel until lead car gets close.
            (2.0 * safe_dist_m, -2)])
          max_vrel_kph = _interp_map(lead_dist_m, max_vrel_kph_map)
          min_kph = lead_absolute_speed_kph - max_vrel_kph
          max_kph = lead_absolute_speed_kph - min_vrel_kph
          new_speed_kph =  clip(new_speed_kph, min_kph, max_kph)
          
        # Enforce limits on speed in the presence of a lead car.
        new_speed_kph = min(new_speed_kph,
                            _max_safe_speed_kph(lead_dist_m),
                            lead_absolute_speed_kph - _min_safe_vrel_kph(lead_dist_m))

      # Enforce limits on speed
      new_speed_kph = clip(new_speed_kph, MIN_PCC_V_KPH, MAX_PCC_V_KPH)
      new_speed_kph = clip(new_speed_kph, MIN_PCC_V_KPH, self.pedal_speed_kph)
      if CS.blinker_on:
        # Don't accelerate during manual turns.
        new_speed_kph = min(new_speed_kph, self.last_speed_kph)
      self.last_speed_kph = new_speed_kph

    return new_speed_kph * CV.KPH_TO_MS

def _visual_radar_adjusted_dist_m(m):
  # visual radar sucks at short distances. It rarely shows readings below 7m.
  # So rescale distances with 7m -> 0m. Maxes out at 1km, if that matters.
  mapping = OrderedDict([
    # (input distance, output distance)
    (7,    0),      # anything below 7m is set to 0m.
    (1000, 1000)])  # values >7m are scaled, maxing out at 1km.
  return _interp_map(m, mapping)

def _safe_distance_m(v_ego_ms):
  return max(FOLLOW_TIME_S * v_ego_ms, MIN_SAFE_DIST_M)

def _max_safe_speed_kph(m):
  return CV.MS_TO_KPH * m / FOLLOW_TIME_S
  
def _min_safe_vrel_kph(m):
  min_vrel_by_distance = OrderedDict([
    # (meters, safe relative velocity in kph)
    # Remember, a negative relative velocity means we are closing the distance.
    (MIN_SAFE_DIST_M, 1),    # If lead is close, it better be pulling away.
    (200,             -15)]) # if lead is far, it's ok if we're closing.
  return _interp_map(m, min_vrel_by_distance)

def _is_present(lead):
  return bool(lead and lead.dRel)

def _sec_til_collision(lead):
  if _is_present(lead) and lead.vRel < 0:
    return _visual_radar_adjusted_dist_m(lead.dRel) / abs(lead.vRel)
  else:
    return 60  # Arbitrary, but better than MAXINT because we can still do math on it.
  
def _interp_map(val, val_map):
  """Helper to call interp with an OrderedDict for the mapping. I find
  this easier to read than interp, which takes two arrays."""
  return interp(val, val_map.keys(), val_map.values())
  
def _accel_limit_multiplier(v_ego, lead):
  """Limits acceleration in the presence of a lead car. The further the lead car
  is, the more accel is allowed. Range: 0 to 1, so that it can be multiplied
  with other accel limits."""
  if _is_present(lead):
    safe_dist_m = _safe_distance_m(v_ego)
    accel_multipliers = OrderedDict([
      # (distance in m, acceleration fraction)
      (0.7 * safe_dist_m, 0.0),
      (1.0 * safe_dist_m, 0.4),
      (3.0 * safe_dist_m, 1.0)])
    return _interp_map(lead.dRel, accel_multipliers)
  else:
    return 1.0

def _decel_limit_multiplier(v_ego, lead):
  if _is_present(lead):
    decel_map = OrderedDict([
      # (sec to collision, decel)
      (2, 1.0),
      (4, 0.4),
      (8, 0.1)])
    return _interp_map(_sec_til_collision(lead), decel_map)
  else:
    return 1.0
    
def _jerk_limits(v_ego, lead, max_speed_kph, lead_last_seen_time_ms):
  # prevent high accel jerk near max speed
  near_max_speed_multipliers = OrderedDict([
    # (kph under max speed, accel jerk multiplier)
    (0, 0.01),
    (4, 1.0)])
  near_max_speed_multiplier = _interp_map(max_speed_kph - v_ego * CV.MS_TO_KPH, near_max_speed_multipliers)
  
  if _is_present(lead):
    # pick decel jerk based on how much time we have til collision
    decel_jerk_map = OrderedDict([
      # (sec to collision, decel jerk)
      (0, -1.00),
      (2, -0.25),
      (4, -0.01),
      (8, -0.001)])
    decel_jerk = _interp_map(_sec_til_collision(lead), decel_jerk_map)
    safe_dist_m = _safe_distance_m(v_ego) 
    accel_jerk_map = OrderedDict([
      # (distance in m, accel jerk)
      (1.0 * safe_dist_m, 0.01),
      (2.8 * safe_dist_m, 0.11)])
    accel_jerk = _interp_map(lead.dRel, accel_jerk_map)
    accel_jerk *= near_max_speed_multiplier
    return decel_jerk, accel_jerk
  else:
    # In the absence of a lead car
    decel_jerk = -0.15
    # Limit accel jerk if the lead was only recently lost, to prevent
    # bucking as a lead is intermittently detected.
    time_since_lead_seen_ms = _current_time_millis() - lead_last_seen_time_ms
    time_since_lead_seen_multipliers = OrderedDict([
      # (ms since last lead sighting, accel jerk multiplier)
      (0,    0.1),
      (2000, 1.0)])
    time_since_lead_seen_multiplier = _interp_map(time_since_lead_seen_ms, time_since_lead_seen_multipliers)
    # Allow higher jerk at low speed, to get started
    accel_jerk_by_speed = OrderedDict([
      # (Speed in m/s, accel jerk)
      (0,  0.18),
      (12, 0.10)])
    accel_jerk = _interp_map(v_ego, accel_jerk_by_speed)
    accel_jerk *= near_max_speed_multiplier * time_since_lead_seen_multiplier
    return decel_jerk, accel_jerk
