from selfdrive.services import service_list
from selfdrive.car.tesla.values import AH, CruiseButtons, CruiseState, CAR
from selfdrive.config import Conversions as CV
import selfdrive.messaging as messaging
import os
import collections
import subprocess
import sys
import time
import zmq
  

class ACCState(object):
  # Possible states of the ACC system, following the DI_cruiseState naming
  # scheme.
  OFF = 0         # Disabled by UI.
  STANDBY = 1     # Ready to be enaged.
  ENABLED = 2     # Engaged.
  NOT_READY = 9   # Not ready to be engaged due to the state of the car.
  
class _Mode(object):
  def __init__(self, label, autoresume, state):
    self.label = label
    self.autoresume = autoresume
    self.state = state
    self.next = None
  
class ACCMode(object):
  # Possible ACC modes, controlling how ACC behaves.
  # This is separate from ACC state. For example, you could
  # have ACC in "Autoresume" mode in "Standby" state.
  OFF  = _Mode(label="off",  autoresume=False, state=ACCState.OFF)
  ON   = _Mode(label="on",   autoresume=False, state=ACCState.STANDBY)
  AUTO = _Mode(label="auto", autoresume=True,  state=ACCState.STANDBY)
  
  BUTTON_NAME = 'acc'
  BUTTON_ABREVIATION = 'ACC'
  
  # Toggle order: OFF -> ON -> AUTO -> OFF
  _all_modes = [OFF, ON, AUTO]
  for index, mode in enumerate(_all_modes):
    mode.next = _all_modes[(index + 1) % len(_all_modes)]
    
  # Map labels to modes for fast lookup by label.
  _label_to_mode = {mode.label: mode for mode in _all_modes}
  @ classmethod
  def from_label(cls, label):
    return cls._label_to_mode.get(label, cls.OFF)  # Default to OFF.
      
  @ classmethod
  def labels(cls):
    return [mode.label for mode in cls._all_modes]

def _current_time_millis():
  return int(round(time.time() * 1000))


class ACCController(object):
  
  # Tesla cruise only functions above 17 MPH
  MIN_CRUISE_SPEED_MS = 17.1 * CV.MPH_TO_MS
    
  def __init__(self):
    self.human_cruise_action_time = 0
    self.automated_cruise_action_time = 0
    context = zmq.Context()
    self.poller = zmq.Poller()
    self.live20 = messaging.sub_sock(context, service_list['live20'].port, conflate=True, poller=self.poller)
    self.last_update_time = 0
    self.enable_adaptive_cruise = False
    self.prev_enable_adaptive_cruise = False
    # Whether to re-engage automatically after being paused due to low speed or
    # user-initated deceleration.
    self.autoresume = False
    self.last_cruise_stalk_pull_time = 0
    self.prev_cruise_buttons = CruiseButtons.IDLE
    self.prev_pcm_acc_status = 0
    self.acc_speed_kph = 0.
    self.user_has_braked = False
    self.has_gone_below_min_speed = False
    self.fast_decel_time = 0
    self.lead_last_seen_time_ms = 0

  # Updates the internal state of this controller based on user input,
  # specifically the steering wheel mounted cruise control stalk, and OpenPilot
  # UI buttons.
  def update_stat(self, CS, enabled):
    # Check if the cruise stalk was double pulled, indicating that adaptive
    # cruise control should be enabled. Twice in .75 seconds counts as a double
    # pull.
    self.prev_enable_adaptive_cruise = self.enable_adaptive_cruise
    acc_string = CS.cstm_btns.get_button_label2(ACCMode.BUTTON_NAME)
    acc_mode = ACCMode.from_label(acc_string)
    CS.cstm_btns.get_button(ACCMode.BUTTON_NAME).btn_label2 = acc_mode.label
    self.autoresume = acc_mode.autoresume
    curr_time_ms = _current_time_millis()
    # Handle pressing the enable button.
    if (CS.cruise_buttons == CruiseButtons.MAIN and
        self.prev_cruise_buttons != CruiseButtons.MAIN):
      double_pull = curr_time_ms - self.last_cruise_stalk_pull_time < 750
      self.last_cruise_stalk_pull_time = curr_time_ms
      ready = (CS.cstm_btns.get_button_status(ACCMode.BUTTON_NAME) > ACCState.OFF
               and enabled
               and CruiseState.is_enabled_or_standby(CS.pcm_acc_status)
               and CS.v_ego > self.MIN_CRUISE_SPEED_MS)
      if ready and double_pull:
        # A double pull enables ACC. updating the max ACC speed if necessary.
        self.enable_adaptive_cruise = True
        # Increase ACC speed to match current, if applicable.
        self.acc_speed_kph = max(CS.v_ego_raw * CV.MS_TO_KPH, self.acc_speed_kph)
        self.user_has_braked = False
        self.has_gone_below_min_speed = False
      else:
        # A single pull disables ACC (falling back to just steering).
        self.enable_adaptive_cruise = False
    # Handle pressing the cancel button.
    elif CS.cruise_buttons == CruiseButtons.CANCEL:
      self.enable_adaptive_cruise = False
      self.acc_speed_kph = 0. 
      self.last_cruise_stalk_pull_time = 0
    # Handle pressing up and down buttons.
    elif (self.enable_adaptive_cruise and
          CS.cruise_buttons != self.prev_cruise_buttons):
      self._update_max_acc_speed(CS)
      
    if CS.brake_pressed:
      self.user_has_braked = True
      if not self.autoresume:
        self.enable_adaptive_cruise = False
        
    if CS.v_ego < self.MIN_CRUISE_SPEED_MS:
      self.has_gone_below_min_speed = True
    
    # If autoresume is not enabled, manually steering or slowing disables ACC.
    if not self.autoresume:
      if not enabled or self.user_has_braked or self.has_gone_below_min_speed:
        self.enable_adaptive_cruise = False
    
    # Notify if ACC was toggled
    if self.prev_enable_adaptive_cruise and not self.enable_adaptive_cruise:
      CS.UE.custom_alert_message(3, "ACC Disabled", 150, 4)
      CS.cstm_btns.set_button_status(ACCMode.BUTTON_NAME, ACCState.STANDBY)
    elif self.enable_adaptive_cruise:
      CS.cstm_btns.set_button_status(ACCMode.BUTTON_NAME, ACCState.ENABLED)
      if not self.prev_enable_adaptive_cruise:
        CS.UE.custom_alert_message(2, "ACC Enabled", 150)

    # Update the UI to show whether the current car state allows ACC.
    if CS.cstm_btns.get_button_status(ACCMode.BUTTON_NAME) in [ACCState.STANDBY, ACCState.NOT_READY]:
      if (enabled
          and CruiseState.is_enabled_or_standby(CS.pcm_acc_status)
          and CS.v_ego > self.MIN_CRUISE_SPEED_MS):
        CS.cstm_btns.set_button_status(ACCMode.BUTTON_NAME, ACCState.STANDBY)
      else:
        CS.cstm_btns.set_button_status(ACCMode.BUTTON_NAME, ACCState.NOT_READY)
          
    # Update prev state after all other actions.
    self.prev_cruise_buttons = CS.cruise_buttons
    self.prev_pcm_acc_status = CS.pcm_acc_status
    
  def _update_max_acc_speed(self, CS):
    # Adjust the max ACC speed based on user cruise stalk actions.
    half_press_kph, full_press_kph = self._get_cc_units_kph(CS.imperial_speed_units)
    speed_change_map = {
      CruiseButtons.RES_ACCEL:     half_press_kph,
      CruiseButtons.RES_ACCEL_2ND: full_press_kph,
      CruiseButtons.DECEL_SET:     -1 * half_press_kph,
      CruiseButtons.DECEL_2ND:     -1 * full_press_kph
    }
    self.acc_speed_kph += speed_change_map.get(CS.cruise_buttons, 0)

    # Clip ACC speed between 0 and 170 KPH.
    self.acc_speed_kph = min(self.acc_speed_kph, 170)
    self.acc_speed_kph = max(self.acc_speed_kph, 0)
    
  # Decide which cruise control buttons to simluate to get the car to the
  # desired speed.
  def update_acc(self, enabled, CS, frame, actuators, pcm_speed):
    # Adaptive cruise control
    current_time_ms = _current_time_millis()
    if CruiseButtons.should_be_throttled(CS.cruise_buttons):
      self.human_cruise_action_time = current_time_ms
    button_to_press = None
    
    # If ACC is disabled, disengage traditional cruise control.
    if (self.prev_enable_adaptive_cruise and not self.enable_adaptive_cruise
        and CS.pcm_acc_status == CruiseState.ENABLED):
      button_to_press = CruiseButtons.CANCEL

    if self.enable_adaptive_cruise and enabled:
      if CS.cstm_btns.get_button_label2(ACCMode.BUTTON_NAME) in ["OP", "AutoOP"]:    
        button_to_press = self._calc_button(CS, pcm_speed)
      else:
        # Alternative speed decision logic that uses the lead car's distance
        # and speed more directly.
        # Bring in the lead car distance from the Live20 feed
        lead_1 = None
        if enabled:
          for socket, _ in self.poller.poll(0):
            if socket is self.live20:
              lead_1 = messaging.recv_one(socket).live20.leadOne
              if lead_1.dRel:
                self.lead_last_seen_time_ms = current_time_ms
        button_to_press = self._calc_follow_button(CS, lead_1)
    if button_to_press:
      self.automated_cruise_action_time = current_time_ms
      # If trying to slow below the min cruise speed, just cancel cruise.
      # This prevents a SCCM crash which is triggered by repeatedly pressing
      # stalk-down when already at min cruise speed.
      if (CruiseButtons.is_decel(button_to_press)
          and CS.v_cruise_actual - 1 < self.MIN_CRUISE_SPEED_MS * CV.MS_TO_KPH):
        button_to_press = CruiseButtons.CANCEL
      if button_to_press == CruiseButtons.CANCEL:
        self.fast_decel_time = current_time_ms
      # Debug logging (disable in production to reduce latency of commands)
      #print "***ACC command: %s***" % button_to_press
    return button_to_press

  # function to calculate the cruise button based on a safe follow distance
  def _calc_follow_button(self, CS, lead_car):
    if lead_car is None:
      return None
    # Desired gap (in seconds) between cars.
    follow_time_s = 2.0
    # v_ego is in m/s, so safe_dist_m is in meters.
    safe_dist_m = CS.v_ego * follow_time_s
    current_time_ms = _current_time_millis()
     # Make sure we were able to populate lead_1.
    # dRel is in meters.
    lead_dist_m = lead_car.dRel
    lead_speed_kph = (lead_car.vRel + CS.v_ego) * CV.MS_TO_KPH
    # Relative velocity between the lead car and our set cruise speed.
    future_vrel_kph = lead_speed_kph - CS.v_cruise_actual
    # How much we can accelerate without exceeding the max allowed speed.
    available_speed_kph = self.acc_speed_kph - CS.v_cruise_actual
    half_press_kph, full_press_kph = self._get_cc_units_kph(CS.imperial_speed_units)
    # button to issue
    button = None
    # debug msg
    msg = None

    # Automatically engage traditional cruise if ACC is active.
    if self._should_autoengage_cc(CS, lead_car=lead_car) and self._no_action_for(milliseconds=100):
      button = CruiseButtons.RES_ACCEL
    # If traditional cruise is engaged, then control it.
    elif CS.pcm_acc_status == CruiseState.ENABLED:
      
      # Disengage cruise control if a slow object is seen ahead. This triggers
      # full regen braking, which is stronger than the braking that happens if
      # you just reduce cruise speed.
      if self._fast_decel_required(CS, lead_car) and self._no_human_action_for(milliseconds=500):
        msg = "Off (Slow traffic)"
        button = CruiseButtons.CANCEL
        
      # if cruise is set to faster than the max speed, slow down
      elif CS.v_cruise_actual > self.acc_speed_kph and self._no_action_for(milliseconds=300):
        msg =  "Slow to max"
        button = CruiseButtons.DECEL_SET
        
      elif (# if we have a populated lead_distance
            lead_dist_m > 0
            and self._no_action_for(milliseconds=300)
            # and we're moving
            and CS.v_cruise_actual > full_press_kph):
        ### Slowing down ###
        # Reduce speed significantly if lead_dist < safe dist
        # and if the lead car isn't already pulling away.
        if lead_dist_m < safe_dist_m * .5 and future_vrel_kph < 2:
          msg =  "-5 (Significantly too close)"
          button = CruiseButtons.DECEL_2ND
        # Don't rush up to lead car
        elif future_vrel_kph < -15:
          msg =  "-5 (approaching too fast)"
          button = CruiseButtons.DECEL_2ND
        elif future_vrel_kph < -8:
          msg =  "-1 (approaching too fast)"
          button = CruiseButtons.DECEL_SET
        elif lead_dist_m < safe_dist_m and future_vrel_kph <= 0:
          msg =  "-1 (Too close)"
          button = CruiseButtons.DECEL_SET
        # Make slow adjustments if close to the safe distance.
        # only adjust every 1 secs
        elif (lead_dist_m < safe_dist_m * 1.3
              and future_vrel_kph < -1 * half_press_kph
              and self._no_action_for(milliseconds=1000)):
          msg =  "-1 (Near safe distance)"
          button = CruiseButtons.DECEL_SET

        ### Speed up ###
        elif (available_speed_kph > half_press_kph
              and lead_dist_m > safe_dist_m
              and self._no_human_action_for(milliseconds=1000)):
          lead_is_far = lead_dist_m > safe_dist_m * 1.75
          closing = future_vrel_kph < -2
          lead_is_pulling_away = future_vrel_kph > 4
          if lead_is_far and not closing or lead_is_pulling_away:
            msg =  "+1 (Beyond safe distance and speed)"
            button = CruiseButtons.RES_ACCEL
          
      # If lead_dist is reported as 0, no one is detected in front of you so you
      # can speed up. Only accel on straight-aways; vision radar often
      # loses lead car in a turn.
      elif (lead_dist_m == 0
            and CS.angle_steers < 2.0
            and half_press_kph < available_speed_kph
            and self._no_action_for(milliseconds=500)
            and self._no_human_action_for(milliseconds=1000)
            and current_time_ms > self.lead_last_seen_time_ms + 4000):
          msg =  "+1 (road clear)"
          button = CruiseButtons.RES_ACCEL

    if (current_time_ms > self.last_update_time + 1000):
      ratio = 0
      if safe_dist_m > 0:
        ratio = (lead_dist_m / safe_dist_m) * 100
      print "Ratio: {0:.1f}%  lead: {1:.1f}m  avail: {2:.1f}kph  vRel: {3:.1f}kph  Angle: {4:.1f}deg".format(
        ratio, lead_dist_m, available_speed_kph, lead_car.vRel * CV.MS_TO_KPH, CS.angle_steers)
      self.last_update_time = current_time_ms
      if msg != None:
        print "ACC: " + msg
    return button
    
  def _should_autoengage_cc(self, CS, lead_car=None):
    # Automatically (re)engage cruise control so long as 
    # 1) The carstate allows cruise control
    # 2) There is no imminent threat of collision
    # 3) The user did not cancel ACC by pressing the brake
    cruise_ready = (self.enable_adaptive_cruise
                    and CS.pcm_acc_status == CruiseState.STANDBY
                    and CS.v_ego >= self.MIN_CRUISE_SPEED_MS
                    and _current_time_millis() > self.fast_decel_time + 2000)
                    
    slow_lead = lead_car and lead_car.dRel > 0 and lead_car.vRel < 0 or self._fast_decel_required(CS, lead_car)
    
    # "Autoresume" mode allows cruise to engage even after brake events, but
    # shouldn't trigger DURING braking.
    autoresume_ready = self.autoresume and CS.a_ego >= 0.1
    
    braked = self.user_has_braked or self.has_gone_below_min_speed
    
    return cruise_ready and not slow_lead and (autoresume_ready or not braked)
    
  def _fast_decel_required(self, CS, lead_car):
    """ Identifies situations which call for rapid deceleration. """
    if not lead_car or not lead_car.dRel:
      return False

    collision_imminent = self._seconds_to_collision(CS, lead_car) < 4
    
    lead_absolute_speed_ms = lead_car.vRel + CS.v_ego
    lead_too_slow = lead_absolute_speed_ms < self.MIN_CRUISE_SPEED_MS
    
    return collision_imminent or lead_too_slow
    
  def _seconds_to_collision(self, CS, lead_car):
    if not lead_car or not lead_car.dRel:
      return sys.maxint
    elif lead_car.vRel >= 0:
      return sys.maxint
    return abs(float(lead_car.dRel) / lead_car.vRel)
    
  def _get_cc_units_kph(self, is_imperial_units):
    # Cruise control buttons behave differently depending on whether the car
    # is configured for metric or imperial units.
    if is_imperial_units:
      # Imperial unit cars adjust cruise in units of 1 and 5 mph.
      half_press_kph = 1 * CV.MPH_TO_KPH
      full_press_kph = 5 * CV.MPH_TO_KPH
    else:
      # Metric cars adjust cruise in units of 1 and 5 kph.
      half_press_kph = 1
      full_press_kph = 5
    return half_press_kph, full_press_kph
    
  # Adjust speed based off OP's longitudinal model. As of OpenPilot 0.5.3, this
  # is inoperable because the planner crashes when given only visual radar
  # inputs. (Perhaps this can be used in the future with a radar install, or if
  # OpenPilot planner changes.)
  def _calc_button(self, CS, desired_speed_ms):
    button_to_press = None
    # Automatically engange traditional cruise if appropriate.
    if self._should_autoengage_cc(CS) and desired_speed_ms >= CS.v_ego:
      button_to_press = CruiseButtons.RES_ACCEL
    # If traditional cruise is engaged, then control it.
    elif (CS.pcm_acc_status == CruiseState.ENABLED
          # But don't make adjustments if a human has manually done so in
          # the last 3 seconds. Human intention should not be overridden.
          and self._no_human_action_for(milliseconds=3000)
          and self._no_automated_action_for(milliseconds=500)):
      # The difference between OP's target speed and the current cruise
      # control speed, in KPH.
      speed_offset_kph = (desired_speed_ms * CV.MS_TO_KPH - CS.v_cruise_actual)
    
      half_press_kph, full_press_kph = self._get_cc_units_kph(CS.imperial_speed_units)
      
      # Reduce cruise speed significantly if necessary. Multiply by a % to
      # make the car slightly more eager to slow down vs speed up.
      if desired_speed_ms < self.MIN_CRUISE_SPEED_MS:
        button_to_press = CruiseButtons.CANCEL
      if speed_offset_kph < -2 * full_press_kph and CS.v_cruise_actual > 0:
        button_to_press = CruiseButtons.CANCEL
      elif speed_offset_kph < -0.6 * full_press_kph and CS.v_cruise_actual > 0:
        # Send cruise stalk dn_2nd.
        button_to_press = CruiseButtons.DECEL_2ND
      # Reduce speed slightly if necessary.
      elif speed_offset_kph < -0.9 * half_press_kph and CS.v_cruise_actual > 0:
        # Send cruise stalk dn_1st.
        button_to_press = CruiseButtons.DECEL_SET
      # Increase cruise speed if possible.
      elif CS.v_ego > self.MIN_CRUISE_SPEED_MS:
        # How much we can accelerate without exceeding max allowed speed.
        available_speed_kph = self.acc_speed_kph - CS.v_cruise_actual
        if speed_offset_kph >= full_press_kph and full_press_kph < available_speed_kph:
          # Send cruise stalk up_2nd.
          button_to_press = CruiseButtons.RES_ACCEL_2ND
        elif speed_offset_kph >= half_press_kph and half_press_kph < available_speed_kph:
          # Send cruise stalk up_1st.
          button_to_press = CruiseButtons.RES_ACCEL
    return button_to_press
    
  def _no_human_action_for(self, milliseconds):
    return _current_time_millis() > self.human_cruise_action_time + milliseconds
    
  def _no_automated_action_for(self, milliseconds):
    return _current_time_millis() > self.automated_cruise_action_time + milliseconds
    
  def _no_action_for(self, milliseconds):
    return self._no_human_action_for(milliseconds) and self._no_automated_action_for(milliseconds)