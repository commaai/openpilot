from selfdrive.services import service_list
from selfdrive.car.tesla.values import ACCState, AH, CruiseButtons, CruiseState, CAR, GetAccMode
from selfdrive.config import Conversions as CV
import selfdrive.messaging as messaging
import os
import subprocess
import time
import zmq
  

def _current_time_millis():
  return int(round(time.time() * 1000))


class ACCController(object):
  
  # Tesla cruise only functions above 17 MPH
  MIN_CRUISE_SPEED_MS = 17.5 * CV.MPH_TO_MS
    
  def __init__(self, carcontroller):
    self.CC = carcontroller
    self.human_cruise_action_time = 0
    self.automated_cruise_action_time = 0
    self.enabled_time = 0
    self.last_angle = 0.
    context = zmq.Context()
    self.poller = zmq.Poller()
    self.live20 = messaging.sub_sock(context, service_list['live20'].port, conflate=True, poller=self.poller)
    self.last_update_time = 0
    self.enable_adaptive_cruise = False
    # Whether to re-engage automatically after being paused due to low speed or
    # user-initated deceleration.
    self.autoresume = False
    self.last_cruise_stalk_pull_time = 0
    self.prev_cruise_buttons = CruiseButtons.IDLE
    self.prev_pcm_acc_status = 0
    self.acc_speed_kph = 0.
    self.last_spotted_lead_time = 0.
  
  # Updates the internal state of this controller based on user input,
  # specifically the steering wheel mounted cruise control stalk, and OpenPilot
  # UI buttons.
  def update_stat(self, CS, enabled):
    # Check if the cruise stalk was double pulled, indicating that adaptive
    # cruise control should be enabled. Twice in .75 seconds counts as a double
    # pull.
    prev_enable_adaptive_cruise = self.enable_adaptive_cruise
    acc_string = CS.cstm_btns.get_button_label2("acc")
    acc_mode = GetAccMode(acc_string)
    CS.cstm_btns.get_button("acc").btn_label2 = acc_mode.name
    self.autoresume = acc_mode.autoresume
    curr_time_ms = _current_time_millis()
    half_press_kph, full_press_kph = self.get_cc_units_kph(CS.imperial_speed_units)
    # Handle pressing the enable button.
    if (CS.cruise_buttons == CruiseButtons.MAIN and
        self.prev_cruise_buttons != CruiseButtons.MAIN):
      double_pull = curr_time_ms - self.last_cruise_stalk_pull_time < 750
      self.last_cruise_stalk_pull_time = curr_time_ms
      ready = (CS.cstm_btns.get_button_status("acc") > ACCState.OFF
               and enabled
               and CruiseState.is_enabled_or_standby(CS.pcm_acc_status)
               and CS.v_ego > self.MIN_CRUISE_SPEED_MS)
      if ready and double_pull:
        # A double pull enables ACC. updating the max ACC speed if necessary.
        self.enable_adaptive_cruise = True
        self.enabled_time = curr_time_ms
        # Increase ACC speed to match current, if applicable.
        self.acc_speed_kph = max(CS.v_ego_raw * CV.MS_TO_KPH, self.acc_speed_kph)
    # Handle pressing the cancel button.
    elif CS.cruise_buttons == CruiseButtons.CANCEL:
      self.enable_adaptive_cruise = False
      self.acc_speed_kph = 0. 
      self.last_cruise_stalk_pull_time = 0
    # Handle pressing up and down buttons.
    elif (self.enable_adaptive_cruise and
          CS.cruise_buttons != self.prev_cruise_buttons):
      # Real stalk command while ACC is already enabled. Adjust the max ACC
      # speed if necessary. For example if max speed is 50 but you're currently
      # only going 30, the cruise speed can be increased without any change to
      # max ACC speed. If actual speed is already 50, the code also increases
      # the max cruise speed.
      if CS.cruise_buttons == CruiseButtons.RES_ACCEL:
        requested_speed_kph = CS.v_ego * CV.MS_TO_KPH + half_press_kph
        self.acc_speed_kph = max(self.acc_speed_kph, requested_speed_kph)
      elif CS.cruise_buttons == CruiseButtons.RES_ACCEL_2ND:
        requested_speed_kph = CS.v_ego * CV.MS_TO_KPH + full_press_kph
        self.acc_speed_kph = max(self.acc_speed_kph, requested_speed_kph)
      elif CS.cruise_buttons == CruiseButtons.DECEL_SET:
        self.acc_speed_kph -= half_press_kph
      elif CS.cruise_buttons == CruiseButtons.DECEL_2ND:
        self.acc_speed_kph -= full_press_kph
      # Clip ACC speed between 0 and 170 KPH.
      self.acc_speed_kph = min(self.acc_speed_kph, 170)
      self.acc_speed_kph = max(self.acc_speed_kph, 0)
    # If autoresume is not enabled, certain user actions disable ACC.
    elif not self.autoresume:
      # If something disabled cruise control (braking), disable ACC too.
      if self.prev_pcm_acc_status == 2 and CS.pcm_acc_status != 2:
        self.enable_adaptive_cruise = False
      # if user took over steering, disable ACC too.
      elif not enabled:
        self.enable_adaptive_cruise = False
    
    # Notify if ACC was toggled
    if prev_enable_adaptive_cruise and not self.enable_adaptive_cruise:
      CS.UE.custom_alert_message(3, "ACC Disabled", 150, 4)
      CS.cstm_btns.set_button_status("acc", ACCState.STANDBY)
    elif self.enable_adaptive_cruise and not prev_enable_adaptive_cruise:
      CS.UE.custom_alert_message(2, "ACC Enabled", 150)
      CS.cstm_btns.set_button_status("acc", ACCState.ENABLED)

    # Update the UI to show whether the current car state allows ACC.
    if CS.cstm_btns.get_button_status("acc") in [ACCState.STANDBY, ACCState.NOT_READY]:
      if (enabled
          and CruiseState.is_enabled_or_standby(CS.pcm_acc_status)
          and CS.v_ego > self.MIN_CRUISE_SPEED_MS):
        CS.cstm_btns.set_button_status("acc", ACCState.STANDBY)
      else:
        CS.cstm_btns.set_button_status("acc", ACCState.NOT_READY)
          
    # Update prev state after all other actions.
    self.prev_cruise_buttons = CS.cruise_buttons
    self.prev_pcm_acc_status = CS.pcm_acc_status
    
  # Decide which cruise control buttons to simluate to get the car to the
  # desired speed.
  def update_acc(self, enabled, CS, frame, actuators, pcm_speed):
    # Adaptive cruise control
    current_time_ms = _current_time_millis()
    if CruiseButtons.should_be_throttled(CS.cruise_buttons):
      self.human_cruise_action_time = current_time_ms
    button_to_press = None

    if (self.enable_adaptive_cruise
        # Only do ACC if OP is steering
        and enabled
        # And adjust infrequently, since sending repeated adjustments makes
        # the car think we're doing a 'long press' on the cruise stalk,
        # resulting in small, jerky speed adjustments.
        and current_time_ms > self.automated_cruise_action_time + 500):
      
      if CS.cstm_btns.get_button_label2("acc") in ["OP", "AutoOP"]:    
        button_to_press = self.calc_button(CS, pcm_speed, current_time_ms)
      else:
        # Alternative speed decision logic that uses the lead car's distance
        # and speed more directly.
        # Bring in the lead car distance from the Live20 feed
        lead_1 = None
        if enabled:
          for socket, _ in self.poller.poll(0):
            if socket is self.live20:
              lead_1 = messaging.recv_one(socket).live20.leadOne
        if CS.cstm_btns.get_button_label2("acc") in ["FOLLOW", "AUTO"]:
          button_to_press = self.calc_follow_button(CS, lead_1)
        elif CS.cstm_btns.get_button_label2("acc") in ["EXPR"]:
          button_to_press = self.calc_experimental_button(CS, lead_1, current_time_ms)
    if button_to_press:
      self.automated_cruise_action_time = current_time_ms
      # If trying to slow below the min cruise speed, just cancel cruise.
      # This prevents a SCCM crash which is triggered by repeatedly pressing
      # stalk-down when already at min cruise speed.
      if (CruiseButtons.is_decel(button_to_press)
          and CS.v_cruise_actual - 1 < self.MIN_CRUISE_SPEED_MS * CV.MS_TO_KPH):
        button_to_press = CruiseButtons.CANCEL
      # Debug logging (disable in production to reduce latency of commands)
      #print "***ACC command: %s***" % button_to_press
    #elif (current_time_ms > self.last_update_time + 1000):
    #  self.last_update_time = current_time_ms
    #  print "Desired ACC speed change: %s" % (speed_offset)
    return button_to_press
    
  def get_cc_units_kph(self, is_imperial_units):
    if is_imperial_units:
      # Imperial unit cars adjust cruise in units of 1 and 5 mph.
      half_press_kph = 1 * CV.MPH_TO_KPH
      full_press_kph = 5 * CV.MPH_TO_KPH
    else:
      # Metric cars adjust cruise in units of 1 and 5 kph.
      half_press_kph = 1
      full_press_kph = 5
    return half_press_kph, full_press_kph
    
  # Adjust speed based off OP's longitudinal model.
  def calc_button(self, CS, desired_speed_ms, current_time_ms):
    # Automatically engange traditional cruise if ACC is active.
    if self.should_autoengage_cc(CS, current_time_ms):
      button_to_press = CruiseButtons.RES_ACCEL
    # If traditional cruise is engaged, then control it.
    elif (CS.pcm_acc_status == 2
          # But don't make adjustments if a human has manually done so in
          # the last 3 seconds. Human intention should not be overridden.
          and current_time_ms > self.human_cruise_action_time + 3000
          and current_time_ms > self.enabled_time + 1000):
      # The difference between OP's target speed and the current cruise
      # control speed, in KPH.
      speed_offset = (desired_speed_ms * CV.MS_TO_KPH - CS.v_cruise_actual)
    
      half_press_kph, full_press_kph = self.get_cc_units_kph(CS.imperial_speed_units)
      
      # Reduce cruise speed significantly if necessary. Multiply by a % to
      # make the car slightly more eager to slow down vs speed up.
      if speed_offset < -0.6 * full_press_kph and CS.v_cruise_actual > 0:
        # Send cruise stalk dn_2nd.
        button_to_press = CruiseButtons.DECEL_2ND
      # Reduce speed slightly if necessary.
      elif speed_offset < -0.9 * half_press_kph and CS.v_cruise_actual > 0:
        # Send cruise stalk dn_1st.
        button_to_press = CruiseButtons.DECEL_SET
      # Increase cruise speed if possible.
      elif CS.v_ego > self.MIN_CRUISE_SPEED_MS:
        # How much we can accelerate without exceeding max allowed speed.
        available_speed = self.acc_speed_kph - CS.v_cruise_actual
        if speed_offset > full_press_kph and full_press_kph < available_speed:
          # Send cruise stalk up_2nd.
          button_to_press = CruiseButtons.RES_ACCEL_2ND
        elif speed_offset > half_press_kph and half_press_kph < available_speed:
          # Send cruise stalk up_1st.
          button_to_press = CruiseButtons.RES_ACCEL
    return button_to_press
    
  def should_autoengage_cc(self, CS, current_time_ms):
    # In auto-resume mode, we must not to engage during deceleration other than
    # during the first moment after ACC was engaged.
    autoresume_supressed_by_braking = (
      self.autoresume
      and CS.a_ego < 0
      and current_time_ms > self.enabled_time + 300)
    return (CS.pcm_acc_status == 1
            and self.enable_adaptive_cruise
            and CS.v_ego >= self.MIN_CRUISE_SPEED_MS
            and not autoresume_supressed_by_braking)

  # function to calculate the cruise button based on a safe follow distance
  def calc_follow_button(self, CS, lead_car):
    follow_time = 2.0 # in seconds
    current_time_ms = _current_time_millis()
     # Make sure we were able to populate lead_1.
    if lead_car is None:
      return None
    # dRel is in meters.
    lead_dist = lead_car.dRel
    # Grab the relative speed.
    rel_speed = lead_car.vRel * CV.MS_TO_KPH
    # Current speed in kph
    cur_speed = CS.v_ego * CV.MS_TO_KPH
    # v_ego is in m/s, so safe_dist_m is in meters.
    safe_dist_m = CS.v_ego * follow_time
    # How much we can accelerate without exceeding the max allowed speed.
    available_speed = self.acc_speed_kph - CS.v_cruise_actual
    half_press_kph, full_press_kph = self.get_cc_units_kph(CS.imperial_speed_units)
    # button to issue
    button = None
    # debug msg
    msg = None

    #print "dRel: ", lead_car.dRel," yRel: ", lead_car.yRel, " vRel: ", lead_car.vRel, " aRel: ", lead_car.aRel, " vLead: ", lead_car.vLead, " vLeadK: ", lead_car.vLeadK, " aLeadK: ",     lead_car.aLeadK

    ###   Logic to determine best cruise speed ###

    # Automatically engange traditional cruise if ACC is active.
    if self.should_autoengage_cc(CS, current_time_ms):
      button = CruiseButtons.RES_ACCEL
    # If traditional cruise is engaged, then control it.
    elif CS.pcm_acc_status == 2:
      # if cruise is set to faster than the max speed, slow down
      if CS.v_cruise_actual > self.acc_speed_kph:
        msg =  "Slow to max"
        button = CruiseButtons.DECEL_SET
      # If lead_dist is reported as 0, no one is detected in front of you so you
      # can speed up don't speed up when steer-angle > 2; vision radar often
      # loses lead car in a turn.
      elif lead_dist == 0 and self.enable_adaptive_cruise and CS.angle_steers < 2.0:
        if full_press_kph < available_speed: 
          msg =  "5 MPH UP   full: ","{0:.1f}kph".format(full_press_kph), "  avail: {0:.1f}kph".format(available_speed)
          button = CruiseButtons.RES_ACCEL_2ND
        elif half_press_kph < available_speed:
          msg =  "1 MPH UP   half: ","{0:.1f}kph".format(half_press_kph), "  avail: {0:.1f}kph".format(available_speed)
          button = CruiseButtons.RES_ACCEL

      # if we have a populated lead_distance
      elif (lead_dist > 0
            # and we only issue commands every 300ms
            and current_time_ms > self.automated_cruise_action_time + 300):
        ### Slowing down ###
        # Reduce speed significantly if lead_dist < 50% of safe dist, no matter
        # the rel_speed
        if CS.v_cruise_actual > full_press_kph:
          if lead_dist < (safe_dist_m * 0.3) and rel_speed < 2:
            msg =  "50pct down"
            button = CruiseButtons.DECEL_2ND
          # Reduce speed significantly if lead_dist < 60% of  safe dist
          # and if the lead car isn't pulling away
          elif lead_dist < (safe_dist_m * 0.5) and rel_speed < 0:
            msg =  "70pct down"
            button = CruiseButtons.DECEL_SET
           #Reduce speed if rel_speed < -15kph so you don't rush up to lead car
          elif rel_speed < -15:
            msg =  "relspd -15 down"
            button = CruiseButtons.DECEL_SET
          # we're close to the safe distance, so make slow adjustments
          # only adjust every 1 secs
          elif (lead_dist < (safe_dist_m * 0.9) and rel_speed < 0
                and current_time_ms > self.automated_cruise_action_time + 1000):
            msg =  "90pct down"
            button = CruiseButtons.DECEL_SET

        ### Speed up ###
        # don't speed up again until you have more than a safe distance in front
        # only adjust every 2 sec
        if ((lead_dist > (safe_dist_m * 0.8) or rel_speed > 5) and half_press_kph < available_speed
              and current_time_ms > self.automated_cruise_action_time + 100):
          msg =  "120pct UP   half: ","{0:.1f}kph".format(half_press_kph), "  avail: {0:.1f}kph".format(available_speed)
          button = CruiseButtons.RES_ACCEL

      # if we don't need to do any of the above, then we're at a pretty good
      # speed make sure if we're at this point that the set cruise speed isn't
      # set too low or high
      if (cur_speed - CS.v_cruise_actual) > 5 and button == None:
        # Send cruise stalk up_1st if the set speed is too low to bring it up
        msg =  "cruise rectify"
        button = CruiseButtons.RES_ACCEL

    if (current_time_ms > self.last_update_time + 1000):
      ratio = 0
      if safe_dist_m > 0:
        ratio = (lead_dist / safe_dist_m) * 100
      print "Ratio: {0:.1f}%".format(ratio), "   lead: ","{0:.1f}m".format(lead_dist),"   avail: ","{0:.1f}kph".format(available_speed), "   Rel Speed: ","{0:.1f}kph".format(rel_speed), "  Angle: {0:.1f}deg".format(CS.angle_steers)
      self.last_update_time = current_time_ms
      if msg != None:
        print msg
        
    return button
    
  # function to calculate the cruise button based on experimental logic.
  def calc_experimental_button(self, CS, lead_car, current_time_ms):
    target_speed_ms = 0.
    
    if not (lead_car and lead_car.dRel):
      # In the absence of a lead car, maintain speed around curves and
      # accelerate to max on straightaways.
      if CS.angle_steers > 2.0:
        target_speed_ms = CS.v_ego
      elif current_time_ms > self.last_spotted_lead_time + 3000:
        target_speed_ms = self.acc_speed_kph * CV.KPH_TO_MS
    else:
      self.last_spotted_lead_time = current_time_ms
      # In the presence of a lead car, attempt to follow the 2-second rule.
      lead_absolute_speed = CS.v_ego + lead_car.vRel
      target_speed_ms = lead_absolute_speed
      min_gap_sec = 2.
      max_gap_sec = 2.5
      actual_gap_sec = lead_car.dRel / CS.v_ego
      half_press_kph, _ = self.get_cc_units_kph(CS.imperial_speed_units)
      if actual_gap_sec < min_gap_sec:
        target_speed_ms -= half_press_kph
      elif actual_gap_sec > max_gap_sec:
        target_speed_ms += half_press_kph
        
    # Clamp between 0 and max ACC speed
    target_speed_ms = max(target_speed_ms, 0)
    target_speed_ms = min(target_speed_ms, self.acc_speed_kph * CS.KPH_TO_MS)
        
    return self.calc_button(CS, target_speed_ms, current_time_ms)