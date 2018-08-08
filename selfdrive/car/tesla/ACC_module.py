from selfdrive.services import service_list
from selfdrive.car.tesla.values import AH, CruiseButtons, CAR
from selfdrive.config import Conversions as CV
import selfdrive.messaging as messaging
import custom_alert as customAlert
import os
import subprocess
import time
import zmq

def _current_time_millis():
  return int(round(time.time() * 1000))

class ACCController(object):
  def __init__(self,carcontroller):
    self.cc = carcontroller
    self.human_cruise_action_time = 0
    self.automated_cruise_action_time = 0
    self.last_angle = 0.
    context = zmq.Context()
    self.poller = zmq.Poller()
    self.live20 = messaging.sub_sock(context, service_list['live20'].port, conflate=True, poller=self.poller)
    self.lead_1 = None
    self.last_update_time = 0
    self.enable_adaptive_cruise = False
    self.last_cruise_stalk_pull_time = 0
    self.prev_steering_wheel_stalk = None
    self.prev_cruise_buttons = CruiseButtons.IDLE
    self.prev_cruise_setting = CruiseButtons.IDLE

  #function to calculate the desired cruise speed based on a safe follow distance
  def calc_follow_speed(self, CS):
    follow_time = 2.5 #in seconds
    current_time_ms = int(round(time.time() * 1000))
    #make sure we were able to populate lead_1
    if self.lead_1 is None:
      return None
    #dRel is in meters
    lead_dist = self.lead_1.dRel
    #grab the relative speed and convert from m/s to kph
    rel_speed = self.lead_1.vRel * 3.6
    #current speed in kph
    cur_speed = CS.v_ego * 3.6
    #v_ego is in m/s, so safe_distance is in meters
    safe_dist = CS.v_ego * follow_time
    # How much we can accelerate without exceeding the max allowed speed.
    available_speed = CS.v_cruise_pcm - CS.v_cruise_actual
    # Tesla cruise only functions above 18 MPH
    min_cruise_speed = 18 * CV.MPH_TO_MS
    # Metric cars adjust cruise in units of 1 and 5 kph
    half_press_kph = 1
    full_press_kph = 5
    # Imperial unit cars adjust cruise in units of 1 and 5 mph
    if CS.imperial_speed_units:
      half_press_kph = 1 * CV.MPH_TO_KPH
      full_press_kph = 5 * CV.MPH_TO_KPH
    #button to issue
    button = None
    #debug msg
    msg = None

    #print "dRel: ", self.lead_1.dRel," yRel: ", self.lead_1.yRel, " vRel: ", self.lead_1.vRel, " aRel: ", self.lead_1.aRel, " vLead: ", self.lead_1.vLead, " vLeadK: ", self.lead_1.vLeadK, " aLeadK: ",     self.lead_1.aLeadK

    ###   Logic to determine best cruise speed ###

    # Automatically engange traditional cruise if it is idle and we are
    # going fast enough.
    if (CS.pcm_acc_status == 1
        and CS.enable_adaptive_cruise
        and CS.v_ego > min_cruise_speed):
      button = CruiseButtons.DECEL_SET
    # If traditional cruise is engaged, then control it.
    elif CS.pcm_acc_status == 2:
      #if lead_dist is reported as 0, no one is detected in front of you so you can speed up
      #don't speed up when steer-angle > 2; vision radar often loses lead car in a turn
      if lead_dist == 0 and CS.enable_adaptive_cruise and CS.angle_steers < 2.0:
        if full_press_kph < (available_speed * 0.9): 
          msg =  "5 MPH UP   full: ","{0:.1f}kph".format(full_press_kph), "  avail: {0:.1f}kph".format(available_speed)
          button = CruiseButtons.RES_ACCEL_2ND
        elif half_press_kph < (available_speed * 0.8):
          msg =  "1 MPH UP   half: ","{0:.1f}kph".format(half_press_kph), "  avail: {0:.1f}kph".format(available_speed)
          button = CruiseButtons.RES_ACCEL

      #if we have a populated lead_distance
      elif (lead_dist > 0
            #and we only issue commands every 300ms
            and current_time_ms > self.automated_cruise_action_time + 300):
        ### Slowing down ###
        #Reduce speed significantly if lead_dist < 50% of safe dist, no matter the rel_speed
        if lead_dist < (safe_dist * 0.5):
          msg =  "50pct down"
          button = CruiseButtons.DECEL_2ND
        #Reduce speed significantly if lead_dist < 60% of  safe dist
        #and if the lead car isn't pulling away
        elif lead_dist < (safe_dist * 0.7) and rel_speed < 5:
          msg =  "70pct down"
          button = CruiseButtons.DECEL_2ND
        #Reduce speed if rel_speed < -15kph so you don't rush up to lead car
        elif rel_speed < -15:
          msg =  "relspd -15 down"
          button = CruiseButtons.DECEL_SET
        #we're close to the safe distance, so make slow adjustments
        #only adjust every 1 secs
        elif (lead_dist < (safe_dist * 0.9) and rel_speed < 0
              and current_time_ms > self.automated_cruise_action_time + 1000):
          msg =  "90pct down"
          button = CruiseButtons.DECEL_SET

        ### Speed up ###
        #don't speed up again until you have more than a safe distance in front
        #only adjust every 2 sec
        elif (lead_dist > safe_dist * 1.2 and half_press_kph < available_speed * 0.8
              and current_time_ms > self.automated_cruise_action_time + 2000):
          msg =  "120pct UP   half: ","{0:.1f}kph".format(half_press_kph), "  avail: {0:.1f}kph".format(available_speed)
          button = CruiseButtons.RES_ACCEL

      #if we don't need to do any of the above, then we're at a pretty good speed
      #make sure if we're at this point that the set cruise speed isn't set too low or high
      if (cur_speed - CS.v_cruise_actual) > 5 and button == None:
        # Send cruise stalk up_1st if the set speed is too low to bring it up
        msg =  "cruise rectify"
        button = CruiseButtons.RES_ACCEL

    
    if (current_time_ms > self.last_update_time + 1000):
      #print "Lead Dist: ", "{0:.1f}".format(lead_dist*3.28), "ft Safe Dist: ", "{0:.1f}".format(safe_dist*3.28), "ft Rel Speed: ","{0:.1f}".format(rel_speed), "kph   SpdOffset: ", "{0:.3f}".format(speed_delta * 1.01)
      ratio = 0
      if safe_dist > 0:
        ratio = (lead_dist / safe_dist) * 100
      print "Ratio: {0:.1f}%".format(ratio), "   lead: ","{0:.1f}m".format(lead_dist),"   avail: ","{0:.1f}kph".format(available_speed), "   Rel Speed: ","{0:.1f}kph".format(rel_speed), "  Angle: {0:.1f}deg".format(CS.angle_steers)
      self.last_update_time = current_time_ms
      if msg != None:
        print msg
        
    return button

  def update_stat(self,CS, enabled):
    # Check if the cruise stalk was double pulled, indicating that adaptive
    # cruise control should be enabled. Twice in .75 seconds counts as a double
    # pull.
    adaptive_cruise_prev = self.enable_adaptive_cruise
    curr_time_ms = _current_time_millis()
    if (CS.cruise_buttons == CruiseButtons.MAIN and
        self.prev_cruise_buttons != CruiseButtons.MAIN):
      self.enable_adaptive_cruise = (curr_time_ms - self.last_cruise_stalk_pull_time < 750) and \
        (CS.cstm_btns.get_button_status("acc")>0) and enabled and \
         ((CS.pcm_acc_status == 2) or (CS.pcm_acc_status == 1))
      if(self.enable_adaptive_cruise):
        customAlert.custom_alert_message("ACC Enabled",self,150)
        CS.cstm_btns.set_button_status("acc",2)
      elif adaptive_cruise_prev == True:
        customAlert.custom_alert_message("ACC Disabled",self,150)
        CS.cstm_btns.set_button_status("acc",1)

      self.last_cruise_stalk_pull_time = curr_time_ms
    elif (CS.cruise_buttons == CruiseButtons.CANCEL and
          self.prev_cruise_buttons != CruiseButtons.CANCEL):
      self.enable_adaptive_cruise = False
      if adaptive_cruise_prev == True:
        customAlert.custom_alert_message("ACC Disabled",self,150)
        CS.cstm_btns.set_button_status("acc",1)
      self.last_cruise_stalk_pull_time = 0
    #if ACC was on and something disabled cruise control, disable ACC too
    elif (self.enable_adaptive_cruise == True and
          CS.pcm_acc_status != 2 and
          curr_time_ms - self.last_cruise_stalk_pull_time >  2000):
      self.enable_adaptive_cruise = False
      customAlert.custom_alert_message("ACC Disabled",self,150)
      CS.cstm_btns.set_button_status("acc",1)
    self.prev_steering_wheel_stalk = CS.steering_wheel_stalk
    #now let's see if the ACC is available
    if (CS.cstm_btns.get_button_status("acc")==1) or (CS.cstm_btns.get_button_status("acc")==9):
        if enabled and ((CS.pcm_acc_status == 2) or (CS.pcm_acc_status == 1)):
            CS.cstm_btns.set_button_status("acc",1)
        else:
            CS.cstm_btns.set_button_status("acc",9)



  def update_acc(self,enabled,CS,frame,actuators,pcm_speed):
    # Adaptive cruise control
    #Bring in the lead car distance from the Live20 feed
    l20 = None
    if enabled:
        for socket, event in self.poller.poll(0):
            if socket is self.live20:
                l20 = messaging.recv_one(socket)
    if l20 is not None:
        self.lead_1 = l20.live20.leadOne

    current_time_ms = int(round(time.time() * 1000))
    if CS.cruise_buttons not in [CruiseButtons.IDLE, CruiseButtons.MAIN]:
        self.human_cruise_action_time = current_time_ms
    button_to_press = None
    # The difference between OP's target speed and the current cruise
    # control speed, in KPH.
    speed_offset = (pcm_speed * CV.MS_TO_KPH - CS.v_cruise_actual)
    # Tesla cruise only functions above 18 MPH
    min_cruise_speed = 18 * CV.MPH_TO_MS

    if (CS.enable_adaptive_cruise
        # Only do ACC if OP is steering
        and enabled
        # And adjust infrequently, since sending repeated adjustments makes
        # the car think we're doing a 'long press' on the cruise stalk,
        # resulting in small, jerky speed adjustments.
        and current_time_ms > self.automated_cruise_action_time + 1000):
        # Automatically engange traditional cruise if it is idle and we are
        # going fast enough and we are accelerating.
        if (CS.pcm_acc_status == 1
            and CS.v_ego > min_cruise_speed
            and CS.a_ego > 0.1):
            button_to_press = CruiseButtons.DECEL_2ND
        # If traditional cruise is engaged, then control it.
        elif (CS.pcm_acc_status == 2
                # But don't make adjustments if a human has manually done so in
                # the last 3 seconds. Human intention should not be overridden.
                and current_time_ms > self.human_cruise_action_time + 3000):
            
            if CS.imperial_speed_units:
                # Imperial unit cars adjust cruise in units of 1 and 5 mph.
                half_press_kph = 1 * CV.MPH_TO_KPH
                full_press_kph = 5 * CV.MPH_TO_KPH
            else:
                # Metric cars adjust cruise in units of 1 and 5 kph.
                half_press_kph = 1
                full_press_kph = 5
            
            # Reduce cruise speed significantly if necessary.
            if speed_offset < (-1 * full_press_kph):
                # Send cruise stalk dn_2nd.
                button_to_press = CruiseButtons.DECEL_2ND
            # Reduce speed slightly if necessary.
            elif speed_offset < (-1 * half_press_kph):
                # Send cruise stalk dn_1st.
                button_to_press = CruiseButtons.DECEL_SET
            # Increase cruise speed if possible.
            elif CS.v_ego > min_cruise_speed:
                # How much we can accelerate without exceeding max allowed speed.
                available_speed = CS.v_cruise_pcm - CS.v_cruise_actual
                if speed_offset > full_press_kph and speed_offset < available_speed:
                    # Send cruise stalk up_2nd.
                    button_to_press = CruiseButtons.RES_ACCEL_2ND
                elif speed_offset > half_press_kph and speed_offset < available_speed:
                    # Send cruise stalk up_1st.
                    button_to_press = CruiseButtons.RES_ACCEL
        if CS.btns[1].btn_label2 == "Mod JJ":
            button_to_press = self.calc_follow_speed(CS)
    if button_to_press:
        self.automated_cruise_action_time = current_time_ms
    return button_to_press
