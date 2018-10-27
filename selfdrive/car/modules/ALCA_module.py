"""
Copyright 2018 BB Solutions, LLC. All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
  * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
  * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in
    the documentation and/or other materials provided with the
    distribution.
  * Neither the name of Google nor the names of its contributors may
    be used to endorse or promote products derived from this software
    without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

HISTORY
-------
v3.6 - moved parameters to carstate.py
v3.5 - changing the start angle to keep turning until we reach MAX_ANGLE_DELTA
v3.4 - read steerRatio from each car parameters file
v3.3 - re-entry logic changed for smoothness
v3.2 - angle adjustment to compesate for road curvature change
v3.1 - new angle logic for a smoother re-entry
v3.0 - better lane dettection logic
v2.0 - detection of lane crossing 
v1.0 - fixed angle move



########################################################
The following parameters need to be added to carstate.py 
and based on each model you want to define
########################################################

# max REAL delta angle for correction vs actuator
CL_MAX_ANGLE_DELTA_BP = [10., 44.]
CL_MAX_ANGLE_DELTA = [2., .8]

# adjustment factor for merging steer angle to actuator; should be over 4; the higher the smoother
CL_ADJUST_FACTOR_BP = [10., 44.]
CL_ADJUST_FACTOR = [16. , 16.]
 # reenrey angle when to let go
CL_REENTRY_ANGLE_BP = [10., 44.]
CL_REENTRY_ANGLE = [5. , 5.]

# a jump in angle above the CL_LANE_DETECT_FACTOR means we crossed the line
CL_LANE_DETECT_BP = [10., 44.]
CL_LANE_DETECT_FACTOR = [1.5, .75]

CL_LANE_PASS_BP = [10., 44.]
CL_LANE_PASS_TIME = [100., 5.] 

# change lane delta angles and other params
CL_MAXD_BP = [10., 32., 44.]
CL_MAXD_A = [.358, 0.084, 0.042] #delta angle based on speed; needs fine tune, based on Tesla steer ratio of 16.75

CL_MIN_V = 8.9 # do not turn if speed less than x m/2; 20 mph = 8.9 m/s
CL_MAX_A = 10. # do not turn if actuator wants more than x deg for going straight; this should be interp based on speed

# define limits for angle change every 0.1 s
# we need to force correction above 10 deg but less than 20
# anything more means we are going to steep or not enough in a turn
MCL_MAX_ACTUATOR_DELTA = 2.
CL_MIN_ACTUATOR_DELTA = 0. 
CL_CORRECTION_FACTOR = 1.

#duration after we cross the line until we release is a factor of speed
CL_TIMEA_BP = [10., 32., 44.]
CL_TIMEA_T = [0.7 ,0.30, 0.20]

"""
from common.numpy_fast import interp
from selfdrive.controls.lib.pid import PIController


class ALCAController(object):
  def __init__(self,carcontroller,alcaEnabled,steerByAngle):
    #import settings
    self.CC = carcontroller  # added to start, will see if we need it actually
    # variables for lane change
    self.alcaEnabled = alcaEnabled
    self.laneChange_strStartFactor = 2.
    self.laneChange_strStartMultiplier = 1.5
    self.laneChange_steerByAngle = steerByAngle # steer only by angle; do not call PID
    self.laneChange_last_actuator_angle = 0.
    self.laneChange_last_actuator_delta = 0.
    self.laneChange_last_sent_angle = 0.
    self.laneChange_over_the_line = 0 # did we cross the line?
    self.laneChange_avg_angle = 0. # used if we do average entry angle over x frames
    self.laneChange_avg_count = 0. # used if we do average entry angle over x frames
    self.laneChange_enabled = 1 # set to zero for no lane change
    self.laneChange_counter = 0 # used to count frames during lane change
    self.laneChange_min_duration = 1. # min time to wait before looking for next lane
    self.laneChange_duration = 5 # how many max seconds to actually do the move; if lane not found after this then send error
    self.laneChange_after_lane_duration_mult = 1.  # multiplier for time after we cross the line before we let OP take over; multiplied with CL_TIMEA_T 
    self.laneChange_wait = 2 # how many seconds to wait before it starts the change
    self.laneChange_lw = 3.0 # lane width in meters
    self.laneChange_angle = 0. # saves the last angle from actuators before lane change starts
    self.laneChange_angled = 0. # angle delta
    self.laneChange_steerr = 16.00 # steer ratio for lane change
    self.laneChange_direction = 0 # direction of the lane change 
    self.prev_right_blinker_on = False # local variable for prev position
    self.prev_left_blinker_on = False # local variable for prev position
    self.keep_angle = False #local variable to keep certain angle delta vs. actuator
    #self.blindspot_blink_counter = 0
    self.pid = None
    self.last10delta = []

  def last10delta_reset(self):
    self.last10delta = []
    for i in range (0,10):
      self.last10delta.append(0.)

  def last10delta_add(self,angle):
    self.last10delta.pop(0)
    self.last10delta.append(angle)
    n = 0
    a = 0.
    for i in self.last10delta:
      if i != 0:
        n += 1
        a += i
    return n,a

  def last10delta_correct(self,c):
    for i in self.last10delta:
        j = self.last10delta.index(i)
        n = self.last10delta.pop(j)
        n += c
        self.last10delta.insert(j,n)


  def update_status(self,alcaEnabled):
    self.alcaEnabled = alcaEnabled

  def set_pid(self,CS):
    self.laneChange_steerr = CS.CP.steerRatio
    self.pid = PIController((CS.CP.steerKpBP, CS.CP.steerKpV),
                            (CS.CP.steerKiBP, CS.CP.steerKiV),
                            k_f=CS.CP.steerKf, pos_limit=1.0)

  def update_angle(self,enabled,CS,frame,actuators):
    alca_m1 = 1.
    alca_m2 = 1.
    alca_m3 = 1.
    if CS.alcaMode == 1:
      alca_m1 =.9 
      alca_m2 = 1.
      alca_m3 = 0.5
    if CS.alcaMode == 2:
      alca_m1 = 0.8
      alca_m2 = 1.7
      alca_m3 = 0.5
    # speed variable parameters
    cl_max_angle_delta = interp(CS.v_ego,CS.CL_MAX_ANGLE_DELTA_BP,CS.CL_MAX_ANGLE_DELTA) * alca_m1
    cl_maxd_a =  interp(CS.v_ego, CS.CL_MAXD_BP, CS.CL_MAXD_A) * alca_m3
    cl_lane_pass_time = interp(CS.v_ego,CS.CL_LANE_PASS_BP,CS.CL_LANE_PASS_TIME) * alca_m2
    cl_timea_t = interp(CS.v_ego,CS.CL_TIMEA_BP,CS.CL_TIMEA_T) * alca_m2
    cl_lane_detect_factor = interp(CS.v_ego, CS.CL_LANE_DETECT_BP, CS.CL_LANE_DETECT_FACTOR)
    cl_max_a = interp(CS.v_ego, CS.CL_MAX_A_BP, CS.CL_MAX_A)
    cl_adjust_factor = interp(CS.v_ego, CS.CL_ADJUST_FACTOR_BP, CS.CL_ADJUST_FACTOR)
    cl_reentry_angle = interp(CS.v_ego, CS.CL_REENTRY_ANGLE_BP, CS.CL_REENTRY_ANGLE)
    cl_correction_factor = interp(CS.v_ego, CS.CL_CORRECTION_FACTOR_BP, CS.CL_CORRECTION_FACTOR)
    cl_min_v = CS.CL_MIN_V
    self.laneChange_wait = CS.CL_WAIT_BEFORE_START
   
    # Basic highway lane change logic
    blindspot = CS.blind_spot_on
    #if changing_lanes:
      #if blindspot:
        #self.blindspot_blink_counter = 0
       # print "debug: blindspot detected in alca"
        #CS.UE.custom_alert_message(3,"Auto Lane Change Canceled! (s)",200,5)
        #self.laneChange_enabled = 1
      #  self.laneChange_counter = 0
        #self.laneChange_direction = 0
     # self.blindspot_blink_counter += 1
    
    #if self.blindspot_blink_counter < 150:
      #self.laneChange_enabled = 0
      #self.laneChange_counter = 0
    #else:
      #self.laneChange_enabled = 1
      #self.laneChange_counter = self.laneChange_counter
     
    #self.laneChange_steerr = CS.CP.steerRatio
    actuator_delta = 0.
    laneChange_angle = 0.
    turn_signal_needed = 0 # send 1 for left, 2 for right 0 for not needed

    if (not CS.right_blinker_on) and (not CS.left_blinker_on) and \
    (self.laneChange_enabled ==7):
        self.laneChange_enabled =1
        self.laneChange_counter =0
        self.laneChange_direction =0
        CS.UE.custom_alert_message(-1,"",0)
    
    if (not CS.right_blinker_on) and (not CS.left_blinker_on) and \
      (self.laneChange_enabled > 1):
      # no blinkers on but we are still changing lane, so we need to send blinker command
      if self.laneChange_direction == -1:
        turn_signal_needed = 1
      elif self.laneChange_direction == 1:
        turn_signal_needed = 2
      else:
        turn_signal_needed = 0


    if (CS.cstm_btns.get_button_status("alca") > 0) and self.alcaEnabled and (self.laneChange_enabled == 1):
      if ((CS.v_ego < cl_min_v) or (abs(actuators.steerAngle) >= cl_max_a) or \
      (abs(CS.angle_steers)>= cl_max_a)  or (not enabled)): 
        CS.cstm_btns.set_button_status("alca",9)
      else:
        CS.cstm_btns.set_button_status("alca",1)

    if self.alcaEnabled and enabled and (((not self.prev_right_blinker_on) and CS.right_blinker_on) or \
      ((not self.prev_left_blinker_on) and CS.left_blinker_on)) and \
      ((CS.v_ego < cl_min_v) or (abs(actuators.steerAngle) >= cl_max_a) or (abs(CS.angle_steers) >=cl_max_a)):
      # something is not right, the speed or angle is limitting
      CS.UE.custom_alert_message(3,"Auto Lane Change Unavailable!",500,3)
      CS.cstm_btns.set_button_status("alca",9)


    if self.alcaEnabled and enabled and (((not self.prev_right_blinker_on) and CS.right_blinker_on) or \
      ((not self.prev_left_blinker_on) and CS.left_blinker_on))  and \
      (CS.v_ego >= cl_min_v) and (abs(actuators.steerAngle) < cl_max_a):
      # start blinker, speed and angle is within limits, let's go
      laneChange_direction = 1
      # changing lanes
      if CS.left_blinker_on:
        laneChange_direction = -1

      if (self.laneChange_enabled > 1) and (self.laneChange_direction <> laneChange_direction):
        # something is not right; signal in oposite direction; cancel
        CS.UE.custom_alert_message(3,"Auto Lane Change Canceled! (s)",200,5)
        self.laneChange_enabled = 1
        self.laneChange_counter = 0
        self.laneChange_direction = 0
        CS.cstm_btns.set_button_status("alca",1)
      elif (self.laneChange_enabled == 1) :
        # compute angle delta for lane change
        CS.UE.custom_alert_message(2,"Auto Lane Change Engaged! (1)",100)
        self.laneChange_enabled = 5
        self.laneChange_counter = 1
        self.laneChange_direction = laneChange_direction
        self.laneChange_avg_angle = 0.
        self.laneChange_avg_count = 0.
        self.laneChange_angled = self.laneChange_direction * self.laneChange_steerr * cl_maxd_a
        self.laneChange_last_actuator_angle = 0.
        self.laneChange_last_actuator_delta = 0.
        self.laneChange_over_the_line = 0 
        CS.cstm_btns.set_button_status("alca",2)
        # reset PID for torque
        self.pid.reset()

    if (not self.alcaEnabled) and self.laneChange_enabled > 1:
      self.laneChange_enabled = 1
      self.laneChange_counter = 0
      self.laneChange_direction = 0

    # lane change in process
    if self.laneChange_enabled > 1:
      if (CS.steer_override or (CS.v_ego < cl_min_v) or (not (CS.right_blinker_on or CS.left_blinker_on))):
        CS.UE.custom_alert_message(4,"Auto Lane Change Canceled! (u)",200,3)
        # if any steer override cancel process or if speed less than min speed
        self.laneChange_counter = 0
        self.laneChange_enabled = 1
        self.laneChange_direction = 0
        CS.cstm_btns.set_button_status("alca",1)
      if blindspot:
        CS.UE.custom_alert_message(4,"Auto Lane Change Held! (b)",200,3)
        # if blindspot detected cancel process
        self.laneChange_counter = 0
        self.laneChange_enabled = 5
        #CS.cstm_btns.set_button_status("alca",1)
            
      # now that we crossed the line, we need to get far enough from the line and then engage OP
      #     1. we wait 0.05s to ensure we don't have back and forth adjustments
      #     2. we check to see if the delta between our steering and the actuator continues to decrease or not
      #     3. we check to see if the delta between our steering and the actuator is less than 5 deg
      #     4. we disengage after 0.5s and let OP take over
      if self.laneChange_enabled ==2:
        laneChange_angle = self.laneChange_angled
        if self.laneChange_counter ==1:
          CS.UE.custom_alert_message(2,"Auto Lane Change Engaged! (5)",800)
        self.laneChange_counter += 1
        # check if angle continues to decrease
        current_delta = abs(self.laneChange_angle + laneChange_angle + (-actuators.steerAngle))
        previous_delta = abs(self.laneChange_last_sent_angle  - self.laneChange_last_actuator_angle)
        if (self.laneChange_counter > cl_lane_pass_time):
          # continue to half the angle between our angle and actuator
          delta_angle = (-actuators.steerAngle - self.laneChange_angle - self.laneChange_angled)/cl_adjust_factor 
          self.laneChange_angle += delta_angle
        # wait 0.05 sec before starting to check if angle increases or if we are within X deg of actuator.angleSteer
        if ((current_delta > previous_delta) or  (current_delta <= cl_reentry_angle)) and (self.laneChange_counter > cl_lane_pass_time):
          self.laneChange_enabled = 7
          self.laneChange_counter = 1
          self.laneChange_direction = 0
        # we crossed the line, so  x sec later give control back to OP
        laneChange_after_lane_duration = cl_timea_t * self.laneChange_after_lane_duration_mult
        if (self.laneChange_counter > laneChange_after_lane_duration * 100):
          self.laneChange_enabled = 7
          self.laneChange_counter = 1
          self.laneChange_direction = 0
      # this is the main stage once we start turning
      # we have to detect when to let go control back to OP or raise alarm if max timer passed
      # there are three conditions we look for:
      #     1. we can detect when we cross the lane, and then we let go control to OP
      #     2. we passed the min timer to cross the line and the delta between actuator and our angle
      #       is less than the release angle, then we let go control to OP
      #     3. the delta between our angle and the actuator is higher than the previous one
      #       (we cross the optimal path), then we let go control to OP
      #     4. max time is achieved: alert and disengage
      # CONTROL: during this time we use ALCA angle to steer (ALCA Control)
      if self.laneChange_enabled ==3:
        if self.laneChange_counter == 1:
          CS.UE.custom_alert_message(2,"Auto Lane Change Engaged! (4)",800)
          self.laneChange_over_the_line = 0
          self.last10delta_reset()
          self.keep_angle = False
        self.laneChange_counter += 1
        laneChange_angle = self.laneChange_angled
        if (self.laneChange_over_the_line == 0):
          # we didn't cross the line, so keep computing the actuator delta until it flips
          actuator_delta = self.laneChange_direction * (-actuators.steerAngle - self.laneChange_last_actuator_angle)
          actuator_ratio = (-actuators.steerAngle)/self.laneChange_last_actuator_angle
        if (actuator_ratio < 1) and (abs(actuator_delta) > 0.5 * cl_lane_detect_factor):
          # sudden change in actuator angle or sign means we are on the other side of the line
          self.laneChange_over_the_line = 1
          self.laneChange_enabled = 2
          self.laneChange_counter = 1
          
          # didn't change the lane yet, check that we are not eversteering or understeering based on road curvature

        """
        # code for v3.1 designed to turn more if lane moves away
        n,a = self.last10delta_add(actuator_delta)
        c = 5.
        if a == 0:
          a = 0.00001
        if (abs(a) < CL_MIN_ACTUATOR_DELTA) and (self.keep_angle == False):
          c = (a/abs(a)) * (CL_MIN_ACTUATOR_DELTA - abs(a)) / 10
          self.laneChange_angle = self.laneChange_angle -self.laneChange_direction * CL_CORRECTION_FACTOR * c * 10 
          self.last10delta_correct(c)
        #print a, c, actuator_delta, self.laneChange_angle
        """
          #print a, c, actuator_delta, self.laneChange_angle
        # code for v3.2 and above
        a_delta = self.laneChange_direction * (self.laneChange_angle + laneChange_angle - (-actuators.steerAngle))
        if (self.laneChange_over_the_line == 0) and ((abs(a_delta) > cl_max_angle_delta * self.laneChange_steerr) or (self.keep_angle)):
          #steering more than what we wanted, need to adjust
          self.keep_angle = True
          self.laneChange_angle  = -actuators.steerAngle + self.laneChange_direction * cl_max_angle_delta * self.laneChange_steerr - laneChange_angle
          self.laneChange_angle  = self.laneChange_angle * (1 - self.laneChange_direction * (1 - cl_correction_factor))
        if self.laneChange_counter >  (self.laneChange_duration) * 100:
          self.laneChange_enabled = 1
          self.laneChange_counter = 0
          CS.UE.custom_alert_message(4,"Auto Lane Change Canceled! (t)",200,5)
          self.laneChange_counter = 0
          CS.cstm_btns.set_button_status("alca",1)
      # this is the critical start of the turn
      # here we will detect the angle to move; it is based on a speed determined angle but we have to also
      # take in consideration what's happening with the delta of consecutive actuators
      # if they go in the same direction with our turn we have to reset our turn angle because the actuator either
      # is compensating for a turn in the road OR for same lane correction for the car
      # CONTROL: during this time we use ALCA angle to steer (ALCA Control)

      # TODO: - when actuator moves in the same direction with lane change, correct starting angle
      if self.laneChange_enabled == 4:
        if self.laneChange_counter == 1:
          self.laneChange_angle = -actuators.steerAngle
          CS.UE.custom_alert_message(2,"Auto Lane Change Engaged! (3)",100)
          self.laneChange_angled = self.laneChange_direction * self.laneChange_steerr * cl_maxd_a
          # if angle more than max angle allowed cancel; last chance to cancel on road curvature
        if self.laneChange_counter == 2:
          CS.UE.custom_alert_message(2,"Auto Lane Change     Engaged! (3)",100)
        if (abs(self.laneChange_angle) > cl_max_a) and (self.laneChange_counter == 1):
            CS.UE.custom_alert_message(4,"Auto Lane Change Canceled! (a)",200,5)
            self.laneChange_enabled = 1
            self.laneChange_counter = 0
            self.laneChange_direction = 0
            CS.cstm_btns.set_button_status("alca",1)
        laneChange_angle = self.laneChange_strStartFactor * self.laneChange_angled *  self.laneChange_counter / 50
        self.laneChange_counter += 1
        #laneChange_angle = self.laneChange_strStartFactor * self.laneChange_angled *  self.laneChange_counter / 50
        #delta_change = abs(self.laneChange_angle+ laneChange_angle + actuators.steerAngle) - self.laneChange_strStartMultiplier * abs(self.laneChange_angled)
        laneChange_angle = self.laneChange_strStartFactor * self.laneChange_angled *  self.laneChange_counter / 100
        delta_change = abs(self.laneChange_angle + laneChange_angle + actuators.steerAngle) - cl_max_angle_delta * self.laneChange_steerr
        if (self.laneChange_counter == 100) or (delta_change >= 0):
          if (delta_change < 0):
            # didn't achieve desired angle yet, so repeat
            self.laneChange_counter = 2
            self.laneChange_angle += laneChange_angle
            laneChange_angle = 0.
          else:
            self.laneChange_enabled = 3
            self.laneChange_counter = 1
            self.laneChange_angled = laneChange_angle
      # this is the first stage of the ALCAS
      # here we wait for x seconds before we start the turn
      # if at any point we detect hand on wheel, we let go of control and notify user
      # the same test for hand on wheel is done at ANY stage throughout the lane change process
      # CONTROL: during this stage we use the actuator angle to steer (OP Control)
      if self.laneChange_enabled == 5:
        if self.laneChange_counter == 1:
          CS.UE.custom_alert_message(2,"Auto Lane Change Engaged! (2)",self.laneChange_wait * 100)
        self.laneChange_counter += 1
        if self.laneChange_counter > (self.laneChange_wait -1) *100:
          self.laneChange_avg_angle +=  -actuators.steerAngle
          self.laneChange_avg_count += 1
        if self.laneChange_counter == self.laneChange_wait * 100:
          self.laneChange_enabled = 4
          self.laneChange_counter = 1
        if self.laneChange_counter == 0:
          if not blindspot:
            if CS.right_blinker_on or CS.left_blinker_on:
              self.laneChange_counter += 1
           
            
      # this is the final stage of the ALCAS
      # this just shows a message that we completed the lane change 
      # CONTROL: during this time we use the actuator angle to steer (OP Control)
      if self.laneChange_enabled == 7:
        if self.laneChange_counter ==1:
          CS.UE.custom_alert_message(2,"Auto Lane Change Complete!",300,4)
          CS.cstm_btns.set_button_status("alca",1)
        self.laneChange_counter +=1
    alca_enabled = (self.laneChange_enabled > 1)
    apply_angle = 0.
    # save position of blinker stalk
    self.prev_right_blinker_on = CS.right_blinker_on
    self.prev_left_blinker_on = CS.left_blinker_on
    # Angle if 0 we need to save it as a very small nonzero with the same sign as the prev one
    self.laneChange_last_actuator_delta = -actuators.steerAngle - self.laneChange_last_actuator_angle
    last_angle_sign = 1
    if (self.laneChange_last_actuator_angle <>0):
      last_angle_sign = self.laneChange_last_actuator_angle / abs(self.laneChange_last_actuator_angle)
    if actuators.steerAngle == 0:
      self.laneChange_last_actuator_angle = last_angle_sign * 0.00001
    else:
      self.laneChange_last_actuator_angle = -actuators.steerAngle
    # determine what angle to send and send it
    if (self.laneChange_enabled > 1) and (self.laneChange_enabled < 5):
      apply_angle = self.laneChange_angle + laneChange_angle
    else:
      apply_angle = -actuators.steerAngle
    self.laneChange_last_sent_angle = apply_angle
    return [-apply_angle,alca_enabled,turn_signal_needed]



  def update(self,enabled,CS,frame,actuators):
    if self.alcaEnabled:
      # ALCA enabled
        new_angle = 0.
        new_ALCA_Enabled = False
        new_turn_signal = 0
        new_angle,new_ALCA_Enabled,new_turn_signal = self.update_angle(enabled,CS,frame,actuators)
        output_steer = 0.
        if new_ALCA_Enabled and (self.laneChange_enabled < 5 ) and not self.laneChange_steerByAngle:
          #self.angle_steers_des = self.angle_steers_des_mpc # not declared
          steers_max = interp(CS.v_ego, CS.CP.steerMaxBP, CS.CP.steerMaxV)
          self.pid.pos_limit = steers_max
          self.pid.neg_limit = -steers_max
          output_steer = self.pid.update(new_angle, CS.angle_steers , check_saturation=(CS.v_ego > 10), override=CS.steer_override, feedforward=new_angle * (CS.v_ego ** 2), speed=CS.v_ego, deadzone=0.0)
        else: 
          output_steer = actuators.steer
        return [new_angle,output_steer,new_ALCA_Enabled,new_turn_signal]
    else:
      # ALCA disabled
      return [actuators.steerAngle,actuators.steer,False,0]
 
