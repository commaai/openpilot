import custom_alert as tcm
from common.numpy_fast import interp


#change lane delta angles and other params
CL_MAXD_BP = [1., 32, 44.]
CL_MAXD_A = [.115, 0.081, 0.042] #delta angle based on speed; needs fine tune, based on Tesla steer ratio of 16.75
#0.477611    [
CL_MIN_V = 0. #8.9 # do not turn if speed less than x m/2; 20 mph = 8.9 m/s
CL_MAX_A = 10. # do not turn if actuator wants more than x deg for going straight; this should be interp based on speed


class ALCAController(object):
  def __init__(self,carcontroller,alcaEnabled,steerByAngle):
    self.CC = carcontroller  #added to start, will see if we need it actually
    #variables for lane change
    self.alcaEnabled = alcaEnabled
    self.laneChange_steerByAngle = steerByAngle
    self.laneChange_last_actuator_angle = 0.
    self.laneChange_last_actuator_delta = 0.
    self.laneChange_last_sent_angle = 0.
    self.laneChange_over_the_line = 0 #did we cross the line?
    self.laneChange_avg_angle = 0. #used if we do average entry angle over x frames
    self.laneChange_avg_count = 0. #used if we do average entry angle over x frames
    self.laneChange_enabled = 1 #set to zero for no lane change
    self.laneChange_counter = 0 #used to count frames during lane change
    self.laneChange_min_duration = 2. #min time to wait before looking for next lane
    self.laneChange_duration = 5.6 #how many max seconds to actually do the move; if lane not found after this then send error
    self.laneChange_wait = 2 #how many seconds to wait before it starts the change
    self.laneChange_lw = 3.7 #lane width in meters
    self.laneChange_angle = 0. #saves the last angle from actuators before lane change starts
    self.laneChange_angled = 0. #angle delta
    self.laneChange_steerr = 16.75 #steer ratio for lane change
    self.laneChange_direction = 0 #direction of the lane change 
  
  def update_angle(self,enabled,CS,frame,actuators):
    # Basic highway lane change logic
    changing_lanes = CS.right_blinker_on or CS.left_blinker_on  

    actuator_delta = 0.
    laneChange_angle = 0.

    if (CS.prev_right_blinker_on and (not CS.right_blinker_on)) or ( CS.prev_left_blinker_on and (not CS.left_blinker_on)):
      if self.laneChange_enabled ==7:
        #if stage 7 (complete) and blinkers turned off we reset
        self.laneChange_enabled =1
        self.laneChange_counter =0
        tcm.custom_alert_message("",CS,0)

    if enabled and (((not CS.prev_right_blinker_on) and CS.right_blinker_on) or \
      ((not CS.prev_left_blinker_on) and CS.left_blinker_on)) and \
      ((CS.v_ego < CL_MIN_V) or (abs(actuators.steerAngle) >= CL_MAX_A)):
      #something is not right, the speed or angle is limitting
      tcm.custom_alert_message("Auto Lane Change Unavailable!",CS,500)

    if enabled and (((not CS.prev_right_blinker_on) and CS.right_blinker_on) or \
      ((not CS.prev_left_blinker_on) and CS.left_blinker_on)) and \
      (CS.v_ego >= CL_MIN_V) and (abs(actuators.steerAngle) < CL_MAX_A):
      # start blinker, speed and angle is within limits, let's go

      laneChange_direction = 1
      #changing lanes
      if CS.left_blinker_on:
        laneChange_direction = -1
      if (self.laneChange_enabled > 1) and (self.laneChange_direction <> laneChange_direction):
        #something is not right; signal in oposite direction; cancel
        tcm.custom_alert_message("Auto Lane Change Canceled! (s)",CS,200)
        self.laneChange_enabled = 1
        self.laneChange_counter = 0
        self.laneChange_direction = 0
      elif (self.laneChange_enabled == 1) :
        #compute angle delta for lane change
        tcm.custom_alert_message("Auto Lane Change Engaged! (1)",CS,100)
        self.laneChange_enabled = 5
        self.laneChange_counter = 1
        self.laneChange_direction = laneChange_direction
        self.laneChange_avg_angle = 0.
        self.laneChange_avg_count = 0.
        self.laneChange_angled = self.laneChange_direction * self.laneChange_steerr *  interp(CS.v_ego, CL_MAXD_BP, CL_MAXD_A)
        self.laneChange_last_actuator_angle = 0.
        self.laneChange_last_actuator_delta = 0.
        self.laneChange_over_the_line = 0 

    #lane change in process
    if self.laneChange_enabled > 1:
      if (CS.steer_override or (CS.v_ego < CL_MIN_V)):
        tcm.custom_alert_message("Auto Lane Change Canceled! (u)",CS,200)
        #if any steer override cancel process or if speed less than min speed
        self.laneChange_counter = 0
        self.laneChange_enabled = 1
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
          tcm.custom_alert_message("Auto Lane Change Engaged! (4)",CS,800)
          self.laneChange_over_the_line = 0
        self.laneChange_counter += 1
        laneChange_angle = self.laneChange_angled
        if (self.laneChange_over_the_line == 0):
          #we didn't cross the line, so keep computing the actuator delta until it flips
          actuator_delta = self.laneChange_direction * (-actuators.steerAngle - self.laneChange_last_actuator_angle)
          actuator_ratio = (-actuators.steerAngle)/self.laneChange_last_actuator_angle
          #actuator_sign_change = (-actuators.steerAngle)*(self.laneChange_last_actuator_angle)
          #angle_now = self.laneChange_angle + self.laneChange_angled
          #angle_correction_increase = abs(self.laneChange_last_sent_angle - self.laneChange_last_actuator_angle) > abs(angle_now + actuators.steerAngle)
        if (actuator_ratio < 1) and (abs(actuator_delta) > 0.5 * abs(self.laneChange_angled)):
          #sudden change in actuator angle or sign means we are on the other side of the line
          tcm.custom_alert_message("Auto Lane Change Engaged! (5)",CS,800)
          self.laneChange_over_the_line = 1
          actuator_delta = 1
        if self.laneChange_over_the_line ==1:
          self.laneChange_enabled = 7
          self.laneChange_counter = 1
          self.laneChange_direction = 0
          #we are on the other side, let control go to OP
        if self.laneChange_counter >  (self.laneChange_duration) * 100:
          self.laneChange_enabled = 1
          self.laneChange_counter = 0
          tcm.custom_alert_message("Auto Lane Change Canceled! (t)",CS,200)
          self.laneChange_counter = 0
      # this is the critical start of the turn
      # here we will detect the angle to move; it is based on a speed determined angle but we have to also
      # take in consideration what's happening with the delta of consecutive actuators
      # if they go in the same direction with our turn we have to reset our turn angle because the actuator either
      # is compensating for a turn in the road OR for same lane correction for the car
      # CONTROL: during this time we use ALCA angle to steer (ALCA Control)

      # TODO: when actuator moves in the same direction with lane change, correct starting angle
      if self.laneChange_enabled == 4:
        if self.laneChange_counter == 1:
          self.laneChange_angle = -actuators.steerAngle
          tcm.custom_alert_message("Auto Lane Change Engaged! (3)",CS,100)
          self.laneChange_angled = self.laneChange_direction * self.laneChange_steerr *  interp(CS.v_ego, CL_MAXD_BP, CL_MAXD_A)
          #if angle more than max angle allowed cancel; last chance to cancel on road curvature
          if (abs(self.laneChange_angle) > CL_MAX_A):
            tcm.custom_alert_message("Auto Lane Change Canceled! (a)",CS,200)
            self.laneChange_enabled = 1
            self.laneChange_counter = 0
            self.laneChange_direction = 0
        laneChange_angle = self.laneChange_angled *  self.laneChange_counter / 50
        self.laneChange_counter += 1
        delta_change = abs(self.laneChange_angle+ laneChange_angle + actuators.steerAngle) - abs(self.laneChange_angled)
        if (self.laneChange_counter == 100) or (delta_change >= 0):
          if (delta_change < 0):
            #didn't achieve desired angle yet, so repeat
            self.laneChange_counter = 1
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
          tcm.custom_alert_message("Auto Lane Change Engaged! (2)",CS,self.laneChange_wait * 100)
        self.laneChange_counter += 1
        if self.laneChange_counter > (self.laneChange_wait -1) *100:
          self.laneChange_avg_angle +=  -actuators.steerAngle
          self.laneChange_avg_count += 1
        if self.laneChange_counter == self.laneChange_wait * 100:
          self.laneChange_enabled = 4
          self.laneChange_counter = 1
      # this is the final stage of the ALCAS
      # this just shows a message that we completed the lane change 
      # CONTROL: during this time we use the actuator angle to steer (OP Control)
      if self.laneChange_enabled == 7:
        if self.laneChange_counter ==1:
          tcm.custom_alert_message("Auto Lane Change Complete!",CS,300)
        self.laneChange_counter +=1
    alca_enabled = (self.laneChange_enabled > 1)
    apply_angle = 0.
    # Angle if 0 we need to save it as a very small nonzero with the same sign as the prev one
    self.laneChange_last_actuator_delta = -actuators.steerAngle - self.laneChange_last_actuator_angle
    last_angle_sign = 1
    if (self.laneChange_last_actuator_angle <>0):
      last_angle_sign = self.laneChange_last_actuator_angle / abs(self.laneChange_last_actuator_angle)
    if actuators.steerAngle == 0:
      self.laneChange_last_actuator_angle = last_angle_sign * 0.00001
    else:
      self.laneChange_last_actuator_angle = -actuators.steerAngle

    if (self.laneChange_enabled > 1) and (self.laneChange_enabled < 5):
      apply_angle = self.laneChange_angle + laneChange_angle
    else:
      apply_angle = -actuators.steerAngle
    self.laneChange_last_sent_angle = apply_angle
    return [-apply_angle,alca_enabled]



  def update(self,enabled,CS,frame,actuators):
    if self.alcaEnabled:
      # ALCA enabled
      if self.laneChange_steerByAngle:
        # steering by angle
        new_angle = 0.
        new_ALCA_enabled = False
        new_angle,new_ALCA_Enabled = self.update_angle(enabled,CS,frame,actuators)
        return [new_angle,new_ALCA_Enabled]
      else:
        # steering by torque
        #TODO: torque ALCA module
        return [actuators.steerAngle,False]
    else:
      # ALCA disabled
      if self.laneChange_steerByAngle:
        #steer by angle
        return [actuators.steerAngle,False]
      else:
        #steer by torque
        #TODO: torque ALCA module
        return [actuators.steerAngle,False]
 
