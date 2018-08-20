from common.numpy_fast import interp
from selfdrive.controls.lib.pid import PIController


#change lane delta angles and other params
CL_MAXD_BP = [1., 32, 44.]
CL_MAXD_A = [.115, 0.081, 0.042] #delta angle based on speed; needs fine tune, based on Tesla steer ratio of 16.75
#0.477611    [
CL_MIN_V = 8.9 # do not turn if speed less than x m/2; 20 mph = 8.9 m/s
CL_MAX_A = 10. # do not turn if actuator wants more than x deg for going straight; this should be interp based on speed


class ALCAController(object):
  def __init__(self,carcontroller,alcaEnabled,steerByAngle):
    self.CC = carcontroller  #added to start, will see if we need it actually
    #variables for lane change
    self.alcaEnabled = alcaEnabled
    self.laneChange_steerByAngle = steerByAngle #steer only by angle; do not call PID
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
    self.prev_right_blinker_on = False #local variable for prev position
    self.prev_left_blinker_on = False #local variable for prev position
    self.pid = None

  def update_status(self,alcaEnabled):
    self.alcaEnabled = alcaEnabled

  def set_pid(self,CS):
    self.pid = PIController((CS.CP.steerKpBP, CS.CP.steerKpV),
                            (CS.CP.steerKiBP, CS.CP.steerKiV),
                            k_f=CS.CP.steerKf, pos_limit=1.0)

  def update_angle(self,enabled,CS,frame,actuators):
    # Basic highway lane change logic
    changing_lanes = CS.right_blinker_on or CS.left_blinker_on  

    actuator_delta = 0.
    laneChange_angle = 0.
    turn_signal_needed = 0 #send 1 for left, 2 for right 0 for not needed

    if (not CS.right_blinker_on) and (not CS.left_blinker_on) and \
    (self.laneChange_enabled ==7):
        self.laneChange_enabled =1
        self.laneChange_counter =0
        self.laneChange_direction =0
        CS.UE.custom_alert_message(-1,"",0)
    
    if (not CS.right_blinker_on) and (not CS.left_blinker_on) and \
      (self.laneChange_enabled > 1):
      #no blinkers on but we are still changing lane, so we need to send blinker command
      if self.laneChange_direction == -1:
        turn_signal_needed = 1
      elif self.laneChange_direction == 1:
        turn_signal_needed = 2
      else:
        turn_signal_needed = 0


    if (CS.cstm_btns.get_button_status("alca") > 0) and self.alcaEnabled and (self.laneChange_enabled == 1):
      if ((CS.v_ego < CL_MIN_V) or (abs(actuators.steerAngle) >= CL_MAX_A) or \
      (abs(CS.angle_steers)>= CL_MAX_A)  or (not enabled)): 
        CS.cstm_btns.set_button_status("alca",9)
      else:
        CS.cstm_btns.set_button_status("alca",1)

    if self.alcaEnabled and enabled and (((not self.prev_right_blinker_on) and CS.right_blinker_on) or \
      ((not self.prev_left_blinker_on) and CS.left_blinker_on)) and \
      ((CS.v_ego < CL_MIN_V) or (abs(actuators.steerAngle) >= CL_MAX_A) or (abs(CS.angle_steers) >=CL_MAX_A)):
      #something is not right, the speed or angle is limitting
      CS.UE.custom_alert_message(3,"Auto Lane Change Unavailable!",500,3)
      CS.cstm_btns.set_button_status("alca",9)


    if self.alcaEnabled and enabled and (((not self.prev_right_blinker_on) and CS.right_blinker_on) or \
      ((not self.prev_left_blinker_on) and CS.left_blinker_on))  and \
      (CS.v_ego >= CL_MIN_V) and (abs(actuators.steerAngle) < CL_MAX_A):
      # start blinker, speed and angle is within limits, let's go
      laneChange_direction = 1
      #changing lanes
      if CS.left_blinker_on:
        laneChange_direction = -1
      if (self.laneChange_enabled > 1) and (self.laneChange_direction <> laneChange_direction):
        #something is not right; signal in oposite direction; cancel
        CS.UE.custom_alert_message(3,"Auto Lane Change Canceled! (s)",200,5)
        self.laneChange_enabled = 1
        self.laneChange_counter = 0
        self.laneChange_direction = 0
        CS.cstm_btns.set_button_status("alca",1)
      elif (self.laneChange_enabled == 1) :
        #compute angle delta for lane change
        CS.UE.custom_alert_message(2,"Auto Lane Change Engaged! (1)",100)
        self.laneChange_enabled = 5
        self.laneChange_counter = 1
        self.laneChange_direction = laneChange_direction
        self.laneChange_avg_angle = 0.
        self.laneChange_avg_count = 0.
        self.laneChange_angled = self.laneChange_direction * self.laneChange_steerr *  interp(CS.v_ego, CL_MAXD_BP, CL_MAXD_A)
        self.laneChange_last_actuator_angle = 0.
        self.laneChange_last_actuator_delta = 0.
        self.laneChange_over_the_line = 0 
        CS.cstm_btns.set_button_status("alca",2)
        #reset PID for torque
        self.pid.reset()

    if (not self.alcaEnabled) and self.laneChange_enabled > 1:
      self.laneChange_enabled = 1
      self.laneChange_counter = 0
      self.laneChange_direction = 0

    #lane change in process
    if self.laneChange_enabled > 1:
      if (CS.steer_override or (CS.v_ego < CL_MIN_V)):
        CS.UE.custom_alert_message(3,"Auto Lane Change Canceled! (u)",200,3)
        #if any steer override cancel process or if speed less than min speed
        self.laneChange_counter = 0
        self.laneChange_enabled = 1
        self.laneChange_direction = 0
        CS.cstm_btns.set_button_status("alca",1)
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

      # TODO: - add some logic once we cross the line to smooth the transition to OP
      #           (needs about 0.1-0.2 s at low angles to get to the point where it's smooth)
      #       - check if the 0.5 * angle_delta is enough of a test for lane change
      #       - check that the delta in angle from actuator is increasing at a constant rate of DA/x
      #          or adjust angle for turns in road
      #       - generate audible alert for errors/take over messages
      if self.laneChange_enabled ==3:
        if self.laneChange_counter == 1:
          CS.UE.custom_alert_message(2,"Auto Lane Change Engaged! (4)",800)
          self.laneChange_over_the_line = 0
        self.laneChange_counter += 1
        laneChange_angle = self.laneChange_angled
        if (self.laneChange_over_the_line == 0):
          #we didn't cross the line, so keep computing the actuator delta until it flips
          actuator_delta = self.laneChange_direction * (-actuators.steerAngle - self.laneChange_last_actuator_angle)
          actuator_ratio = (-actuators.steerAngle)/self.laneChange_last_actuator_angle
        if (actuator_ratio < 1) and (abs(actuator_delta) > 0.5 * abs(self.laneChange_angled)):
          #sudden change in actuator angle or sign means we are on the other side of the line
          CS.UE.custom_alert_message(2,"Auto Lane Change Engaged! (5)",800)
          self.laneChange_over_the_line = 1
        if self.laneChange_over_the_line ==1:
          self.laneChange_enabled = 7
          self.laneChange_counter = 1
          self.laneChange_direction = 0
          #we are on the other side, let control go to OP
        if self.laneChange_counter >  (self.laneChange_duration) * 100:
          self.laneChange_enabled = 1
          self.laneChange_counter = 0
          CS.UE.custom_alert_message(3,"Auto Lane Change Canceled! (t)",200,5)
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
          self.laneChange_angled = self.laneChange_direction * self.laneChange_steerr *  interp(CS.v_ego, CL_MAXD_BP, CL_MAXD_A)
          #if angle more than max angle allowed cancel; last chance to cancel on road curvature
          if (abs(self.laneChange_angle) > CL_MAX_A):
            CS.UE.custom_alert_message(3,"Auto Lane Change Canceled! (a)",200,5)
            self.laneChange_enabled = 1
            self.laneChange_counter = 0
            self.laneChange_direction = 0
            CS.cstm_btns.set_button_status("alca",1)
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
          CS.UE.custom_alert_message(2,"Auto Lane Change Engaged! (2)",self.laneChange_wait * 100)
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
    #determine what angle to send and send it
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
          output_steer = self.pid.update(new_angle, CS.angle_steers , check_saturation=(CS.v_ego > 10), override=CS.steer_override, feedforward=new_angle, speed=CS.v_ego, deadzone=0.0)
        else: 
          output_steer = actuators.steer
        return [new_angle,output_steer,new_ALCA_Enabled,new_turn_signal]
    else:
      # ALCA disabled
      return [actuators.steerAngle,actuators.steer,False,0]
 
