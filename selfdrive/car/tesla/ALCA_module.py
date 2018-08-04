import custom_alert as tcm


#change lane delta angles and other params
CL_MAXD_BP = [1., 32, 44.]
CL_MAXD_A = [.24, 0.17, 0.09] #delta angle based on speed; needs fine tune
CL_MIN_V = 8.9 # do not turn if speed less than x m/2; 20 mph = 8.9 m/s
CL_MAX_A = 10. # do not turn if actuator wants more than x deg for going straight; this should be interp based on speed


class ALCAController(object):
  def __init__(self,carcontroller):
    self.CC = carcontroller
    #variables for lane change
    self.last_angle = 0.
    self.laneChange_last_actuator_angle = 0.
    self.laneChange_last_actuator_delta = 0.
    self.laneChange_over_the_line = 0
    self.laneChange_avg_angle = 0.
    self.laneChange_avg_count = 0.
    self.laneChange_enabled = 1 #set to zero for no lane change
    self.laneChange_counter = 0 #used to count frames during lane change
    self.laneChange_duration = 6.6 #how many max seconds to actually do the move
    self.laneChange_wait = 2 #how many seconds to wait before it starts the change
    self.laneChange_lw = 3.7 #lane width in meters
    self.laneChange_angle = 0. #saves the last angle from actuators before lane change starts
    self.laneChange_angled = 0. #angle delta
    self.laneChange_steerr = 8 #steer ratio for lane changes : ck if we can use the same
    self.laneChange_direction = 0 #direction of the lane change 
  
  def update(self,enabled,CS,frame,actuators):
    # Basic highway lane change logic
    changing_lanes = CS.right_blinker_on or CS.left_blinker_on  

    actuator_delta = 0.
    laneChange_angle = 0.

    if (CS.prev_right_blinker_on and (not CS.right_blinker_on)) or ( CS.prev_left_blinker_on and (not CS.left_blinker_on)):
      if CS.laneChange_enabled ==7:
        CS.laneChange_enabled =1
        CS.laneChange_counter =0
        tcm.custom_alert_message("")
    if (((not CS.prev_right_blinker_on) and CS.right_blinker_on) or ((not CS.prev_left_blinker_on) and CS.left_blinker_on)) and ((CS.v_ego < CL_MIN_V) or (abs(actuators.steerAngle) >= CL_MAX_A)):
      #something is not right; signal in oposite direction; cancel
      tcm.custom_alert_message("Auto Lane Change Unavailable!")
      CS.custom_alert_counter = 500
    if (((not CS.prev_right_blinker_on) and CS.right_blinker_on) or ((not CS.prev_left_blinker_on) and CS.left_blinker_on)) and (CS.v_ego >= CL_MIN_V) and (abs(actuators.steerAngle) < CL_MAX_A):
      laneChange_direction = 1 # for some reason right turns for me nees more angke
      #changing lanes
      if CS.left_blinker_on:
        laneChange_direction = -1
      if (CS.laneChange_enabled > 1) and (CS.laneChange_direction <> laneChange_direction):
        #something is not right; signal in oposite direction; cancel
        tcm.custom_alert_message("Auto Lane Change Canceled! (s)")
        CS.custom_alert_counter = 200
        CS.laneChange_enabled = 1
        CS.laneChange_counter = 0
        CS.laneChange_direction = 0
      elif (CS.laneChange_enabled == 1) :
        #compute angle delta for lane change
        tcm.custom_alert_message("Auto Lane Change Engaged! (1)")
        CS.custom_alert_counter = 100
        CS.laneChange_enabled = 5
        CS.laneChange_counter = 1
        CS.laneChange_direction = laneChange_direction
        self.laneChange_avg_angle = 0.
        self.laneChange_avg_count = 0.
        CS.laneChange_angled = CS.laneChange_direction * CS.laneChange_steerr *  interp(CS.v_ego, CL_MAXD_BP, CL_MAXD_A)
    #lane change in process
    if CS.laneChange_enabled > 1:
      if (CS.steer_override or (CS.v_ego < CL_MIN_V)):
        tcm.custom_alert_message("Auto Lane Change Canceled! (u)")
        CS.custom_alert_counter = 200
        #if any steer override cancel process or if speed less than min speed
        CS.laneChange_counter = 0
        CS.laneChange_enabled = 1
        CS.laneChange_direction = 0
      # this is the main stage once we start turning
      # we have to detect when to let go control back to OP or raise alarm if max timer passed
      # there are three conditions we look for:
      #     1. we can detect when we cross the lane, and then we let go control to OP
      #     2. we passed the min timer to cross the line and the delta between actuator and our angle
      #       is less than the release angle, then we let go control to OP
      #     3. the delta between our angle and the actuator is higher than the previous one
      #       (we cross the optimal path), then we let go control to OP
      #     4. max time is achieved: alert and disengage
      if CS.laneChange_enabled ==3:
        if CS.laneChange_counter == 1:
          tcm.custom_alert_message("Auto Lane Change Engaged! (4)")
          CS.custom_alert_counter = 800
          self.laneChange_last_actuator_delta = 0.
          self.laneChange_last_actuator_angle = - actuators.steerAngle
          self.laneChange_over_the_line = 0
        CS.laneChange_counter += 1
        laneChange_angle = CS.laneChange_angled
        if (self.laneChange_over_the_line == 0):
          #we didn't cross the line, so keep computing the actuator delta until it flips
          curvature = 1
          if abs(CS.laneChange_angle) > 2.:
            curvature = CS.laneChange_angle / abs(CS.laneChange_angle)
          actuator_sign = -1. if CS.laneChange_angled >= 0 else 1.
          actuator_delta =  curvature*(-actuators.steerAngle - self.laneChange_last_actuator_angle)/actuator_sign
          angle_now = CS.laneChange_angle +CS.laneChange_angled
          #actuator_delta = (angle_now - self.laneChange_last_actuator_angle)*(angle_now + actuators.steerAngle)
          self.laneChange_last_actuator_angle = - actuators.steerAngle
        if (actuator_delta < 0) and (abs(actuator_delta) > abs(CS.laneChange_angled)):
          #sudden change in actuator angle sign means we are on the other side of the line
          tcm.custom_alert_message("Auto Lane Change Engaged! (5)")
          CS.custom_alert_counter = 800
          self.laneChange_over_the_line = 1
          self.laneChange_last_actuator_delta = abs (CS.laneChange_angle + CS.laneChange_angled + actuators.steerAngle)
          actuator_delta = 1
        if self.laneChange_over_the_line ==1:
          CS.laneChange_enabled = 7
          CS.laneChange_counter = 1
          CS.laneChange_direction = 0
          """#we are on the other side, let's try to find either a min angle to let go
          #or we need to find the inflection point and let go
          actuator_delta_end = abs(CS.laneChange_angle + CS.laneChange_angled + actuators.steerAngle)
          if (actuator_delta_end < 8.) or (abs(self.laneChange_last_actuator_delta) < abs(actuator_delta_end)):
            #found the release point
            CS.laneChange_enabled = 2
            CS.laneChange_counter = 1
          else:
            self.laneChange_last_actuator_delta = actuator_delta_end"""
        if CS.laneChange_counter >  (CS.laneChange_duration) * 100:
          CS.laneChange_enabled = 1
          CS.laneChange_counter = 0
          tcm.custom_alert_message("Auto Lane Change Canceled! (t)")
          CS.custom_alert_counter = 200
          CS.laneChange_counter = 0
      # this is the critical start of the turn
      # here we will detect the angle to move; it is based on a speed determined angle but we have to also
      # take in consideration what's happening with the delta of consecutive actuators
      # if they go in the same direction with our turn we have to reset our turn angle because the actuator either
      # is compensating for a turn in the road OR for same lane correction for the car
      if CS.laneChange_enabled == 4:
        if CS.laneChange_counter == 1:
          CS.laneChange_angle = -actuators.steerAngle
          tcm.custom_alert_message("Auto Lane Change Engaged! (3)")
          CS.custom_alert_counter=100
          CS.laneChange_angled = CS.laneChange_direction * CS.laneChange_steerr *  interp(CS.v_ego, CL_MAXD_BP, CL_MAXD_A)
          #if angle more than max angle allowed cancel; last chance to cancel on road curvature
          if (abs(CS.laneChange_angle) > CL_MAX_A):
            tcm.custom_alert_message("Auto Lane Change Canceled! (a)")
            CS.custom_alert_counter = 200
            CS.laneChange_enabled = 1
            CS.laneChange_counter = 0
            CS.laneChange_direction = 0
        laneChange_angle = CS.laneChange_angled *  CS.laneChange_counter / 50
        CS.laneChange_counter += 1
        delta_change = abs(CS.laneChange_angle+ laneChange_angle + actuators.steerAngle) - abs(CS.laneChange_angled)
        if (CS.laneChange_counter == 100) or (delta_change >= 0):
          if (delta_change < 0):
            #didn't achieve desired angle yet, so repeat
            CS.laneChange_counter = 1
          else:
            CS.laneChange_enabled = 3
            CS.laneChange_counter = 1
            CS.laneChange_angled = laneChange_angle
      # this is the first stage of the ALCAS
      # here we wait for x seconds before we start the turn
      # if at any point we detect hand on wheel, we let go of control and notify user
      # the same test for hand on wheel is done at ANY stage throughout the lane change process
      if CS.laneChange_enabled == 5:
        if CS.laneChange_counter == 1:
          tcm.custom_alert_message("Auto Lane Change Engaged! (2)")
          CS.custom_alert_counter = CS.laneChange_wait * 100
        CS.laneChange_counter += 1
        if CS.laneChange_counter > (CS.laneChange_wait -1) *100:
          self.laneChange_avg_angle +=  -actuators.steerAngle
          self.laneChange_avg_count += 1
        if CS.laneChange_counter == CS.laneChange_wait * 100:
          CS.laneChange_enabled = 4
          CS.laneChange_counter = 1
      if CS.laneChange_enabled == 7:
        if CS.laneChange_counter ==1:
          tcm.custom_alert_message("Auto Lane Change Complete!")
          CS.custom_alert_counter = 200
        CS.laneChange_counter +=1
    alca_enabled = (CS.laneChange_enabled > 1)
    apply_angle = 0.
    # Angle
    if (CS.laneChange_enabled > 1) and (CS.laneChange_enabled < 5):
      apply_angle = CS.laneChange_angle + laneChange_angle
    else:
      apply_angle = -actuators.steerAngle
    return [apply_angle,alca_enabled]
