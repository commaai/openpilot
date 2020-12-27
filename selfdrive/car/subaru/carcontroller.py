from selfdrive.car import apply_std_steer_torque_limits
from selfdrive.car.subaru import subarucan
from selfdrive.car.subaru.values import DBC, PREGLOBAL_CARS
from opendbc.can.packer import CANPacker
import time


class CarControllerParams():
  def __init__(self):
    self.STEER_MAX = 2047              # max_steer 4095
    self.STEER_STEP = 2                # how often we update the steer cmd
    self.STEER_DELTA_UP = 50           # torque increase per refresh, 0.8s to max
    self.STEER_DELTA_DOWN = 70         # torque decrease per refresh
    self.STEER_DRIVER_ALLOWANCE = 60   # allowed driver torque before start limiting
    self.STEER_DRIVER_MULTIPLIER = 10  # weight driver torque heavily
    self.STEER_DRIVER_FACTOR = 1       # from dbc

    #SUBARU STOP AND GO - Global
    self.SNG_DISTANCE_LIMIT = 120      # distance trigger value limit for stop and go (0-255)
    self.SNG_DISTANCE_DEADBAND = 10     # deadband for SNG lead car refence distance to cater for Close_Distance sensor noises
    self.THROTTLE_TAP_LIMIT = 5        # send a maximum of 5 throttle tap messages (trial and error)
    self.THROTTLE_TAP_LEVEL = 5        # send a throttle message with value of 5 (trial and error)
    self.ES_CLOSE_DISTANCE_SETTLE_TIME = 250000000  #(250ms) time taken (in nanoseconds) for ES's Close_Distance signal to settle (taking care of noise after stopping)

    #SUBARU STOP AND GO - Pre-Global
    self.SNG_DISTANCE_THRESHOLD_PREGLOBAL = 3 #SnG trigger when lead car distance > 3m
    self.SNG_DISTANCE_LIMIT_PREGLOBAL = 4 #SnG only trigger if close distance is less than 4

class CarController():
  def __init__(self, dbc_name, CP, VM):
    self.apply_steer_last = 0
    self.es_distance_cnt = -1
    self.es_accel_cnt = -1
    self.es_lkas_cnt = -1
    self.fake_button_prev = 0
    self.steer_rate_limited = False
    self.throttle_cnt = -1

    self.params = CarControllerParams()
    self.packer = CANPacker(DBC[CP.carFingerprint]['pt'])

    #SUBARU STOP AND GO flags and vars
    self.prev_cruise_state = -1
    self.cruise_state_change_time = -1
    self.sng_throttle_tap_cnt = 0
    self.sng_resume_acc = False
    self.sng_has_recorded_distance = False
    self.sng_distance_threshold = self.params.SNG_DISTANCE_LIMIT

    #SUBARU STOP AND GO - Pre-Global only
    self.prev_close_distance = -1

  def update(self, enabled, CS, frame, actuators, pcm_cancel_cmd, visual_alert, left_line, right_line):

    can_sends = []

    # *** steering ***
    if (frame % self.params.STEER_STEP) == 0:

      apply_steer = int(round(actuators.steer * self.params.STEER_MAX))

      # limits due to driver torque

      new_steer = int(round(apply_steer))
      apply_steer = apply_std_steer_torque_limits(new_steer, self.apply_steer_last, CS.out.steeringTorque, self.params)
      self.steer_rate_limited = new_steer != apply_steer

      if not enabled:
        apply_steer = 0

      if CS.CP.carFingerprint in PREGLOBAL_CARS:
        can_sends.append(subarucan.create_preglobal_steering_control(self.packer, apply_steer, frame, self.params.STEER_STEP))
      else:
        can_sends.append(subarucan.create_steering_control(self.packer, apply_steer, frame, self.params.STEER_STEP))

      self.apply_steer_last = apply_steer

    #---------------------------------------------STOP AND GO---------------------------------------------------
    if CS.CP.carFingerprint in PREGLOBAL_CARS:
      #PRE-GLOBAL STOP AND GO
      #Activate ACC Resume with throttle tap
      if (enabled
          and CS.car_follow                                                    #Must have lead car
          and CS.close_distance > self.params.SNG_DISTANCE_THRESHOLD_PREGLOBAL #Distance with lead car > 3m (this is due to Preglobal ES's unreliable Close Distance signal)
          and CS.close_distance < 4.5                                          #For safety, SnG will not operate if Close Distance reads more than 4.5m (Pre-global ES's unreliability, sometimes Close Distance shows max-5m when there is a stationary object ahead)
          and CS.close_distance > self.prev_close_distance                     #Distance with lead car is increasing
         ):
        self.sng_resume_acc = True

      throttle_cmd = -1 #normally, just forward throttle msg from ECU
      if self.sng_resume_acc:
        #Send Maximum <THROTTLE_TAP_LIMIT> to get car out of HOLD
        if self.sng_throttle_tap_cnt < 5:
          throttle_cmd = 5
          self.sng_throttle_tap_cnt += 1
        else:
          self.sng_throttle_tap_cnt = -1
          self.sng_resume_acc = False

      self.prev_close_distance = CS.close_distance

      #Send throttle message
      if self.throttle_cnt != CS.throttle_msg["Counter"]:
        can_sends.append(subarucan.create_preglobal_throttle_control(self.packer, CS.throttle_msg, throttle_cmd))
        self.throttle_cnt = CS.throttle_msg["Counter"]
    else:
      #GLOBAL STOP AND GO
      #Car can only be in HOLD state (3) if it is standing still
      # => if not in HOLD state car has to be moving or driver has taken action
      if CS.cruise_state != 3:
        self.sng_throttle_tap_cnt = 0           #Reset throttle tap message count when car starts moving
        self.sng_resume_acc = False             #Cancel throttle tap when car starts moving
        self.sng_has_recorded_distance = False  #Reset has_recorded_distance flag once car started moving

      #Reset SNG distance threshold to limit value if we havent recorded a reference distance threshold
      #This is to make sure car will always move forward when lead car moves before SnG reference distance
      #threshold is recorded
      if not self.sng_has_recorded_distance:
        self.sng_distance_threshold = self.params.SNG_DISTANCE_LIMIT

      #Record the time at which CruiseState change to HOLD (3)
      if self.prev_cruise_state != 3 and CS.cruise_state == 3:
        self.cruise_state_change_time = time.time_ns()

      #While in HOLD, wait <ES_CLOSE_DISTANCE_SETTLE_TIME> nanoseconds (since Cruise state changes to HOLD)
      #before recording SnG lead car reference distance
      if (enabled
          and CS.cruise_state == 3                #in HOLD state
          and not self.sng_has_recorded_distance  #has not recorded reference distance
          and time.time_ns() > self.cruise_state_change_time + self.params.ES_CLOSE_DISTANCE_SETTLE_TIME): #wait 200ms before recording reference distance
        self.sng_distance_threshold = CS.close_distance
        self.sng_has_recorded_distance = True     #Set flag to true so sng_distance_threshold wont be recorded again until car moves
        #Limit lead car reference distance to <SNG_DISTANCE_LIMIT>
        if self.sng_distance_threshold > self.params.SNG_DISTANCE_LIMIT:
          self.sng_distance_threshold = self.params.SNG_DISTANCE_LIMIT

      #Trigger THROTTLE TAP when in hold and close_distance increases > SNG distance threshold (with deadband)
      #false positives caused by pedestrians/cyclists crossing the street in front of car
      self.sng_resume_acc = False
      if (enabled
          and CS.cruise_state == 3 #cruise state == 3 => ACC HOLD state
          and CS.close_distance > self.sng_distance_threshold + self.params.SNG_DISTANCE_DEADBAND #lead car distance is within SnG operating range
          and CS.close_distance < 255
          and CS.car_follow == 1):
        self.sng_resume_acc = True

      #Send a throttle tap to resume ACC
      throttle_cmd = -1 #normally, just forward throttle msg from ECU
      if self.sng_resume_acc:
        #Send Maximum <THROTTLE_TAP_LIMIT> to get car out of HOLD
        if self.sng_throttle_tap_cnt < self.params.THROTTLE_TAP_LIMIT:
          throttle_cmd = self.params.THROTTLE_TAP_LEVEL
          self.sng_throttle_tap_cnt += 1
        else:
          self.sng_throttle_tap_cnt = -1
          self.sng_resume_acc = False
      #TODO: Send cruise throttle to get car up to speed. There is a 2-3 seconds delay after
      # throttle tap is sent and car start moving. EDIT: This is standard with Toyota OP's SnG
      #pseudo: !!!WARNING!!! Dangerous, proceed with CARE
      #if sng_resume_acc is True && has been 1 second since sng_resume_acc turns to True && current ES_Throttle < 2000
      #    send ES_Throttle = 2000

      #Update prev values
      self.prev_cruise_state = CS.cruise_state

      #Send throttle message
      if self.throttle_cnt != CS.throttle_msg["Counter"]:
        can_sends.append(subarucan.create_throttle(self.packer, CS.throttle_msg, throttle_cmd))
        self.throttle_cnt = CS.throttle_msg["Counter"] 
    #--------------------------------------------------------------------------------------------------------------

    # *** alerts and pcm cancel ***

    if CS.CP.carFingerprint in PREGLOBAL_CARS:
      if self.es_accel_cnt != CS.es_accel_msg["Counter"]:
        # 1 = main, 2 = set shallow, 3 = set deep, 4 = resume shallow, 5 = resume deep
        # disengage ACC when OP is disengaged
        if pcm_cancel_cmd:
          fake_button = 1
        # turn main on if off and past start-up state
        elif not CS.out.cruiseState.available and CS.ready:
          fake_button = 1
        else:
          fake_button = CS.button

        # unstick previous mocked button press
        if fake_button == 1 and self.fake_button_prev == 1:
          fake_button = 0
        self.fake_button_prev = fake_button

        can_sends.append(subarucan.create_es_throttle_control(self.packer, fake_button, CS.es_accel_msg))
        self.es_accel_cnt = CS.es_accel_msg["Counter"]

    else:
      if self.es_distance_cnt != CS.es_distance_msg["Counter"]:
        can_sends.append(subarucan.create_es_distance(self.packer, CS.es_distance_msg, pcm_cancel_cmd))
        self.es_distance_cnt = CS.es_distance_msg["Counter"]

      if self.es_lkas_cnt != CS.es_lkas_msg["Counter"]:
        can_sends.append(subarucan.create_es_lkas(self.packer, CS.es_lkas_msg, visual_alert, left_line, right_line))
        self.es_lkas_cnt = CS.es_lkas_msg["Counter"]

    return can_sends
