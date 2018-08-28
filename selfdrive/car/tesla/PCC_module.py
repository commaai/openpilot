from selfdrive.car.tesla import teslacan
from selfdrive.services import service_list
from selfdrive.car.tesla.values import AH, CruiseButtons, CAR
from selfdrive.boardd.boardd import can_list_to_can_capnp
from selfdrive.config import Conversions as CV
from common.numpy_fast import clip
import selfdrive.messaging as messaging
import os
import subprocess
import time
import zmq

# Accel limits
ACCEL_HYST_GAP = 0.02  # don't change accel command for small oscilalitons within this value
ACCEL_MAX = 1
ACCEL_MIN = -1
ACCEL_SCALE = max(ACCEL_MAX, -ACCEL_MIN)
ACCEL_REWIND_MAX = 0.04
PEDAL_DEADZONE = 0.05

#BB
BRAKE_THRESHOLD =0.1
ACCEL_THRESHOLD = 0.1
#BBTODO: move the vehicle variables; maybe make them speed variable
TORQUE_LEVEL_ACC = 30.
TORQUE_LEVEL_DECEL = -30.

STOPPING_EGO_SPEED = 0.5
MIN_CAN_SPEED = 0.3  #TODO: parametrize this in car interface
STOPPING_TARGET_SPEED = MIN_CAN_SPEED + 0.01
STARTING_TARGET_SPEED = 0.5
BRAKE_THRESHOLD_TO_PID = 0.2

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


def get_compute_gb_models(accel, speed):
  creep_brake = 0.0
  creep_speed = 2.3
  creep_brake_value = 0.15
  if speed < creep_speed:
    creep_brake = (creep_speed - speed) / creep_speed * creep_brake_value
  return float(accel) / 4.8 - creep_brake

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
    self.prev_steering_wheel_stalk = None
    self.prev_cruise_buttons = CruiseButtons.IDLE
    self.prev_cruise_setting = CruiseButtons.IDLE
    self.pedal_hardware_present = False
    self.pedal_hardware_first_check = True
    self.pedal_speed_kph = 0.
    self.pedal_idx = 0
    self.accel_steady = 0.
    self.prev_actuator_gas = 0.
    self.user_gas_state = 0
    self.torqueLevel_last = 0.
    self.prev_v_ego = 0.
    self.lastPedalForZeroTorque = 0.
    self.follow_time = 2.0 # in seconds
    self.pid = None
    self.v_pid = 0.
    self.last_output_gb = 0.
    
  def reset(self, v_pid):
    self.pid.reset()
    self.v_pid = v_pid

  def update_stat(self,CS, enabled, sendcan):
    if self.pid = None:
      CP = CS.CP
      self.PIController((CP.longitudinalKpBP, CP.longitudinalKpV),
                            (CP.longitudinalKiBP, CP.longitudinalKiV),
                            rate=100.0,
                            sat_limit=0.8)
      self.reset(0.):

    #if self.LoC == None:
    #  self.LoC = LongControl(CS.CP,get_compute_gb_models)
    can_sends = []
    #BBTODO: a better way to engage the pedal early and reset its CAN
    # on first brake press check if hardware present; being on CAN2 values are not included in fingerprinting
    self.pedal_hardware_present = CS.pedal_hardware_present
    
    if not self.pedal_hardware_present:
      if (CS.cstm_btns.get_button_status("pedal")>0):
        #no pedal hardware, disable button
        CS.cstm_btns.set_button_status("pedal",0)
      return
    if self.pedal_hardware_present:
      if (CS.cstm_btns.get_button_status("pedal")==0):
        #pedal hardware, enable button
        CS.cstm_btns.set_button_status("pedal",1)
      return
    # check if we had error before
    if self.user_gas_state != CS.user_gas_state:
      self.user_gas_state = CS.user_gas_state
      CS.cstm_btns.set_button_status("pedal", 1 if self.user_gas_state > 0 else 0)
      if self.user_gas_state > 0:
        CS.UE.custom_alert_message(3,"Pedal Interceptor Error (" + `self.user_gas_state` + ")",150,4)
        # send reset command
        idx = self.pedal_idx
        self.pedal_idx = (self.pedal_idx + 1) % 16
        can_sends.append(teslacan.create_gas_command_msg(0,0,idx))
        sendcan.send(can_list_to_can_capnp(can_sends, msgtype='sendcan').to_bytes())
    # disable on brake
    if CS.brake_pressed and self.enable_pedal_cruise:
      self.enable_pedal_cruise = False
      self.reset(0.)
      CS.UE.custom_alert_message(3,"PDL Disabled",150,4)
      CS.cstm_btns.set_button_status("pedal",1)
    # process any stalk movement
    curr_time_ms = _current_time_millis()
    enable_pedal_cruise = self.enable_pedal_cruise
    speed_uom = 1.
    if CS.imperial_speed_units:
      speed_uom = 1.609
    if (CS.cruise_buttons == CruiseButtons.MAIN and
        self.prev_cruise_buttons != CruiseButtons.MAIN):
      double_pull = (curr_time_ms - self.last_cruise_stalk_pull_time < 750) and \
        (CS.cstm_btns.get_button_status("pedal")>0) and enabled and \
         (CS.pcm_acc_status == 0) and self.pedal_hardware_present
      if(not self.enable_pedal_cruise) and double_pull:
        self.enable_pedal_cruise = True
        CS.UE.custom_alert_message(2,"PDL Enabled",150)
        CS.cstm_btns.set_button_status("pedal",2)
        if self.pedal_speed_kph < CS.v_ego_raw * 3.6:
          self.pedal_speed_kph = CS.v_ego_raw * 3.6
      elif (self.enable_pedal_cruise) and double_pull:
        #already enabled, reset speed to current speed if speed is grater than previous one
        if self.pedal_speed_kph < CS.v_ego_raw * 3.6:
          CS.UE.custom_alert_message(2,"PDL Speed Updated",150)
          self.pedal_speed_kph = CS.v_ego_raw * 3.6
      elif self.enable_pedal_cruise and not double_pull:
        self.enable_pedal_cruise = False
        CS.UE.custom_alert_message(3,"PDL Disabled",150,4)
        CS.cstm_btns.set_button_status("pedal",1)
        self.reset(0.)
      self.last_cruise_stalk_pull_time = curr_time_ms
    elif (CS.cruise_buttons == CruiseButtons.CANCEL and
          self.prev_cruise_buttons != CruiseButtons.CANCEL):
      self.enable_pedal_cruise = False
      if enable_pedal_cruise == True:
        CS.UE.custom_alert_message(3,"PDL Disabled",150,4)
        CS.cstm_btns.set_button_status("pedal",1)
        self.reset(0.)
      self.last_cruise_stalk_pull_time = 0
    elif (self.enable_pedal_cruise and CS.cruise_buttons !=self.prev_cruise_buttons):
      #enabled and new stalk command, let's see what we do with speed
      if CS.cruise_buttons == CruiseButtons.RES_ACCEL:
        self.pedal_speed_kph += speed_uom
        if self.pedal_speed_kph > 170:
          self.pedal_speed_kph = 170
      if CS.cruise_buttons == CruiseButtons.RES_ACCEL_2ND:
        self.pedal_speed_kph += 5 * speed_uom
        if self.pedal_speed_kph > 170:
          self.pedal_speed_kph = 170
      if CS.cruise_buttons == CruiseButtons.DECEL_SET:
        self.pedal_speed_kph -= speed_uom
        if self.pedal_speed_kph < 0:
          self.pedal_speed_kph = 0
      if CS.cruise_buttons == CruiseButtons.DECEL_2ND:
        self.pedal_speed_kph -= 5 * speed_uom
        if self.pedal_speed_kph < 0:
          self.pedal_speed_kph = 0
    #if PDL was on and something disabled cruise control, disable PDL too
    elif (self.enable_pedal_cruise == True and
          CS.pcm_acc_status != 0 and
          curr_time_ms - self.last_cruise_stalk_pull_time >  2000):
      self.enable_pedal_cruise = False
      CS.UE.custom_alert_message(3,"PDL Disabled",150,4)
      CS.cstm_btns.set_button_status("pedal",1)
      self.reset(0.)
    self.prev_steering_wheel_stalk = CS.steering_wheel_stalk
    self.prev_cruise_buttons = CS.cruise_buttons
    #now let's see if the PDL is available
    if (CS.cstm_btns.get_button_status("pedal")==1) or (CS.cstm_btns.get_button_status("pedal")==9):
        if enabled and (CS.pcm_acc_status == 0):
            CS.cstm_btns.set_button_status("pedal",1)
        else:
            CS.cstm_btns.set_button_status("pedal",9)
    

  def update_pdl(self,enabled,CS,frame,actuators,pcm_speed):
    #Pedal cruise control
    #if no hardware present, return -1
    #tesla_gas, tesla_brake = self.LoC.update(active, CS.vEgo, CS.brakePressed, CS.standstill, CS.cruiseState.standstill,
    #                                          v_cruise_kph, plan.vTarget, plan.vTargetFuture, plan.aTarget,
    #                                          CS.CP, PL.lead_1)
    idx = self.pedal_idx
    self.pedal_idx = (self.pedal_idx + 1) % 16
    if not self.pedal_hardware_present or not enabled:
      return 0.,0,idx


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

    prevent_overshoot = False #not CS.CP.stoppingControl and CS.v_ego < 1.5 and v_target_future < 0.7
    gas_max = interp(CS.v_ego, CS.CP.gasMaxBP, CS.CP.gasMaxV)
    brake_max = interp(CS.v_ego, CS.CP.brakeMaxBP, CS.CP.brakeMaxV)
    self.v_pid = self.calc_follow_speed(CS)
    self.pid.pos_limit = gas_max
    self.pid.neg_limit = - brake_max
    v_ego_pid = max(CS.v_ego, MIN_CAN_SPEED)
    deadzone = interp(v_ego_pid, CS.CP.longPidDeadzoneBP, CS.CP.longPidDeadzoneV)
    #BBAD adding overridet to pid to see if we can engage sooner
    # we will try to feed forward the pedal position.... we might want to feed the last output_gb....
    # it's all about testing now.
    output_gb = self.pid.update(self.v_pid, v_ego_pid, speed=v_ego_pid, override=override, deadzone=deadzone, feedforward=pedal_gas, freeze_integrator=prevent_overshoot)
    if prevent_overshoot:
      output_gb = min(output_gb, 0.0)

    self.last_output_gb = output_gb

    # gas and brake
    apply_accel = clip(output_gb, 0., gas_max)
    apply_brake = -clip(output_gb, -brake_max, 0.)

    #save position for brake
    if (CS.torqueLevel < TORQUE_LEVEL_ACC) and (CS.torqueLevel > TORQUE_LEVEL_DECEL):
      self.lastPedalForZeroTorque = self.prev_actuator_gas

    #slow deceleration
    if (apply_accel > PEDAL_DEADZONE) and (apply_accel < self.prev_actuator_gas) and (apply_brake <= BRAKE_THRESHOLD):
      if (CS.torqueLevel < TORQUE_LEVEL_ACC) and (CS.v_ego < self.prev_v_ego):
        tesla_accel = self.prev_actuator_gas
      else:
        tesla_accel = clip(apply_accel,self.prev_actuator_gas - ACCEL_REWIND_MAX,self.prev_actuator_gas)
    elif (apply_accel <= PEDAL_DEADZONE) and (apply_brake <= BRAKE_THRESHOLD):
      if (CS.torqueLevel < TORQUE_LEVEL_ACC):
        #we are in the torque dead zone, 
        tesla_accel = self.prev_actuator_gas
      else:
        tesla_accel = self.prev_actuator_gas - ACCEL_REWIND_MAX
    elif (apply_brake > BRAKE_THRESHOLD) and (apply_accel <= ACCEL_THRESHOLD):
      if self.lastPedalForZeroTorque > 0:
        tesla_accel = (.8 - apply_brake) * self.lastPedalForZeroTorque
        self.lastPedalForZeroTorque = 0.
      else:
        tesla_accel = (.8 - apply_brake) * self.prev_actuator_gas
    else:
      tesla_accel = actuators.gas
    tesla_accel, self.accel_steady = accel_hysteresis(tesla_accel, self.accel_steady, enabled)
    
    tesla_accel = clip(tesla_accel, 0., 1.) if self.enable_pedal_cruise else 0.
    enable_gas = 1 if self.enable_pedal_cruise else 0
    self.torqueLevel_last = CS.torqueLevel
    self.prev_actuator_gas = tesla_accel * enable_gas
    self.prev_v_ego = CS.v_ego
    return tesla_accel,enable_gas,idx


  # function to calculate the cruise speed based on a safe follow distance
  def calc_follow_speed(self, CS):
    
    current_time_ms = _current_time_millis()
     # Make sure we were able to populate lead_1.
    if self.lead_1 is None:
      return None
    # dRel is in meters.
    lead_dist = self.lead_1.dRel
    # Grab the relative speed.
    rel_speed = self.lead_1.vRel * CV.MS_TO_KPH
    # v_ego is in m/s, so safe_dist_mance is in meters.
    safe_dist_m = CS.v_ego * self.follow_time
    # Current speed in kph
    actual_speed = CS.v_ego * CV.MS_TO_KPH
    available_speed = self.pedal_speed_kph - actual_speed
    # Pedal only functions above 1 kph.
    min_cruise_speed_ms = 1 
    # button to issue
    new_speed = 0.
    # debug msg
    msg = None

    #print "dRel: ", self.lead_1.dRel," yRel: ", self.lead_1.yRel, " vRel: ", self.lead_1.vRel, " aRel: ", self.lead_1.aRel, " vLead: ", self.lead_1.vLead, " vLeadK: ", self.lead_1.vLeadK, " aLeadK: ",     self.lead_1.aLeadK

    ###   Logic to determine best cruise speed ###

    if self.enable_pedal_cruise:
      # if cruise is set to faster than the max speed, slow down
      if actual_speed > self.pedal_speed_kph:
        msg =  "Slow to max"
        new_speed = self.pedal_speed_kph
      # If lead_dist is reported as 0, no one is detected in front of you so you
      # can speed up don't speed up when steer-angle > 2; vision radar often
      # loses lead car in a turn.
      elif lead_dist == 0 and CS.angle_steers < 2.0:
        if actual_speed < self.pedal_speed_kph: 
          msg =  "Accel to max"
          new_speed = self.pedal_speed_kph
      # if we have a populated lead_distance
      elif (lead_dist > 0):
        ### Slowing down ###
        # Reduce speed significantly if lead_dist < 50% of safe dist, no matter
        # the rel_speed

        #Reduce speed if rel_speed < -15kph so you don't rush up to lead car
        if rel_speed < -15  and lead_dist > 2. * safe_dist_m:
          msg =  "relspd -15 down"
          new_speed = actual_speed + rel_speed/2
        #Reduce speed if rel_speed < -5kph so you don't rush up to lead car
        elif rel_speed < -5  and lead_dist >  1.5 safe_dist_m:
          msg =  "relspd to 0"
          new_speed = actual_speed + rel_speed
        # Reduce speed significantly if lead_dist < 60% of  safe dist
        if lead_dist < (safe_dist_m * 0.3) and rel_speed < 2:
          msg =  "50pct down"
          new_speed = actual_speed * 0.5
        # and if the lead car isn't pulling away
        elif lead_dist < (safe_dist_m * 0.5) and rel_speed < 0:
          msg =  "70pct down"
          new_speed = actual_speed * 0.3
        # we're close to the safe distance, so make slow adjustments
        # only adjust every 1 secs
        elif lead_dist < (safe_dist_m * 0.9) and rel_speed < 0:
          msg =  "90pct down"
          new_speed = actual_speed * 0.9

        ### Speed up ###
        # don't speed up again until you have more than a safe distance in front
        # only adjust every 2 sec
        elif lead_dist > (safe_dist_m * 0.8) or rel_speed > 5:
          msg =  "120pct UP   half: ","{0:.1f}kph".format(half_press_kph), "  avail: {0:.1f}kph".format(available_speed)
          new_speed = actual_speed  + 1
        
    return new_speed * CV.KPH_TO_MS
