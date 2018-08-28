from selfdrive.car.tesla import teslacan
from selfdrive.controls.lib.pid import PIController
from common.numpy_fast import clip, interp
from selfdrive.services import service_list
from selfdrive.car.tesla.values import AH,CruiseState, CruiseButtons, CAR
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
FOLLOW_UP_TIME = 1.5 #time in seconds to follow car in front
MIN_PCC_V = 0. #
MAX_PCC_V = 170.

STOPPING_EGO_SPEED = 0.5
MIN_CAN_SPEED = 0.3  #TODO: parametrize this in car interface
STOPPING_TARGET_SPEED = MIN_CAN_SPEED + 0.01
STARTING_TARGET_SPEED = 0.5
BRAKE_THRESHOLD_TO_PID = 0.2


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
    self.follow_time = FOLLOW_UP_TIME # in seconds
    self.pid = None
    self.v_pid = 0.
    self.last_output_gb = 0.
    
  def reset(self, v_pid):
    self.pid.reset()
    self.v_pid = v_pid

  def update_stat(self,CS, enabled, sendcan):
    if self.pid == None:
      CP = CS.CP
      self.pid = PIController((CP.longitudinalKpBP, CP.longitudinalKpV),
                            (CP.longitudinalKiBP, CP.longitudinalKiV),
                            rate=100.0,
                            sat_limit=0.8)
      self.reset(0.)

    can_sends = []
    #BBTODO: a better way to engage the pedal early and reset its CAN
    # on first brake press check if hardware present; being on CAN2 values are not included in fingerprinting
    self.pedal_hardware_present = CS.pedal_hardware_present
    
    if not self.pedal_hardware_present:
      if (CS.cstm_btns.get_button_status("pedal")>0):
        #no pedal hardware, disable button
        CS.cstm_btns.set_button_status("pedal",0)
        print "disabling pedal"
      print "no pedal hardware"
      return
    if self.pedal_hardware_present:
      if (CS.cstm_btns.get_button_status("pedal")==0):
        #pedal hardware, enable button
        CS.cstm_btns.set_button_status("pedal",1)
        print "enabling pedal"
    # check if we had error before
    if self.user_gas_state != CS.user_gas_state:
      self.user_gas_state = CS.user_gas_state
      CS.cstm_btns.set_button_status("pedal", 1 if self.user_gas_state > 0 else 0)
      if self.user_gas_state > 0:
        CS.UE.custom_alert_message(3,"Pedal Interceptor Error (" + `self.user_gas_state` + ")",150,4)
        # send reset command
        idx = self.pedal_idx
        self.pedal_idx = (self.pedal_idx + 1) % 16
        can_sends.append(teslacan.create_pedal_command_msg(0,0,idx))
        sendcan.send(can_list_to_can_capnp(can_sends, msgtype='sendcan').to_bytes())
    # disable on brake
    if CS.brake_pressed and self.enable_pedal_cruise:
      self.enable_pedal_cruise = False
      self.reset(0.)
      CS.UE.custom_alert_message(3,"PDL Disabled",150,4)
      CS.cstm_btns.set_button_status("pedal",1)
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
      ready = (CS.cstm_btns.get_button_status("pedal") > PCCState.OFF and
               enabled and
               CruiseState.is_off(CS.pcm_acc_status))
      if ready and double_pull:
        # A double pull enables ACC. updating the max ACC speed if necessary.
        self.enable_pedal_cruise = True
        # Increase ACC speed to match current, if applicable.
        self.pedal_speed_kph = max(CS.v_ego_raw * CV.MS_TO_KPH, self.pedal_speed_kph)
      else:
        # A single pull disables ACC (falling back to just steering).
        self.enable_pedal_cruise = False
    # Handle pressing the cancel button.
    elif CS.cruise_buttons == CruiseButtons.CANCEL:
      self.enable_pedal_cruise = False
      self.pedal_speed_kph = 0. 
      self.last_cruise_stalk_pull_time = 0
    # Handle pressing up and down buttons.
    elif (self.enable_pedal_cruise and 
          CS.cruise_buttons !=self.prev_cruise_buttons):
      # Real stalk command while ACC is already enabled. Adjust the max ACC
      # speed if necessary. For example if max speed is 50 but you're currently
      # only going 30, the cruise speed can be increased without any change to
      # max ACC speed. If actual speed is already 50, the code also increases
      # the max cruise speed.
      if CS.cruise_buttons == CruiseButtons.RES_ACCEL:
        requested_speed_kph = CS.v_ego_raw * CV.MS_TO_KPH + speed_uom_kph
        self.pedal_speed_kph = max(self.pedal_speed_kph, requested_speed_kph)
      elif CS.cruise_buttons == CruiseButtons.RES_ACCEL_2ND:
        requested_speed_kph = CS.v_ego_raw * CV.MS_TO_KPH + 5 * speed_uom_kph
        self.pedal_speed_kph = max(self.pedal_speed_kph, requested_speed_kph)
      elif CS.cruise_buttons == CruiseButtons.DECEL_SET:
        self.pedal_speed_kph -= speed_uom_kph
      elif CS.cruise_buttons == CruiseButtons.DECEL_2ND:
        self.pedal_speed_kph -= 5 * speed_uom_kph
      # Clip ACC speed between 0 and 170 KPH.
      self.pedal_speed_kph = min(self.pedal_speed_kph, 170)
      self.pedal_speed_kph = max(self.pedal_speed_kph, 1)
    # If something disabled cruise control, disable PCC too
    elif (self.enable_pedal_cruise == True and
          CS.pcm_acc_status != 0):
      self.enable_pedal_cruise = False
    
    # Notify if PCC was toggled
    if prev_enable_pedal_cruise and not self.enable_pedal_cruise:
      CS.UE.custom_alert_message(3, "PCC Disabled", 150, 4)
      CS.cstm_btns.set_button_status("pedal", PCCState.STANDBY)
    elif self.enable_pedal_cruise and not prev_enable_pedal_cruise:
      CS.UE.custom_alert_message(2, "PCC Enabled", 150)
      CS.cstm_btns.set_button_status("pedal", PCCState.ENABLED)

    # Update the UI to show whether the current car state allows ACC.
    if CS.cstm_btns.get_button_status("pedal") in [PCCState.STANDBY, PCCState.NOT_READY]:
      if (enabled
          and CruiseState.is_off(CS.pcm_acc_status)):
        CS.cstm_btns.set_button_status("pedal", PCCState.STANDBY)
      else:
        CS.cstm_btns.set_button_status("pedal", PCCState.NOT_READY)
          
    # Update prev state after all other actions.
    self.prev_cruise_buttons = CS.cruise_buttons
    self.prev_pcm_acc_status = CS.pcm_acc_status
    

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
    v_ego_pid = max(CS.v_ego_raw, MIN_CAN_SPEED)
    deadzone = interp(v_ego_pid, CS.CP.longPidDeadzoneBP, CS.CP.longPidDeadzoneV)
    #BBAD adding overridet to pid to see if we can engage sooner
    override = self.enable_pedal_cruise and CS.v_ego_raw *  CV.MS_TO_KPH > self.pedal_speed_kph and CS.user_gas_pressed
    # we will try to feed forward the pedal position.... we might want to feed the last output_gb....
    # it's all about testing now.
    output_gb = self.pid.update(self.v_pid, v_ego_pid, speed=v_ego_pid, override=override, deadzone=deadzone, feedforward=last_output_gb, freeze_integrator=prevent_overshoot)
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
      if (CS.torqueLevel < TORQUE_LEVEL_ACC) and (CS.v_ego_raw < self.prev_v_ego):
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
      tesla_accel = apply_accel
    tesla_accel, self.accel_steady = accel_hysteresis(tesla_accel, self.accel_steady, enabled)
    
    tesla_accel = clip(tesla_accel, 0., 1.) if self.enable_pedal_cruise else 0.
    enable_gas = 1 if self.enable_pedal_cruise else 0
    self.torqueLevel_last = CS.torqueLevel
    self.prev_actuator_gas = tesla_accel * enable_gas
    self.prev_v_ego = CS.v_ego_raw
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
    safe_dist_m = CS.v_ego_raw * self.follow_time
    # Current speed in kph
    actual_speed = CS.v_ego_raw * CV.MS_TO_KPH
    available_speed = self.pedal_speed_kph - actual_speed
    # button to issue
    new_speed = 0.
    # debug msg
    msg = None

    #print "dRel: ", self.lead_1.dRel," yRel: ", self.lead_1.yRel, " vRel: ", self.lead_1.vRel, " aRel: ", self.lead_1.aRel, " vLead: ", self.lead_1.vLead, " vLeadK: ", self.lead_1.vLeadK, " aLeadK: ",     self.lead_1.aLeadK

    ###   Logic to determine best cruise speed ###

    if self.enable_pedal_cruise:
      # if cruise is set to faster than the max speed, slow down
      new_speed = actual_speed
      if actual_speed > self.pedal_speed_kph:
        msg =  "Slow to max"
        new_speed = actual_speed - 1 
      # If lead_dist is reported as 0, no one is detected in front of you so you
      # can speed up don't speed up when steer-angle > 2; vision radar often
      # loses lead car in a turn.
      elif lead_dist == 0 and CS.angle_steers < 5.0:
        if actual_speed < self.pedal_speed_kph: 
          msg =  "Accel to max"
          new_speed = actual_speed + 0.5
      # if we have a populated lead_distance
      # TODO: make angle dependent on speed
      elif (lead_dist == 0 or lead_dist >= safe_dist_m) and CS.angle_steers >= 5.0:
        new_speed = actual_speed
        msg = "Safe distance & turning: steady speed"
      elif (lead_dist > 0):
        ### Slowing down ###
        #Reduce speed if rel_speed < -15kph so you don't rush up to lead car
        if rel_speed < -15  and lead_dist > 2. * safe_dist_m:
          msg =  "Approaching fast (-15), still twice the safe distance, slow down"
          new_speed = actual_speed + rel_speed/2
        #Reduce speed if rel_speed < -5kph so you don't rush up to lead car
        elif rel_speed < -5  and lead_dist >  1.5 * safe_dist_m:
          msg =  "Approaching fast (-5), still 1.5 the safe distance, slow down"
          new_speed = actual_speed + rel_speed * 2
        # Reduce speed significantly if lead_dist < 60% of  safe dist
        elif lead_dist < (safe_dist_m * 0.3) and rel_speed < 2:
          msg =  "90pct down"
          new_speed = actual_speed * 0.1
        # and if the lead car isn't pulling away
        elif lead_dist < (safe_dist_m * 0.5) and rel_speed < 0:
          msg =  "70pct down"
          new_speed = actual_speed * 0.3
        # we're close to the safe distance, so make slow adjustments
        # only adjust every 1 secs
        elif lead_dist < (safe_dist_m * 0.9) and rel_speed < 0:
          msg =  "10pct down"
          new_speed = actual_speed + rel_speed * 2
        ### Speed up ###
        # don't speed up again until you have more than a safe distance in front
        # only adjust every 2 sec
        elif lead_dist > (safe_dist_m * 1.2) or rel_speed > 5:
          msg = "Lead moving ahead fast: increase speed"
          new_speed = actual_speed  + 1
        else:
          msg = "Have lead and do nothing"
          new_speed = actual_speed
      else:
        msg = "No lead and do nothing"
        new_speed = actual_speed
      if msg:
        print msg  
      new_speed = clip(new_speed,MIN_PCC_V,MAX_PCC_V)
    return new_speed * CV.KPH_TO_MS
