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
from selfdrive.car.tesla.tesla_longcontrol import LongControl, STARTING_TARGET_SPEED

# Accel limits
ACCEL_HYST_GAP = 0.02  # don't change accel command for small oscilalitons within this value
ACCEL_MAX = 1
ACCEL_MIN = -1
ACCEL_SCALE = max(ACCEL_MAX, -ACCEL_MIN)
ACCEL_REWIND_MAX = 0.04
PEDAL_DEADZONE = 0.1

#BBTODO: move the vehicle variables; maybe make them speed variable
TORQUE_LEVEL_ACC = 30.
TORQUE_LEVEL_DECEL = -30.

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

def calc_plan(CS, CP, events, PL, LaC, LoC, v_cruise_kph, driver_status, geofence):
   # plan runs always, independently of the state
   force_decel = False #driver_status.awareness < 0. or (geofence is not None and not geofence.in_geofence)
   plan_packet = PL.update(CS, LaC, LoC, v_cruise_kph, force_decel)
   plan = plan_packet.plan
   plan_ts = plan_packet.logMonoTime

   # disable if lead isn't close when system is active and brake is pressed to avoid
   # unexpected vehicle accelerations
   if CS.brakePressed and plan.vTargetFuture >= STARTING_TARGET_SPEED and not CP.radarOffCan and CS.vEgo < 0.3:
     #events.append(create_event('noTarget', [ET.NO_ENTRY, ET.IMMEDIATE_DISABLE]))
     #BBTODOsend special message
     print "disabled due to condition in calc_plans"

   return plan, plan_ts

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
    self.LoC = None
    self.torqueLevel_last = 0.
    self.prev_v_ego = 0.
    self.lastPedalForZeroTorque = 0.
    
    

  def update_stat(self,CS, enabled, sendcan):
    if self.LoC == None:
      self.LoC = LongControl(CS.CP,get_compute_gb_models)
    can_sends = []
    #BBTODO: a better way to engage the pedal early and reset its CAN
    # on first brake press check if hardware present; being on CAN2 values are not included in fingerprinting
    if (CS.brake_pressed) and (CS.user_gas >= 0 ) and (not self.pedal_hardware_present) and (self.pedal_hardware_first_check):
      self.pedal_hardware_present = True
      CS.config_ui_buttons(True)
      self.pedal_hardware_first_check = False
    if (CS.brake_pressed) and (not self.pedal_hardware_present) and (self.pedal_hardware_first_check):
      self.pedal_hardware_first_check = False
      CS.config_ui_buttons(False)
    if not self.pedal_hardware_present:
      if (CS.cstm_btns.get_button_status("pedal")>0):
        #no pedal hardware, disable button
        CS.cstm_btns.set_button_status("pedal",0)
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
      CS.UE.custom_alert_message(3,"PDL Disabled",150,4)
      CS.cstm_btns.set_button_status("pedal",1)
    # process any stalk movement
    curr_time_ms = _current_time_millis()
    adaptive_cruise_prev = self.enable_pedal_cruise
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
      self.last_cruise_stalk_pull_time = curr_time_ms
    elif (CS.cruise_buttons == CruiseButtons.CANCEL and
          self.prev_cruise_buttons != CruiseButtons.CANCEL):
      self.enable_pedal_cruise = False
      if adaptive_cruise_prev == True:
        CS.UE.custom_alert_message(3,"PDL Disabled",150,4)
        CS.cstm_btns.set_button_status("pedal",1)
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
    # gas and brake
    apply_accel = actuators.gas
    apply_brake = actuators.brake

    #slow deceleration
    if (apply_accel > PEDAL_DEADZON) and (apply_accel < self.prev_actuator_gas) and (apply_brake == 0):
      if (CS.torqueLevel < TORQUE_LEVEL_ACC) and (CS.v_ego < self.prev_v_ego):
        tesla_accel = self.prev_actuator_gas
      else:
        tesla_accel = clip(apply_accel,self.prev_actuator_gas - ACCEL_REWIND_MAX,self.prev_actuator_gas)
    else:
        tesla_acces = apply_accel

    #coast
    if (apply_accel <= PEDAL_DEADZONE) and (apply_brake == 0):
      if (CS.torqueLevel < TORQUE_LEVEL_ACC):
        #we are in the torque dead zone, 
        tesla_accel = self.prev_actuator_gas
      else:
        tesla_accel = self.prev_actuator_gas - ACCEL_REWIND_MAX

    #save position for brake
    if (CS.torqueLevel < TORQUE_LEVEL_ACC) and (CS.torqueLevel > TORQUE_LEVEL_DECEL):
      self.lastPedalForZeroTorque = self.prev_actuator_gas

    if (apply_brake > 0) and (apply_gas == 0):
      if self.lastPedalForZeroTorque > 0:
        tesla_accel = (1 - apply_brake) * self.lastPedalForZeroTorque
        self.lastPedalForZeroTorque = 0.
      else:
        tesla_accel = (1 - apply_brake) * self.prev_actuator_gas

    tesla_accel, self.accel_steady = accel_hysteresis(tesla_accel, self.accel_steady, enabled)
    
    apply_gas = clip(apply_accel, 0., 1.) if self.enable_pedal_cruise else 0.
    enable_gas = 1 if self.enable_pedal_cruise else 0
    self.torqueLevel_last = CS.torqueLevel
    self.prev_actuator_gas = tesla_accel * enable_gas
    self.prev_v_ego = CS.v_ego
    return apply_gas,enable_gas,idx
