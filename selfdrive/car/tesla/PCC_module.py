from selfdrive.services import service_list
from selfdrive.car.tesla.values import AH, CruiseButtons, CAR
from selfdrive.config import Conversions as CV
from common.numpy_fast import clip
import selfdrive.messaging as messaging
import os
import subprocess
import time
import zmq

def _current_time_millis():
  return int(round(time.time() * 1000))

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

  def update_stat(self,CS, enabled):
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
    idx = self.pedal_idx
    self.pedal_idx = (self.pedal_idx + 1) % 16
    if not self.pedal_hardware_present or not enabled:
      return 0.,0,idx
    apply_gas = clip(actuators.gas, 0., 1.) if self.enable_pedal_cruise else 0.
    enable_gas = 1 if self.enable_pedal_cruise else 0
    return apply_gas,enable_gas,idx
