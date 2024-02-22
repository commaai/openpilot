from cereal import car
from openpilot.common.numpy_fast import clip, interp
from openpilot.common.conversions import Conversions as CV
from openpilot.common.realtime import DT_CTRL
from opendbc.can.packer import CANPacker
from openpilot.selfdrive.car import apply_std_steer_angle_limits
from openpilot.selfdrive.car.ford import fordcan
from openpilot.selfdrive.car.ford.values import CANFD_CAR, CAR, CarControllerParams

LongCtrlState = car.CarControl.Actuators.LongControlState
VisualAlert = car.CarControl.HUDControl.VisualAlert


def apply_ford_curvature_limits(apply_curvature, apply_curvature_last, current_curvature, v_ego_raw):
  # No blending at low speed due to lack of torque wind-up and inaccurate current curvature
  if v_ego_raw > 9:
    apply_curvature = clip(apply_curvature, current_curvature - CarControllerParams.CURVATURE_ERROR,
                           current_curvature + CarControllerParams.CURVATURE_ERROR)

  # Curvature rate limit after driver torque limit
  apply_curvature = apply_std_steer_angle_limits(apply_curvature, apply_curvature_last, v_ego_raw, CarControllerParams)

  return clip(apply_curvature, -CarControllerParams.CURVATURE_MAX, CarControllerParams.CURVATURE_MAX)

def hysteresis(current_value, old_value, target, stdDevLow: float, stdDevHigh: float):
  if target - stdDevLow < current_value < target + stdDevHigh:
    result = old_value
  elif current_value <= target - stdDevLow:
    result = 1
  elif current_value >= target + stdDevHigh:
    result = 0

  return result

# Use hysteresis to determine if brake/pre-charger actiator needs to be activated or not.
def actuators_calc(self, brake):
  ts = self.frame * DT_CTRL
  brake_actuate = hysteresis(brake, self.brake_actuate_last, self.brake_actutator_target, self.brake_actutator_stdDevLow, self.brake_actutator_stdDevHigh)
  self.brake_actuate_last = brake_actuate

  precharge_actuate = hysteresis(brake, self.precharge_actuate_last, self.precharge_actutator_target, self.precharge_actutator_stdDevLow, self.precharge_actutator_stdDevHigh)
  if precharge_actuate and not self.precharge_actuate_last:
    self.precharge_actuate_ts = ts
  elif not precharge_actuate:
    self.precharge_actuate_ts = 0

  if (
      precharge_actuate and 
      not brake_actuate and
      self.precharge_actuate_ts > 0 and 
      brake > (self.precharge_actutator_target - self.precharge_actutator_stdDevLow) and 
      (ts - self.precharge_actuate_ts) > (200 * DT_CTRL) 
    ):
    # release the pre-charge if active for more than 2 seconds and the brake actuator was not activated
    precharge_actuate = False

  self.precharge_actuate_last = precharge_actuate

  return precharge_actuate, brake_actuate

class CarController:
  def __init__(self, dbc_name, CP, VM):
    self.CP = CP
    self.VM = VM
    self.packer = CANPacker(dbc_name)
    self.CAN = fordcan.CanBus(CP)
    self.frame = 0

    self.apply_curvature_last = 0
    self.main_on_last = False
    self.lkas_enabled_last = False
    self.steer_alert_last = False
    self.brake_actuate_last = 0
    self.precharge_actuate_last = 0
    self.precharge_actuate_ts = 0

    ## Brake Actuator ##

    # If no hysteresis is applied - at what accel shold the the actuator activate
    self.brake_actutator_target = -0.1

    # Activates at self.brake_actutator_target - self.brake_actutator_stdDevLow
    # Resulted default value: -0.3
    self.brake_actutator_stdDevLow = 0.2
  
    # Deactivates at self.brake_actutator_target + self.brake_actutator_stdDevHigh
    # Resulted default value: .0
    self.brake_actutator_stdDevHigh = 0.1

    ## Pre charge Actuator ##

    # If no hysteresis is applied - at what accel shold the the actuator activate
    self.precharge_actutator_target = -0.1

    # Activates at self.precharge_actutator_target - self.precharge_actutator_stdDevLow
    # Resulted default value: -0.2
    self.precharge_actutator_stdDevLow = 0.1

    # Deactivates at self.precharge_actutator_target + self.precharge_actutator_stdDevHigh
    # Resulted default value: 0
    self.precharge_actutator_stdDevHigh = 0.1

    # The value we want the brake to start from.
    self.brake_0_point = 0

    # At what bake accel should the custom point 
    self.brake_converge_at = -1.5

    if self.CP.carFingerprint in  {CAR.F_150_LIGHTNING_MK1, CAR.MUSTANG_MACH_E_MK1}:
      # Different values for EV - we still have to pre-charge we precharge and brake at the same time
      self.brake_actutator_target = self.precharge_actutator_target = -0.1
      self.brake_actutator_stdDevLow = self.precharge_actutator_stdDevLow = 0.2
      self.brake_actutator_stdDevHigh = self.precharge_actutator_stdDevHigh = 0.1

  def update(self, CC, CS, now_nanos):
    can_sends = []

    actuators = CC.actuators
    hud_control = CC.hudControl

    main_on = CS.out.cruiseState.available
    steer_alert = hud_control.visualAlert in (VisualAlert.steerRequired, VisualAlert.ldw)
    fcw_alert = hud_control.visualAlert == VisualAlert.fcw

    ### acc buttons ###
    if CC.cruiseControl.cancel:
      can_sends.append(fordcan.create_button_msg(self.packer, self.CAN.camera, CS.buttons_stock_values, cancel=True))
      can_sends.append(fordcan.create_button_msg(self.packer, self.CAN.main, CS.buttons_stock_values, cancel=True))
    elif CC.cruiseControl.resume and (self.frame % CarControllerParams.BUTTONS_STEP) == 0:
      can_sends.append(fordcan.create_button_msg(self.packer, self.CAN.camera, CS.buttons_stock_values, resume=True))
      can_sends.append(fordcan.create_button_msg(self.packer, self.CAN.main, CS.buttons_stock_values, resume=True))
    # if stock lane centering isn't off, send a button press to toggle it off
    # the stock system checks for steering pressed, and eventually disengages cruise control
    elif CS.acc_tja_status_stock_values["Tja_D_Stat"] != 0 and (self.frame % CarControllerParams.ACC_UI_STEP) == 0:
      can_sends.append(fordcan.create_button_msg(self.packer, self.CAN.camera, CS.buttons_stock_values, tja_toggle=True))

    ### lateral control ###
    # send steer msg at 20Hz
    if (self.frame % CarControllerParams.STEER_STEP) == 0:
      if CC.latActive:
        # apply rate limits, curvature error limit, and clip to signal range
        current_curvature = -CS.out.yawRate / max(CS.out.vEgoRaw, 0.1)
        apply_curvature = apply_ford_curvature_limits(actuators.curvature, self.apply_curvature_last, current_curvature, CS.out.vEgoRaw)
      else:
        apply_curvature = 0.

      self.apply_curvature_last = apply_curvature

      if self.CP.carFingerprint in CANFD_CAR:
        # TODO: extended mode
        mode = 1 if CC.latActive else 0
        counter = (self.frame // CarControllerParams.STEER_STEP) % 0xF
        can_sends.append(fordcan.create_lat_ctl2_msg(self.packer, self.CAN, mode, 0., 0., -apply_curvature, 0., counter))
      else:
        can_sends.append(fordcan.create_lat_ctl_msg(self.packer, self.CAN, CC.latActive, 0., 0., -apply_curvature, 0.))

    # send lka msg at 33Hz
    if (self.frame % CarControllerParams.LKA_STEP) == 0:
      can_sends.append(fordcan.create_lka_msg(self.packer, self.CAN))

    ### longitudinal control ###
    # send acc msg at 50Hz
    if self.CP.openpilotLongitudinalControl and (self.frame % CarControllerParams.ACC_CONTROL_STEP) == 0:
      # Both gas and accel are in m/s^2, accel is used solely for braking
      accel = clip(actuators.accel, CarControllerParams.ACCEL_MIN, CarControllerParams.ACCEL_MAX)
      gas = accel
      if not CC.longActive or gas < CarControllerParams.MIN_GAS:
        gas = CarControllerParams.INACTIVE_GAS

      stopping = CC.actuators.longControlState == LongCtrlState.stopping

      # Calculate targetSpeed
      targetSpeed = actuators.speed
      if not CC.longActive and hud_control.setSpeed:
        targetSpeed = hud_control.setSpeed

      precharge_actuate, brake_actuate = actuators_calc(self, accel)
      brake = accel
      if brake < 0 :
        brake_clip = self.brake_actutator_target - self.brake_actutator_stdDevLow
        brake = interp(accel, [ CarControllerParams.ACCEL_MIN, self.brake_converge_at, brake_clip], [CarControllerParams.ACCEL_MIN, self.brake_converge_at, self.brake_0_point])

      can_sends.append(fordcan.create_acc_msg(self.packer, self.CAN, CC.longActive, gas, brake, stopping, brake_actuate, precharge_actuate, v_ego_kph=targetSpeed * CV.MS_TO_KPH))

    ### ui ###
    send_ui = (self.main_on_last != main_on) or (self.lkas_enabled_last != CC.latActive) or (self.steer_alert_last != steer_alert)
    # send lkas ui msg at 1Hz or if ui state changes
    if (self.frame % CarControllerParams.LKAS_UI_STEP) == 0 or send_ui:
      can_sends.append(fordcan.create_lkas_ui_msg(self.packer, self.CAN, main_on, CC.latActive, steer_alert, hud_control, CS.lkas_status_stock_values))
    # send acc ui msg at 5Hz or if ui state changes
    if (self.frame % CarControllerParams.ACC_UI_STEP) == 0 or send_ui:
      can_sends.append(fordcan.create_acc_ui_msg(self.packer, self.CAN, self.CP, main_on, CC.latActive,
                                         fcw_alert, CS.out.cruiseState.standstill, hud_control,
                                         CS.acc_tja_status_stock_values))

    self.main_on_last = main_on
    self.lkas_enabled_last = CC.latActive
    self.steer_alert_last = steer_alert

    new_actuators = actuators.copy()
    new_actuators.curvature = self.apply_curvature_last

    self.frame += 1
    return new_actuators, can_sends
