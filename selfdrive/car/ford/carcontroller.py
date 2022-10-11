import math
from cereal import car
from common.numpy_fast import clip, interp
from opendbc.can.packer import CANPacker
from selfdrive.car.ford import fordcan
from selfdrive.car.ford.values import CANBUS, CarControllerParams

VisualAlert = car.CarControl.HUDControl.VisualAlert


def apply_ford_steer_angle_limits(apply_angle, apply_angle_last, vEgo):
  # rate limit
  steer_up = apply_angle_last * apply_angle > 0. and abs(apply_angle) > abs(apply_angle_last)
  rate_limit = CarControllerParams.RATE_LIMIT_UP if steer_up else CarControllerParams.RATE_LIMIT_DOWN
  max_angle_diff = interp(vEgo, rate_limit.speed_points, rate_limit.max_angle_diff_points)
  apply_angle = clip(apply_angle, (apply_angle_last - max_angle_diff), (apply_angle_last + max_angle_diff))

  # absolute limit (LatCtlPath_An_Actl)
  apply_path_angle = math.radians(apply_angle) / CarControllerParams.LKAS_STEER_RATIO
  apply_path_angle = clip(apply_path_angle, -0.5, 0.5235)
  apply_angle = math.degrees(apply_path_angle) * CarControllerParams.LKAS_STEER_RATIO

  return apply_angle


class CarController:
  def __init__(self, dbc_name, CP, VM):
    self.CP = CP
    self.VM = VM
    self.packer = CANPacker(dbc_name)
    self.frame = 0

    self.apply_angle_last = 0
    self.main_on_last = False
    self.lkas_enabled_last = False
    self.steer_alert_last = False

  def update(self, CC, CS):
    can_sends = []

    actuators = CC.actuators
    hud_control = CC.hudControl

    main_on = CS.out.cruiseState.available
    steer_alert = hud_control.visualAlert in (VisualAlert.steerRequired, VisualAlert.ldw)

    ### acc buttons ###
    if CC.cruiseControl.cancel:
      can_sends.append(fordcan.create_button_command(self.packer, CS.buttons_stock_values, cancel=True))
      can_sends.append(fordcan.create_button_command(self.packer, CS.buttons_stock_values, cancel=True, bus=CANBUS.main))
    elif CC.cruiseControl.resume and (self.frame % CarControllerParams.BUTTONS_STEP) == 0:
      can_sends.append(fordcan.create_button_command(self.packer, CS.buttons_stock_values, resume=True))
      can_sends.append(fordcan.create_button_command(self.packer, CS.buttons_stock_values, resume=True, bus=CANBUS.main))
    # if stock lane centering isn't off, send a button press to toggle it off
    # the stock system checks for steering pressed, and eventually disengages cruise control
    elif CS.acc_tja_status_stock_values["Tja_D_Stat"] != 0 and (self.frame % CarControllerParams.ACC_UI_STEP) == 0:
      can_sends.append(fordcan.create_button_command(self.packer, CS.buttons_stock_values, tja_toggle=True))


    ### lateral control ###
    if CC.latActive:
      lca_rq = 1
      apply_angle = apply_ford_steer_angle_limits(actuators.steeringAngleDeg, self.apply_angle_last, CS.out.vEgo)
    else:
      lca_rq = 0
      apply_angle = 0.

    # send steering commands at 20Hz
    if (self.frame % CarControllerParams.LKAS_STEER_STEP) == 0:
      # use LatCtlPath_An_Actl to actuate steering
      path_angle = math.radians(apply_angle) / CarControllerParams.LKAS_STEER_RATIO

      # set slower ramp type when small steering angle change
      # 0=Slow, 1=Medium, 2=Fast, 3=Immediately
      steer_change = abs(CS.out.steeringAngleDeg - actuators.steeringAngleDeg)
      if steer_change < 2.0:
        ramp_type = 0
      elif steer_change < 4.0:
        ramp_type = 1
      elif steer_change < 6.0:
        ramp_type = 2
      else:
        ramp_type = 3
      precision = 1  # 0=Comfortable, 1=Precise (the stock system always uses comfortable)

      self.apply_angle_last = apply_angle
      can_sends.append(fordcan.create_lka_command(self.packer, 0, 0))
      can_sends.append(fordcan.create_tja_command(self.packer, lca_rq, ramp_type, precision,
                                                  0, path_angle, 0, 0))


    ### ui ###
    send_ui = (self.main_on_last != main_on) or (self.lkas_enabled_last != CC.latActive) or (self.steer_alert_last != steer_alert)

    # send lkas ui command at 1Hz or if ui state changes
    if (self.frame % CarControllerParams.LKAS_UI_STEP) == 0 or send_ui:
      can_sends.append(fordcan.create_lkas_ui_command(self.packer, main_on, CC.latActive, steer_alert, hud_control, CS.lkas_status_stock_values))

    # send acc ui command at 20Hz or if ui state changes
    if (self.frame % CarControllerParams.ACC_UI_STEP) == 0 or send_ui:
      can_sends.append(fordcan.create_acc_ui_command(self.packer, main_on, CC.latActive, hud_control, CS.acc_tja_status_stock_values))

    self.main_on_last = main_on
    self.lkas_enabled_last = CC.latActive
    self.steer_alert_last = steer_alert

    new_actuators = actuators.copy()
    new_actuators.steeringAngleDeg = self.apply_angle_last

    self.frame += 1
    return new_actuators, can_sends
