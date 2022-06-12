import math
from cereal import car
from common.numpy_fast import clip, interp
from selfdrive.car.ford import fordcan
from selfdrive.car.ford.values import CarControllerParams
from opendbc.can.packer import CANPacker

VisualAlert = car.CarControl.HUDControl.VisualAlert


class CarController():
  def __init__(self, dbc_name, CP, VM):
    self.CP = CP
    self.VM = VM
    self.packer = CANPacker(dbc_name)

    self.apply_steer_last = 0
    self.steer_rate_limited = False
    self.main_on_last = False
    self.lkas_enabled_last = False
    self.steer_alert_last = False

  def apply_ford_steer_angle_limits(self, desired_curvature, CS):
    # convert curvature to steering angle
    apply_steer = self.VM.get_steer_from_curvature(desired_curvature, CS.out.vEgo, 0.0)

    # rate limit steering angle
    steer_up = apply_steer * self.apply_steer_last > 0. and abs(apply_steer) > abs(self.apply_steer_last)
    rate_limit = CarControllerParams.STEER_RATE_LIMIT_UP if steer_up else CarControllerParams.STEER_RATE_LIMIT_DOWN
    max_angle_diff = interp(CS.out.vEgo, rate_limit.speed_points, rate_limit.max_angle_diff_points)
    apply_steer = clip(apply_steer, self.apply_steer_last - max_angle_diff, self.apply_steer_last + max_angle_diff)

    # convert back to curvature
    apply_curvature = self.VM.calc_curvature(math.radians(apply_steer), CS.out.vEgo, 0.0)
    self.steer_rate_limited = apply_curvature != desired_curvature

    return apply_curvature, apply_steer

  def update(self, CC, CS, frame):
    can_sends = []

    actuators = CC.actuators
    hud_control = CC.hudControl

    main_on = CS.out.cruiseState.available
    steer_alert = hud_control.visualAlert in (VisualAlert.steerRequired, VisualAlert.ldw)

    if CC.cruiseControl.cancel:
      # cancel stock ACC
      can_sends.append(fordcan.spam_cancel_button(self.packer))


    ### lateral control ###

    # apply rate limits
    apply_curvature, apply_steer = self.apply_ford_steer_angle_limits(actuators, CS.out.vEgo)

    # send steering commands at 20Hz
    if (frame % CarControllerParams.LKAS_STEER_STEP) == 0:
      lca_rq = 1 if CC.latActive else 0

      # calculate path angle from rate limited curvature
      path_angle = (CS.out.vEgo * apply_curvature) / 2

      # ramp rate: 0=Slow, 1=Medium, 2=Fast, 3=Immediately
      ramp_type = 2
      # precision: 0=Comfortable, 1=Precise
      precision = 0

      can_sends.append(fordcan.create_lkas_command(self.packer, apply_steer, apply_curvature))
      can_sends.append(fordcan.create_tja_command(self.packer, lca_rq, ramp_type, precision,
                                                  0, path_angle, 0, 0))

      self.apply_steer_last = apply_steer


    ### ui ###
    send_ui = (self.main_on_last != main_on) or (self.lkas_enabled_last != CC.latActive) or (self.steer_alert_last != steer_alert)

    # send lkas ui command at 1Hz or if ui state changes
    if (frame % CarControllerParams.LKAS_UI_STEP) == 0 or send_ui:
      can_sends.append(fordcan.create_lkas_ui_command(self.packer, main_on, CC.latActive, steer_alert, CS.lkas_status_stock_values))

    # send acc ui command at 20Hz or if ui state changes
    if (frame % CarControllerParams.ACC_UI_STEP) == 0 or send_ui:
      can_sends.append(fordcan.create_acc_ui_command(self.packer, main_on, CC.latActive, CS.acc_tja_status_stock_values))

    self.main_on_last = main_on
    self.lkas_enabled_last = CC.latActive
    self.steer_alert_last = steer_alert

    new_actuators = actuators.copy()
    new_actuators.curvature = apply_curvature

    return new_actuators, can_sends
