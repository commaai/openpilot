import math
from cereal import car
from common.numpy_fast import clip, interp
from opendbc.can.packer import CANPacker
from selfdrive.car.ford import fordcan
from selfdrive.car.ford.values import CarControllerParams

VisualAlert = car.CarControl.HUDControl.VisualAlert


def apply_ford_steer_angle_limits(apply_angle, apply_angle_last, vEgo):
  # rate limit
  steer_up = apply_angle_last * apply_angle > 0. and abs(apply_angle) > abs(apply_angle_last)
  rate_limit = CarControllerParams.RATE_LIMIT_UP if steer_up else CarControllerParams.RATE_LIMIT_DOWN
  max_angle_diff = interp(vEgo, rate_limit.speed_points, rate_limit.max_angle_diff_points)
  apply_angle = clip(apply_angle, (apply_angle_last - max_angle_diff), (apply_angle_last + max_angle_diff))

  # absolute limit (LatCtlPath_An_Actl)
  apply_path_angle = math.radians(apply_angle) / CarControllerParams.STEER_RATIO
  apply_path_angle = clip(apply_path_angle, -0.4995, 0.5240)
  apply_angle = math.degrees(apply_path_angle) * CarControllerParams.STEER_RATIO

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
    elif CC.cruiseControl.resume:
      can_sends.append(fordcan.create_button_command(self.packer, CS.buttons_stock_values, resume=True))

    # if stock lane centering is active or in standby, toggle it off
    # the stock system checks for steering pressed, and eventually disengages cruise control
    if (self.frame % 200) == 0 and CS.acc_tja_status_stock_values["Tja_D_Stat"] != 0:
      can_sends.append(fordcan.create_button_command(self.packer, CS.buttons_stock_values, tja_toggle=True))


    ### lateral control ###
    # Ford lane centering command uses a third order polynomial to describe the road centerline.
    # The polynomial is defined by the following coefficients:
    #   c0: lateral offset between the vehicle and the centerline
    #   c1: heading angle between the vehicle and the centerline
    #   c2: curvature of the centerline
    #   c3: rate of change of curvature of the centerline
    # As the EPS controller combines this information with other sensor data, the steering angle
    # cannot be easily controlled. A closed loop controller is used to achieve the desired steering
    # angle.

    # send steering commands at 20Hz
    if (self.frame % CarControllerParams.LKAS_STEER_STEP) == 0:
      if CC.latActive:
        new_steer = actuators.steeringAngleDeg + actuators.steer * CarControllerParams.STEER_MAX

        lca_rq = 1
        apply_angle = apply_ford_steer_angle_limits(new_steer, self.apply_angle_last, CS.out.vEgo)
      else:
        lca_rq = 0
        apply_angle = CS.out.steeringAngleDeg

      # use LatCtlPath_An_Actl to actuate steering
      path_angle = math.radians(apply_angle) / CarControllerParams.STEER_RATIO

      # set slower ramp type when small steering angle change
      # 0=Slow, 1=Medium, 2=Fast, 3=Immediately
      angle_error = abs(CS.out.steeringAngleDeg - actuators.steeringAngleDeg)
      if angle_error < 3.0:
        ramp_type = 0
      elif angle_error < 6.0:
        ramp_type = 1
      else:
        ramp_type = 2
      precision = 1  # 0=Comfortable, 1=Precise

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
