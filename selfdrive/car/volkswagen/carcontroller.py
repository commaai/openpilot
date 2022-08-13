from cereal import car
from opendbc.can.packer import CANPacker
from selfdrive.car import apply_std_steer_torque_limits
from selfdrive.car.volkswagen import mqbcan
from selfdrive.car.volkswagen.values import CANBUS, CarControllerParams

VisualAlert = car.CarControl.HUDControl.VisualAlert


class CarController:
  def __init__(self, dbc_name, CP, VM):
    self.CP = CP
    self.CCP = CarControllerParams(CP)
    self.CCS = mqbcan
    self.packer_pt = CANPacker(dbc_name)

    self.apply_steer_last = 0
    self.frame = 0
    self.hcaSameTorqueCount = 0
    self.hcaEnabledFrameCount = 0

  def update(self, CC, CS, ext_bus):
    actuators = CC.actuators
    hud_control = CC.hudControl

    can_sends = []

    # **** Steering Controls ************************************************ #

    if self.frame % self.CCP.HCA_STEP == 0:
      # Logic to avoid HCA state 4 "refused":
      #   * Don't steer unless HCA is in state 3 "ready" or 5 "active"
      #   * Don't steer at standstill
      #   * Don't send > 3.00 Newton-meters torque
      #   * Don't send the same torque for > 6 seconds
      #   * Don't send uninterrupted steering for > 360 seconds
      # One frame of HCA disabled is enough to reset the timer, without zeroing the
      # torque value. Do that anytime we happen to have 0 torque, or failing that,
      # when exceeding ~1/3 the 360 second timer.

      if CC.latActive:
        new_steer = int(round(actuators.steer * self.CCP.STEER_MAX))
        apply_steer = apply_std_steer_torque_limits(new_steer, self.apply_steer_last, CS.out.steeringTorque, self.CCP)
        if apply_steer == 0:
          hcaEnabled = False
          self.hcaEnabledFrameCount = 0
        else:
          self.hcaEnabledFrameCount += 1
          if self.hcaEnabledFrameCount >= 118 * (100 / self.CCP.HCA_STEP):  # 118s
            hcaEnabled = False
            self.hcaEnabledFrameCount = 0
          else:
            hcaEnabled = True
            if self.apply_steer_last == apply_steer:
              self.hcaSameTorqueCount += 1
              if self.hcaSameTorqueCount > 1.9 * (100 / self.CCP.HCA_STEP):  # 1.9s
                apply_steer -= (1, -1)[apply_steer < 0]
                self.hcaSameTorqueCount = 0
            else:
              self.hcaSameTorqueCount = 0
      else:
        hcaEnabled = False
        apply_steer = 0

      self.apply_steer_last = apply_steer
      can_sends.append(self.CCS.create_steering_control(self.packer_pt, CANBUS.pt, apply_steer, hcaEnabled))

    # **** HUD Controls ***************************************************** #

    if self.frame % self.CCP.LDW_STEP == 0:
      hud_alert = 0
      if hud_control.visualAlert in (VisualAlert.steerRequired, VisualAlert.ldw):
        hud_alert = self.CCP.LDW_MESSAGES["laneAssistTakeOver"]
      can_sends.append(self.CCS.create_lka_hud_control(self.packer_pt, CANBUS.pt, CS.ldw_stock_values, CC.enabled,
                                                       CS.out.steeringPressed, hud_alert, hud_control))

    # **** Stock ACC Button Controls **************************************** #

    if self.CP.pcmCruise and self.frame % self.CCP.GRA_ACC_STEP == 0:
      idx = (CS.gra_stock_values["COUNTER"] + 1) % 16
      if CC.cruiseControl.cancel:
        can_sends.append(self.CCS.create_acc_buttons_control(self.packer_pt, ext_bus, CS.gra_stock_values, idx, cancel=True))
      elif CC.cruiseControl.resume:
        can_sends.append(self.CCS.create_acc_buttons_control(self.packer_pt, ext_bus, CS.gra_stock_values, idx, resume=True))

    new_actuators = actuators.copy()
    new_actuators.steer = self.apply_steer_last / self.CCP.STEER_MAX

    self.frame += 1
    return new_actuators, can_sends
