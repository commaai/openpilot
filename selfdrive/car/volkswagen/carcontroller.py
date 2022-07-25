from cereal import car
from opendbc.can.packer import CANPacker
from common.numpy_fast import clip
from common.conversions import Conversions as CV
from selfdrive.car import apply_std_steer_torque_limits
from selfdrive.car.volkswagen import volkswagencan, pqcan
from selfdrive.car.volkswagen.values import PQ_CARS, CANBUS, MQB_LDW_MESSAGES, PQ_LDW_MESSAGES, CarControllerParams as P

VisualAlert = car.CarControl.HUDControl.VisualAlert


class CarController:
  def __init__(self, dbc_name, CP, VM):
    self.CP = CP
    self.apply_steer_last = 0
    self.frame = 0
    self.packer_pt = CANPacker(dbc_name)

    if CP.carFingerprint in PQ_CARS:
      self.create_steering_control = pqcan.create_steering_control
      self.create_lka_hud_control = pqcan.create_lka_hud_control
      self.create_acc_buttons_control = pqcan.create_acc_buttons_control
      self.create_acc_accel_control = pqcan.create_acc_accel_control
      self.create_acc_hud_control = pqcan.create_acc_hud_control
      self.ldw_step = P.PQ_LDW_STEP
      self.ldw_messages = PQ_LDW_MESSAGES
    else:
      self.create_steering_control = volkswagencan.create_steering_control
      self.create_lka_hud_control = volkswagencan.create_lka_hud_control
      self.create_acc_buttons_control = volkswagencan.create_acc_buttons_control
      self.create_acc_accel_control = None
      self.create_acc_hud_control = None
      self.ldw_step = P.MQB_LDW_STEP
      self.ldw_messages = MQB_LDW_MESSAGES

    self.hcaSameTorqueCount = 0
    self.hcaEnabledFrameCount = 0

  def update(self, CC, CS, ext_bus):
    actuators = CC.actuators
    hud_control = CC.hudControl

    can_sends = []

    # **** Steering Controls ************************************************ #

    if self.frame % P.HCA_STEP == 0:
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
        new_steer = int(round(actuators.steer * P.STEER_MAX))
        apply_steer = apply_std_steer_torque_limits(new_steer, self.apply_steer_last, CS.out.steeringTorque, P)
        if apply_steer == 0:
          hcaEnabled = False
          self.hcaEnabledFrameCount = 0
        else:
          self.hcaEnabledFrameCount += 1
          if self.hcaEnabledFrameCount >= 118 * (100 / P.HCA_STEP):  # 118s
            hcaEnabled = False
            self.hcaEnabledFrameCount = 0
          else:
            hcaEnabled = True
            if self.apply_steer_last == apply_steer:
              self.hcaSameTorqueCount += 1
              if self.hcaSameTorqueCount > 1.9 * (100 / P.HCA_STEP):  # 1.9s
                apply_steer -= (1, -1)[apply_steer < 0]
                self.hcaSameTorqueCount = 0
            else:
              self.hcaSameTorqueCount = 0
      else:
        hcaEnabled = False
        apply_steer = 0

      self.apply_steer_last = apply_steer
      can_sends.append(self.create_steering_control(self.packer_pt, CANBUS.pt, apply_steer, hcaEnabled))

    # **** Acceleration Controls ******************************************** #

    if self.CP.openpilotLongitudinalControl:
      if self.frame % P.ACC_CONTROL_STEP:
        if CC.longActive:
          adr_status = 1
        elif CS.out.cruiseState.available:
          adr_status = 2
        else:
          adr_status = 0
        accel = clip(actuators.accel, P.ACCEL_MIN, P.ACCEL_MAX) if CC.longActive else 0
        can_sends.append(self.create_acc_accel_control(self.packer_pt, CANBUS.pt, adr_status, accel))
      if self.frame % P.ACC_HUD_STEP:
        if CC.longActive:
          acc_status = 3
        elif CS.out.cruiseState.available:
          acc_status = 2
        elif CS.out.accFaulted:
          acc_status = 6
        else:
          acc_status = 0
        set_speed = hud_control.setSpeed * CV.MS_TO_KPH  # FIXME: follow the recent displayed-speed updates, also use mph_kmh toggle to fix display rounding problem?
        can_sends.append(self.create_acc_hud_control(self.packer_pt, CANBUS.pt, acc_status, set_speed,
                                                     hud_control.leadVisible))

    # **** HUD Controls ***************************************************** #

    if self.frame % self.ldw_step == 0:
      hud_alert = 0
      if hud_control.visualAlert in (VisualAlert.steerRequired, VisualAlert.ldw):
        hud_alert = self.ldw_messages["laneAssistTakeOver"]

      can_sends.append(self.create_lka_hud_control(self.packer_pt, CANBUS.pt, CC.enabled, CS.out.steeringPressed,
                                                   hud_alert, hud_control.leftLaneVisible, hud_control.rightLaneVisible,
                                                   CS.ldw_stock_values, hud_control.leftLaneDepart, hud_control.rightLaneDepart))

    # **** ACC Button Controls ********************************************** #

    if self.CP.pcmCruise and self.frame % P.GRA_ACC_STEP == 0:
      idx = (CS.gra_stock_values["COUNTER"] + 1) % 16
      if CC.cruiseControl.cancel:
        can_sends.append(self.create_acc_buttons_control(self.packer_pt, ext_bus, CS.gra_stock_values, idx, cancel=True))
      elif CC.cruiseControl.resume:
        can_sends.append(self.create_acc_buttons_control(self.packer_pt, ext_bus, CS.gra_stock_values, idx, resume=True))

    new_actuators = actuators.copy()
    new_actuators.steer = self.apply_steer_last / P.STEER_MAX

    self.frame += 1
    return new_actuators, can_sends
