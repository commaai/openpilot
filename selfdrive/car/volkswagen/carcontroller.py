from cereal import car
from opendbc.can.packer import CANPacker
from common.numpy_fast import clip
from common.conversions import Conversions as CV
from selfdrive.car import apply_std_steer_torque_limits
from selfdrive.car.volkswagen import mqbcan, pqcan
from selfdrive.car.volkswagen.values import CANBUS, PQ_CARS, CarControllerParams

VisualAlert = car.CarControl.HUDControl.VisualAlert
LongCtrlState = car.CarControl.Actuators.LongControlState


class CarController:
  def __init__(self, dbc_name, CP, VM):
    self.CP = CP
    self.CCP = CarControllerParams(CP)
    self.CCS = pqcan if CP.carFingerprint in PQ_CARS else mqbcan
    self.packer_pt = CANPacker(dbc_name)

    self.apply_steer_last = 0
    self.frame = 0
    self.hcaSameTorqueCount = 0
    self.hcaEnabledFrameCount = 0
    self.acc_starting = False
    self.acc_stopping = False

  def update(self, CC, CS, ext_bus):
    actuators = CC.actuators
    hud_control = CC.hudControl

    can_sends = []

    # **** Acceleration and Braking Controls ******************************** #

    if CS.CP.openpilotLongitudinalControl:
      if self.frame % self.CCP.ACC_CONTROL_STEP == 0:
        if CS.tsk_status in [2, 3, 4, 5]:
          acc_status = 3 if CC.longActive else 2
        else:
          acc_status = CS.tsk_status

        accel = clip(actuators.accel, self.CCP.ACCEL_MIN, self.CCP.ACCEL_MAX) if CC.longActive else 0

        # FIXME: this needs to become a proper state machine
        acc_hold_request, acc_hold_release, acc_hold_type, stopping_distance = False, False, 0, 20.46
        if actuators.longControlState == LongCtrlState.stopping and CS.out.vEgo < 0.2:
          self.acc_stopping = True
          acc_hold_request = True
          if CS.out.cruiseState.standstill:
            acc_hold_type = 1  # hold
            stopping_distance = 3.5
          else:
            acc_hold_type = 3  # hold_standby
            stopping_distance = 0.5
        elif CC.longActive:
          if self.acc_stopping:
            self.acc_starting = True
            self.acc_stopping = False
          self.acc_starting &= CS.out.vEgo < 1.5
          acc_hold_release = self.acc_starting
          acc_hold_type = 4 if self.acc_starting and CS.out.vEgo < 0.1 else 0  # startup
        else:
          self.acc_stopping, self.acc_starting = False, False

        cb_pos = 0.0 if hud_control.leadVisible or CS.out.vEgo < 2.0 else 0.1  # react faster to lead cars, also don't get hung up at DSG clutch release/kiss points when creeping to stop
        cb_neg = 0.0 if accel < 0 else 0.2  # IDK why, but stock likes to zero this out when accel is negative

        can_sends.append(self.CCS.create_acc_06_control(self.packer_pt, CANBUS.pt, CC.longActive, acc_status,
                                                                 accel, self.acc_stopping, self.acc_starting,
                                                                 cb_pos, cb_neg, CS.acc_type))
        can_sends.append(self.CCS.create_acc_07_control(self.packer_pt, CANBUS.pt, CC.longActive,
                                                                 accel, acc_hold_request, acc_hold_release,
                                                                 acc_hold_type, stopping_distance))

      if self.frame % self.CCP.ACC_HUD_STEP == 0:
        if hud_control.leadVisible:
          lead_distance = 512 if CS.digital_cluster_installed else 8  # TODO: look up actual distance to lead
        else:
          lead_distance = 0

        can_sends.append(self.CCS.create_acc_02_control(self.packer_pt, CANBUS.pt, CS.tsk_status,
                                                                 hud_control.setSpeed * CV.MS_TO_KPH, lead_distance))
        can_sends.append(self.CCS.create_acc_04_control(self.packer_pt, CANBUS.pt, CS.acc_04_stock_values))

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

    # **** Acceleration Controls ******************************************** #

    if self.frame % self.CCP.ACC_CONTROL_STEP == 0 and self.CP.openpilotLongitudinalControl:
      tsk_status = self.CCS.tsk_status_value(CS.out.cruiseState.available, CS.out.accFaulted, CC.longActive)
      accel = clip(actuators.accel, self.CCP.ACCEL_MIN, self.CCP.ACCEL_MAX) if CC.longActive else 0
      can_sends.extend(self.CCS.create_acc_accel_control(self.packer_pt, CANBUS.pt, tsk_status, accel))

    # **** HUD Controls ***************************************************** #

    if self.frame % self.CCP.LDW_STEP == 0:
      hud_alert = 0
      if hud_control.visualAlert in (VisualAlert.steerRequired, VisualAlert.ldw):
        hud_alert = self.CCP.LDW_MESSAGES["laneAssistTakeOver"]
      can_sends.append(self.CCS.create_lka_hud_control(self.packer_pt, CANBUS.pt, CS.ldw_stock_values, CC.enabled,
                                                       CS.out.steeringPressed, hud_alert, hud_control))

    if self.frame % self.CCP.ACC_HUD_STEP == 0 and self.CP.openpilotLongitudinalControl:
      acc_hud_status = self.CCS.acc_hud_status_value(CS.out.cruiseState.available, CS.out.accFaulted, CC.longActive)
      set_speed = hud_control.setSpeed * CV.MS_TO_KPH  # FIXME: follow the recent displayed-speed updates, also use mph_kmh toggle to fix display rounding problem?
      can_sends.append(self.CCS.create_acc_hud_control(self.packer_pt, CANBUS.pt, acc_hud_status, set_speed,
                                                       hud_control.leadVisible))

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
