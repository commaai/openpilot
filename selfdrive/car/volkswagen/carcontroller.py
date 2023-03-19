from cereal import car
from opendbc.can.packer import CANPacker
from common.numpy_fast import clip
from common.conversions import Conversions as CV
from common.realtime import DT_CTRL
from selfdrive.car import apply_driver_steer_torque_limits
from selfdrive.car.volkswagen import mlbcan, mqbcan, pqcan
from selfdrive.car.volkswagen.values import CANBUS, MLB_CARS, PQ_CARS, CarControllerParams

VisualAlert = car.CarControl.HUDControl.VisualAlert
LongCtrlState = car.CarControl.Actuators.LongControlState


class CarController:
  def __init__(self, dbc_name, CP, VM):
    self.CP = CP
    self.CCP = CarControllerParams(CP)
    self.packer_pt = CANPacker(dbc_name)

    if CP.carFingerprint in PQ_CARS:
      self.CCS = pqcan
    elif CP.carFingerprint in MLB_CARS:
      self.CCS = mlbcan
    else:
      self.CCS = mqbcan

    self.apply_steer_last = 0
    self.gra_acc_counter_last = None
    self.frame = 0
    self.eps_timer_workaround = True  # For testing, replace with CP.carFingerprint in (PQ_CARS, MLB_CARS)
    self.hca_frame_timer_running = 0
    self.hca_frame_timer_resetting = 0
    self.hca_frame_low_torque = 0
    self.hca_frame_same_torque = 0
    self.hca_output_steer = 0

  def update(self, CC, CS, ext_bus, now_nanos):
    actuators = CC.actuators
    hud_control = CC.hudControl
    can_sends = []

    # **** Steering Controls ************************************************ #

    if self.frame % self.CCP.STEER_STEP == 0:
      # Logic to avoid HCA state 4 "refused":
      #   * Don't steer unless HCA is in state 3 "ready" or 5 "active"
      #   * Don't steer at standstill
      #   * Don't send > 3.00 Newton-meters torque
      #   * Don't send the same torque for > 6 seconds
      #   * Don't send uninterrupted steering for > 360 seconds
      # MQB racks reset the uninterrupted steering timer after a single frame
      # of HCA disabled; this is done whenever output happens to be zero.
      # PQ35, PQ46, NMS and MLB racks need >1 second to reset. Try to perform
      # resets when engaged for >240 seconds and output stays under 20%.

      if CC.latActive:
        new_steer = int(round(actuators.steer * self.CCP.STEER_MAX))
        apply_steer = apply_driver_steer_torque_limits(new_steer, self.apply_steer_last, CS.out.steeringTorque, self.CCP)
        self.hca_frame_timer_running += self.CCP.STEER_STEP
        if self.apply_steer_last == apply_steer:
          self.hca_frame_same_torque += self.CCP.STEER_STEP
          if self.hca_frame_same_torque > 1.9 / DT_CTRL:
            apply_steer -= (1, -1)[apply_steer < 0]
            self.hca_frame_same_torque = 0
        else:
          self.hca_frame_same_torque = 0
        hca_enabled = abs(apply_steer) > 0
        self.hca_output_steer = apply_steer
        if self.eps_timer_workaround and self.hca_frame_timer_running >= 240 / DT_CTRL:
          if abs(apply_steer) <= self.CCP.STEER_MAX * 0.2:
            self.hca_frame_low_torque += self.CCP.STEER_STEP
            if self.hca_frame_low_torque >= 0.5 / DT_CTRL:
              hca_enabled = False
              self.hca_output_steer = 0
          else:
            self.hca_frame_low_torque = 0
            if self.hca_frame_timer_resetting > 0:
              apply_steer = clip(apply_steer, -self.CCP.STEER_DELTA_UP, self.CCP.STEER_DELTA_UP)
              self.hca_output_steer = apply_steer
      else:
        hca_enabled = False
        apply_steer = 0
        self.hca_frame_low_torque = 0
        self.hca_output_steer = 0

      if hca_enabled:
        self.hca_frame_timer_resetting = 0
      else:
        self.hca_frame_timer_resetting += self.CCP.STEER_STEP
        if self.hca_frame_timer_resetting >= 1.1 / DT_CTRL:
          self.hca_frame_timer_running = 0
          apply_steer = 0

      can_sends.append(self.CCS.create_steering_control(self.packer_pt, CANBUS.pt, self.hca_output_steer, hca_enabled))
      self.apply_steer_last = apply_steer

    # **** Acceleration Controls ******************************************** #

    if self.frame % self.CCP.ACC_CONTROL_STEP == 0 and self.CP.openpilotLongitudinalControl:
      acc_control = self.CCS.acc_control_value(CS.out.cruiseState.available, CS.out.accFaulted, CC.longActive)
      accel = clip(actuators.accel, self.CCP.ACCEL_MIN, self.CCP.ACCEL_MAX) if CC.longActive else 0
      stopping = actuators.longControlState == LongCtrlState.stopping
      starting = actuators.longControlState == LongCtrlState.starting
      can_sends.extend(self.CCS.create_acc_accel_control(self.packer_pt, CANBUS.pt, CS.acc_type, CC.longActive, accel,
                                                         acc_control, stopping, starting, CS.esp_hold_confirmation))

    # **** HUD Controls ***************************************************** #

    if self.frame % self.CCP.LDW_STEP == 0:
      hud_alert = 0
      if hud_control.visualAlert in (VisualAlert.steerRequired, VisualAlert.ldw):
        hud_alert = self.CCP.LDW_MESSAGES["laneAssistTakeOver"]
      can_sends.append(self.CCS.create_lka_hud_control(self.packer_pt, CANBUS.pt, CS.ldw_stock_values, CC.enabled,
                                                       CS.out.steeringPressed, hud_alert, hud_control))

    if self.frame % self.CCP.ACC_HUD_STEP == 0 and self.CP.openpilotLongitudinalControl:
      lead_distance = 0
      if hud_control.leadVisible and self.frame * DT_CTRL > 1.0:  # Don't display lead until we know the scaling factor
        lead_distance = 512 if CS.upscale_lead_car_signal else 8
      acc_hud_status = self.CCS.acc_hud_status_value(CS.out.cruiseState.available, CS.out.accFaulted, CC.longActive)
      set_speed = hud_control.setSpeed * CV.MS_TO_KPH  # FIXME: follow the recent displayed-speed updates, also use mph_kmh toggle to fix display rounding problem?
      can_sends.append(self.CCS.create_acc_hud_control(self.packer_pt, CANBUS.pt, acc_hud_status, set_speed,
                                                       lead_distance))

    # **** Stock ACC Button Controls **************************************** #

    gra_send_ready = self.CP.pcmCruise and CS.gra_stock_values["COUNTER"] != self.gra_acc_counter_last
    if gra_send_ready and (CC.cruiseControl.cancel or CC.cruiseControl.resume):
      counter = (CS.gra_stock_values["COUNTER"] + 1) % 16
      can_sends.append(self.CCS.create_acc_buttons_control(self.packer_pt, ext_bus, CS.gra_stock_values, counter,
                                                           cancel=CC.cruiseControl.cancel, resume=CC.cruiseControl.resume))

    new_actuators = actuators.copy()
    new_actuators.steer = self.hca_output_steer / self.CCP.STEER_MAX
    new_actuators.steerOutputCan = self.apply_steer_last

    self.gra_acc_counter_last = CS.gra_stock_values["COUNTER"]
    self.frame += 1
    return new_actuators, can_sends
