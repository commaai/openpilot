from common.numpy_fast import clip
from cereal import car
from selfdrive.config import Conversions as CV
from selfdrive.car import apply_std_steer_torque_limits
from selfdrive.car.volkswagen import volkswagencan
from selfdrive.car.volkswagen.values import DBC_FILES, CANBUS, MQB_LDW_MESSAGES, BUTTON_STATES, CarControllerParams as P
from opendbc.can.packer import CANPacker

VisualAlert = car.CarControl.HUDControl.VisualAlert
LongCtrlState = car.CarControl.Actuators.LongControlState

class CarController():
  def __init__(self, dbc_name, CP, VM):
    self.apply_steer_last = 0

    self.packer_pt = CANPacker(DBC_FILES.mqb)

    self.hcaSameTorqueCount = 0
    self.hcaEnabledFrameCount = 0
    self.graButtonStatesToSend = None
    self.graMsgSentCount = 0
    self.graMsgStartFramePrev = 0
    self.graMsgBusCounterPrev = 0

    self.steer_rate_limited = False
    self.acc_starting = False
    self.acc_stopping = False

  def update(self, enabled, CS, frame, ext_bus, actuators, visual_alert, left_lane_visible, right_lane_visible, left_lane_depart,
             right_lane_depart, lead_visible, set_speed, speed_visible):
    """ Controls thread """

    can_sends = []

    # **** Acceleration and Braking Controls ******************************** #

    if CS.CP.openpilotLongitudinalControl:
      if frame % P.ACC_CONTROL_STEP == 0:
        if CS.tsk_status in [2, 3, 4, 5]:
          acc_status = 3 if enabled else 2
        else:
          acc_status = CS.tsk_status

        accel = clip(actuators.accel, P.ACCEL_MIN, P.ACCEL_MAX) if enabled else 0

        # FIXME: this needs to become a proper state machine
        acc_hold_request, acc_hold_release, acc_hold_type, stopping_distance = False, False, 0, 20.46
        if actuators.longControlState == LongCtrlState.stopping and CS.out.vEgo < 0.2:
          self.acc_stopping = True
          acc_hold_request = True
          if CS.esp_hold_confirmation:
            acc_hold_type = 1  # hold
            stopping_distance = 3.5
          else:
            acc_hold_type = 3  # hold_standby
            stopping_distance = 0.5
        elif enabled:
          if self.acc_stopping:
            self.acc_starting = True
            self.acc_stopping = False
          self.acc_starting &= CS.out.vEgo < 1.5
          acc_hold_release = self.acc_starting
          acc_hold_type = 4 if self.acc_starting and CS.out.vEgo < 0.1 else 0  # startup
        else:
          self.acc_stopping, self.acc_starting = False, False

        cb_pos = 0.0 if lead_visible or CS.out.vEgo < 2.0 else 0.1  # react faster to lead cars, also don't get hung up at DSG clutch release/kiss points when creeping to stop
        cb_neg = 0.0 if accel < 0 else 0.2  # IDK why, but stock likes to zero this out when accel is negative
        secondary_accel = accel if lead_visible else 0  # try matching part of stock behavior, presumably drag/compression brake only if we can get away with it

        idx = (frame / P.ACC_CONTROL_STEP) % 16
        can_sends.append(volkswagencan.create_mqb_acc_06_control(self.packer_pt, CANBUS.pt, enabled, acc_status,
                                                                 accel, self.acc_stopping, self.acc_starting,
                                                                 cb_pos, cb_neg, idx))
        can_sends.append(volkswagencan.create_mqb_acc_07_control(self.packer_pt, CANBUS.pt, enabled,
                                                                 accel, secondary_accel, acc_hold_request, acc_hold_release,
                                                                 acc_hold_type, stopping_distance, idx))

      if frame % P.ACC_HUD_STEP == 0:
        idx = (frame / P.ACC_HUD_STEP) % 16
        can_sends.append(volkswagencan.create_mqb_acc_02_control(self.packer_pt, CANBUS.pt, CS.tsk_status,
                                                                  set_speed * CV.MS_TO_KPH, speed_visible, lead_visible, idx))
        can_sends.append(volkswagencan.create_mqb_acc_04_control(self.packer_pt, CANBUS.pt, CS.acc_04_stock_values, idx))

    # **** Steering Controls ************************************************ #

    if frame % P.HCA_STEP == 0:
      # Logic to avoid HCA state 4 "refused":
      #   * Don't steer unless HCA is in state 3 "ready" or 5 "active"
      #   * Don't steer at standstill
      #   * Don't send > 3.00 Newton-meters torque
      #   * Don't send the same torque for > 6 seconds
      #   * Don't send uninterrupted steering for > 360 seconds
      # One frame of HCA disabled is enough to reset the timer, without zeroing the
      # torque value. Do that anytime we happen to have 0 torque, or failing that,
      # when exceeding ~1/3 the 360 second timer.

      if enabled and CS.out.vEgo > CS.CP.minSteerSpeed and not (CS.out.standstill or CS.out.steerError or CS.out.steerWarning):
        new_steer = int(round(actuators.steer * P.STEER_MAX))
        apply_steer = apply_std_steer_torque_limits(new_steer, self.apply_steer_last, CS.out.steeringTorque, P)
        self.steer_rate_limited = new_steer != apply_steer
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
      idx = (frame / P.HCA_STEP) % 16
      can_sends.append(volkswagencan.create_mqb_steering_control(self.packer_pt, CANBUS.pt, apply_steer,
                                                                 idx, hcaEnabled))

    # **** HUD Controls ***************************************************** #

    if frame % P.LDW_STEP == 0:
      if visual_alert in [VisualAlert.steerRequired, VisualAlert.ldw]:
        hud_alert = MQB_LDW_MESSAGES["laneAssistTakeOverSilent"]
      else:
        hud_alert = MQB_LDW_MESSAGES["none"]

      can_sends.append(volkswagencan.create_mqb_hud_control(self.packer_pt, CANBUS.pt, enabled,
                                                            CS.out.steeringPressed, hud_alert, left_lane_visible,
                                                            right_lane_visible, CS.ldw_stock_values,
                                                            left_lane_depart, right_lane_depart))

    # **** ACC Button Controls ********************************************** #

    # FIXME: this entire section is in desperate need of refactoring

    if CS.CP.pcmCruise:
      if frame > self.graMsgStartFramePrev + P.GRA_VBP_STEP:
        if not enabled and CS.out.cruiseState.enabled:
          # Cancel ACC if it's engaged with OP disengaged.
          self.graButtonStatesToSend = BUTTON_STATES.copy()
          self.graButtonStatesToSend["cancel"] = True
        elif enabled and CS.esp_hold_confirmation:
          # Blip the Resume button if we're engaged at standstill.
          # FIXME: This is a naive implementation, improve with visiond or radar input.
          self.graButtonStatesToSend = BUTTON_STATES.copy()
          self.graButtonStatesToSend["resumeCruise"] = True

      if CS.graMsgBusCounter != self.graMsgBusCounterPrev:
        self.graMsgBusCounterPrev = CS.graMsgBusCounter
        if self.graButtonStatesToSend is not None:
          if self.graMsgSentCount == 0:
            self.graMsgStartFramePrev = frame
          idx = (CS.graMsgBusCounter + 1) % 16
          can_sends.append(volkswagencan.create_mqb_acc_buttons_control(self.packer_pt, ext_bus, self.graButtonStatesToSend, CS, idx))
          self.graMsgSentCount += 1
          if self.graMsgSentCount >= P.GRA_VBP_COUNT:
            self.graButtonStatesToSend = None
            self.graMsgSentCount = 0

    return can_sends
