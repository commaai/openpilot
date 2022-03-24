from cereal import car
from selfdrive.car import apply_std_steer_torque_limits
from selfdrive.car.volkswagen import volkswagencan
from selfdrive.car.volkswagen.values import DBC_FILES, CANBUS, MQB_LDW_MESSAGES, BUTTON_STATES, CarControllerParams as P
from opendbc.can.packer import CANPacker

VisualAlert = car.CarControl.HUDControl.VisualAlert

class CarController():
  def __init__(self, dbc_name, CP, VM):
    self.apply_steer_last = 0
    self.CP = CP

    self.packer_pt = CANPacker(DBC_FILES.mqb)

    self.hcaSameTorqueCount = 0
    self.hcaEnabledFrameCount = 0
    self.graButtonStatesToSend = None
    self.graMsgSentCount = 0
    self.graMsgStartFramePrev = 0
    self.graMsgBusCounterPrev = 0

    self.steer_rate_limited = False

  def update(self, c, CS, frame, ext_bus, actuators, visual_alert, left_lane_visible, right_lane_visible, left_lane_depart, right_lane_depart):
    """ Controls thread """

    can_sends = []

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

      if c.latActive:
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
      if visual_alert in (VisualAlert.steerRequired, VisualAlert.ldw):
        hud_alert = MQB_LDW_MESSAGES["laneAssistTakeOverSilent"]
      else:
        hud_alert = MQB_LDW_MESSAGES["none"]

      can_sends.append(volkswagencan.create_mqb_hud_control(self.packer_pt, CANBUS.pt, c.enabled,
                                                            CS.out.steeringPressed, hud_alert, left_lane_visible,
                                                            right_lane_visible, CS.ldw_stock_values,
                                                            left_lane_depart, right_lane_depart))

    # **** ACC Button Controls ********************************************** #

    # FIXME: this entire section is in desperate need of refactoring

    if self.CP.pcmCruise:
      if frame > self.graMsgStartFramePrev + P.GRA_VBP_STEP:
        if c.cruiseControl.cancel:
          # Cancel ACC if it's engaged with OP disengaged.
          self.graButtonStatesToSend = BUTTON_STATES.copy()
          self.graButtonStatesToSend["cancel"] = True
        elif c.enabled and CS.esp_hold_confirmation:
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

    new_actuators = actuators.copy()
    new_actuators.steer = self.apply_steer_last / P.STEER_MAX

    return new_actuators, can_sends
