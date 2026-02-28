# Opel Corsa F (PSA CMP Platform) - Car Controller
# Sends CAN bus commands for steering, HUD, and ACC control
from cereal import car
from selfdrive.car import apply_std_steer_torque_limits
from selfdrive.car.opel import opelcan
from selfdrive.car.opel.values import DBC_FILES, CANBUS, PSA_LDW_MESSAGES, BUTTON_STATES, CarControllerParams as P
from opendbc.can.packer import CANPacker

VisualAlert = car.CarControl.HUDControl.VisualAlert


class CarController():
  def __init__(self, dbc_name, CP, VM):
    self.apply_steer_last = 0

    self.packer_pt = CANPacker(DBC_FILES.psa_cmp)

    self.lkasSameTorqueCount = 0
    self.lkasEnabledFrameCount = 0
    self.accButtonStatesToSend = None
    self.accMsgSentCount = 0
    self.accMsgStartFramePrev = 0
    self.accMsgBusCounterPrev = 0

    self.steer_rate_limited = False

  def update(self, c, CS, frame, ext_bus, actuators, visual_alert, left_lane_visible, right_lane_visible, left_lane_depart, right_lane_depart):
    """ Controls thread """

    can_sends = []

    # **** Steering Controls ************************************************ #

    if frame % P.LKAS_STEP == 0:
      # Logic to avoid EPS rejection:
      #   * Don't steer unless EPS is in ready or active state
      #   * Don't steer at standstill
      #   * Don't send > 3.00 Newton-meters torque
      #   * Don't send the same torque for > 6 seconds
      #   * Don't send uninterrupted steering for > 360 seconds

      if c.latActive:
        new_steer = int(round(actuators.steer * P.STEER_MAX))
        apply_steer = apply_std_steer_torque_limits(new_steer, self.apply_steer_last, CS.out.steeringTorque, P)
        self.steer_rate_limited = new_steer != apply_steer
        if apply_steer == 0:
          lkas_enabled = False
          self.lkasEnabledFrameCount = 0
        else:
          self.lkasEnabledFrameCount += 1
          if self.lkasEnabledFrameCount >= 118 * (100 / P.LKAS_STEP):  # 118s
            lkas_enabled = False
            self.lkasEnabledFrameCount = 0
          else:
            lkas_enabled = True
            if self.apply_steer_last == apply_steer:
              self.lkasSameTorqueCount += 1
              if self.lkasSameTorqueCount > 1.9 * (100 / P.LKAS_STEP):  # 1.9s
                apply_steer -= (1, -1)[apply_steer < 0]
                self.lkasSameTorqueCount = 0
            else:
              self.lkasSameTorqueCount = 0
      else:
        lkas_enabled = False
        apply_steer = 0

      self.apply_steer_last = apply_steer
      idx = (frame / P.LKAS_STEP) % 16
      can_sends.append(opelcan.create_psa_steering_control(self.packer_pt, CANBUS.pt, apply_steer,
                                                            idx, lkas_enabled))

    # **** HUD Controls ***************************************************** #

    if frame % P.LDW_STEP == 0:
      if visual_alert in (VisualAlert.steerRequired, VisualAlert.ldw):
        hud_alert = PSA_LDW_MESSAGES["laneAssistTakeOver"]
      else:
        hud_alert = PSA_LDW_MESSAGES["none"]

      can_sends.append(opelcan.create_psa_hud_control(self.packer_pt, CANBUS.pt, c.enabled,
                                                       CS.out.steeringPressed, hud_alert,
                                                       left_lane_visible, right_lane_visible,
                                                       left_lane_depart, right_lane_depart))

    # **** ACC Button Controls ********************************************** #

    if CS.CP.pcmCruise:
      if frame > self.accMsgStartFramePrev + P.ACC_VBP_STEP:
        if c.cruiseControl.cancel:
          # Cancel ACC if it's engaged with OP disengaged
          self.accButtonStatesToSend = BUTTON_STATES.copy()
          self.accButtonStatesToSend["cancel"] = True
        elif c.enabled and CS.out.standstill:
          # Blip the Resume button if we're engaged at standstill
          self.accButtonStatesToSend = BUTTON_STATES.copy()
          self.accButtonStatesToSend["resumeCruise"] = True

      if CS.accMsgBusCounter != self.accMsgBusCounterPrev:
        self.accMsgBusCounterPrev = CS.accMsgBusCounter
        if self.accButtonStatesToSend is not None:
          if self.accMsgSentCount == 0:
            self.accMsgStartFramePrev = frame
          idx = (CS.accMsgBusCounter + 1) % 16
          can_sends.append(opelcan.create_psa_acc_buttons_control(self.packer_pt, ext_bus,
                                                                   self.accButtonStatesToSend, CS, idx))
          self.accMsgSentCount += 1
          if self.accMsgSentCount >= P.ACC_VBP_COUNT:
            self.accButtonStatesToSend = None
            self.accMsgSentCount = 0

    new_actuators = actuators.copy()
    new_actuators.steer = self.apply_steer_last / P.STEER_MAX

    return new_actuators, can_sends
