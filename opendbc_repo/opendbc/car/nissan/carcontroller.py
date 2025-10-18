from opendbc.can import CANPacker
from opendbc.car import Bus, structs
from opendbc.car.lateral import apply_std_steer_angle_limits
from opendbc.car.interfaces import CarControllerBase
from opendbc.car.nissan import nissancan
from opendbc.car.nissan.values import CAR, CarControllerParams

VisualAlert = structs.CarControl.HUDControl.VisualAlert


class CarController(CarControllerBase):
  def __init__(self, dbc_names, CP):
    super().__init__(dbc_names, CP)
    self.car_fingerprint = CP.carFingerprint

    self.apply_angle_last = 0

    self.packer = CANPacker(dbc_names[Bus.pt])

  def update(self, CC, CS, now_nanos):
    actuators = CC.actuators
    hud_control = CC.hudControl
    pcm_cancel_cmd = CC.cruiseControl.cancel

    can_sends = []

    ### STEER ###
    steer_hud_alert = 1 if hud_control.visualAlert in (VisualAlert.steerRequired, VisualAlert.ldw) else 0

    # windup slower
    self.apply_angle_last = apply_std_steer_angle_limits(actuators.steeringAngleDeg, self.apply_angle_last, CS.out.vEgoRaw,
                                                         CS.out.steeringAngleDeg, CC.latActive, CarControllerParams.ANGLE_LIMITS)

    lkas_max_torque = 0
    if CC.latActive:
      # Max torque from driver before EPS will give up and not apply torque
      if not bool(CS.out.steeringPressed):
        lkas_max_torque = CarControllerParams.LKAS_MAX_TORQUE
      else:
        # Scale max torque based on how much torque the driver is applying to the wheel
        lkas_max_torque = max(
          # Scale max torque down to half LKAX_MAX_TORQUE as a minimum
          CarControllerParams.LKAS_MAX_TORQUE * 0.5,
          # Start scaling torque at STEER_THRESHOLD
          CarControllerParams.LKAS_MAX_TORQUE - 0.6 * max(0, abs(CS.out.steeringTorque) - CarControllerParams.STEER_THRESHOLD)
        )

    if self.CP.carFingerprint in (CAR.NISSAN_ROGUE, CAR.NISSAN_XTRAIL, CAR.NISSAN_ALTIMA) and pcm_cancel_cmd:
      can_sends.append(nissancan.create_acc_cancel_cmd(self.packer, self.car_fingerprint, CS.cruise_throttle_msg))

    # TODO: Find better way to cancel!
    # For some reason spamming the cancel button is unreliable on the Leaf
    # We now cancel by making propilot think the seatbelt is unlatched,
    # this generates a beep and a warning message every time you disengage
    if self.CP.carFingerprint in (CAR.NISSAN_LEAF, CAR.NISSAN_LEAF_IC) and self.frame % 2 == 0:
      can_sends.append(nissancan.create_cancel_msg(self.packer, CS.cancel_msg, pcm_cancel_cmd))

    can_sends.append(nissancan.create_steering_control(
      self.packer, self.apply_angle_last, self.frame, CC.latActive, lkas_max_torque))

    # Below are the HUD messages. We copy the stock message and modify
    if self.CP.carFingerprint != CAR.NISSAN_ALTIMA:
      if self.frame % 2 == 0:
        can_sends.append(nissancan.create_lkas_hud_msg(self.packer, CS.lkas_hud_msg, CC.enabled, hud_control.leftLaneVisible, hud_control.rightLaneVisible,
                                                       hud_control.leftLaneDepart, hud_control.rightLaneDepart))

      if self.frame % 50 == 0:
        can_sends.append(nissancan.create_lkas_hud_info_msg(
          self.packer, CS.lkas_hud_info_msg, steer_hud_alert
        ))

    new_actuators = actuators.as_builder()
    new_actuators.steeringAngleDeg = self.apply_angle_last

    self.frame += 1
    return new_actuators, can_sends
