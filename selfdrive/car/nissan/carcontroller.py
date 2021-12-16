from cereal import car
from common.numpy_fast import clip, interp
from selfdrive.car.nissan import nissancan
from opendbc.can.packer import CANPacker
from selfdrive.car.nissan.values import CAR, CarControllerParams


VisualAlert = car.CarControl.HUDControl.VisualAlert


class CarController():
  def __init__(self, dbc_name, CP, VM):
    self.CP = CP
    self.car_fingerprint = CP.carFingerprint

    self.lkas_max_torque = 0
    self.last_angle = 0

    self.packer = CANPacker(dbc_name)

  def update(self, enabled, CS, frame, actuators, cruise_cancel, hud_alert,
             left_line, right_line, left_lane_depart, right_lane_depart):
    """ Controls thread """

    # Send CAN commands.
    can_sends = []

    ### STEER ###
    acc_active = bool(CS.out.cruiseState.enabled)
    lkas_hud_msg = CS.lkas_hud_msg
    lkas_hud_info_msg = CS.lkas_hud_info_msg
    apply_angle = actuators.steeringAngleDeg

    steer_hud_alert = 1 if hud_alert in [VisualAlert.steerRequired, VisualAlert.ldw] else 0

    if enabled:
      # # windup slower
      if self.last_angle * apply_angle > 0. and abs(apply_angle) > abs(self.last_angle):
        angle_rate_lim = interp(CS.out.vEgo, CarControllerParams.ANGLE_DELTA_BP, CarControllerParams.ANGLE_DELTA_V)
      else:
        angle_rate_lim = interp(CS.out.vEgo, CarControllerParams.ANGLE_DELTA_BP, CarControllerParams.ANGLE_DELTA_VU)

      apply_angle = clip(apply_angle, self.last_angle - angle_rate_lim, self.last_angle + angle_rate_lim)

      # Max torque from driver before EPS will give up and not apply torque
      if not bool(CS.out.steeringPressed):
        self.lkas_max_torque = CarControllerParams.LKAS_MAX_TORQUE
      else:
        # Scale max torque based on how much torque the driver is applying to the wheel
        self.lkas_max_torque = max(
          # Scale max torque down to half LKAX_MAX_TORQUE as a minimum
          CarControllerParams.LKAS_MAX_TORQUE * 0.5,
          # Start scaling torque at STEER_THRESHOLD
          CarControllerParams.LKAS_MAX_TORQUE - 0.6 * max(0, abs(CS.out.steeringTorque) - CarControllerParams.STEER_THRESHOLD)
        )

    else:
      apply_angle = CS.out.steeringAngleDeg
      self.lkas_max_torque = 0

    self.last_angle = apply_angle

    if not enabled and acc_active:
      # send acc cancel cmd if drive is disabled but pcm is still on, or if the system can't be activated
      cruise_cancel = 1

    if self.CP.carFingerprint in [CAR.ROGUE, CAR.XTRAIL, CAR.ALTIMA] and cruise_cancel:
        can_sends.append(nissancan.create_acc_cancel_cmd(self.packer, self.car_fingerprint, CS.cruise_throttle_msg, frame))

    # TODO: Find better way to cancel!
    # For some reason spamming the cancel button is unreliable on the Leaf
    # We now cancel by making propilot think the seatbelt is unlatched,
    # this generates a beep and a warning message every time you disengage
    if self.CP.carFingerprint in [CAR.LEAF, CAR.LEAF_IC] and frame % 2 == 0:
        can_sends.append(nissancan.create_cancel_msg(self.packer, CS.cancel_msg, cruise_cancel))

    can_sends.append(nissancan.create_steering_control(
        self.packer, apply_angle, frame, enabled, self.lkas_max_torque))

    if lkas_hud_msg and lkas_hud_info_msg:
      if frame % 2 == 0:
        can_sends.append(nissancan.create_lkas_hud_msg(
          self.packer, lkas_hud_msg, enabled, left_line, right_line, left_lane_depart, right_lane_depart))

      if frame % 50 == 0:
        can_sends.append(nissancan.create_lkas_hud_info_msg(
          self.packer, lkas_hud_info_msg, steer_hud_alert
        ))

    new_actuators = actuators.copy()
    new_actuators.steeringAngleDeg = apply_angle

    return new_actuators, can_sends
