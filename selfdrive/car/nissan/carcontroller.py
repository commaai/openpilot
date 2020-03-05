from common.numpy_fast import clip, interp
from selfdrive.car.nissan import nissancan
from opendbc.can.packer import CANPacker

# Steer angle limits
ANGLE_MAX_BP = [1.3, 10., 30.]
ANGLE_MAX_V = [540., 120., 23.]
ANGLE_DELTA_BP = [0., 5., 15.]
ANGLE_DELTA_V = [5., .8, .15]     # windup limit
ANGLE_DELTA_VU = [5., 3.5, 0.4]   # unwind limit
LKAS_MAX_TORQUE = 100             # A value of 100 is easy to overpower


class CarController():
  def __init__(self, dbc_name, CP, VM):
    self.car_fingerprint = CP.carFingerprint

    self.lkas_max_torque = 0
    self.last_angle = 0

    self.packer = CANPacker(dbc_name)

  def update(self, enabled, CS, frame, actuators, cruise_cancel):
    """ Controls thread """

    # Send CAN commands.
    can_sends = []

    ### STEER ###
    acc_active = bool(CS.out.cruiseState.enabled)
    cruise_throttle_msg = CS.cruise_throttle_msg
    apply_angle = actuators.steerAngle

    if enabled:
      # # windup slower
      if self.last_angle * apply_angle > 0. and abs(apply_angle) > abs(self.last_angle):
        angle_rate_lim = interp(CS.out.vEgo, ANGLE_DELTA_BP, ANGLE_DELTA_V)
      else:
        angle_rate_lim = interp(CS.out.vEgo, ANGLE_DELTA_BP, ANGLE_DELTA_VU)

      apply_angle = clip(apply_angle, self.last_angle -
                         angle_rate_lim, self.last_angle + angle_rate_lim)

      # steer angle
      angle_lim = interp(CS.out.vEgo, ANGLE_MAX_BP, ANGLE_MAX_V)
      apply_angle = clip(apply_angle, -angle_lim, angle_lim)

      # Max torque from driver before EPS will give up and not apply torque
      if not bool(CS.out.steeringPressed):
        self.lkas_max_torque = LKAS_MAX_TORQUE
      else:
        # Scale max torque based on how much torque the driver is applying to the wheel
        self.lkas_max_torque = max(
            0, LKAS_MAX_TORQUE - 0.4 * abs(CS.out.steeringTorque))

    else:
      apply_angle = CS.out.steeringAngle
      self.lkas_max_torque = 0

    self.last_angle = apply_angle

    if not enabled and acc_active:
      # send acc cancel cmd if drive is disabled but pcm is still on, or if the system can't be activated
      cruise_cancel = 1

    if cruise_cancel:
      can_sends.append(nissancan.create_acc_cancel_cmd(
          self.packer, cruise_throttle_msg, frame))

    can_sends.append(nissancan.create_steering_control(
        self.packer, self.car_fingerprint, apply_angle, frame, acc_active, self.lkas_max_torque))

    return can_sends
