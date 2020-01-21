from selfdrive.car.mazda import mazdacan
from selfdrive.car.mazda.values import DBC, SteerLimitParams
from opendbc.can.packer import CANPacker
from selfdrive.car import apply_std_steer_torque_limits

class CarController():
  def __init__(self, dbc_name, CP, VM):
    self.steer_idx = 0
    self.apply_steer_last = 0
    self.packer = CANPacker(dbc_name)
    self.steer_rate_limited = False

  def update(self, enabled, CS, frame, actuators):
    """ Controls thread """

    can_sends = []

    ### STEER ###

    if enabled and not CS.steer_not_allowed:
      # calculate steer and also set limits due to driver torque
      new_steer = int(round(actuators.steer * SteerLimitParams.STEER_MAX))
      apply_steer = apply_std_steer_torque_limits(new_steer, self.apply_steer_last,
                                                  CS.out.steer_torque_driver, SteerLimitParams)
      self.steer_rate_limited = new_steer != apply_steer
    else:
      apply_steer = 0
      self.steer_rate_limited = False

    self.apply_steer_last = apply_steer
  

    can_sends.append(mazdacan.create_steering_control(self.packer, CS.CP.carFingerprint,
                                                      frame, apply_steer, CS.cam_lkas))
    return can_sends
