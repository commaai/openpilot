from selfdrive.car import apply_std_steer_torque_limits
from selfdrive.car.faw import fawcan
from selfdrive.car.faw.values import DBC_FILES, CANBUS, CarControllerParams as P
from opendbc.can.packer import CANPacker

class CarController():
  def __init__(self, dbc_name, CP, VM):
    self.apply_steer_last = 0
    self.CP = CP

    self.packer_pt = CANPacker(DBC_FILES.faw)

    self.steer_rate_limited = False

  def update(self, c, CS, frame, actuators):
    """ Controls thread """

    can_sends = []

    # **** Steering Controls ************************************************ #

    if frame % P.LKAS_STEP == 0:
      if c.latActive:
        new_steer = int(round(actuators.steer * P.STEER_MAX))
        apply_steer = apply_std_steer_torque_limits(new_steer, self.apply_steer_last, CS.out.steeringTorque, P)
        self.steer_rate_limited = new_steer != apply_steer
        lkas_torque_enabled = True if apply_steer != 0 else False
      else:
        apply_steer = 0
        lkas_torque_enabled = False

      self.apply_steer_last = apply_steer
      can_sends.append(fawcan.create_steering_control(self.packer_pt, CANBUS.pt, apply_steer, lkas_torque_enabled))

    # **** HUD Controls ***************************************************** #

    # TODO

    # **** ACC Button Controls ********************************************** #

    # TODO

    new_actuators = actuators.copy()
    new_actuators.steer = self.apply_steer_last / P.STEER_MAX

    return new_actuators, can_sends
