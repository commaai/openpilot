from selfdrive.car import apply_std_steer_torque_limits
from selfdrive.car.hongqi import hongqican
from selfdrive.car.hongqi.values import DBC_FILES, CANBUS, CarControllerParams as P
from opendbc.can.packer import CANPacker

class CarController():
  def __init__(self, dbc_name, CP, VM):
    self.apply_steer_last = 0
    self.CP = CP

    self.packer_pt = CANPacker(DBC_FILES.hongqi)

  def update(self, c, CS, frame, actuators):
    """ Controls thread """

    can_sends = []

    # **** Steering Controls ************************************************ #

    if frame % P.LKAS_STEP == 0:
      if c.latActive:
        new_steer = int(round(actuators.steer * P.STEER_MAX))
        apply_steer = apply_std_steer_torque_limits(new_steer, self.apply_steer_last, CS.out.steeringTorque, P)
      else:
        apply_steer = 0

      self.apply_steer_last = apply_steer
      can_sends.append(hongqican.create_steering_control(self.packer_pt, CANBUS.pt, apply_steer, c.latActive))

    # **** HUD Controls ***************************************************** #

    # TODO

    # **** ACC Button Controls ********************************************** #

    # TODO

    new_actuators = actuators.copy()
    new_actuators.steer = self.apply_steer_last / P.STEER_MAX

    return new_actuators, can_sends
