from selfdrive.car.mazda import mazdacan
from selfdrive.car.mazda.values import  DBC
from selfdrive.can.packer import CANPacker
from selfdrive.car import apply_std_steer_torque_limits


class CarControllerParams():
  def __init__(self, car_fingerprint):
    self.STEER_MAX = 600                 # max_steer 2048
    self.STEER_STEP = 1                  # how often we update the steer cmd
    self.STEER_DELTA_UP = 10             # torque increase per refresh
    self.STEER_DELTA_DOWN = 20           # torque decrease per refresh
    self.STEER_DRIVER_ALLOWANCE = 15     # allowed driver torque before start limiting
    self.STEER_DRIVER_MULTIPLIER = 1     # weight driver torque heavily
    self.STEER_DRIVER_FACTOR = 1         # from dbc

class CarController():
  def __init__(self, canbus, car_fingerprint):
    self.steer_idx = 0
    self.apply_steer_last = 0
    self.car_fingerprint = car_fingerprint

    # Setup detection helper. Routes commands to
    # an appropriate CAN bus number.
    self.canbus = canbus
    self.params = CarControllerParams(car_fingerprint)
    self.packer_pt = CANPacker(DBC[car_fingerprint]['pt'])

  def update(self, enabled, CS, frame, actuators):
    """ Controls thread """

    P = self.params
    can_sends = []
    canbus = self.canbus

    ### STEER ###
    if (frame % P.STEER_STEP) == 0:

      if enabled and not CS.steer_not_allowed:
        # calculate steer and also set limits due to driver torque
        apply_steer = apply_std_steer_torque_limits(int(round(actuators.steer * P.STEER_MAX)),
                                                  self.apply_steer_last, CS.steer_torque_driver, P)
      else:
        apply_steer = 0

      self.apply_steer_last = apply_steer
      ctr = (frame // P.STEER_STEP) % 16

      can_sends.append(mazdacan.create_steering_control(self.packer_pt, canbus.powertrain,
                                                        CS.CP.carFingerprint, ctr, apply_steer,
                                                        CS.cam_lkas))
    return can_sends
