from cereal import car
from opendbc.can.packer import CANPacker
from selfdrive.car import apply_driver_steer_torque_limits
from selfdrive.car.simulator import simcan
from selfdrive.car.simulator.values import CarControllerParams

VisualAlert = car.CarControl.HUDControl.VisualAlert


class CarController:
  def __init__(self, dbc_name, CP, VM):
    self.CP = CP
    self.apply_steer_last = 0
    self.packer = CANPacker(dbc_name)

  def update(self, CC, CS, now_nanos):
    can_sends = []

    apply_steer = 0

    if CC.latActive:
      # calculate steer and also set limits due to driver torque
      new_steer = int(round(CC.actuators.steer))
      apply_steer = apply_driver_steer_torque_limits(new_steer, self.apply_steer_last,
                                                  CS.out.steeringTorque, CarControllerParams)
      
    self.apply_steer_last = apply_steer

    # send steering command
    can_sends.append(simcan.create_steering_control(self.packer, self.CP.carFingerprint,
                                                      apply_steer))

    new_actuators = CC.actuators.copy()
    new_actuators.steer = apply_steer
    new_actuators.steerOutputCan = apply_steer

    return new_actuators, can_sends
