from cereal import car
from common.numpy_fast import clip, interp
from selfdrive.car.body import bodycan
from opendbc.can.packer import CANPacker
from selfdrive.car.body.values import CAR, CarControllerParams


VisualAlert = car.CarControl.HUDControl.VisualAlert


class CarController():
  def __init__(self, dbc_name, CP, VM):
    self.CP = CP
    self.car_fingerprint = CP.carFingerprint

    self.lkas_max_torque = 0
    self.last_angle = 0

    self.packer = CANPacker(dbc_name)

  def update(self, c, CS, frame, actuators, cruise_cancel, hud_alert,
             left_line, right_line, left_lane_depart, right_lane_depart):

    can_sends = []

    apply_angle = actuators.steeringAngleDeg

    torque_l = 60
    torque_r = 60
    can_sends.append(bodycan.create_control(
        self.packer, torque_l, torque_r))

    new_actuators = actuators.copy()
    new_actuators.steeringAngleDeg = apply_angle

    return new_actuators, can_sends
