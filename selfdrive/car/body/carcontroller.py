import numpy as np

from common.realtime import DT_CTRL
from selfdrive.car.body import bodycan
from opendbc.can.packer import CANPacker
from selfdrive.car.body.values import SPEED_FROM_RPM

MAX_TORQUE = 500
MAX_TORQUE_RATE = 50
MAX_ANGLE_ERROR = 7
MAX_POS_INTEGRATOR = 0.2 # meters
MAX_TURN_INTEGRATOR = 0.1 # meters


class CarController():
  def __init__(self, dbc_name, CP, VM):
    self.frame = 0
    self.packer = CANPacker(dbc_name)

    self.i_speed = 0
    self.i_speed_diff = 0

    self.i_balance = 0
    self.d_balance = 0


    self.speed_measured = 0.
    self.speed_desired = 0.

    self.torque_r_filtered = 0.
    self.torque_l_filtered = 0.

  @staticmethod
  def deadband_filter(torque, deadband):
    if torque > 0:
      torque += deadband
    else:
      torque -= deadband
    return torque

  def update(self, CC, CS):

    torque_l = 0
    torque_r = 0

    llk_valid = len(CC.orientationNED) > 0 and len(CC.angularVelocity) > 0
    if CC.enabled and llk_valid:

      # Steer and accel mixin. Speed should be used as a target? (speed should be in m/s! now it is in RPM)
      # Mix steer into torque_diff
      # self.steerRatio = 0.5
      # torque_r = int(np.clip((CC.actuators.accel*1000) - (CC.actuators.steer*1000) * self.steerRatio, -1000, 1000))
      # torque_l = int(np.clip((CC.actuators.accel*1000) + (CC.actuators.steer*1000) * self.steerRatio, -1000, 1000))
      # ////

      # Setpoint speed PID
      kp_speed = 0.001 / SPEED_FROM_RPM
      ki_speed = 0.001 / SPEED_FROM_RPM
      alpha_speed = 1.0

      self.speed_measured = SPEED_FROM_RPM * (CS.out.wheelSpeeds.fl + CS.out.wheelSpeeds.fr) / 2.
      self.speed_desired = (1. - alpha_speed) * self.speed_desired
      speed_error = self.speed_desired - self.speed_measured
      self.i_speed += speed_error * DT_CTRL
      self.i_speed = np.clip(self.i_speed, -MAX_POS_INTEGRATOR, MAX_POS_INTEGRATOR)
      set_point = kp_speed * speed_error + ki_speed * self.i_speed

      # Balancing PID
      kp_balance = 1300
      ki_balance = 0
      kd_balance = 280

      # Clip angle error, this is enough to get up from stands
      p_balance = np.clip((-CC.orientationNED[1]) - set_point, np.radians(-MAX_ANGLE_ERROR), np.radians(MAX_ANGLE_ERROR))
      self.i_balance += CS.out.wheelSpeeds.fl + CS.out.wheelSpeeds.fr
      self.d_balance = np.clip(-CC.angularVelocity[1], -1., 1.)
      torque = int(np.clip((p_balance*kp_balance + self.i_balance*ki_balance + self.d_balance*kd_balance), -1000, 1000))

      # yaw recovery PID
      kp_turn = 0.1 / SPEED_FROM_RPM
      ki_turn = 0.1 / SPEED_FROM_RPM

      speed_diff_measured = SPEED_FROM_RPM * (CS.out.wheelSpeeds.fl - CS.out.wheelSpeeds.fr)
      self.i_speed_diff += speed_diff_measured * DT_CTRL
      self.i_speed_diff = np.clip(self.i_speed_diff, -MAX_TURN_INTEGRATOR, MAX_TURN_INTEGRATOR)
      torque_diff = int(np.clip(kp_turn * speed_diff_measured + ki_turn * self.i_speed_diff, -100, 100))

      # Combine 2 PIDs outputs
      torque_r = torque + torque_diff
      torque_l = torque - torque_diff

      # Torque rate limits
      self.torque_r_filtered = np.clip(self.deadband_filter(torque_r, 10),
                                       self.torque_r_filtered - MAX_TORQUE_RATE,
                                       self.torque_r_filtered + MAX_TORQUE_RATE)
      self.torque_l_filtered = np.clip(self.deadband_filter(torque_l, 10),
                                       self.torque_l_filtered - MAX_TORQUE_RATE,
                                       self.torque_l_filtered + MAX_TORQUE_RATE)
      torque_r = int(np.clip(self.torque_r_filtered, -MAX_TORQUE, MAX_TORQUE))
      torque_l = int(np.clip(self.torque_l_filtered, -MAX_TORQUE, MAX_TORQUE))

    can_sends = []
    can_sends.append(bodycan.create_control(self.packer, torque_l, torque_r, self.frame // 2))

    new_actuators = CC.actuators.copy()
    new_actuators.accel = torque_l
    new_actuators.steer = torque_r

    self.frame += 1
    return new_actuators, can_sends
