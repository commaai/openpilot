import numpy as np

from common.realtime import DT_CTRL
from opendbc.can.packer import CANPacker
from selfdrive.car.body import bodycan
from selfdrive.car.body.values import SPEED_FROM_RPM
from selfdrive.controls.lib.pid import PIDController


MAX_TORQUE = 700
MAX_TORQUE_RATE = 50
MAX_ANGLE_ERROR = np.radians(7)
MAX_POS_INTEGRATOR = 0.2   # meters
MAX_TURN_INTEGRATOR = 0.2  # meters

MAX_KNEE_TORQUE_LEFT = 300 # knee motor
MAX_KNEE_TORQUE_RIGHT = 100 # hip motor
MAX_KNEE_TORQUE_RATE = 20
MAX_KNEE_ANGLE_INTEGRATOR_LEFT = 10
MAX_KNEE_ANGLE_INTEGRATOR_RIGHT = 10


class CarController:
  def __init__(self, dbc_name, CP, VM):
    self.packer = CANPacker(dbc_name)

    # Speed, balance and turn PIDs
    self.speed_pid = PIDController(0.025, k_i=0.11, rate=1/DT_CTRL)
    # self.balance_pid = PIDController(3200, k_i=0, k_d=230, rate=1/DT_CTRL)
    self.balance_pid = PIDController(2400, k_i=0, k_d=180, rate=1/DT_CTRL)
    self.turn_pid = PIDController(110, k_i=11.5, rate=1/DT_CTRL)

    self.torque_l_filtered = 0.
    self.torque_r_filtered = 0.

    self.knee_angle_left_pid = PIDController(40, k_i=12, rate=1/DT_CTRL)
    self.knee_angle_right_pid = PIDController(40, k_i=4, rate=1/DT_CTRL)

    self.torque_knee_l_filtered = 0.
    self.torque_knee_r_filtered = 0.

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
    knee_torque_l = 0
    knee_torque_r = 0

    knee_angle_measured_left = 0
    knee_angle_measured_right = 0

    llk_valid = len(CC.orientationNED) > 0 and len(CC.angularVelocity) > 0
    if CC.enabled and llk_valid:
      # Read these from the joystick
      # TODO: this isn't acceleration, okay?
      speed_desired = CC.actuators.accel / 5.
      speed_diff_desired = -CC.actuators.steer

      speed_measured = SPEED_FROM_RPM * (CS.out.wheelSpeeds.fl + CS.out.wheelSpeeds.fr) / 2.
      speed_error = speed_desired - speed_measured

      freeze_integrator = ((speed_error < 0 and self.speed_pid.error_integral <= -MAX_POS_INTEGRATOR) or
                           (speed_error > 0 and self.speed_pid.error_integral >= MAX_POS_INTEGRATOR))
      angle_setpoint = self.speed_pid.update(speed_error, freeze_integrator=freeze_integrator)
      # print(angle_setpoint, (-CC.orientationNED[1]))
      # angle_setpoint = 0

      # Clip angle error, this is enough to get up from stands
      angle_error = np.clip((-CC.orientationNED[1]) - angle_setpoint + np.radians(CS.knee_angle_r - CS.knee_angle_r), -MAX_ANGLE_ERROR, MAX_ANGLE_ERROR)
      # angle_error -= 0.02
      angle_error_rate = np.clip(-CC.angularVelocity[1], -1., 1.)
      # print(angle_error, angle_setpoint)
      torque = self.balance_pid.update(angle_error, error_rate=angle_error_rate)

      speed_diff_measured = SPEED_FROM_RPM * (CS.out.wheelSpeeds.fl - CS.out.wheelSpeeds.fr)
      turn_error = speed_diff_measured - speed_diff_desired
      freeze_integrator = ((turn_error < 0 and self.turn_pid.error_integral <= -MAX_TURN_INTEGRATOR) or
                           (turn_error > 0 and self.turn_pid.error_integral >= MAX_TURN_INTEGRATOR))
      torque_diff = self.turn_pid.update(turn_error, freeze_integrator=freeze_integrator)
      # torque_diff = 0

      # Combine 2 PIDs outputs
      torque_l = torque - torque_diff
      torque_r = torque + torque_diff

      # Torque rate limits
      self.torque_l_filtered = np.clip(self.deadband_filter(torque_l, 10),
                                       self.torque_l_filtered - MAX_TORQUE_RATE,
                                       self.torque_l_filtered + MAX_TORQUE_RATE)
      self.torque_r_filtered = np.clip(self.deadband_filter(torque_r, 10),
                                       self.torque_r_filtered - MAX_TORQUE_RATE,
                                       self.torque_r_filtered + MAX_TORQUE_RATE)
      
      torque_l = int(np.clip(self.torque_l_filtered, -MAX_TORQUE, MAX_TORQUE))
      torque_r = int(np.clip(self.torque_r_filtered, -MAX_TORQUE, MAX_TORQUE))
      # print(f"angle_error: {angle_error}, gyro: {(-CC.orientationNED[1])}, knee: {np.radians(CS.knee_angle_error)}, setpoint: {angle_setpoint}, trq_l/r:{torque_l}, {torque_r}")

    # Knee angle controls
    if CC.enabled:
      knee_angle_measured_left = CS.knee_angle_l
      knee_angle_measured_right = CS.knee_angle_r

      # TODO: add joystick support
      knee_angle_desired_left = 180
      knee_angle_desired_right = knee_angle_measured_left

      knee_angle_error_left = knee_angle_desired_left - knee_angle_measured_left
      knee_angle_error_right = knee_angle_desired_right - knee_angle_measured_right

      freeze_integrator = ((knee_angle_error_left < 0 and self.knee_angle_left_pid.error_integral <= -MAX_KNEE_ANGLE_INTEGRATOR_LEFT) or
                           (knee_angle_error_left > 0 and self.knee_angle_left_pid.error_integral >= MAX_KNEE_ANGLE_INTEGRATOR_LEFT))
      knee_torque_l = self.knee_angle_left_pid.update(knee_angle_error_left, freeze_integrator=freeze_integrator)

      freeze_integrator = ((knee_angle_error_right < 0 and self.knee_angle_right_pid.error_integral <= -MAX_KNEE_ANGLE_INTEGRATOR_RIGHT) or
                           (knee_angle_error_right > 0 and self.knee_angle_right_pid.error_integral >= MAX_KNEE_ANGLE_INTEGRATOR_RIGHT))
      knee_torque_r = self.knee_angle_right_pid.update(knee_angle_error_right, freeze_integrator=freeze_integrator)

    # Torque rate limits
      self.torque_knee_l_filtered = np.clip(knee_torque_l,
                                        self.torque_knee_l_filtered - MAX_KNEE_TORQUE_RATE,
                                        self.torque_knee_l_filtered + MAX_KNEE_TORQUE_RATE)
      self.torque_knee_r_filtered = np.clip(knee_torque_r,
                                        self.torque_knee_r_filtered - MAX_KNEE_TORQUE_RATE,
                                        self.torque_knee_r_filtered + MAX_KNEE_TORQUE_RATE)
      
      knee_torque_l = int(np.clip(self.torque_knee_l_filtered, -MAX_KNEE_TORQUE_LEFT, MAX_KNEE_TORQUE_LEFT))
      knee_torque_r = int(np.clip(self.torque_knee_r_filtered, -MAX_KNEE_TORQUE_RIGHT, MAX_KNEE_TORQUE_RIGHT))

    can_sends = []
    torque_l = torque_r = 0
    can_sends.append(bodycan.create_control(self.packer, torque_l, torque_r))

    print(knee_torque_l, knee_angle_measured_left)
    # knee_torque_l = 0
    # knee_torque_r = 11
    
    can_sends.append(bodycan.create_knee_control(self.packer, knee_torque_l, knee_torque_r))

    new_actuators = CC.actuators.copy()
    new_actuators.accel = torque_l
    new_actuators.steer = torque_r

    return new_actuators, can_sends
