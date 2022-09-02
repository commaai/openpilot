import numpy as np

from common.realtime import DT_CTRL
from opendbc.can.packer import CANPacker
from selfdrive.car.body import bodycan
from selfdrive.car.body.values import CAR, SPEED_FROM_RPM
from selfdrive.controls.lib.pid import PIDController


MAX_TORQUE = 700
MAX_TORQUE_RATE = 50
MAX_ANGLE_ERROR = np.radians(7)
MAX_POS_INTEGRATOR = 10.0   # meters
MAX_TURN_INTEGRATOR = 10.0  # meters

MAX_TORQUE_KNEE = 400
MAX_TORQUE_HIP = 200
MAX_KNEE_TORQUE_RATE = 5
MAX_ANGLE_INTEGRATOR_KNEE = 1.0
MAX_ANGLE_INTEGRATOR_HIP = 1.0


class CarController:
  def __init__(self, dbc_name, CP, VM):
    self.CP = CP
    self.packer = CANPacker(dbc_name)

    # Speed, balance and turn PIDs
    if self.CP.carFingerprint == CAR.BODY:
      self.speed_pid = PIDController(0.115, k_i=0.23, rate=1/DT_CTRL)
      self.balance_pid = PIDController(1300, k_i=0, k_d=280, rate=1/DT_CTRL)
    else:
      self.speed_pid = PIDController(2e-2, k_i=8e-2, rate=1/DT_CTRL)
      self.balance_pid = PIDController(2400, k_i=0, k_d=180, rate=1/DT_CTRL)
    self.turn_pid = PIDController(110, k_i=11.5, rate=1/DT_CTRL)

    self.torque_left_wheel_filtered = 0.
    self.torque_right_wheel_filtered = 0.

    self.knee_angle_pid = PIDController(10, k_i=0.2, rate=1/DT_CTRL)
    self.hip_angle_pid = PIDController(10, k_i=0.2, rate=1/DT_CTRL)

    self.torque_knee_filtered = 0.
    self.torque_hip_filtered = 0.

  @staticmethod
  def deadband_filter(torque, deadband):
    if torque > 0:
      torque += deadband
    else:
      torque -= deadband
    return torque

  def update(self, CC, CS):
    torque_left_wheel = 0
    torque_right_wheel = 0
    knee_angle_error = 0

    can_sends = []

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

      # if self.CP.carFingerprint == CAR.BODY_KNEE:
        # knee_angle_error = np.radians(CS.knee_angle - CS.hip_angle)

      # Clip angle error, this is enough to get up from stands
      angle_error = np.clip((-CC.orientationNED[1]) - angle_setpoint + knee_angle_error, -MAX_ANGLE_ERROR, MAX_ANGLE_ERROR)
      angle_error_rate = np.clip(-CC.angularVelocity[1], -1., 1.)
      torque = self.balance_pid.update(angle_error, error_rate=angle_error_rate)

      speed_diff_measured = SPEED_FROM_RPM * (CS.out.wheelSpeeds.fl - CS.out.wheelSpeeds.fr)
      turn_error = speed_diff_measured - speed_diff_desired
      freeze_integrator = ((turn_error < 0 and self.turn_pid.error_integral <= -MAX_TURN_INTEGRATOR) or
                           (turn_error > 0 and self.turn_pid.error_integral >= MAX_TURN_INTEGRATOR))
      torque_diff = self.turn_pid.update(turn_error, freeze_integrator=freeze_integrator)

      # Combine 2 PIDs outputs
      torque_left_wheel = torque - torque_diff
      torque_right_wheel = torque + torque_diff

      # Torque rate limits
      self.torque_left_wheel_filtered = np.clip(self.deadband_filter(torque_left_wheel, 10),
                                       self.torque_left_wheel_filtered - MAX_TORQUE_RATE,
                                       self.torque_left_wheel_filtered + MAX_TORQUE_RATE)
      self.torque_right_wheel_filtered = np.clip(self.deadband_filter(torque_right_wheel, 10),
                                       self.torque_right_wheel_filtered - MAX_TORQUE_RATE,
                                       self.torque_right_wheel_filtered + MAX_TORQUE_RATE)

      torque_left_wheel = int(np.clip(self.torque_left_wheel_filtered, -MAX_TORQUE, MAX_TORQUE))
      torque_right_wheel = int(np.clip(self.torque_right_wheel_filtered, -MAX_TORQUE, MAX_TORQUE))

    # Knee angle controls
    if self.CP.carFingerprint == CAR.BODY_KNEE and CC.enabled:
      angle_desired_knee = CC.actuators.gas
      angle_desired_hip = CC.actuators.brake

      angle_measured_knee = CS.knee_angle
      angle_measured_hip = CS.hip_angle

      angle_error_knee = angle_desired_knee - angle_measured_knee
      angle_error_hip = angle_desired_hip - angle_measured_hip

      freeze_integrator = ((angle_error_knee < 0 and self.knee_angle_pid.error_integral <= -MAX_ANGLE_INTEGRATOR_KNEE) or
                           (angle_error_knee > 0 and self.knee_angle_pid.error_integral >= MAX_ANGLE_INTEGRATOR_KNEE))
      torque_knee = self.knee_angle_pid.update(angle_error_knee, freeze_integrator=freeze_integrator)

      freeze_integrator = ((angle_error_hip < 0 and self.hip_angle_pid.error_integral <= -MAX_ANGLE_INTEGRATOR_HIP) or
                           (angle_error_hip > 0 and self.hip_angle_pid.error_integral >= MAX_ANGLE_INTEGRATOR_HIP))
      torque_hip = self.hip_angle_pid.update(angle_error_hip, freeze_integrator=freeze_integrator)

      # Torque rate limits
      self.torque_knee_filtered = np.clip(torque_knee,
                                        self.torque_knee_filtered - MAX_KNEE_TORQUE_RATE,
                                        self.torque_knee_filtered + MAX_KNEE_TORQUE_RATE)
      self.torque_hip_filtered = np.clip(torque_hip,
                                        self.torque_hip_filtered - MAX_KNEE_TORQUE_RATE,
                                        self.torque_hip_filtered + MAX_KNEE_TORQUE_RATE)

      torque_knee = int(np.clip(self.torque_knee_filtered, -MAX_TORQUE_KNEE, MAX_TORQUE_KNEE))
      torque_hip = int(np.clip(self.torque_hip_filtered, -MAX_TORQUE_HIP, MAX_TORQUE_HIP))

      # Extra safety for when training stands attached(temporary):
      if angle_measured_knee <= 30 and torque_knee > 0:
        torque_knee = 0
      elif angle_measured_knee >= 330 and torque_knee < 0:
        torque_knee = 0

      # Do not try to balance if knee and hip have excessive angle on start
      if abs(angle_desired_knee - angle_measured_knee) > 5 or abs(angle_desired_hip - angle_measured_hip) > 5:
        torque_left_wheel = 0
        torque_right_wheel = 0

      can_sends.append(bodycan.create_knee_control(self.packer, torque_knee, torque_hip))

    can_sends.append(bodycan.create_control(self.packer, torque_left_wheel, torque_right_wheel))
    # can_sends.append(bodycan.create_rpm_limit(self.packer,500,500)) # Increase speed limit to ~5m/s

    new_actuators = CC.actuators.copy()
    new_actuators.accel = torque_left_wheel
    new_actuators.steer = torque_right_wheel

    return new_actuators, can_sends
