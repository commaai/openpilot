from selfdrive.car.body import bodycan
from opendbc.can.packer import CANPacker

import cereal.messaging as messaging
import numpy as np

class CarController():
  def __init__(self, dbc_name, CP, VM):
    self.CP = CP
    self.car_fingerprint = CP.carFingerprint

    self.lkas_max_torque = 0
    self.last_angle = 0

    self.packer = CANPacker(dbc_name)
    # ////////////////////////////////
    self.sm = messaging.SubMaster(['liveLocationKalman'])

    self.kp = 1300
    self.ki = 0
    self.kd = 280
    self.i = 0
    self.d = 0
    self.i_speed = 0
    self.i_tq = 0

    self.set_point = np.deg2rad(-0)

    self.accel_err = 0

    self.speed_measured = 0
    self.speed_desired = 0
    self.torque_right_filtered = 0.0
    self.torque_left_filtered = 0.0
    # ////////////////////////////////

  def update(self, c, CS, frame, actuators, cruise_cancel, hud_alert,
             left_line, right_line, left_lane_depart, right_lane_depart):

    # print(c.pitch) # Value from sm['liveLocationKalman'].orientationNED.value[1]

    # ///////////////////////////////////////
    self.sm.update()

    alpha = 1.0
    self.speed_desired = (1. - alpha)*self.speed_desired
    kp_speed = 0.001
    ki_speed = 0
    self.i_speed += ki_speed * (self.speed_desired - self.speed_measured)
    self.i_speed = np.clip(self.i_speed, -0.1, 0.1)
    self.set_point = kp_speed * (self.speed_desired - self.speed_measured) + self.i_speed
    try:
      angle_err = (-self.sm['liveLocationKalman'].orientationNED.value[1]) - self.set_point
      d_new = -self.sm['liveLocationKalman'].angularVelocityDevice.value[1]
      alpha_d = 1.0
      self.d = (1. - alpha_d) * self.d + alpha * d_new
      self.d =  np.clip(self.d, -1., 1.)
    except Exception:
      print("can't subscribe?")
      # Send 0 torque if can't read?
      pass

    self.i += angle_err
    self.i = np.clip(self.i, -2, 2)

    self.speed_measured = (CS.out.wheelSpeeds.fl + CS.out.wheelSpeeds.fr) / 2

    speed = int(np.clip(angle_err*self.kp + self.accel_err*self.ki + self.d*self.kd, -1000, 1000))

    kp_diff = 0.95
    kd_diff = 0.1
    p_tq = (CS.out.wheelSpeeds.fl - CS.out.wheelSpeeds.fr)

    torque_diff = int(np.clip(p_tq*kp_diff + self.i_tq*kd_diff, -100, 100))

    self.i_tq += (CS.out.wheelSpeeds.fl - CS.out.wheelSpeeds.fr)
    torque_r = speed + torque_diff
    torque_l = speed - torque_diff

    if torque_r > 0: torque_r += 10
    else: torque_r -= 10
    if torque_l > 0: torque_l += 10
    else: torque_l -= 10

    alpha_torque = 1.
    self.torque_right_filtered = (1. - alpha_torque) * self.torque_right_filtered + alpha_torque * torque_r
    self.torque_left_filtered = (1. - alpha_torque) * self.torque_left_filtered + alpha_torque * torque_l
    torque_r = int(np.clip(self.torque_right_filtered, -1000, 1000))
    torque_l = int(np.clip(self.torque_left_filtered, -1000, 1000))
    self.accel_err += CS.out.wheelSpeeds.fl + CS.out.wheelSpeeds.fr
    # ///////////////////////////////////////
    can_sends = []

    apply_angle = actuators.steeringAngleDeg

    #torque_l = 60
    #torque_r = 60
    can_sends.append(bodycan.create_control(
        self.packer, torque_l, torque_r))

    new_actuators = actuators.copy()
    new_actuators.steeringAngleDeg = apply_angle

    return new_actuators, can_sends
