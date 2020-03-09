#!/usr/bin/env python3
import math

import cereal.messaging as messaging
from selfdrive.locationd.kalman.models.car_kf import CarKalman, ObservationKind, States

CARSTATE_DECIMATION = 5


class ParamsLearner:
  def __init__(self):
    self.kf = CarKalman()
    self.active = False

    self.speed = 0
    self.steering_pressed = False
    self.steering_angle = 0
    self.carstate_counter = 0

  def update_active(self):
    self.active = (abs(self.steering_angle) < 45 or not self.steering_pressed) and self.speed > 5

  def handle_log(self, t, which, msg):
    if which == 'liveLocationKalman':

      v_device = msg.velocityDevice.value
      # v_device_std = msg.velocityDevice.std
      self.speed = v_device[0]

      yaw_rate = msg.angularVelocityDevice.value[2]
      # yaw_rate_std = msg.angularVelocityDevice.std[2]

      self.update_active()
      if self.active:
        self.kf.predict_and_observe(t, ObservationKind.CAL_DEVICE_FRAME_YAW_RATE, [-yaw_rate])
        self.kf.predict_and_observe(t, ObservationKind.CAL_DEVICE_FRAME_XY_SPEED, [[v_device[0], -v_device[1]]])

        # Clamp values
        x = self.kf.x
        if not (10 < x[States.STEER_RATIO] < 25):
          self.kf.predict_and_observe(t, ObservationKind.STEER_RATIO, [15.0])

        if not (0.5 < x[States.STIFFNESS] < 3.0):
          self.kf.predict_and_observe(t, ObservationKind.STIFFNESS, [1.0])

      else:
        self.kf.filter.filter_time = t - 0.1

    elif which == 'carState':
      self.carstate_counter += 1
      if self.carstate_counter % CARSTATE_DECIMATION == 0:
        self.steering_angle = msg.steeringAngle
        self.steering_pressed = msg.steeringPressed

        self.update_active()
        if self.active:
          self.kf.predict_and_observe(t, ObservationKind.STEER_ANGLE, [math.radians(msg.steeringAngle)])
          self.kf.predict_and_observe(t, ObservationKind.ANGLE_OFFSET_FAST, [0])
        else:
          self.kf.filter.filter_time = t - 0.1


def main(sm=None, pm=None):
  if sm is None:
    sm = messaging.SubMaster(['liveLocationKalman', 'carState'])
  if pm is None:
    pm = messaging.PubMaster(['liveParameters'])

  learner = ParamsLearner()

  while True:
    sm.update()

    for which, updated in sm.updated.items():
      if not updated:
        continue
      t = sm.logMonoTime[which] * 1e-9
      learner.handle_log(t, which, sm[which])

    # TODO: set valid to false when locationd stops sending
    # TODO: make sure controlsd knows when there is no gyro
    # TODO: move posenetValid somewhere else to show the model uncertainty alert
    # TODO: Save and resume values from param
    # TODO: Change KF to allow mass, etc to be inputs in predict step

    if sm.updated['carState']:
      msg = messaging.new_message('liveParameters')
      msg.logMonoTime = sm.logMonoTime['carState']

      msg.liveParameters.valid = True  # TODO: Check if learned values are sane
      msg.liveParameters.posenetValid = True
      msg.liveParameters.sensorValid = True

      x = learner.kf.x
      msg.liveParameters.steerRatio = float(x[States.STEER_RATIO])
      msg.liveParameters.stiffnessFactor = float(x[States.STIFFNESS])
      msg.liveParameters.angleOffsetAverage = math.degrees(x[States.ANGLE_OFFSET])
      msg.liveParameters.angleOffset = math.degrees(x[States.ANGLE_OFFSET_FAST])

      # P = learner.kf.P
      # print()
      # print("sR", float(x[States.STEER_RATIO]), float(P[States.STEER_RATIO, States.STEER_RATIO])**0.5)
      # print("x ", float(x[States.STIFFNESS]), float(P[States.STIFFNESS, States.STIFFNESS])**0.5)
      # print("ao avg ", math.degrees(x[States.ANGLE_OFFSET]), math.degrees(P[States.ANGLE_OFFSET, States.ANGLE_OFFSET])**0.5)
      # print("ao ", math.degrees(x[States.ANGLE_OFFSET_FAST]), math.degrees(P[States.ANGLE_OFFSET_FAST, States.ANGLE_OFFSET_FAST])**0.5)

      pm.send('liveParameters', msg)


if __name__ == "__main__":
  main()
