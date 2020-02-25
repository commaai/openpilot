#!/usr/bin/env python3
import math

import cereal.messaging as messaging
import common.transformations.orientation as orient
from selfdrive.locationd.kalman.models.car_kf import CarKalman, ObservationKind, States


class ParamsLearner:
  def __init__(self):
    self.kf = CarKalman()
    self.active = False

  def handle_log(self, t, which, msg):
    if self.kf.filter.filter_time is None or t < self.kf.filter.filter_time - 10.0:
      print("Resetting time")
      self.kf.filter_filter_time = t

    if which == 'liveLocation':
      yaw_rate = -msg.gyro[2]
      roll, pitch, yaw = math.radians(msg.roll), math.radians(msg.pitch), math.radians(-msg.heading)
      v_device = orient.rot_from_euler([roll, pitch, yaw]).dot(msg.vNED)

      self.active = v_device[0] > 5

      if self.active:
        self.kf.predict_and_observe(t, ObservationKind.CAL_DEVICE_FRAME_YAW_RATE, [yaw_rate])
        self.kf.predict_and_observe(t, ObservationKind.CAL_DEVICE_FRAME_XY_SPEED, [[v_device[0], -v_device[1]]])
      else:
        self.kf.filter.filter_time = t

    elif which == 'carState':
      if self.active:
        self.kf.predict_and_observe(t, ObservationKind.STEER_ANGLE, [math.radians(msg.steeringAngle)])
        self.kf.predict_and_observe(t, ObservationKind.ANGLE_OFFSET_FAST, [0])
      else:
        self.kf.filter.filter_time = t


def main(sm=None, pm=None):
  if sm is None:
    sm = messaging.SubMaster(['liveLocation', 'carState'])
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

    if sm.updated['carState']:
      msg = messaging.new_message()
      msg.logMonoTime = sm.logMonoTime['carState']

      msg.init('liveParameters')
      msg.liveParameters.valid = bool(learner.active)
      msg.liveParameters.posenetValid = True
      msg.liveParameters.sensorValid = True

      x = learner.kf.x
      msg.liveParameters.steerRatio = float(x[States.STEER_RATIO])
      msg.liveParameters.stiffnessFactor = float(x[States.STIFFNESS])
      msg.liveParameters.angleOffsetAverage = math.degrees(x[States.ANGLE_OFFSET])
      msg.liveParameters.angleOffset = math.degrees(x[States.ANGLE_OFFSET_FAST])

      pm.send('liveParameters', msg)


if __name__ == "__main__":
  main()
