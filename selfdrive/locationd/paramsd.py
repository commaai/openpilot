#!/usr/bin/env python3
import gc
import math

import json
import numpy as np

import cereal.messaging as messaging
from cereal import car
from common.params import Params, put_nonblocking
from common.realtime import set_realtime_priority, DT_MDL
from common.numpy_fast import clip
from selfdrive.locationd.models.car_kf import CarKalman, ObservationKind, States
from selfdrive.locationd.models.constants import GENERATED_DIR
from selfdrive.swaglog import cloudlog


MAX_ANGLE_OFFSET_DELTA = 20 * DT_MDL  # Max 20 deg/s
ROLL_MIN, ROLL_MAX = math.radians(-10), math.radians(10)

class ParamsLearner:
  def __init__(self, CP, steer_ratio, stiffness_factor, angle_offset, P_initial=None):
    self.kf = CarKalman(GENERATED_DIR, steer_ratio, stiffness_factor, angle_offset, P_initial)

    self.kf.filter.set_global("mass", CP.mass)
    self.kf.filter.set_global("rotational_inertia", CP.rotationalInertia)
    self.kf.filter.set_global("center_to_front", CP.centerToFront)
    self.kf.filter.set_global("center_to_rear", CP.wheelbase - CP.centerToFront)
    self.kf.filter.set_global("stiffness_front", CP.tireStiffnessFront)
    self.kf.filter.set_global("stiffness_rear", CP.tireStiffnessRear)

    self.active = False

    self.speed = 0
    self.steering_pressed = False
    self.steering_angle = 0

    self.valid = True

  def handle_log(self, t, which, msg):
    if which == 'liveLocationKalman':
      yaw_rate = msg.angularVelocityCalibrated.value[2]
      yaw_rate_std = msg.angularVelocityCalibrated.std[2]

      roll = msg.orientationNED.value[0]
      roll_valid = msg.orientationNED.valid and ROLL_MIN < roll < ROLL_MAX

      yaw_rate_valid = msg.angularVelocityCalibrated.valid
      yaw_rate_valid = yaw_rate_valid and 0 < yaw_rate_std < 10  # rad/s
      yaw_rate_valid = yaw_rate_valid and abs(yaw_rate) < 1  # rad/s

      if self.active:
        if msg.inputsOK and msg.posenetOK:

          if yaw_rate_valid:
            self.kf.predict_and_observe(t,
                                        ObservationKind.ROAD_FRAME_YAW_RATE,
                                        np.array([[-yaw_rate]]),
                                        np.array([np.atleast_2d(yaw_rate_std**2)]))

          if roll_valid:
            self.kf.predict_and_observe(t,
                                        ObservationKind.ROAD_ROLL,
                                        np.array([[roll]]))
          else:
            # This is done to bound the road roll estimate when localizer values are invalid
            self.kf.predict_and_observe(t, ObservationKind.ROAD_ROLL_INVALID, np.array([[0]]))

        self.kf.predict_and_observe(t, ObservationKind.ANGLE_OFFSET_FAST, np.array([[0]]))

    elif which == 'carState':
      self.steering_angle = msg.steeringAngleDeg
      self.steering_pressed = msg.steeringPressed
      self.speed = msg.vEgo

      in_linear_region = abs(self.steering_angle) < 45 or not self.steering_pressed
      self.active = self.speed > 5 and in_linear_region

      if self.active:
        self.kf.predict_and_observe(t, ObservationKind.STEER_ANGLE, np.array([[math.radians(msg.steeringAngleDeg)]]))
        self.kf.predict_and_observe(t, ObservationKind.ROAD_FRAME_X_SPEED, np.array([[self.speed]]))

    if not self.active:
      # Reset time when stopped so uncertainty doesn't grow
      self.kf.filter.set_filter_time(t)
      self.kf.filter.reset_rewind()


def main(sm=None, pm=None):
  gc.disable()
  set_realtime_priority(5)

  if sm is None:
    sm = messaging.SubMaster(['liveLocationKalman', 'carState'], poll=['liveLocationKalman'])
  if pm is None:
    pm = messaging.PubMaster(['liveParameters'])

  params_reader = Params()
  # wait for stats about the car to come in from controls
  cloudlog.info("paramsd is waiting for CarParams")
  CP = car.CarParams.from_bytes(params_reader.get("CarParams", block=True))
  cloudlog.info("paramsd got CarParams")

  min_sr, max_sr = 0.5 * CP.steerRatio, 2.0 * CP.steerRatio

  params = params_reader.get("LiveParameters")

  # Check if car model matches
  if params is not None:
    params = json.loads(params)
    if params.get('carFingerprint', None) != CP.carFingerprint:
      cloudlog.info("Parameter learner found parameters for wrong car.")
      params = None

  # Check if starting values are sane
  if params is not None:
    try:
      angle_offset_sane = abs(params.get('angleOffsetAverageDeg')) < 10.0
      steer_ratio_sane = min_sr <= params['steerRatio'] <= max_sr
      params_sane = angle_offset_sane and steer_ratio_sane
      if not params_sane:
        cloudlog.info(f"Invalid starting values found {params}")
        params = None
    except Exception as e:
      cloudlog.info(f"Error reading params {params}: {str(e)}")
      params = None

  # TODO: cache the params with the capnp struct
  if params is None:
    params = {
      'carFingerprint': CP.carFingerprint,
      'steerRatio': CP.steerRatio,
      'stiffnessFactor': 1.0,
      'angleOffsetAverageDeg': 0.0,
    }
    cloudlog.info("Parameter learner resetting to default values")

  # When driving in wet conditions the stiffness can go down, and then be too low on the next drive
  # Without a way to detect this we have to reset the stiffness every drive
  params['stiffnessFactor'] = 1.0
  learner = ParamsLearner(CP, params['steerRatio'], params['stiffnessFactor'], math.radians(params['angleOffsetAverageDeg']))
  angle_offset_average = params['angleOffsetAverageDeg']
  angle_offset = angle_offset_average

  while True:
    sm.update()
    for which in sorted(sm.updated.keys(), key=lambda x: sm.logMonoTime[x]):
      if sm.updated[which]:
        t = sm.logMonoTime[which] * 1e-9
        learner.handle_log(t, which, sm[which])

    if sm.updated['liveLocationKalman']:
      x = learner.kf.x
      P = np.sqrt(learner.kf.P.diagonal())
      if not all(map(math.isfinite, x)):
        cloudlog.error("NaN in liveParameters estimate. Resetting to default values")
        learner = ParamsLearner(CP, CP.steerRatio, 1.0, 0.0)
        x = learner.kf.x

      angle_offset_average = clip(math.degrees(x[States.ANGLE_OFFSET]), angle_offset_average - MAX_ANGLE_OFFSET_DELTA, angle_offset_average + MAX_ANGLE_OFFSET_DELTA)
      angle_offset = clip(math.degrees(x[States.ANGLE_OFFSET] + x[States.ANGLE_OFFSET_FAST]), angle_offset - MAX_ANGLE_OFFSET_DELTA, angle_offset + MAX_ANGLE_OFFSET_DELTA)

      msg = messaging.new_message('liveParameters')
      msg.logMonoTime = sm.logMonoTime['carState']

      msg.liveParameters.posenetValid = True
      msg.liveParameters.sensorValid = True
      msg.liveParameters.steerRatio = float(x[States.STEER_RATIO])
      msg.liveParameters.stiffnessFactor = float(x[States.STIFFNESS])
      msg.liveParameters.roll = float(x[States.ROAD_ROLL])
      msg.liveParameters.angleOffsetAverageDeg = angle_offset_average
      msg.liveParameters.angleOffsetDeg = angle_offset
      msg.liveParameters.valid = all((
        abs(msg.liveParameters.angleOffsetAverageDeg) < 10.0,
        abs(msg.liveParameters.angleOffsetDeg) < 10.0,
        0.2 <= msg.liveParameters.stiffnessFactor <= 5.0,
        min_sr <= msg.liveParameters.steerRatio <= max_sr,
      ))
      msg.liveParameters.steerRatioStd = float(P[States.STEER_RATIO])
      msg.liveParameters.stiffnessFactorStd = float(P[States.STIFFNESS])
      msg.liveParameters.angleOffsetAverageStd = float(P[States.ANGLE_OFFSET])
      msg.liveParameters.angleOffsetFastStd = float(P[States.ANGLE_OFFSET_FAST])

      if sm.frame % 1200 == 0:  # once a minute
        params = {
          'carFingerprint': CP.carFingerprint,
          'steerRatio': msg.liveParameters.steerRatio,
          'stiffnessFactor': msg.liveParameters.stiffnessFactor,
          'angleOffsetAverageDeg': msg.liveParameters.angleOffsetAverageDeg,
        }
        put_nonblocking("LiveParameters", json.dumps(params))

      pm.send('liveParameters', msg)


if __name__ == "__main__":
  main()
