#!/usr/bin/env python3
import os
import math
import json
import numpy as np

import cereal.messaging as messaging
from cereal import car
from cereal import log
from openpilot.common.params import Params, put_nonblocking
from openpilot.common.realtime import config_realtime_process, DT_MDL
from openpilot.common.numpy_fast import clip
from openpilot.selfdrive.locationd.models.car_kf import CarKalman, ObservationKind, States
from openpilot.selfdrive.locationd.models.constants import GENERATED_DIR
from openpilot.system.swaglog import cloudlog


MAX_ANGLE_OFFSET_DELTA = 20 * DT_MDL  # Max 20 deg/s
ROLL_MAX_DELTA = math.radians(20.0) * DT_MDL  # 20deg in 1 second is well within curvature limits
ROLL_MIN, ROLL_MAX = math.radians(-10), math.radians(10)
ROLL_LOWERED_MAX = math.radians(8)
ROLL_STD_MAX = math.radians(1.5)
LATERAL_ACC_SENSOR_THRESHOLD = 4.0
OFFSET_MAX = 10.0
OFFSET_LOWERED_MAX = 8.0


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

    self.speed = 0.0
    self.yaw_rate = 0.0
    self.yaw_rate_std = 0.0
    self.roll = 0.0
    self.steering_angle = 0.0
    self.roll_valid = False

  def handle_log(self, t, which, msg):
    if which == 'liveLocationKalman':
      self.yaw_rate = msg.angularVelocityCalibrated.value[2]
      self.yaw_rate_std = msg.angularVelocityCalibrated.std[2]

      localizer_roll = msg.orientationNED.value[0]
      localizer_roll_std = np.radians(1) if np.isnan(msg.orientationNED.std[0]) else msg.orientationNED.std[0]
      self.roll_valid = (localizer_roll_std < ROLL_STD_MAX) and (ROLL_MIN < localizer_roll < ROLL_MAX) and msg.sensorsOK
      if self.roll_valid:
        roll = localizer_roll
        # Experimentally found multiplier of 2 to be best trade-off between stability and accuracy or similar?
        roll_std = 2 * localizer_roll_std
      else:
        # This is done to bound the road roll estimate when localizer values are invalid
        roll = 0.0
        roll_std = np.radians(10.0)
      self.roll = clip(roll, self.roll - ROLL_MAX_DELTA, self.roll + ROLL_MAX_DELTA)

      yaw_rate_valid = msg.angularVelocityCalibrated.valid
      yaw_rate_valid = yaw_rate_valid and 0 < self.yaw_rate_std < 10  # rad/s
      yaw_rate_valid = yaw_rate_valid and abs(self.yaw_rate) < 1  # rad/s

      if self.active:
        if msg.posenetOK:

          if yaw_rate_valid:
            self.kf.predict_and_observe(t,
                                        ObservationKind.ROAD_FRAME_YAW_RATE,
                                        np.array([[-self.yaw_rate]]),
                                        np.array([np.atleast_2d(self.yaw_rate_std**2)]))

          self.kf.predict_and_observe(t,
                                      ObservationKind.ROAD_ROLL,
                                      np.array([[self.roll]]),
                                      np.array([np.atleast_2d(roll_std**2)]))
        self.kf.predict_and_observe(t, ObservationKind.ANGLE_OFFSET_FAST, np.array([[0]]))

        # We observe the current stiffness and steer ratio (with a high observation noise) to bound
        # the respective estimate STD. Otherwise the STDs keep increasing, causing rapid changes in the
        # states in longer routes (especially straight stretches).
        stiffness = float(self.kf.x[States.STIFFNESS].item())
        steer_ratio = float(self.kf.x[States.STEER_RATIO].item())
        self.kf.predict_and_observe(t, ObservationKind.STIFFNESS, np.array([[stiffness]]))
        self.kf.predict_and_observe(t, ObservationKind.STEER_RATIO, np.array([[steer_ratio]]))

    elif which == 'carState':
      self.steering_angle = msg.steeringAngleDeg
      self.speed = msg.vEgo

      in_linear_region = abs(self.steering_angle) < 45
      self.active = self.speed > 1 and in_linear_region

      if self.active:
        self.kf.predict_and_observe(t, ObservationKind.STEER_ANGLE, np.array([[math.radians(msg.steeringAngleDeg)]]))
        self.kf.predict_and_observe(t, ObservationKind.ROAD_FRAME_X_SPEED, np.array([[self.speed]]))

    if not self.active:
      # Reset time when stopped so uncertainty doesn't grow
      self.kf.filter.set_filter_time(t)
      self.kf.filter.reset_rewind()


def check_valid_with_hysteresis(current_valid: bool, val: float, threshold: float, lowered_threshold: float):
  if current_valid:
    current_valid = abs(val) < threshold
  else:
    current_valid = abs(val) < lowered_threshold
  return current_valid


def main(sm=None, pm=None):
  config_realtime_process([0, 1, 2, 3], 5)

  DEBUG = bool(int(os.getenv("DEBUG", "0")))
  REPLAY = bool(int(os.getenv("REPLAY", "0")))

  if sm is None:
    sm = messaging.SubMaster(['liveLocationKalman', 'carState'], poll=['liveLocationKalman'])
  if pm is None:
    pm = messaging.PubMaster(['liveParameters'])

  params_reader = Params()
  # wait for stats about the car to come in from controls
  cloudlog.info("paramsd is waiting for CarParams")
  with car.CarParams.from_bytes(params_reader.get("CarParams", block=True)) as msg:
    CP = msg
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
      steer_ratio_sane = min_sr <= params['steerRatio'] <= max_sr
      if not steer_ratio_sane:
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

  if not REPLAY:
    # When driving in wet conditions the stiffness can go down, and then be too low on the next drive
    # Without a way to detect this we have to reset the stiffness every drive
    params['stiffnessFactor'] = 1.0

  pInitial = None
  if DEBUG:
    pInitial = np.array(params['filterState']['std']) if 'filterState' in params else None

  learner = ParamsLearner(CP, params['steerRatio'], params['stiffnessFactor'], math.radians(params['angleOffsetAverageDeg']), pInitial)
  angle_offset_average = params['angleOffsetAverageDeg']
  angle_offset = angle_offset_average
  roll = 0.0
  avg_offset_valid = True
  total_offset_valid = True
  roll_valid = True

  while True:
    sm.update()
    if sm.all_checks():
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

      angle_offset_average = clip(math.degrees(x[States.ANGLE_OFFSET].item()),
                                  angle_offset_average - MAX_ANGLE_OFFSET_DELTA, angle_offset_average + MAX_ANGLE_OFFSET_DELTA)
      angle_offset = clip(math.degrees(x[States.ANGLE_OFFSET].item() + x[States.ANGLE_OFFSET_FAST].item()),
                          angle_offset - MAX_ANGLE_OFFSET_DELTA, angle_offset + MAX_ANGLE_OFFSET_DELTA)
      roll = clip(float(x[States.ROAD_ROLL].item()), roll - ROLL_MAX_DELTA, roll + ROLL_MAX_DELTA)
      roll_std = float(P[States.ROAD_ROLL].item())
      # Account for the opposite signs of the yaw rates
      sensors_valid = bool(abs(learner.speed * (x[States.YAW_RATE].item() + learner.yaw_rate)) < LATERAL_ACC_SENSOR_THRESHOLD)
      avg_offset_valid = check_valid_with_hysteresis(avg_offset_valid, angle_offset_average, OFFSET_MAX, OFFSET_LOWERED_MAX)
      total_offset_valid = check_valid_with_hysteresis(total_offset_valid, angle_offset, OFFSET_MAX, OFFSET_LOWERED_MAX)
      roll_valid = check_valid_with_hysteresis(roll_valid, roll, ROLL_MAX, ROLL_LOWERED_MAX)

      msg = messaging.new_message('liveParameters')

      liveParameters = msg.liveParameters
      liveParameters.posenetValid = True
      liveParameters.sensorValid = sensors_valid
      liveParameters.steerRatio = float(x[States.STEER_RATIO].item())
      liveParameters.stiffnessFactor = float(x[States.STIFFNESS].item())
      liveParameters.roll = roll
      liveParameters.angleOffsetAverageDeg = angle_offset_average
      liveParameters.angleOffsetDeg = angle_offset
      liveParameters.valid = all((
        avg_offset_valid,
        total_offset_valid,
        roll_valid,
        roll_std < ROLL_STD_MAX,
        0.2 <= liveParameters.stiffnessFactor <= 5.0,
        min_sr <= liveParameters.steerRatio <= max_sr,
      ))
      liveParameters.steerRatioStd = float(P[States.STEER_RATIO].item())
      liveParameters.stiffnessFactorStd = float(P[States.STIFFNESS].item())
      liveParameters.angleOffsetAverageStd = float(P[States.ANGLE_OFFSET].item())
      liveParameters.angleOffsetFastStd = float(P[States.ANGLE_OFFSET_FAST].item())
      if DEBUG:
        liveParameters.filterState = log.LiveLocationKalman.Measurement.new_message()
        liveParameters.filterState.value = x.tolist()
        liveParameters.filterState.std = P.tolist()
        liveParameters.filterState.valid = True

      msg.valid = sm.all_checks()

      if sm.frame % 1200 == 0:  # once a minute
        params = {
          'carFingerprint': CP.carFingerprint,
          'steerRatio': liveParameters.steerRatio,
          'stiffnessFactor': liveParameters.stiffnessFactor,
          'angleOffsetAverageDeg': liveParameters.angleOffsetAverageDeg,
        }
        put_nonblocking("LiveParameters", json.dumps(params))

      pm.send('liveParameters', msg)


if __name__ == "__main__":
  main()
