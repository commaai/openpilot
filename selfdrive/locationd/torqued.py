#!/usr/bin/env python3
import sys
import signal
import numpy as np
from collections import deque, defaultdict

import cereal.messaging as messaging
from cereal import car, log
from common.params import Params
from common.realtime import config_realtime_process, DT_MDL
from common.filter_simple import FirstOrderFilter
from system.swaglog import cloudlog
from selfdrive.controls.lib.vehicle_model import ACCELERATION_DUE_TO_GRAVITY


HISTORY = 5  # secs
POINTS_PER_BUCKET = 2000
MIN_POINTS_TOTAL = 4000
FIT_POINTS_TOTAL = 3000
MIN_VEL = 15  # m/s
FRICTION_FACTOR = 1.5  # ~85% of data coverage
SANITY_FACTOR = 0.5
STEER_MIN_THRESHOLD = 0.02
MIN_FILTER_DECAY = 50
MAX_FILTER_DECAY = 250
LAT_ACC_THRESHOLD = 1
STEER_BUCKET_BOUNDS = [(-0.5, -0.3), (-0.3, -0.2), (-0.2, -0.1), (-0.1, 0), (0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5)]
MIN_BUCKET_POINTS = [100, 300, 500, 500, 500, 500, 300, 100]
MAX_RESETS = 5.0
MIN_DATA_PERC = 0.25
MIN_ENGAGE_BUFFER = 2  # secs
TORQUED_TAG = "v0.1"


def slope2rot(slope):
  sin = np.sqrt(slope**2 / (slope**2 + 1))
  cos = np.sqrt(1 / (slope**2 + 1))
  return np.array([[cos, -sin], [sin, cos]])


class PointBuckets:
  def __init__(self, x_bounds, min_points):
    self.x_bounds = x_bounds
    self.buckets = {bounds: deque(maxlen=POINTS_PER_BUCKET) for bounds in x_bounds}
    self.buckets_min_points = {bounds: min_point for bounds, min_point in zip(x_bounds, min_points)}

  def bucket_lengths(self):
    return [len(v) for v in self.buckets.values()]

  def __len__(self):
    return sum(self.bucket_lengths())

  def is_valid(self):
    return all([len(v) >= min_pts for v, min_pts in zip(self.buckets.values(), self.buckets_min_points.values())]) and (self.__len__() >= MIN_POINTS_TOTAL)

  def add_point(self, x, y):
    for bound_min, bound_max in self.x_bounds:
      if (x >= bound_min) and (x < bound_max):
        self.buckets[(bound_min, bound_max)].append([x, y])
        break

  def get_points(self, num_points=None):
    points = [v for sublist in self.buckets.values() for v in list(sublist)]
    if num_points is None:
      return points
    return np.array(points)[np.random.choice(len(points), min(len(points), num_points), replace=False)]

  def load_points(self, points):
    for x, y in points:
      self.add_point(x, y)


class TorqueEstimator:
  def __init__(self, CP, params):
    self.hist_len = int(HISTORY / DT_MDL)
    self.lag = CP.steerActuatorDelay + .2   # from controlsd

    self.offline_friction = 0.0
    self.offline_latAccelFactor = 0.0
    self.resets = 0.0

    if CP.lateralTuning.which() == 'torque':
      self.offline_friction = CP.lateralTuning.torque.friction
      self.offline_latAccelFactor = CP.lateralTuning.torque.latAccelFactor

    self.reset()
    params = log.Event.from_bytes(params).liveTorqueParameters if params is not None else None
    if params is not None and\
       self.is_sane(params.latAccelFactorFiltered, params.latAccelOffsetFiltered, params.frictionCoefficientFiltered) and\
       params.tag == TORQUED_TAG:
       # ToDo: check if car has changed and reset (do not load old points if this is a new car)
      initial_params = {
        'latAccelFactor': params.latAccelFactorFiltered,
        'latAccelOffset': params.latAccelOffsetFiltered,
        'frictionCoefficient': params.frictionCoefficientFiltered,
        'points': params.points
      }
      self.decay = params.decay
      self.filtered_points.load_points(initial_params['points'])
    else:
      initial_params = {
        'latAccelFactor': self.offline_latAccelFactor,
        'latAccelOffset': 0.0,
        'frictionCoefficient': self.offline_friction,
        'points': []
      }
      self.decay = MIN_FILTER_DECAY
    self.filtered_params = {}
    for param in initial_params:
      self.filtered_params[param] = FirstOrderFilter(initial_params[param], self.decay, DT_MDL)

  def reset(self):
    self.resets += 1.0
    self.raw_points = defaultdict(lambda: deque(maxlen=self.hist_len))
    self.filtered_points = PointBuckets(x_bounds=STEER_BUCKET_BOUNDS, min_points=MIN_BUCKET_POINTS)

  def estimate_params(self):
    points = self.filtered_points.get_points(FIT_POINTS_TOTAL)
    # total least square solution as both x and y are noisy observations
    # this is empirically the slope of the hysteresis parallelogram as opposed to the line through the diagonals
    points = np.insert(points, 1, 1.0, axis=1)
    _, _, v = np.linalg.svd(points, full_matrices=False)
    slope, offset = -v.T[0:2, 2] / v.T[2, 2]
    _, spread = np.einsum("ik,kj -> ji", np.column_stack((points[:, 0], points[:, 2] - offset)), slope2rot(slope))
    friction_coeff = np.std(spread) * FRICTION_FACTOR
    steer_data_distribution = np.array([np.mean(points[:, 0] < 0), np.mean(points[:, 0] > 0)])
    lat_acc_data_distribution = np.array([np.mean(points[:, 2] < offset), np.mean(points[:, 2] > offset)])
    return slope, offset, friction_coeff, steer_data_distribution, lat_acc_data_distribution

  def update_params(self, params):
    self.decay = min(self.decay + DT_MDL, MAX_FILTER_DECAY)
    for param, value in params.items():
      self.filtered_params[param].update(value)
      self.filtered_params[param].update_alpha(self.decay)

  def is_sane(self, latAccelFactor, latAccelOffset, friction_coeff, steer_data_distribution=np.array([0.5, 0.5]), lat_acc_data_distribution=np.array([0.5, 0.5])):
    min_factor, max_factor = 1.0 - SANITY_FACTOR, 1.0 + SANITY_FACTOR
    if any(steer_data_distribution < MIN_DATA_PERC) or any(lat_acc_data_distribution < MIN_DATA_PERC):
      return False
    if latAccelFactor is None or latAccelOffset is None or friction_coeff is None:
      return False
    if np.isnan(latAccelFactor) or np.isnan(latAccelOffset) or np.isnan(friction_coeff):
      return False
    return ((max_factor * self.offline_latAccelFactor) >= latAccelFactor >= (min_factor * self.offline_latAccelFactor)) & \
      ((max_factor * self.offline_friction) >= friction_coeff >= (min_factor * self.offline_friction))

  def handle_log(self, t, which, msg):
    if which == "carControl":
      self.raw_points["carControl_t"].append(t)
      self.raw_points["steer_torque"].append(-msg.actuatorsOutput.steer)
      self.raw_points["active"].append(msg.latActive)
    elif which == "carState":
      self.raw_points["carState_t"].append(t)
      self.raw_points["vego"].append(msg.vEgo)
      self.raw_points["steer_override"].append(msg.steeringPressed)
    elif which == "liveLocationKalman":
      if len(self.raw_points['steer_torque']) == self.hist_len:
        yaw_rate = msg.angularVelocityCalibrated.value[2]
        roll = msg.orientationNED.value[0]
        active = np.interp(np.arange(t - MIN_ENGAGE_BUFFER, t, DT_MDL), np.array(self.raw_points['carControl_t']) + self.lag, self.raw_points['active']).astype(bool)
        steer_override = np.interp(np.arange(t - MIN_ENGAGE_BUFFER, t, DT_MDL), np.array(self.raw_points['carState_t']) + self.lag, self.raw_points['steer_override']).astype(bool)
        vego = np.interp(t, np.array(self.raw_points['carState_t']) + self.lag, self.raw_points['vego'])
        steer = np.interp(t, np.array(self.raw_points['carControl_t']) + self.lag, self.raw_points['steer_torque'])
        lateral_acc = (vego * yaw_rate) - (np.sin(roll) * ACCELERATION_DUE_TO_GRAVITY)
        if all(active) and (not any(steer_override)) and (vego > MIN_VEL) and (abs(steer) > STEER_MIN_THRESHOLD) and (abs(lateral_acc) <= LAT_ACC_THRESHOLD):
          self.filtered_points.add_point(float(steer), float(lateral_acc))

  def get_msg(self, valid=True, with_points=False):
    msg = messaging.new_message('liveTorqueParameters')
    msg.valid = valid
    liveTorqueParameters = msg.liveTorqueParameters
    liveTorqueParameters.tag = TORQUED_TAG

    if self.filtered_points.is_valid():
      try:
        latAccelFactor, latAccelOffset, friction_coeff, steer_data_distribution, lat_acc_data_distribution = self.estimate_params()
      except np.linalg.LinAlgError as e:
        latAccelFactor = latAccelOffset = friction_coeff = None
        cloudlog.exception(f"Error computing live torque params: {e}")

      if self.is_sane(latAccelFactor, latAccelOffset, friction_coeff, steer_data_distribution, lat_acc_data_distribution):
        liveTorqueParameters.liveValid = True
        liveTorqueParameters.latAccelFactorRaw = float(latAccelFactor)
        liveTorqueParameters.latAccelOffsetRaw = float(latAccelOffset)
        liveTorqueParameters.frictionCoefficientRaw = float(friction_coeff)
        self.update_params({'latAccelFactor': latAccelFactor, 'latAccelOffset': latAccelOffset, 'frictionCoefficient': friction_coeff})
      else:
        cloudlog.exception("live torque params are numerically unstable")
        liveTorqueParameters.liveValid = False
        # estimator.reset()
    else:
      liveTorqueParameters.liveValid = False

    if with_points:
      liveTorqueParameters.points = self.filtered_points.get_points()

    liveTorqueParameters.latAccelFactorFiltered = float(self.filtered_params['latAccelFactor'].x)
    liveTorqueParameters.latAccelOffsetFiltered = float(self.filtered_params['latAccelOffset'].x)
    liveTorqueParameters.frictionCoefficientFiltered = float(self.filtered_params['frictionCoefficient'].x)
    liveTorqueParameters.totalBucketPoints = len(self.filtered_points)
    liveTorqueParameters.decay = self.decay
    liveTorqueParameters.maxResets = self.resets
    return msg


def main(sm=None, pm=None):
  config_realtime_process([0, 1, 2, 3], 5)

  if sm is None:
    sm = messaging.SubMaster(['carControl', 'carState', 'liveLocationKalman'], poll=['liveLocationKalman'])

  if pm is None:
    pm = messaging.PubMaster(['liveTorqueParameters'])

  params = Params()
  CP = car.CarParams.from_bytes(params.get("CarParams", block=True))
  torque_params = params.get("LiveTorqueParameters")
  estimator = TorqueEstimator(CP, torque_params)

  def cache_params(sig, torque_estimator):
    signal.signal(sig, signal.SIG_DFL)
    cloudlog.warning("caching torque params")
    msg = torque_estimator.get_msg(with_points=True)
    Params().put("LiveTorqueParameters", msg.to_bytes())
    sys.exit(0)
  signal.signal(signal.SIGINT, lambda sig, frame: cache_params(sig, estimator))

  while True:
    sm.update()
    if sm.all_checks():
      for which in sm.updated.keys():
        if sm.updated[which]:
          t = sm.logMonoTime[which] * 1e-9
          estimator.handle_log(t, which, sm[which])

    # 4Hz driven by liveLocationKalman
    if sm.frame % 5 == 0:
      pm.send('liveTorqueParameters', estimator.get_msg(valid=sm.all_checks()))


if __name__ == "__main__":
  main()
