#!/usr/bin/env python3
from collections import deque, defaultdict
import json
import numpy as np
import cereal.messaging as messaging
from cereal import car
from common.params import Params, put_nonblocking
from common.realtime import set_realtime_priority, DT_MDL
from common.filter_simple import FirstOrderFilter

HISTORY = 5  # secs
RAW_QUEUE_FIELDS = ['active', 'steer_override', 'steer_torque', 'vego']
POINTS_PER_BUCKET = 500
MIN_VEL = 10  # m/s
FRICTION_FACTOR = 1.5  # ~85% of data coverage
SANITY_FACTOR = 0.2
CALC_FREQ = 100


def slope2rot(slope):
  sin = np.sqrt(slope**2 / (slope**2 + 1))
  cos = np.sqrt(1 / (slope**2 + 1))
  return np.array([[cos, -sin], [sin, cos]])


class PointBuckets:
  def __init__(self, x_bounds):
    self.x_bounds = x_bounds
    self.buckets = defaultdict(lambda: deque(maxlen=POINTS_PER_BUCKET))

  def __len__(self):
    return sum([len(v) for v in self.buckets.values()])

  def is_valid(self):
    return np.all(np.array([len(v) for v in self.buckets.values()]) == POINTS_PER_BUCKET)

  def add_point(self, x, y):
    for bound_min, bound_max in self.x_bounds:
      if (x >= bound_min) and (x < bound_max):
        self.buckets[(bound_min, bound_max)].append([x, y])
        break

  def get_points(self):
    return np.array([v for sublist in self.buckets.items() for v in sublist])


class TorqueEstimator:
  def __init__(self):
    self.hist_len = int(HISTORY / DT_MDL)
    params_reader = Params()
    CP = car.CarParams.from_bytes(params_reader.get("CarParams", block=True))
    self.lag = CP.steerActuatorDelay + .2   # from controlsd
    self.offline_friction_coeff = CP.lateralTuning.torque.friction
    self.offline_slope = 1.0 / CP.lateralTuning.torque.kp

    params = params_reader.get("LiveTorqueParameters")
    params = json.loads(params) if params is not None else None
    if self.car_sane(params, CP.carFingerprint):
      try:
        if not self.is_sane(
          params.get('slope'),
          params.get('intercept'),
          params.get('frictionCoefficient')
        ):
          params = None
      except Exception as e:
        print(e)
        params = None

    if params is None:
      params = {
        'slope': self.offline_slope,
        'intercept': 0.0,
        'frictionCoefficient': self.offline_friction_coeff
      }
    self.slopeFiltered = FirstOrderFilter(params['slope'], 19.0, 1.0)
    self.interceptFiltered = FirstOrderFilter(params['intercept'], 19.0, 1.0)
    self.frictionCoefficientFiltered = FirstOrderFilter(params['frictionCoefficient'], 19.0, 1.0)

    self.reset()

  def reset(self):
    self.raw_points = {k: deque(maxlen=self.hist_len) for k in RAW_QUEUE_FIELDS}
    self.raw_points_t = {k: deque(maxlen=self.hist_len) for k in RAW_QUEUE_FIELDS}
    self.filtered_points = PointBuckets([(-0.5, -0.25), (-0.25, 0), (0, 0.25), (0.25, 0.5)])

  def estimate_params(self):
    points = self.filtered_points.get_points()
    points = np.insert(points, 1, 1.0, axis=1)
    # total least square solution as both x and y are noisy observations
    # this is emperically the slope of the hysteresis parallelogram as opposed to the line through the diagonals
    _, _, v = np.linalg.svd(points, full_matrices=False)
    slope, intercept = -v.T[0:2, 2] / v.T[2, 2]
    _, spread = np.einsum("ik,kj -> ji", np.column_stack((points[:, 0], points[:, 2] - intercept)), slope2rot(slope))
    friction_coeff = np.std(spread) * FRICTION_FACTOR
    return slope, intercept, friction_coeff

  def car_sane(self, params, fingerprint):
    return False if params.get('carFingerprint', None) != fingerprint else True

  def is_sane(self, slope, intercept, friction_coeff):
    return (slope > (1.0 - SANITY_FACTOR) * self.offline_slope) & \
      (slope < (1.0 + SANITY_FACTOR) * self.offline_slope) & \
      (friction_coeff > (1.0 - SANITY_FACTOR) * self.offline_friction_coeff) & \
      (friction_coeff < (1.0 + SANITY_FACTOR) * self.offline_friction_coeff) & \
      (intercept > (1.0 - SANITY_FACTOR) * self.offline_friction_coeff) & \
      (intercept < (1.0 + SANITY_FACTOR) * self.offline_friction_coeff)

  def handle_log(self, t, which, msg):
    if which == "carControl":
      self.raw_points_t["steer_torque"].append(t)
      self.raw_points["steer_torque"].append(msg.actuatorsOutput.steer)
    elif which == "carState":
      self.raw_points_t["vego"].append(t)
      self.raw_points["vego"].append(msg.vEgo)
      self.raw_points_t["steer_override"].append(t)
      self.raw_points["steer_override"].append(msg.steeringPressed)
    elif which == "controlsState":
      self.raw_points_t["active"].append(t)
      self.raw_points["active"].append(msg.active)
    elif which == "liveLocationKalman":
      if len(self.raw_points['steer_torque']) == self.hist_len:
        yaw_rate = msg.msg.angularVelocityCalibrated.value[2]
        active = bool(np.interp(t, np.array(self.raw_points_t['active']) + self.lag, self.raw_points['active']))
        steer_override = bool(np.interp(t, np.array(self.raw_points_t['steer_override']) + self.lag, self.raw_points['steer_override']))
        vego = np.interp(t, np.array(self.raw_points_t['vego']) + self.lag, self.raw_points['vego'])
        if active and (not steer_override) and (vego > MIN_VEL):
          steer = np.interp(t, np.array(self.raw_points_t['steer_torque']) + self.lag, self.raw_points['steer_torque'])
          lateral_acc = vego * yaw_rate
          self.filtered_points.add_point(steer, lateral_acc)


def torque_params_thread(sm=None, pm=None):
  set_realtime_priority(1)

  if sm is None:
    sm = messaging.SubMaster(['controlsState', 'carState', 'liveLocationKalman'], poll=['liveLocationKalman'])

  if pm is None:
    pm = messaging.PubMaster(['liveTorqueParameters'])

  estimator = TorqueEstimator()

  while True:
    sm.update()
    if sm.all_checks():
      for which in sm.updated.keys():
        if sm.updated[which]:
          t = sm.logMonoTime[which] * 1e-9
          estimator.handle_log(t, which, sm[which])

    if sm.updated['liveLocationKalman']:
      msg = messaging.new_message('liveTorqueParameters')
      msg.valid = sm.all_checks()
      liveTorqueParameters = msg.liveTorqueParameters

      if estimator.filtered_points.is_valid():
        slope, intercept, friction_coeff = estimator.estimate_params()
        if estimator.is_sane(slope, intercept, friction_coeff):
          liveTorqueParameters.liveValid = True
          liveTorqueParameters.slopeRaw = slope
          liveTorqueParameters.interceptRaw = intercept
          liveTorqueParameters.frictionCoefficientRaw = friction_coeff
          estimator.slopeFiltered.update(slope)
          estimator.interceptFiltered.update(intercept)
          estimator.frictionCoefficientFiltered.update(friction_coeff)
        else:
          liveTorqueParameters.liveValid = False
          estimator.reset()
      else:
        liveTorqueParameters.liveValid = False
      liveTorqueParameters.slopeFiltered = estimator.slopeFiltered.x
      liveTorqueParameters.interceptFiltered = estimator.interceptFiltered.x
      liveTorqueParameters.frictionCoefficientFiltered = estimator.frictionCoefficientFiltered.x
      liveTorqueParameters.totalBucketPoints = len(estimator.filtered_points)

      if sm.frame % 1200 == 0:  # once a minute
        params_to_write = {
          "slope": estimator.slopeFiltered.x,
          "intercept": estimator.interceptFiltered.x,
          "frictionCoefficient": estimator.frictionCoefficientFiltered.x
        }
        put_nonblocking("LiveTorqueParameters", json.dumps(params_to_write))

      pm.send('liveTorqueParameters', msg)


if __name__ == "__main__":
  torque_params_thread()
