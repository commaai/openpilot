#!/usr/bin/env python3
from collections import deque
import json
import numpy as np
import cereal.messaging as messaging
from cereal import car
from common.params import Params, put_nonblocking
from common.realtime import set_realtime_priority, DT_MDL
from common.filter_simple import FirstOrderFilter
from system.swaglog import cloudlog
from selfdrive.controls.lib.vehicle_model import ACCELERATION_DUE_TO_GRAVITY


HISTORY = 5  # secs
RAW_QUEUE_FIELDS = ['active', 'steer_override', 'steer_torque', 'vego']
POINTS_PER_BUCKET = 300
MIN_POINTS_PER_BUCKET = 100
MIN_POINTS_TOTAL = 500
MIN_VEL = 10  # m/s
FRICTION_FACTOR = 1.5  # ~85% of data coverage
SANITY_FACTOR = 0.5
STEER_MIN_THRESHOLD = 0.05
FILTER_RC = 50

def slope2rot(slope):
  sin = np.sqrt(slope**2 / (slope**2 + 1))
  cos = np.sqrt(1 / (slope**2 + 1))
  return np.array([[cos, -sin], [sin, cos]])


class PointBuckets:
  def __init__(self, x_bounds):
    self.x_bounds = x_bounds
    self.buckets = {bounds: deque(maxlen=POINTS_PER_BUCKET) for bounds in x_bounds}

  def __len__(self):
    return sum([len(v) for v in self.buckets.values()])

  def is_valid(self):
    return np.all(np.array([len(v) for v in self.buckets.values()]) >= MIN_POINTS_PER_BUCKET) and (self.__len__() >= MIN_POINTS_TOTAL)

  def add_point(self, x, y):
    for bound_min, bound_max in self.x_bounds:
      if (x >= bound_min) and (x < bound_max):
        self.buckets[(bound_min, bound_max)].append([x, y])
        break

  def get_points(self):
    return np.array([v for sublist in self.buckets.values() for v in list(sublist)])


class TorqueEstimator:
  def __init__(self, CP, params):
    self.hist_len = int(HISTORY / DT_MDL)
    self.lag = CP.steerActuatorDelay + .2   # from controlsd

    self.offline_friction_coeff = 0
    self.offline_slope = 0
    if CP.lateralTuning.which() == 'torque':
      self.offline_friction_coeff = CP.lateralTuning.torque.friction
      self.offline_slope = CP.lateralTuning.torque.slope

    params = json.loads(params) if params is not None else None
    if params is not None and self.car_sane(params, CP.carFingerprint):
      try:
        if not self.is_sane(
          params.get('slope'),
          params.get('offset'),
          params.get('frictionCoefficient')
        ):
          params = None
      except Exception:
        # print(e)
        params = None

    if params is None:
      params = {
        'slope': self.offline_slope,
        'offset': 0.0,
        'frictionCoefficient': self.offline_friction_coeff
      }
    self.slopeFiltered = FirstOrderFilter(params['slope'], FILTER_RC, DT_MDL)
    self.offsetFiltered = FirstOrderFilter(params['offset'], FILTER_RC, DT_MDL)
    self.frictionCoefficientFiltered = FirstOrderFilter(params['frictionCoefficient'], FILTER_RC, DT_MDL)

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
    slope, offset = -v.T[0:2, 2] / v.T[2, 2]
    _, spread = np.einsum("ik,kj -> ji", np.column_stack((points[:, 0], points[:, 2] - offset)), slope2rot(slope))
    friction_coeff = np.std(spread) * FRICTION_FACTOR
    return slope, offset, friction_coeff

  def car_sane(self, params, fingerprint):
    return False if params.get('carFingerprint', None) != fingerprint else True

  def is_sane(self, slope, offset, friction_coeff):
    min_factor, max_factor = 1.0 - SANITY_FACTOR, 1.0 + SANITY_FACTOR
    if slope is None or offset is None or friction_coeff is None:
      return False
    if np.isnan(slope) or np.isnan(offset) or np.isnan(friction_coeff):
      return False
    return ((max_factor * self.offline_slope) >= slope >= (min_factor * self.offline_slope)) & \
      ((max_factor * self.offline_friction_coeff) >= friction_coeff >= (min_factor * self.offline_friction_coeff))

  def handle_log(self, t, which, msg):
    if which == "carControl":
      self.raw_points_t["steer_torque"].append(t)
      self.raw_points["steer_torque"].append(-msg.actuatorsOutput.steer)
      self.raw_points_t["active"].append(t)
      self.raw_points["active"].append(msg.latActive)
    elif which == "carState":
      self.raw_points_t["vego"].append(t)
      self.raw_points["vego"].append(msg.vEgo)
      self.raw_points_t["steer_override"].append(t)
      self.raw_points["steer_override"].append(msg.steeringPressed)
    elif which == "liveLocationKalman":
      if len(self.raw_points['steer_torque']) == self.hist_len:
        yaw_rate = msg.angularVelocityCalibrated.value[2]
        roll = msg.orientationNED.value[0]
        active = bool(np.interp(t, np.array(self.raw_points_t['active']) + self.lag, self.raw_points['active']))
        steer_override = bool(np.interp(t, np.array(self.raw_points_t['steer_override']) + self.lag, self.raw_points['steer_override']))
        vego = np.interp(t, np.array(self.raw_points_t['vego']) + self.lag, self.raw_points['vego'])
        steer = np.interp(t, np.array(self.raw_points_t['steer_torque']) + self.lag, self.raw_points['steer_torque'])
        if active and (not steer_override) and (vego > MIN_VEL) and (abs(steer) > STEER_MIN_THRESHOLD):
          lateral_acc = (vego * yaw_rate) - (np.sin(roll) * ACCELERATION_DUE_TO_GRAVITY)
          self.filtered_points.add_point(steer, lateral_acc)


def main(sm=None, pm=None):
  set_realtime_priority(1)

  if sm is None:
    sm = messaging.SubMaster(['carControl', 'carState', 'liveLocationKalman'], poll=['liveLocationKalman'])

  if pm is None:
    pm = messaging.PubMaster(['liveTorqueParameters'])

  params_reader = Params()
  CP = car.CarParams.from_bytes(params_reader.get("CarParams", block=True))
  params = params_reader.get("LiveTorqueParameters")
  estimator = TorqueEstimator(CP, params)

  while True:
    sm.update()
    if sm.all_checks():
      for which in sm.updated.keys():
        if sm.updated[which]:
          t = sm.logMonoTime[which] * 1e-9
          estimator.handle_log(t, which, sm[which])

    if sm.updated['liveLocationKalman']:
      # print(sm.frame, [len(v) for v in estimator.filtered_points.buckets.values()])
      msg = messaging.new_message('liveTorqueParameters')
      msg.valid = sm.all_checks()
      liveTorqueParameters = msg.liveTorqueParameters

      if estimator.filtered_points.is_valid():
        try:
          slope, offset, friction_coeff = estimator.estimate_params()
          # print(slope, offset, friction_coeff)
        except Exception as e:
          # print(e)
          slope = offset = friction_coeff = None
          cloudlog.exception(f"Error computing live torque params: {e}")

        if estimator.is_sane(slope, offset, friction_coeff):
          liveTorqueParameters.liveValid = True
          liveTorqueParameters.slopeRaw = slope
          liveTorqueParameters.offsetRaw = offset
          liveTorqueParameters.frictionCoefficientRaw = friction_coeff
          estimator.slopeFiltered.update(slope)
          estimator.offsetFiltered.update(offset)
          estimator.frictionCoefficientFiltered.update(friction_coeff)
        else:
          cloudlog.exception("live torque params are numerically unstable")
          liveTorqueParameters.liveValid = False
          # estimator.reset()
      else:
        liveTorqueParameters.liveValid = False
      liveTorqueParameters.slopeFiltered = estimator.slopeFiltered.x
      liveTorqueParameters.offsetFiltered = estimator.offsetFiltered.x
      liveTorqueParameters.frictionCoefficientFiltered = estimator.frictionCoefficientFiltered.x
      liveTorqueParameters.totalBucketPoints = len(estimator.filtered_points)

      if sm.frame % 1200 == 0:  # once a minute
        params_to_write = {
          "slope": estimator.slopeFiltered.x,
          "offset": estimator.offsetFiltered.x,
          "frictionCoefficient": estimator.frictionCoefficientFiltered.x
        }
        put_nonblocking("LiveTorqueParameters", json.dumps(params_to_write))

      pm.send('liveTorqueParameters', msg)


if __name__ == "__main__":
  main()
