#!/usr/bin/env python3
import os
import sys
import signal
import numpy as np
from collections import deque, defaultdict
# from itertools import product

import cereal.messaging as messaging
from cereal import car, log
from openpilot.common.params import Params
from openpilot.common.realtime import config_realtime_process, DT_MDL
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.locationd.helpers import PointBuckets
from openpilot.system.swaglog import cloudlog

from selfdrive.car.interfaces import ACCEL_MAX, ACCEL_MIN

HISTORY = 5  # secs
POINTS_PER_BUCKET = 1500
MIN_POINTS_TOTAL = 4000
MIN_POINTS_TOTAL_QLOG = 600
FIT_POINTS_TOTAL = 2500
FIT_POINTS_TOTAL_QLOG = 200
MIN_VEL = 0.5  # m/s
FACTOR_SANITY = 0.3
VEGO_MAX = 40.0  # m/s
FACTOR_SANITY_BRAKE = 0.1
ACCEL_MIN_THRESHOLD = 0.02
MIN_FILTER_DECAY = 50
MAX_FILTER_DECAY = 250
# ACCEL_BUCKET_BOUNDS = [(-4.0, -1.5), (-1.5, -0.5), (-0.5, -0.2), (-0.2, 0.2), (0.2, 0.5), (0.5, 1.0), (1.0, 1.5)]
# V_EGO_BUCKET_BOUNDS = [(0, 2), (2, 12), (12, 20), (20, 40)]
# PITCH_BUCKET_BOUNDS = [(-0.1, 0.1)]  # TODO: Ensure mean is really 0
# ALL_BUCKET_BOUNDS = list(product(ACCEL_BUCKET_BOUNDS, V_EGO_BUCKET_BOUNDS, PITCH_BUCKET_BOUNDS))
BUCKETS = {
  # a_ego        v_ego   pitch           joint gas/brake
  ((-4.0, -1.5), (0, 2), (-0.1, 0.1)): 10,
  ((-4.0, -1.5), (2, 12), (-0.1, 0.1)): 50,
  ((-4.0, -1.5), (12, 20), (-0.1, 0.1)): 10,
  ((-4.0, -1.5), (20, 40), (-0.1, 0.1)): 10,
  ((-1.5, -0.5), (0, 2), (-0.1, 0.1)): 0,
  ((-1.5, -0.5), (2, 12), (-0.1, 0.1)): 50,
  ((-1.5, -0.5), (12, 20), (-0.1, 0.1)): 50,
  ((-1.5, -0.5), (20, 40), (-0.1, 0.1)): 0,
  ((-0.5, -0.2), (0, 2), (-0.1, 0.1)): 0,
  ((-0.5, -0.2), (2, 12), (-0.1, 0.1)): 20,
  ((-0.5, -0.2), (12, 20), (-0.1, 0.1)): 20,
  ((-0.5, -0.2), (20, 40), (-0.1, 0.1)): 20,
  ((-0.2, 0.2), (0, 2), (-0.1, 0.1)): 500,
  ((-0.2, 0.2), (2, 12), (-0.1, 0.1)): 500,
  ((-0.2, 0.2), (12, 20), (-0.1, 0.1)): 500,
  ((-0.2, 0.2), (20, 40), (-0.1, 0.1)): 500,
  ((0.2, 0.5), (0, 2), (-0.1, 0.1)): 0,
  ((0.2, 0.5), (2, 12), (-0.1, 0.1)): 10,
  ((0.2, 0.5), (12, 20), (-0.1, 0.1)): 10,
  ((0.2, 0.5), (20, 40), (-0.1, 0.1)): 0,
  ((0.5, 1.0), (0, 2), (-0.1, 0.1)): 0,
  ((0.5, 1.0), (2, 12), (-0.1, 0.1)): 5,
  ((0.5, 1.0), (12, 20), (-0.1, 0.1)): 0,
  ((0.5, 1.0), (20, 40), (-0.1, 0.1)): 0,
  ((1.0, 1.5), (0, 2), (-0.1, 0.1)): 0,
  ((1.0, 1.5), (2, 12), (-0.1, 0.1)): 0,
  ((1.0, 1.5), (12, 20), (-0.1, 0.1)): 0,
  ((1.0, 1.5), (20, 40), (-0.1, 0.1)): 0,
}
ALL_BUCKET_BOUNDS = list(BUCKETS.keys())
MIN_BUCKET_POINTS = list(BUCKETS.values())

MIN_ENGAGE_BUFFER = 1  # secs

VERSION = 1  # bump this to invalidate old parameter caches


class GasBuckets(PointBuckets):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.x_size = len(self.x_bounds[0])

  def is_valid(self):
    return all(
      len(v) >= min_pts for v, min_pts in zip(self.buckets.values(), self.buckets_min_points.values(), strict=True)) \
      and (self.__len__() >= self.min_points_total)

  @staticmethod
  def is_jointly_valid(*pbs: 'GasBuckets'):
    # TODO: this is kind of a hack
    x_bounds = pbs[0].x_bounds
    if not all(pb.x_bounds == x_bounds for pb in pbs):
      raise ValueError("x_bounds must be the same for all PointBuckets")
    return all(
      sum(len(pb.buckets[bounds]) for pb in pbs) >= min_pts
      for bounds, min_pts in zip(x_bounds, pbs[0].buckets_min_points.values(), strict=True)) \
      and (sum(len(pb) for pb in pbs) >= pbs[0].min_points_total)

  def add_point(self, x, y):
    for bounds in self.x_bounds:
      if all((x[i] >= bounds[i][0]) and (x[i] < bounds[i][1]) for i in range(self.x_size)):
        self.buckets[bounds].append([*x, 1.0, y])
        break


class GasBrakeEstimator:
  def __init__(self, CP, decimated=False):
    self.gas_command_offset = CP.gasCommandOffset
    self.hist_len = int(HISTORY / DT_MDL)
    self.lag = CP.longitudinalActuatorDelayLowerBound + .2   # from controlsd
    if decimated:
      self.min_points_total = MIN_POINTS_TOTAL_QLOG
      self.fit_points = FIT_POINTS_TOTAL_QLOG

    else:
      self.min_points_total = MIN_POINTS_TOTAL
      self.fit_points = FIT_POINTS_TOTAL

    self.offline_gasFactor = CP.longitudinalTuning.gasFactor
    self.offline_brakeFactor = CP.longitudinalTuning.brakeFactor

    self.resets = 0.0
    self.use_params = True

    self.reset()

    initial_params = {
      'gasFactor': self.offline_gasFactor,
      'brakeFactor': self.offline_brakeFactor,
      'points': []
    }
    self.decay = MIN_FILTER_DECAY

    # try to restore cached params
    params = Params()
    params_cache = params.get("LiveGasCarParams")
    gas_cache = params.get("LiveGasParameters")
    if params_cache is not None and gas_cache is not None:
      try:
        with log.Event.from_bytes(gas_cache) as log_evt:
          cache_lgp = log_evt.liveGasParameters
        with car.CarParams.from_bytes(params_cache) as msg:
          cache_CP = msg
        if self.get_restore_key(cache_CP, cache_lgp.version) == self.get_restore_key(CP, VERSION):
          if cache_lgp.liveValid:
            initial_params = {
              'gasFactor': cache_lgp.gasFactor,
              'brakeFactor': cache_lgp.brakeFactor,
            }
          self.decay = cache_lgp.decay
          # TODO: how to simplify this? Slicing doesn't seem to work
          #  type is <class 'capnp.lib.capnp._DynamicListReader'>
          self.filtered_gas.load_points([([p[0], p[1], p[2]], p[3]) for p in cache_lgp.gasPoints])
          self.filtered_brake.load_points([([p[0], p[1], p[2]], p[-1]) for p in cache_lgp.brakePoints])
          cloudlog.info("restored gas params from cache")
      except Exception:
        cloudlog.exception("failed to restore cached gas params")
        params.remove("LiveGasCarParams")
        params.remove("LiveGasParameters")

    self.filtered_params = {}
    for param in initial_params:
      self.filtered_params[param] = FirstOrderFilter(initial_params[param], self.decay, DT_MDL)

  def get_restore_key(self, CP, version):
    return (CP.carFingerprint, version)

  def reset(self):
    self.resets += 1.0
    self.decay = MIN_FILTER_DECAY
    self.raw_points = defaultdict(lambda: deque(maxlen=self.hist_len))
    self.filtered_gas = GasBuckets(x_bounds=ALL_BUCKET_BOUNDS, min_points_total=self.min_points_total,
                                     min_points=MIN_BUCKET_POINTS, points_per_bucket=POINTS_PER_BUCKET)
    self.filtered_brake = GasBuckets(x_bounds=ALL_BUCKET_BOUNDS, min_points_total=self.min_points_total,
                                       min_points=MIN_BUCKET_POINTS, points_per_bucket=POINTS_PER_BUCKET)

  def sanity_check(self):
    cases = [
      # (accel, vego, pitch)
      (0.0, 0.0, 0.0),
      (0.0, VEGO_MAX, 0.0),
      (ACCEL_MAX, 1.0, 0.0),
      (ACCEL_MAX, VEGO_MAX, 0.0),
      (ACCEL_MIN, 1.0, 0.0),
      (ACCEL_MIN, VEGO_MAX, 0.0),
      (-1.0, 10.0, 0.0),
      (-1.0, 10.0, 0.25),
      (0.0, 10.0, 0.25),
      (0.0, 10.0, -0.25),
    ]
    for accel, vego, pitch in cases:
      x = [accel, vego, pitch, 1]
      offline_gas = np.dot(self.offline_gasFactor, x)
      offline_brake = np.dot(self.offline_brakeFactor, x)

      live_gas = np.dot(self.filtered_params['gasFactor'].x, x)
      live_brake = np.dot(self.filtered_params['brakeFactor'].x, x)

      if (abs(live_gas - offline_gas) > FACTOR_SANITY * offline_gas
          or abs(live_brake - offline_brake) > FACTOR_SANITY_BRAKE * offline_brake):
        return False
    return True

  def estimate_params(self) -> np.ndarray:
    # TODO: can we cat these together for a single solve?
    A_gas = self.filtered_gas.get_points(self.fit_points)[:, :-1]
    y_gas = self.filtered_gas.get_points(self.fit_points)[:, -1]
    A_brake = self.filtered_brake.get_points(self.fit_points)[:, :-1]
    y_brake = self.filtered_brake.get_points(self.fit_points)[:, -1]
    try:
      x_gas = np.linalg.lstsq(A_gas, y_gas, rcond=None)[0]
      x_brake = np.linalg.lstsq(A_brake, y_brake, rcond=None)[0]
      x = np.vstack([x_gas, x_brake])
    except np.linalg.LinAlgError as e:
      cloudlog.exception(f"Error computing live torque params: {e}")
      x = np.nan * np.ones((2, A_gas.shape[1]))
    return x

  def update_params(self, params):
    self.decay = min(self.decay + DT_MDL, MAX_FILTER_DECAY)
    for param, value in params.items():
      self.filtered_params[param].update(value)
      self.filtered_params[param].update_alpha(self.decay)

  def handle_log(self, t, which, msg):
    if which == "carControl":
      self.raw_points["carControl_t"].append(t + self.lag)
      self.raw_points["actuator_accel"].append(msg.actuators.accel)
      self.raw_points["gas"].append(msg.actuatorsOutput.gas)
      self.raw_points["brake"].append(msg.actuatorsOutput.brake)
      self.raw_points["active"].append(msg.latActive)
    elif which == "carState":
      self.raw_points["carState_t"].append(t + self.lag)
      self.raw_points["vego"].append(msg.vEgo)  # consider using velocityCalibrated from liveLocationKalman?
      self.raw_points["aego"].append(msg.aEgo)  # consider using accelerationCalibrated from liveLocationKalman?
      self.raw_points["gas_override"].append(msg.gasPressed)
    elif which == "liveLocationKalman":
      if len(self.raw_points['gas']) == self.hist_len:
        pitch = msg.calibratedOrientationNED.value[1]
        active = np.interp(np.arange(t - MIN_ENGAGE_BUFFER, t, DT_MDL), self.raw_points['carControl_t'], self.raw_points['active']).astype(bool)
        gas_override = np.interp(np.arange(t - MIN_ENGAGE_BUFFER, t, DT_MDL), self.raw_points['carState_t'], self.raw_points['gas_override']).astype(bool)
        vego = np.interp(t, self.raw_points['carState_t'], self.raw_points['vego'])
        aego = np.interp(t, self.raw_points['carState_t'], self.raw_points['aego'])
        gas = np.interp(t, self.raw_points['carControl_t'], self.raw_points['gas'])
        brake = np.interp(t, self.raw_points['carControl_t'], self.raw_points['brake'])
        if all(active) and (not any(gas_override)) and (vego > MIN_VEL) and (abs(aego) > ACCEL_MIN_THRESHOLD):
          x = np.array([aego, vego, pitch])
          if brake > 0:
            self.filtered_brake.add_point(x, brake)
          else:
            self.filtered_gas.add_point(x, gas)

  def get_msg(self, valid=True, with_points=False):
    msg = messaging.new_message('liveGasParameters')
    msg.valid = valid
    liveGasParameters = msg.liveGasParameters
    liveGasParameters.version = VERSION
    liveGasParameters.useParams = self.use_params

    if GasBuckets.is_jointly_valid(self.filtered_gas, self.filtered_brake):
      x = self.estimate_params()

      if any(val is None or np.isnan(val) for val in x.flatten()) or not self.sanity_check():
        cloudlog.exception("Live gas parameters are invalid.")
        liveGasParameters.liveValid = False
        self.reset()
      else:
        liveGasParameters.liveValid = True
        self.update_params({
          'gasFactor': [float(x[0, i]) for i in range(x.shape[1])],
          'brakeFactor': [float(x[1, i]) for i in range(x.shape[1])],
        })
    else:
      liveGasParameters.liveValid = False

    if with_points:
      liveGasParameters.gasPoints = self.filtered_gas.get_points()[:, [0, 1, 2, 4]].tolist()
      liveGasParameters.brakePoints = self.filtered_brake.get_points()[:, [0, 1, 2, 4]].tolist()

    liveGasParameters.gasFactor = self.filtered_params['gasFactor'].x
    liveGasParameters.brakeFactor = self.filtered_params['brakeFactor'].x
    liveGasParameters.totalBucketPoints = len(self.filtered_gas) + len(self.filtered_brake)
    liveGasParameters.decay = self.decay
    liveGasParameters.maxResets = self.resets
    return msg

def main(sm=None, pm=None):
  config_realtime_process([0, 1, 2, 3], 5)

  if sm is None:
    sm = messaging.SubMaster(['carControl', 'carState', 'liveLocationKalman'], poll=['liveLocationKalman'])

  if pm is None:
    pm = messaging.PubMaster(['liveGasParameters'])

  params = Params()
  with car.CarParams.from_bytes(params.get("CarParams", block=True)) as CP:
    estimator = GasBrakeEstimator(CP)

  def cache_params(sig, frame):
    signal.signal(sig, signal.SIG_DFL)
    cloudlog.warning("caching gas params")

    params = Params()
    params.put("LiveGasCarParams", CP.as_builder().to_bytes())

    msg = estimator.get_msg(with_points=True)
    params.put("LiveGasParameters", msg.to_bytes())

    sys.exit(0)

  if "REPLAY" not in os.environ:
    signal.signal(signal.SIGINT, cache_params)

  while True:
    sm.update()
    if sm.all_checks():
      for which in sm.updated.keys():
        if sm.updated[which]:
          t = sm.logMonoTime[which] * 1e-9
          estimator.handle_log(t, which, sm[which])

    # 4Hz driven by liveLocationKalman
    if sm.frame % 5 == 0:
      pm.send('liveGasParameters', estimator.get_msg(valid=sm.all_checks()))


if __name__ == "__main__":
  main()
