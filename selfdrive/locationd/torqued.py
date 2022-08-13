#!/usr/bin/env python3
from collections import deque, defaultdict
import numpy as np
import cereal.messaging as messaging
from cereal import car
from common.params import Params
from common.realtime import set_realtime_priority, DT_MDL

HISTORY = 5  # secs
RAW_QUEUE_FIELDS = ['active', 'steer_override', 'steer_torque', 'vego']
POINTS_PER_BUCKET = 500
MIN_VEL = 10  # m/s
FRICTION_FACTOR = 1.5  # ~85% of data coverage


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

  def is_sane(self, slope, intercept, friction_coeff):
    return True

  def update_params(self, slope, intercept, friction_coeff):
    pass

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

      if self.filtered_points.is_valid():
        slope, intercept, friction_coeff = self.estimate_params()
        if self.is_sane(slope, intercept, friction_coeff):
          self.update_params(slope, intercept, friction_coeff)
        else:
          self.reset()


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

    # if sm.updated['cameraOdometry']:
    #   calibrator.handle_v_ego(sm['carState'].vEgo)
    #   new_rpy = calibrator.handle_cam_odom(sm['cameraOdometry'].trans,
    #                                        sm['cameraOdometry'].rot,
    #                                        sm['cameraOdometry'].transStd)

    #   if DEBUG and new_rpy is not None:
    #     print('got new rpy', new_rpy)

    # # 4Hz driven by cameraOdometry
    # if sm.frame % 5 == 0:
    #   calibrator.send_data(pm)


if __name__ == "__main__":
  torque_params_thread()
