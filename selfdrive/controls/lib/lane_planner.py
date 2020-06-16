from common.numpy_fast import interp
from common.realtime import DT_MDL
from common.transformations.orientation import rot_from_euler

import numpy as np
from cereal import log

CAMERA_OFFSET = 0.06  # m from center car to camera
PLOT = False

if PLOT:
  import matplotlib.pyplot as plt


def compute_path_pinv(l=50):
  deg = 3
  x = np.arange(l*1.0)
  X = np.vstack(tuple(x**n for n in range(deg, -1, -1))).T
  pinv = np.linalg.pinv(X)
  return pinv


def model_polyfit(points, path_pinv):
  return np.dot(path_pinv, [float(x) for x in points])


def eval_poly(poly, x):
  return poly[3] + poly[2]*x + poly[1]*x**2 + poly[0]*x**3


def calc_d_poly(l_poly, r_poly, p_poly, l_prob, r_prob, lane_width, v_ego):
  # This will improve behaviour when lanes suddenly widen
  # these numbers were tested on 2000segments and found to work well
  lane_width = min(4.0, lane_width)
  width_poly = l_poly - r_poly
  prob_mods = []
  for t_check in [0.0, 1.5, 3.0]:
    width_at_t = eval_poly(width_poly, t_check * (v_ego + 7))
    prob_mods.append(interp(width_at_t, [4.0, 5.0], [1.0, 0.0]))
  mod = min(prob_mods)
  l_prob = mod * l_prob
  r_prob = mod * r_prob

  path_from_left_lane = l_poly.copy()
  path_from_left_lane[3] -= lane_width / 2.0
  path_from_right_lane = r_poly.copy()
  path_from_right_lane[3] += lane_width / 2.0

  lr_prob = l_prob + r_prob - l_prob * r_prob

  d_poly_lane = (l_prob * path_from_left_lane + r_prob * path_from_right_lane) / (l_prob + r_prob + 0.0001)
  return lr_prob * d_poly_lane + (1.0 - lr_prob) * p_poly


class LanePlanner():
  def __init__(self):
    self.l_poly = np.array([0., 0., 0., 0.])
    self.r_poly = np.array([0., 0., 0., 0.])
    self.p_poly = np.array([0., 0., 0., 0.])
    self.d_poly = np.array([0., 0., 0., 0.])

    self.lane_width_estimate = 3.7
    self.lane_width_certainty = 1.0
    self.lane_width = 3.7

    self.l_prob = 0.
    self.r_prob = 0.

    self.l_lane_change_prob = 0.
    self.r_lane_change_prob = 0.

    self._path_pinv = compute_path_pinv()
    self.x_points = np.arange(50)
    self.cnt = 0

    self.ll_time = None

    if PLOT:
      plt.ion()
      self.fig = plt.figure(figsize=(15, 20))
      self.ax = self.fig.add_subplot(111)
      # self.ax.set_xlim([-150, 150])
      self.ax.set_xlim([-5, 5])
      self.ax.set_ylim([-50., 200.])
      self.ax.set_xlabel('x [m]')
      self.ax.set_ylabel('y [m]')
      self.ax.grid(True)

      self.points_l_x = np.arange(192)
      self.points_l_y = 0 * self.points_l_x
      self.line_l, = self.ax.plot(-self.points_l_y, self.points_l_x, 'C0.', markersize=1)
      self.line_r, = self.ax.plot(-self.points_l_y, self.points_l_x, 'C1.', markersize=1)

      self.line_l_poly, = self.ax.plot(-self.points_l_y, self.points_l_x, 'C0--')
      self.line_r_poly, = self.ax.plot(-self.points_l_y, self.points_l_x, 'C1--')
      plt.show()

  def parse_model(self, md, ll):
    if len(ll.angularVelocityCalibrated.value) == 0:
      return

    if len(md.meta.desireState):
      self.l_lane_change_prob = md.meta.desireState[log.PathPlan.Desire.laneChangeLeft - 1]
      self.r_lane_change_prob = md.meta.desireState[log.PathPlan.Desire.laneChangeRight - 1]

    orient = np.array(ll.calibratedOrientationECEF.value)
    ecef = np.atleast_2d(np.array(ll.positionECEF.value)).T
    ecef_from_local = rot_from_euler(orient).T

    if self.cnt % int(2.0 / DT_MDL) == 0:
      self.l_prob = 1
      self.r_prob = 1

      self.points_l_x = np.arange(192, dtype=np.float32)
      self.points_l_y = np.polyval(md.leftLane.poly, self.points_l_x)

      self.points_r_x = np.arange(192, dtype=np.float32)
      self.points_r_y = np.polyval(md.rightLane.poly, self.points_r_x)

      z = np.zeros_like(self.points_l_x)
      points_l = np.vstack([self.points_l_x, -self.points_l_y, z])
      self.points_l_ecef = ecef_from_local.dot(points_l) + ecef

      points_r = np.vstack([self.points_r_x, -self.points_r_y, z])
      self.points_r_ecef = ecef_from_local.dot(points_r) + ecef

    points_l = ecef_from_local.T.dot(self.points_l_ecef - ecef)
    self.points_l_x = points_l[0, :]
    self.points_l_y = -points_l[1, :]

    points_r = ecef_from_local.T.dot(self.points_r_ecef - ecef)
    self.points_r_x = points_r[0, :]
    self.points_r_y = -points_r[1, :]

    self.l_poly = np.polyfit(self.points_l_x, self.points_l_y, 3)
    self.r_poly = np.polyfit(self.points_r_x, self.points_r_y, 3)

    if PLOT:
      self.line_l.set_xdata(-self.points_l_y)
      self.line_l.set_ydata(self.points_l_x)

      self.line_r.set_xdata(-self.points_r_y)
      self.line_r.set_ydata(self.points_r_x)

      x = np.arange(192)
      self.line_l_poly.set_xdata(-np.polyval(self.l_poly, x))
      self.line_r_poly.set_xdata(-np.polyval(self.r_poly, x))

      self.fig.canvas.draw()
      self.fig.canvas.flush_events()

    self.cnt += 1

  def update_d_poly(self, v_ego):
    # only offset left and right lane lines; offsetting p_poly does not make sense
    self.l_poly[3] += CAMERA_OFFSET
    self.r_poly[3] += CAMERA_OFFSET

    # Find current lanewidth
    self.lane_width_certainty += 0.05 * (self.l_prob * self.r_prob - self.lane_width_certainty)
    current_lane_width = abs(self.l_poly[3] - self.r_poly[3])
    self.lane_width_estimate += 0.005 * (current_lane_width - self.lane_width_estimate)
    speed_lane_width = interp(v_ego, [0., 31.], [2.8, 3.5])
    self.lane_width = self.lane_width_certainty * self.lane_width_estimate + \
                      (1 - self.lane_width_certainty) * speed_lane_width

    self.d_poly = (self.l_poly + self.r_poly) / 2
