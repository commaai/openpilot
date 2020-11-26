from common.numpy_fast import interp
import numpy as np
from cereal import log

CAMERA_OFFSET = 0.06  # m from center car to camera


def compute_path_pinv(length=50):
  deg = 3
  x = np.arange(length*1.0)
  X = np.vstack(tuple(x**n for n in range(deg, -1, -1))).T
  pinv = np.linalg.pinv(X)
  return pinv


def model_polyfit(points, path_pinv):
  return np.dot(path_pinv, [float(x) for x in points])


def eval_poly(poly, x):
  return poly[3] + poly[2]*x + poly[1]*x**2 + poly[0]*x**3


class LanePlanner:
  def __init__(self):
    self.l_poly = [0., 0., 0., 0.]
    self.r_poly = [0., 0., 0., 0.]
    self.p_poly = [0., 0., 0., 0.]
    self.d_poly = [0., 0., 0., 0.]

    self.lane_width_estimate = 3.7
    self.lane_width_certainty = 1.0
    self.lane_width = 3.7

    self.l_prob = 0.
    self.r_prob = 0.

    self.l_std = 0.
    self.r_std = 0.

    self.l_lane_change_prob = 0.
    self.r_lane_change_prob = 0.

    self._path_pinv = compute_path_pinv()
    self.x_points = np.arange(50)

  def parse_model(self, md):
    if len(md.leftLane.poly):
      self.l_poly = np.array(md.leftLane.poly)
      self.l_std = float(md.leftLane.std)
      self.r_poly = np.array(md.rightLane.poly)
      self.r_std = float(md.rightLane.std)
      self.p_poly = np.array(md.path.poly)
    else:
      self.l_poly = model_polyfit(md.leftLane.points, self._path_pinv)  # left line
      self.r_poly = model_polyfit(md.rightLane.points, self._path_pinv)  # right line
      self.p_poly = model_polyfit(md.path.points, self._path_pinv)  # predicted path
    self.l_prob = md.leftLane.prob  # left line prob
    self.r_prob = md.rightLane.prob  # right line prob

    if len(md.meta.desireState):
      self.l_lane_change_prob = md.meta.desireState[log.PathPlan.Desire.laneChangeLeft]
      self.r_lane_change_prob = md.meta.desireState[log.PathPlan.Desire.laneChangeRight]

  def update_d_poly(self, v_ego):
    # only offset left and right lane lines; offsetting p_poly does not make sense
    self.l_poly[3] += CAMERA_OFFSET
    self.r_poly[3] += CAMERA_OFFSET

    # Reduce reliance on lanelines that are too far apart or
    # will be in a few seconds
    l_prob, r_prob = self.l_prob, self.r_prob
    width_poly = self.l_poly - self.r_poly
    prob_mods = []
    for t_check in [0.0, 1.5, 3.0]:
      width_at_t = eval_poly(width_poly, t_check * (v_ego + 7))
      prob_mods.append(interp(width_at_t, [4.0, 5.0], [1.0, 0.0]))
    mod = min(prob_mods)
    l_prob *= mod
    r_prob *= mod

    # Reduce reliance on uncertain lanelines
    l_std_mod = interp(self.l_std, [.15, .3], [1.0, 0.0])
    r_std_mod = interp(self.r_std, [.15, .3], [1.0, 0.0])
    l_prob *= l_std_mod
    r_prob *= r_std_mod

    # Find current lanewidth
    self.lane_width_certainty += 0.05 * (l_prob * r_prob - self.lane_width_certainty)
    current_lane_width = abs(self.l_poly[3] - self.r_poly[3])
    self.lane_width_estimate += 0.005 * (current_lane_width - self.lane_width_estimate)
    speed_lane_width = interp(v_ego, [0., 31.], [2.8, 3.5])
    self.lane_width = self.lane_width_certainty * self.lane_width_estimate + \
                      (1 - self.lane_width_certainty) * speed_lane_width

    clipped_lane_width = min(4.0, self.lane_width)
    path_from_left_lane = self.l_poly.copy()
    path_from_left_lane[3] -= clipped_lane_width / 2.0
    path_from_right_lane = self.r_poly.copy()
    path_from_right_lane[3] += clipped_lane_width / 2.0

    lr_prob = l_prob + r_prob - l_prob * r_prob

    d_poly_lane = (l_prob * path_from_left_lane + r_prob * path_from_right_lane) / (l_prob + r_prob + 0.0001)
    self.d_poly = lr_prob * d_poly_lane + (1.0 - lr_prob) * self.p_poly.copy()
