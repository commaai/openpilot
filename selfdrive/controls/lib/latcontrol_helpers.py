import numpy as np
import math
from common.numpy_fast import interp

_K_CURV_V = [1., 0.6]
_K_CURV_BP = [0., 0.002]

# lane width http://safety.fhwa.dot.gov/geometric/pubs/mitigationstrategies/chapter3/3_lanewidth.cfm
_LANE_WIDTH_V = [2.7, 3.5]

# break points of speed
_LANE_WIDTH_BP = [0., 31.]


def calc_d_lookahead(v_ego, d_poly):
  # this function computes how far too look for lateral control
  # howfar we look ahead is function of speed and how much curvy is the path
  offset_lookahead = 1.
  k_lookahead = 7.
  # integrate abs value of second derivative of poly to get a measure of path curvature
  pts_len = 50.  # m
  if len(d_poly) > 0:
    pts = np.polyval([6 * d_poly[0], 2 * d_poly[1]], np.arange(0, pts_len))
  else:
    pts = 0.
  curv = np.sum(np.abs(pts)) / pts_len

  k_curv = interp(curv, _K_CURV_BP, _K_CURV_V)

  # sqrt on speed is needed to keep, for a given curvature, the y_des
  # proportional to speed. Indeed, y_des is prop to d_lookahead^2
  # 36m at 25m/s
  d_lookahead = offset_lookahead + math.sqrt(max(v_ego, 0)) * k_lookahead * k_curv
  return d_lookahead


def calc_lookahead_offset(v_ego, angle_steers, d_lookahead, VM, angle_offset):
  # this function returns the lateral offset given the steering angle, speed and the lookahead distance
  sa = math.radians(angle_steers - angle_offset)
  curvature = VM.calc_curvature(sa, v_ego)
  # clip is to avoid arcsin NaNs due to too sharp turns
  y_actual = d_lookahead * np.tan(np.arcsin(np.clip(d_lookahead * curvature, -0.999, 0.999)) / 2.)
  return y_actual, curvature


def calc_desired_steer_angle(v_ego, y_des, d_lookahead, VM, angle_offset):
  # inverse of the above function
  curvature = np.sin(np.arctan(y_des / d_lookahead) * 2.) / d_lookahead
  steer_des = math.degrees(VM.get_steer_from_curvature(curvature, v_ego)) + angle_offset
  return steer_des, curvature


def compute_path_pinv(l=50):
  deg = 3
  x = np.arange(l*1.0)
  X = np.vstack(tuple(x**n for n in range(deg, -1, -1))).T
  pinv = np.linalg.pinv(X)
  return pinv


def model_polyfit(points, path_pinv):
  return np.dot(path_pinv, map(float, points))


def calc_desired_path(l_poly,
                      r_poly,
                      p_poly,
                      l_prob,
                      r_prob,
                      p_prob,
                      speed,
                      lane_width=None):
  # this function computes the poly for the center of the lane, averaging left and right polys
  if lane_width is None:
    lane_width = interp(speed, _LANE_WIDTH_BP, _LANE_WIDTH_V)

  # lanes in Germany are ~2.75-3.5m wide
  half_lane_poly = np.array([0., 0., 0., lane_width / 2.])
  if l_prob + r_prob > 0.01:
    c_poly = ((l_poly - half_lane_poly) * l_prob +
              (r_poly + half_lane_poly) * r_prob) / (l_prob + r_prob)
    c_prob = l_prob + r_prob - l_prob * r_prob
  else:
    c_poly = np.zeros(4)
    c_prob = 0.

  p_weight = 1.  # predicted path weight relatively to the center of the lane
  d_poly = list((c_poly * c_prob + p_poly * p_prob * p_weight) / (c_prob + p_prob * p_weight))
  return d_poly, c_poly, c_prob
