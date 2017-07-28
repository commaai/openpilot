import math
import numpy as np

from common.numpy_fast import interp
import selfdrive.messaging as messaging


def compute_path_pinv():
  deg = 3
  x = np.arange(50.0)
  X = np.vstack(tuple(x**n for n in range(deg, -1, -1))).T
  pinv = np.linalg.pinv(X)
  return pinv

def model_polyfit(points, path_pinv):
  return np.dot(path_pinv, map(float, points))

# lane width http://safety.fhwa.dot.gov/geometric/pubs/mitigationstrategies/chapter3/3_lanewidth.cfm
_LANE_WIDTH_V = [3., 3.8]

# break points of speed
_LANE_WIDTH_BP = [0., 31.]

def calc_desired_path(l_poly, r_poly, p_poly, l_prob, r_prob, p_prob, speed):
  #*** this function computes the poly for the center of the lane, averaging left and right polys
  lane_width = interp(speed, _LANE_WIDTH_BP, _LANE_WIDTH_V)

  # lanes in US are ~3.6m wide
  half_lane_poly = np.array([0., 0., 0., lane_width / 2.])
  if l_prob + r_prob > 0.01:
    c_poly = ((l_poly - half_lane_poly) * l_prob +
              (r_poly + half_lane_poly) * r_prob) / (l_prob + r_prob)
    c_prob = math.sqrt((l_prob**2 + r_prob**2) / 2.)
  else:
    c_poly = np.zeros(4)
    c_prob = 0.

  p_weight = 1. # predicted path weight relatively to the center of the lane
  d_poly =  list((c_poly*c_prob + p_poly*p_prob*p_weight ) / (c_prob + p_prob*p_weight))
  return d_poly, c_poly, c_prob

class OptPathPlanner(object):
  def __init__(self, model):
    self.model = model
    self.dead = True
    self.d_poly = [0., 0., 0., 0.]
    self.last_model = 0.
    self._path_pinv = compute_path_pinv()

  def update(self, cur_time, v_ego, md):
    if md is not None:
      # simple compute of the center of the lane
      pts = [(x+y)/2 for x,y in zip(md.model.leftLane.points, md.model.rightLane.points)]
      self.d_poly = model_polyfit(pts, self._path_pinv)

      self.last_model = cur_time
      self.dead = False
    elif cur_time - self.last_model > 0.5:
      self.dead = True

class PathPlanner(object):
  def __init__(self):
    self.dead = True
    self.d_poly = [0., 0., 0., 0.]
    self.last_model = 0.
    self.lead_dist, self.lead_prob, self.lead_var = 0, 0, 1
    self._path_pinv = compute_path_pinv()

  def update(self, cur_time, v_ego, md):
    if md is not None:
      p_poly = model_polyfit(md.model.path.points, self._path_pinv)       # predicted path
      l_poly = model_polyfit(md.model.leftLane.points, self._path_pinv)   # left line
      r_poly = model_polyfit(md.model.rightLane.points, self._path_pinv)  # right line

      p_prob = 1.                       # model does not tell this probability yet, so set to 1 for now
      l_prob = md.model.leftLane.prob   # left line prob
      r_prob = md.model.rightLane.prob  # right line prob

      self.lead_dist = md.model.lead.dist
      self.lead_prob = md.model.lead.prob
      self.lead_var = md.model.lead.std**2

      # compute target path
      self.d_poly, _, _ = calc_desired_path(l_poly, r_poly, p_poly, l_prob, r_prob, p_prob, v_ego)

      self.last_model = cur_time
      self.dead = False
    elif cur_time - self.last_model > 0.5:
      self.dead = True
