from common.numpy_fast import interp
import numpy as np
from cereal import log

CAMERA_OFFSET = 0.06  # m from center car to camera
NORM_THRESHOLD = 2  # 60 degrees


def eval_poly(poly, x):
  return poly[3] + poly[2]*x + poly[1]*x**2 + poly[0]*x**3


def monotonic_increasing_around_idx(now_idx, x):
  monotonic = np.ones(x.shape, dtype=np.bool)
  x_diff = np.diff(x, prepend=x[0])
  neg_grad_idxs_before = np.where(x_diff[:now_idx] < 0)[0]
  neg_grad_idxs_after = np.where(x_diff[now_idx:] < 0)[0] + now_idx
  if len(neg_grad_idxs_before) > 0:
    monotonic[:neg_grad_idxs_before[-1]] = False
  if len(neg_grad_idxs_after) > 0:
    monotonic[neg_grad_idxs_after[0]:] = False
  return monotonic


def monotonic_increasing_around_t(t, line_t, x):
  assert len(line_t) == len(x)
  now_idx = np.argmin(abs(line_t - t))
  return monotonic_increasing_around_idx(now_idx, x)


def clean_path_for_polyfit(path_xyz):
  valid = np.isfinite(path_xyz).all(axis=1)
  path_xyz = path_xyz[valid]
  if sum(valid) == 0:
    return path_xyz
  monotonic = monotonic_increasing_around_idx(0, path_xyz[:,0])
  path_xyz = path_xyz[monotonic]
  clip_cnt = 0
  idx = 1
  while idx < len(path_xyz) and (clip_cnt < 2 or idx < 5):
    x_diff = path_xyz[idx,0] - path_xyz[idx - 1,0]
    mini = path_xyz[idx-1,1] - NORM_THRESHOLD*x_diff
    maxi = path_xyz[idx-1,1] + NORM_THRESHOLD*x_diff
    if not mini <= path_xyz[idx,1] <= maxi:
      clip_cnt += 1
    path_xyz[idx,1] = np.clip(path_xyz[idx,1], mini, maxi)
    idx += 1
  return path_xyz[:idx]


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

  def parse_model(self, md):
    path_xyz = np.column_stack([md.position.x, md.position.y, md.position.z])
    path_xyz_stds = np.column_stack([md.position.xStd, md.position.yStd, md.postion.zStd])
    path_xyz = clean_path_for_polyfit(path_xyz)
    # mpc only goes till 2.5s anyway
    path_xyz = path_xyz[:16]

    if len(path_xyz) > 5 and path_xyz[-1,0] > 1:
      #TODO hacky use exact same code as runtime
      weights = 1/path_xyz_stds[:len(path_xyz),1]
      weights[0] = 1e3
      self.p_poly = np.polyfit(path_xyz[:,0], path_xyz[:,1], 3, w=weights)
    else:
      self.p_poly = np.zeros((4,))
    self.l_prob = 0.0
    self.r_prob = 0.0

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
