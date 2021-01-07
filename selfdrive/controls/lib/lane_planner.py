from common.numpy_fast import interp
import numpy as np
from cereal import log
from xx.uncommon.numpy_helpers import deep_interp_np

CAMERA_OFFSET = 0.06  # m from center car to camera



class LanePlanner:
  def __init__(self):
    self.path_xyz = np.zeros((33,3))

    self.lll_xyz = np.zeros((33,3))
    self.rll_xyz = np.zeros((33,3))
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
    self.lane_t = (np.array(md.laneLines[1].t) + np.array(md.laneLines[1].t))/2
    self.path_t = np.array(md.position.t)
    self.path_xyz = np.column_stack([md.position.x, md.position.y, md.position.z])

    self.lll_xyz = np.column_stack([md.laneLines[1].x, md.laneLines[1].y, md.laneLines[1].z])
    self.rll_xyz = np.column_stack([md.laneLines[2].x, md.laneLines[2].y, md.laneLines[2].z])
    self.lll_prob = md.laneLineProbs[1]
    self.rll_prob = md.laneLineProbs[2]
    self.lll_std = md.laneLineStds[1]
    self.rll_std = md.laneLineStds[2]

    if len(md.meta.desireState):
      self.l_lane_change_prob = md.meta.desireState[log.PathPlan.Desire.laneChangeLeft]
      self.r_lane_change_prob = md.meta.desireState[log.PathPlan.Desire.laneChangeRight]

  def update_d_path(self, v_ego):
    # only offset left and right lane lines; offsetting path does not make sense
    self.lll_xyz[:,1] -= CAMERA_OFFSET
    self.rll_xyz[:,1] -= CAMERA_OFFSET

    # Reduce reliance on lanelines that are too far apart or
    # will be in a few seconds
    l_prob, r_prob = self.lll_prob, self.rll_prob
    width_pts = (self.rll_xyz[:,1] - self.lll_xyz[:,1])
    x_pts = (self.lll_xyz[:,0] + self.rll_xyz[:,0])/2
    prob_mods = []
    for t_check in [0.0, 1.5, 3.0]:
      width_at_t = interp(t_check * (v_ego + 7), x_pts, width_pts)
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
    current_lane_width = abs(self.rll_xyz[0,1] - self.lll_xyz[0,1])
    self.lane_width_estimate += 0.005 * (current_lane_width - self.lane_width_estimate)
    speed_lane_width = interp(v_ego, [0., 31.], [2.8, 3.5])
    self.lane_width = self.lane_width_certainty * self.lane_width_estimate + \
                      (1 - self.lane_width_certainty) * speed_lane_width

    clipped_lane_width = min(4.0, self.lane_width)
    path_from_left_lane = self.lll_xyz.copy()
    path_from_left_lane[:,1] += clipped_lane_width / 2.0
    path_from_right_lane = self.rll_xyz.copy()
    path_from_right_lane[:,1] -= clipped_lane_width / 2.0

    lr_prob = l_prob + r_prob - l_prob * r_prob
    lane_path_xyz = (l_prob * path_from_left_lane + r_prob * path_from_right_lane) / (l_prob + r_prob + 0.0001)
    path_xyz_interp = deep_interp_np(self.lane_t, self.path_t, self.path_xyz)
    self.d_path_xyz = lr_prob * lane_path_xyz + (1.0 - lr_prob) * path_xyz_interp
