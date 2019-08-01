from common.numpy_fast import interp
import numpy as np
from selfdrive.controls.lib.latcontrol_helpers import model_polyfit, calc_desired_path, compute_path_pinv

CAMERA_OFFSET = 0.06

class ModelParser(object):
  def __init__(self):
    self.d_poly = [0., 0., 0., 0.]
    self.c_poly = [0., 0., 0., 0.]
    self.p_poly = np.array([0., 0., 0., 0.])
    self.c_prob = 0.
    self.p_prob = 0.
    self.last_model = 0.
    self.lead_dist, self.lead_prob, self.lead_var = 0, 0, 1
    self._path_pinv = compute_path_pinv()

    self.lane_width_estimate = 3.0
    self.lane_width_certainty = 1.0
    self.lane_width = 3.0
    self.l_prob = 0.
    self.r_prob = 0.
    self.x_points = np.arange(50)

  def update(self, v_ego, md):
    if len(md.leftLane.poly):
      l_poly = np.array(md.leftLane.poly)
      r_poly = np.array(md.rightLane.poly)
      #p_poly = np.array(md.path.poly)
      p_poly = np.array([0., 0., 0., 0.])
    else:
      l_poly = model_polyfit(md.leftLane.points, self._path_pinv)  # left line
      r_poly = model_polyfit(md.rightLane.points, self._path_pinv)  # right line
      p_poly = model_polyfit(md.path.points, self._path_pinv)  # predicted path

    # only offset left and right lane lines; offsetting p_poly does not make sense
    l_poly[3] += CAMERA_OFFSET
    r_poly[3] += CAMERA_OFFSET

    l_prob = md.leftLane.prob  # left line prob
    r_prob = md.rightLane.prob  # right line prob

    # Find current lanewidth
    lr_prob = l_prob * r_prob
    self.lane_width_certainty += 0.05 * (lr_prob - self.lane_width_certainty)
    current_lane_width = abs(l_poly[3] - r_poly[3])
    self.lane_width_estimate += 0.005 * (current_lane_width - self.lane_width_estimate)
    speed_lane_width = interp(v_ego, [0., 31.], [2.8, 3.5])
    self.lane_width = np.clip(self.lane_width_certainty * self.lane_width_estimate + \
                      (1 - self.lane_width_certainty) * speed_lane_width,  \
                      self.lane_width - 0.025, self.lane_width + 0.025)

    half_lane_width = self.lane_width / 2.0
    l_center = l_prob * (l_poly[3] - half_lane_width)
    r_center = r_prob * (r_poly[3] + half_lane_width)
    p_prob = 0.0001 + (self.c_prob**2 + self.p_prob**2) / (self.c_prob + self.p_prob + 0.0001)

    self.p_poly[3] = np.clip(0.0, self.p_poly[3] - 0.005, self.p_poly[3] + 0.005)
    self.p_poly[2] = np.clip(0.0, self.p_poly[2] - 0.001, self.p_poly[2] + 0.001)
    self.p_poly[1] = np.clip(0.0, self.p_poly[1] - 0.0001, self.p_poly[1] + 0.0001)
    #self.p_poly[0] = np.clip(0.0, self.p_poly[0] - 0.00001, self.p_poly[0] + 0.00001)

    if l_center < 0.0 or r_center > 0.0:
      if l_center > -r_center:
        p_poly[3] = (r_center + p_prob * self.p_poly[3]) / (r_prob + p_prob + 0.0001)
      else:
        p_poly[3] = (l_center + p_prob * self.p_poly[3]) / (l_prob + p_prob + 0.0001)

      race_line_adjust = np.interp(abs(p_poly[3]), [0.0, 0.3], [0.0, 1.0])
      l_race_poly = (race_line_adjust * l_poly * l_prob + p_prob * self.p_poly) / (race_line_adjust * l_prob + p_prob + 0.0001)
      r_race_poly = (race_line_adjust * r_poly * r_prob + p_prob * self.p_poly) / (race_line_adjust * r_prob + p_prob + 0.0001)
      if self.d_poly[1] < 0.0:
        p_poly[2] = min(l_race_poly[2], r_race_poly[2], self.p_poly[2])
        p_poly[1] = min(l_race_poly[1], r_race_poly[1], self.d_poly[1])
        #p_poly[0] = min(l_race_poly[0], r_race_poly[0], self.d_poly[0])
      else:
        p_poly[2] = max(l_race_poly[2], r_race_poly[2], self.p_poly[2])
        p_poly[1] = max(l_race_poly[1], r_race_poly[1], self.d_poly[1])
        #p_poly[0] = max(l_race_poly[0], r_race_poly[0], self.d_poly[0])
    else:
      p_poly[3] = self.p_poly[3]
      p_poly[2] = self.p_poly[2]
      p_poly[1] = self.d_poly[1]
      #p_poly[0] = self.d_poly[0]

    self.lead_dist = md.lead.dist
    self.lead_prob = md.lead.prob
    self.lead_var = md.lead.std**2

    # compute target path
    self.d_poly, self.c_poly, self.c_prob = calc_desired_path(
      l_poly, r_poly, p_poly, l_prob, r_prob, p_prob, v_ego, self.lane_width)

    self.r_poly = r_poly
    self.r_prob = r_prob

    self.l_poly = l_poly
    self.l_prob = l_prob

    self.p_poly = p_poly
    self.p_prob = p_prob
