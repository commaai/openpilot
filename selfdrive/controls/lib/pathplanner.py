import numpy as np
import math
from common.numpy_fast import interp
from selfdrive.controls.lib.latcontrol_helpers import model_polyfit, calc_desired_path, compute_path_pinv

CAMERA_OFFSET = 0.06  # m from center car to camera

class PathPlanner(object):
  def __init__(self):
    self.d_poly = [0., 0., 0., 0.]
    self.c_poly = [0., 0., 0., 0.]
    self.c_prob = 0.
    self.last_model = 0.
    self.lead_dist, self.lead_prob, self.lead_var = 0, 0, 1
    self._path_pinv = compute_path_pinv()

    self.lane_width_estimate = 3.2
    self.lane_width_certainty = 1.0
    self.lane_width = 3.7
    self.l_prob = 0.
    self.r_prob = 0.

  def update(self, v_ego, md, LaC=None):
    if md is not None:
      p_poly = model_polyfit(md.model.path.points, self._path_pinv)  # predicted path
      l_poly = model_polyfit(md.model.leftLane.points, self._path_pinv)  # left line
      r_poly = model_polyfit(md.model.rightLane.points, self._path_pinv)  # right line

      try:
        if LaC is not None and LaC.angle_steers_des_mpc != 0.0:
          angle_error = LaC.angle_steers_des_mpc - (0.05 * LaC.avg_angle_steers + LaC.steerActuatorDelay * LaC.projected_angle_steers) / (LaC.steerActuatorDelay + 0.05)
        else:
          angle_error = 0.0
        if angle_error != 0.0:
          LaC.lateral_error = 1.0 * np.clip(v_ego * (LaC.steerActuatorDelay + 0.05) * math.tan(math.radians(angle_error)), -1.2, 1.2)
          lateral_error = LaC.lateral_error
        else:
          lateral_error = 0.0
      except:
        lateral_error = 0.0

      # only offset left and right lane lines; offsetting p_poly does not make sense
      l_poly[3] += CAMERA_OFFSET - lateral_error
      r_poly[3] += CAMERA_OFFSET - lateral_error

      p_prob = 1.  # model does not tell this probability yet, so set to 1 for now
      l_prob = md.model.leftLane.prob  # left line prob
      r_prob = md.model.rightLane.prob  # right line prob

      # Find current lanewidth
      lr_prob = l_prob * r_prob
      self.lane_width_certainty += 0.05 * (lr_prob - self.lane_width_certainty)
      current_lane_width = abs(l_poly[3] - r_poly[3])
      self.lane_width_estimate += 0.005 * (current_lane_width - self.lane_width_estimate)
      speed_lane_width = interp(v_ego, [0., 31.], [3., 3.8])
      self.lane_width = self.lane_width_certainty * self.lane_width_estimate + \
                        (1 - self.lane_width_certainty) * speed_lane_width

      lane_width_diff = abs(self.lane_width - current_lane_width)
      lane_r_prob = interp(lane_width_diff, [0.3, 1.0], [1.0, 0.0])

      r_prob *= lane_r_prob

      self.lead_dist = md.model.lead.dist
      self.lead_prob = md.model.lead.prob
      self.lead_var = md.model.lead.std**2

      # compute target path
      self.d_poly, self.c_poly, self.c_prob = calc_desired_path(
        l_poly, r_poly, p_poly, l_prob, r_prob, p_prob, v_ego, self.lane_width)

      self.r_poly = r_poly
      self.r_prob = r_prob

      self.l_poly = l_poly
      self.l_prob = l_prob

      self.p_poly = p_poly
      self.p_prob = p_prob
