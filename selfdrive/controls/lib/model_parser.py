from common.numpy_fast import interp
import numpy as np
from selfdrive.controls.lib.latcontrol_helpers import model_polyfit, calc_desired_path, compute_path_pinv

CAMERA_OFFSET = 0.06  # m from center car to camera

def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)

class ModelParser(object):
  def __init__(self):
    self.d_poly = [0., 0., 0., 0.]
    self.c_poly = [0., 0., 0., 0.]
    self.c_prob = 0.
    self.last_model = 0.
    self.lead_dist, self.lead_prob, self.lead_var = 0, 0, 1
    self._path_pinv = compute_path_pinv()

    self.lane_width = 3.0
    self.readings = []
    self.frame = 0
    self.l_prob = 0.
    self.r_prob = 0.
    self.x_points = np.arange(50)

  def update(self, v_ego, md):
    if len(md.leftLane.poly):
      l_poly = np.array(md.leftLane.poly)
      r_poly = np.array(md.rightLane.poly)
      p_poly = np.array(md.path.poly)
    else:
      l_poly = model_polyfit(md.leftLane.points, self._path_pinv)  # left line
      r_poly = model_polyfit(md.rightLane.points, self._path_pinv)  # right line
      p_poly = model_polyfit(md.path.points, self._path_pinv)  # predicted path

    # only offset left and right lane lines; offsetting p_poly does not make sense
    l_poly[3] += CAMERA_OFFSET
    r_poly[3] += CAMERA_OFFSET

    p_prob = 1.  # model does not tell this probability yet, so set to 1 for now
    l_prob = md.leftLane.prob  # left line prob
    r_prob = md.rightLane.prob  # right line prob

    # Find current lanewidth
    if l_prob > 0.49 and r_prob > 0.49:
        self.frame += 1
        if self.frame % 20 == 0:
            self.frame = 0
            current_lane_width = sorted((2.8, abs(l_poly[3] - r_poly[3]), 3.6))[1]
            max_samples = 30
            self.readings.append(current_lane_width)
            avg = mean(self.readings)
            self.lane_width = avg
            if len(self.readings) == max_samples:
                self.readings.pop(0)

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
