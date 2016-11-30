import numpy as np

import common.filters as filters
from selfdrive.controls.lib.latcontrol import calc_curvature


# Calibration Status
class CalibStatus(object):
  INCOMPLETE = 0
  VALID = 1
  INVALID = 2


def line_intersection(line1, line2, no_int_sub = [0,0]):
  xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
  ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

  def det(a, b):
    return a[0] * b[1] - a[1] * b[0]

  div = det(xdiff, ydiff)
  if div == 0:
    # since we are in float domain, this should really never happen
    return no_int_sub

  d = (det(*line1), det(*line2))
  x = det(d, xdiff) / div
  y = det(d, ydiff) / div
  return [x, y]

def points_inside_hit_box(pts, box):
  """Determine which points lie inside a box.

     Inputs:
      pts: An nx2 array of points to hit test.
      box: An array [[x_left, y_top], [x_right, y_bottom]] describing a box to
           use for hit testing.
     Returns:
      A logical array with true for every member of pts inside box.
  """
  hits = np.all(np.logical_and(pts > box[0, :], pts < box[1, :]), axis=1)
  return hits

def warp_points(pt_s, warp_matrix):
  # pt_s are the source points, nx2 array.
  pt_d = np.dot(warp_matrix[:, :2], pt_s.T) + warp_matrix[:, 2][:, np.newaxis]

  # divide by third dimension for representation in image space.
  return (pt_d[:2, :] / pt_d[2, :]).T

class ViewCalibrator(object):
  def __init__(self, box_size, big_box_size, vp_r, warp_matrix_start, vp_f=None, cal_cycle=0, cal_status=0):
    self.calibration_threshold = 3000
    self.box_size = box_size
    self.big_box_size = big_box_size

    self.warp_matrix_start = warp_matrix_start
    self.vp_r = list(vp_r)

    if vp_f is None:
      self.vp_f = list(vp_r)
    else:
      self.vp_f = list(vp_f)

    # slow filter fot the vanishing point 
    vp_fr = 0.005    # Hz, slow filter 
    self.dt = 0.05  # camera runs at 20Hz

    self.update_warp_matrix()

    self.vp_x_filter = filters.FirstOrderLowpassFilter(vp_fr, self.dt, self.vp_f[0])
    self.vp_y_filter = filters.FirstOrderLowpassFilter(vp_fr, self.dt, self.vp_f[1])

    self.cal_cycle = cal_cycle
    self.cal_status = cal_status
    self.cal_perc = int(np.minimum(self.cal_cycle*100./self.calibration_threshold, 100))

  def vanishing_point_process(self, old_ps, new_ps, v_ego, steer_angle, VP):
    # correct diffs by yaw rate 
    cam_fov = 23.06*np.pi/180. # deg
    curvature = calc_curvature(v_ego, steer_angle, VP)
    yaw_rate = curvature * v_ego
    hor_angle_shift = yaw_rate * self.dt * self.box_size[0] / cam_fov
    old_ps += [hor_angle_shift, 0]  # old points have moved in the image due to yaw rate

    pos_ps = [None]*len(new_ps)
    for ii in range(len(old_ps)):
      xo = old_ps[ii][0]
      yo = old_ps[ii][1]
      yn = new_ps[ii][1]

      # don't consider points with low flow in y
      if abs(yn - yo) > 1:
        if xo > (self.vp_f[0] + 20):
          pos_ps[ii] = 'r'   # right lane point
        elif xo < (self.vp_f[0] - 20):
          pos_ps[ii] = 'l'   # left lane point

        # intersect all the right lines with the left lines
    idxs_l = [i for i, x in enumerate(pos_ps) if x == 'l']
    idxs_r = [i for i, x in enumerate(pos_ps) if x == 'r']

    old_ps_l, new_ps_l = old_ps[idxs_l], new_ps[idxs_l]
    old_ps_r, new_ps_r = old_ps[idxs_r], new_ps[idxs_r]
    # return None if there is one side with no lines, the speed is low or the steer angle is high 
    if len(old_ps_l) == 0 or len(old_ps_r) == 0 or v_ego < 20 or abs(steer_angle) > 5:
      return None

    int_ps = [[None] * len(old_ps_r)] * len(old_ps_l)
    for ll in range(len(old_ps_l)):
      for rr in range(len(old_ps_r)):
        old_p_l, old_p_r, new_p_l, new_p_r = old_ps_l[ll], old_ps_r[
          rr], new_ps_l[ll], new_ps_r[rr]
        line_l = [[old_p_l[0], old_p_l[1]], [new_p_l[0], new_p_l[1]]]
        line_r = [[old_p_r[0], old_p_r[1]], [new_p_r[0], new_p_r[1]]]
        int_ps[ll][rr] = line_intersection(
          line_l, line_r, no_int_sub=self.vp_f)
        # saturate outliers that are too far from the estimated vp
        int_ps[ll][rr][0] = np.clip(int_ps[ll][rr][0], self.vp_f[0] - 20, self.vp_f[0] + 20)
        int_ps[ll][rr][1] = np.clip(int_ps[ll][rr][1], self.vp_f[1] - 30, self.vp_f[1] + 30)
    vp = np.mean(np.mean(np.array(int_ps), axis=0), axis=0)

    return vp

  def calibration_validity(self):
    # this function sanity checks that the small box is contained in the big box.
    # otherwise the warp function will generate black spots on the small box
    cp = np.asarray([[0, 0],
                    [self.box_size[0], 0],
                    [self.box_size[0], self.box_size[1]],
                    [0, self.box_size[1]]])

    cpw = warp_points(cp, self.warp_matrix)

    # pixel margin for validity hysteresys: 
    # - if calibration is good, keep it good until small box is inside the big box
    # - if calibration isn't good, then make it good again if small box is in big box with margin
    margin_px = 0 if self.cal_status == CalibStatus.VALID else 5
    big_hit_box = np.asarray(
      [[margin_px, margin_px],
       [self.big_box_size[0], self.big_box_size[1] - margin_px]])

    cpw_outside_big_box = np.logical_not(points_inside_hit_box(cpw, big_hit_box))
    return not np.any(cpw_outside_big_box)


  def get_calibration_hit_box(self):
    """Returns an axis-aligned hit box in canonical image space.
       Points which do not fall within this box should not be used for
       calibration.

       Returns:
        An array [[x_left, y_top], [x_right, y_bottom]] describing a box inside
        which all calibration points should lie.
    """
    # We mainly care about feature from lanes, so removed points from sky.
    y_filter = 50.
    return np.asarray([[0, y_filter], [self.box_size[0], self.box_size[1]]])


  def update_warp_matrix(self):
    translation_matrix = np.asarray(
      [[1, 0, self.vp_f[0] - self.vp_r[0]],
       [0, 1, self.vp_f[1] - self.vp_r[1]],
       [0, 0, 1]])
    self.warp_matrix = np.dot(translation_matrix, self.warp_matrix_start)
    self.warp_matrix_inv = np.linalg.inv(self.warp_matrix)

  def calibration(self, p0, p1, st, v_ego, steer_angle, VP):
    # convert to np array first thing
    p0 = np.asarray(p0)
    p1 = np.asarray(p1)
    st = np.asarray(st)

    p0 = p0.reshape((-1,2))
    p1 = p1.reshape((-1,2))

    # filter out pts with bad status
    p0 = p0[st==1]
    p1 = p1[st==1]

    calib_hit_box = self.get_calibration_hit_box()
    # remove all the points outside the small box and above the horizon line
    good_idxs = points_inside_hit_box(
      warp_points(p0, self.warp_matrix_inv), calib_hit_box)
    p0 = p0[good_idxs]
    p1 = p1[good_idxs]

    # print("unwarped points: {}".format(warp_points(p0, self.warp_matrix_inv)))
    # print("good_idxs {}:".format(good_idxs))

    # get instantaneous vp
    vp = self.vanishing_point_process(p0, p1, v_ego, steer_angle, VP)

    if vp is not None:
      # filter the vanishing point
      self.vp_f = [self.vp_x_filter(vp[0]), self.vp_y_filter(vp[1])]
      self.cal_cycle += 1

    if not self.calibration_validity():
      self.cal_status = CalibStatus.INVALID
    else:
      # 10 minutes @5Hz TODO: make this threshold function of convergency speed
      self.cal_status = CalibStatus.VALID
      #self.cal_status = CalibStatus.VALID if self.cal_cycle > self.calibration_threshold else CalibStatus.INCOMPLETE
    self.cal_perc = int(np.minimum(self.cal_cycle*100./self.calibration_threshold, 100))

    self.update_warp_matrix()
