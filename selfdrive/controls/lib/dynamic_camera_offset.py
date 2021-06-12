import numpy as np
from cereal import messaging
from common.numpy_fast import interp
from common.op_params import opParams
from common.realtime import sec_since_boot
from selfdrive.config import Conversions as CV
from cereal.messaging import SubMaster, PubMaster


class DynamicCameraOffset:  # keeps away from oncoming traffic
  def __init__(self):
    self.sm = SubMaster(['laneSpeed'])
    self.pm = PubMaster(['dynamicCameraOffset'])
    self.op_params = opParams()
    self.camera_offset = self.op_params.get('camera_offset')

    self.left_lane_oncoming = False  # these variables change
    self.right_lane_oncoming = False
    self.last_left_lane_oncoming = False
    self.last_right_lane_oncoming = False
    self.last_oncoming_time = 0
    self.i = 0.0

    self._setup_static()

  def _setup_static(self):  # these variables are static
    self._enabled = self.op_params.get('dynamic_camera_offset')
    self._min_enable_speed = 35 * CV.MPH_TO_MS
    self._min_lane_width_certainty = 0.4
    hug = 0.115  # how much to hug
    self._center_ratio = 0.5
    self._hug_left_ratio = self._center_ratio - hug
    self._hug_right_ratio = self._center_ratio + hug

    self._keep_offset_for = self.op_params.get('dynamic_camera_offset_time')  # seconds after losing oncoming lane
    self._ramp_angles = [0, 12.5, 25]
    self._ramp_angle_mods = [1, 0.85, 0.1]  # multiply offset by this based on angle

    self._ramp_down_times = [self._keep_offset_for, self._keep_offset_for * 1.3]

    self._poly_prob_speeds = [0, 25 * CV.MPH_TO_MS, 35 * CV.MPH_TO_MS, 60 * CV.MPH_TO_MS]
    self._poly_probs = [0.2, 0.25, 0.45, 0.55]  # we're good if only one line is above this

    self._k_p = 1.5
    _i_rate = 1 / 20
    self._k_i = 1.2 * _i_rate

  def update(self, v_ego, active, angle_steers, lane_width_estimate, lane_width_certainty, polys, probs):
    self.camera_offset = self.op_params.get('camera_offset')  # update base offset from user
    if self._enabled:  # if feature enabled
      self.sm.update(0)
      self.left_lane_oncoming = self.sm['laneSpeed'].leftLaneOncoming
      self.right_lane_oncoming = self.sm['laneSpeed'].rightLaneOncoming
      self.lane_width_estimate, self.lane_width_certainty = lane_width_estimate, lane_width_certainty
      self.l_poly, self.r_poly = polys
      self.l_prob, self.r_prob = probs

      dynamic_offset = self._get_camera_offset(v_ego, active, angle_steers)
      self._send_state()  # for alerts, before speed check so alerts don't get stuck on
      if dynamic_offset is not None:
        return self.camera_offset + dynamic_offset

      self.i = 0  # reset when not active
    return self.camera_offset  # don't offset if no lane line in direction we're going to hug

  def _get_camera_offset(self, v_ego, active, angle_steers):
    self.keeping_left, self.keeping_right = False, False  # reset keeping
    time_since_oncoming = sec_since_boot() - self.last_oncoming_time
    if not active:  # no alert when not engaged
      return
    if np.isnan(self.l_poly[3]) or np.isnan(self.r_poly[3]):
      return
    if v_ego < self._min_enable_speed:
      return
    _min_poly_prob = interp(v_ego, self._poly_prob_speeds, self._poly_probs)
    if self.l_prob < _min_poly_prob and self.r_prob < _min_poly_prob:  # we only need one line and an accurate current lane width
      return

    left_lane_oncoming = self.left_lane_oncoming
    right_lane_oncoming = self.right_lane_oncoming

    if self.have_oncoming:
      if self.lane_width_certainty < self._min_lane_width_certainty:
        return
      self.last_oncoming_time = sec_since_boot()
      self.last_left_lane_oncoming = self.left_lane_oncoming  # only update last oncoming vars when currently have oncoming. one should always be True for the 2 second ramp down
      self.last_right_lane_oncoming = self.right_lane_oncoming
    elif time_since_oncoming > self._ramp_down_times[-1]:  # return if it's x+ seconds after last oncoming, no need to offset
      return
    else:  # no oncoming and not yet x seconds after we lost an oncoming lane. use last oncoming lane until we complete full offset time
      left_lane_oncoming = self.last_left_lane_oncoming
      right_lane_oncoming = self.last_right_lane_oncoming

    estimated_lane_position = self._get_camera_position()

    hug_modifier = interp(abs(angle_steers), self._ramp_angles, self._ramp_angle_mods)  # don't offset as much when angle is high
    if left_lane_oncoming:
      self.keeping_right = True
      hug_ratio = (self._hug_right_ratio * hug_modifier) + (self._center_ratio * (1 - hug_modifier))  # weighted average
    elif right_lane_oncoming:
      self.keeping_left = True
      hug_ratio = (self._hug_left_ratio * hug_modifier) + (self._center_ratio * (1 - hug_modifier))
    else:
      raise Exception('Error, no lane is oncoming but we\'re here!')

    error = estimated_lane_position - hug_ratio
    self.i += error * self._k_i  # PI controller
    offset = self.i + error * self._k_p

    if time_since_oncoming <= self._ramp_down_times[-1] and not self.have_oncoming:
      offset = interp(time_since_oncoming, self._ramp_down_times, [offset, 0])  # we have passed initial full offset time, start to ramp down
    return offset

  def _send_state(self):
    dco_send = messaging.new_message('dynamicCameraOffset')
    dco_send.dynamicCameraOffset.keepingLeft = self.keeping_left
    dco_send.dynamicCameraOffset.keepingRight = self.keeping_right
    self.pm.send('dynamicCameraOffset', dco_send)

  @property
  def have_oncoming(self):
    return self.left_lane_oncoming != self.right_lane_oncoming  # only one lane oncoming

  def _get_camera_position(self):
    """
    Returns the position of the camera in the lane as a percentage. left to right: [0, 1]; 0.5 is centered
    You MUST verify that either left or right polys and lane width are accurate before calling this function.
    """
    left_line_pos = self.l_poly[3] + self.camera_offset  # polys have not been offset yet
    right_line_pos = self.r_poly[3] + self.camera_offset
    cam_pos_left = left_line_pos / self.lane_width_estimate  # estimated position of car in lane based on left line
    cam_pos_right = 1 - abs(right_line_pos) / self.lane_width_estimate  # estimated position of car in lane based on right line

    # find car's camera position using weighted average of lane poly certainty
    # if certainty of both lines are high, then just average ~equally
    l_prob = self.l_prob / (self.l_prob + self.r_prob)  # this and next line sums to 1
    r_prob = self.r_prob / (self.l_prob + self.r_prob)
    # be biased towards position found from most probable lane line
    return cam_pos_left * l_prob + cam_pos_right * r_prob
