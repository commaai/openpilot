from common.numpy_fast import interp
from math import atan2, sqrt
from common.realtime import DT_DMON
from selfdrive.controls.lib.drive_helpers import create_event, EventTypes as ET
from common.filter_simple import FirstOrderFilter
from common.stat_live import RunningStatFilter

_AWARENESS_TIME = 70.  # one minute limit without user touching steering wheels make the car enter a terminal status
_AWARENESS_PRE_TIME_TILL_TERMINAL = 15.  # a first alert is issued 25s before expiration
_AWARENESS_PROMPT_TIME_TILL_TERMINAL = 6.  # a second alert is issued 15s before start decelerating the car
_DISTRACTED_TIME = 11.
_DISTRACTED_PRE_TIME_TILL_TERMINAL = 8.
_DISTRACTED_PROMPT_TIME_TILL_TERMINAL = 6.

_FACE_THRESHOLD = 0.4
_EYE_THRESHOLD = 0.6
_BLINK_THRESHOLD = 0.5 # 0.225
_BLINK_THRESHOLD_SLACK = 0.65
_BLINK_THRESHOLD_STRICT = 0.5
_PITCH_WEIGHT = 1.35 # 1.5  # pitch matters a lot more
_POSESTD_THRESHOLD = 0.14
_METRIC_THRESHOLD = 0.4
_METRIC_THRESHOLD_SLACK = 0.55
_METRIC_THRESHOLD_STRICT = 0.4
_PITCH_POS_ALLOWANCE = 0.12 # rad, to not be too sensitive on positive pitch
_PITCH_NATURAL_OFFSET = 0.02 # people don't seem to look straight when they drive relaxed, rather a bit up
_YAW_NATURAL_OFFSET = 0.08 # people don't seem to look straight when they drive relaxed, rather a bit to the right (center of car)

_HI_STD_TIMEOUT = 5
_HI_STD_FALLBACK_TIME = 10 # fall back to wheel touch if model is uncertain for a long time
_DISTRACTED_FILTER_TS = 0.25  # 0.6Hz

_POSE_CALIB_MIN_SPEED = 13 # 30 mph
_POSE_OFFSET_MIN_COUNT = 60 # valid data counts before calibration completes, 1 seg is 600 counts
_POSE_OFFSET_MAX_COUNT = 360 # stop deweighting new data after 6 min, aka "short term memory"

_RECOVERY_FACTOR_MAX = 5. # relative to minus step change
_RECOVERY_FACTOR_MIN = 1.25 # relative to minus step change

MAX_TERMINAL_ALERTS = 3 # not allowed to engage after 3 terminal alerts
MAX_TERMINAL_DURATION = 300 # 30s

# model output refers to center of cropped image, so need to apply the x displacement offset
RESIZED_FOCAL = 320.0
H, W, FULL_W = 320, 160, 426

class DistractedType():
  NOT_DISTRACTED = 0
  BAD_POSE = 1
  BAD_BLINK = 2

def face_orientation_from_net(angles_desc, pos_desc, rpy_calib):
  # the output of these angles are in device frame
  # so from driver's perspective, pitch is up and yaw is right

  pitch_net = angles_desc[0]
  yaw_net = angles_desc[1]
  roll_net = angles_desc[2]

  face_pixel_position = ((pos_desc[0] + .5)*W - W + FULL_W, (pos_desc[1]+.5)*H)
  yaw_focal_angle = atan2(face_pixel_position[0] - FULL_W//2, RESIZED_FOCAL)
  pitch_focal_angle = atan2(face_pixel_position[1] - H//2, RESIZED_FOCAL)

  roll = roll_net
  pitch = pitch_net + pitch_focal_angle
  yaw = -yaw_net + yaw_focal_angle

  # no calib for roll
  pitch -= rpy_calib[1]
  yaw -= rpy_calib[2]
  return roll, pitch, yaw

class DriverPose():
  def __init__(self):
    self.yaw = 0.
    self.pitch = 0.
    self.roll = 0.
    self.yaw_std = 0.
    self.pitch_std = 0.
    self.roll_std = 0.
    self.pitch_offseter = RunningStatFilter(max_trackable=_POSE_OFFSET_MAX_COUNT)
    self.yaw_offseter = RunningStatFilter(max_trackable=_POSE_OFFSET_MAX_COUNT)
    self.low_std = True
    self.cfactor = 1.

class DriverBlink():
  def __init__(self):
    self.left_blink = 0.
    self.right_blink = 0.
    self.cfactor = 1.

class DriverStatus():
  def __init__(self):
    self.pose = DriverPose()
    self.pose_calibrated = self.pose.pitch_offseter.filtered_stat.n > _POSE_OFFSET_MIN_COUNT and \
                            self.pose.yaw_offseter.filtered_stat.n > _POSE_OFFSET_MIN_COUNT
    self.blink = DriverBlink()
    self.awareness = 1.
    self.awareness_active = 1.
    self.awareness_passive = 1.
    self.driver_distracted = False
    self.driver_distraction_filter = FirstOrderFilter(0., _DISTRACTED_FILTER_TS, DT_DMON)
    self.face_detected = False
    self.terminal_alert_cnt = 0
    self.terminal_time = 0
    self.step_change = 0.
    self.active_monitoring_mode = True
    self.hi_stds = 0
    self.threshold_prompt = _DISTRACTED_PROMPT_TIME_TILL_TERMINAL / _DISTRACTED_TIME

    self.is_rhd_region = False
    self.is_rhd_region_checked = False

    self._set_timers(active_monitoring=True)

  def _set_timers(self, active_monitoring):
    if self.active_monitoring_mode and self.awareness <= self.threshold_prompt:
      if active_monitoring:
        self.step_change = DT_DMON / _DISTRACTED_TIME
      else:
        self.step_change = 0.
      return # no exploit after orange alert
    elif self.awareness <= 0.:
      return

    if active_monitoring:
      # when falling back from passive mode to active mode, reset awareness to avoid false alert
      if not self.active_monitoring_mode:
        self.awareness_passive = self.awareness
        self.awareness = self.awareness_active

      self.threshold_pre = _DISTRACTED_PRE_TIME_TILL_TERMINAL / _DISTRACTED_TIME
      self.threshold_prompt = _DISTRACTED_PROMPT_TIME_TILL_TERMINAL / _DISTRACTED_TIME
      self.step_change = DT_DMON / _DISTRACTED_TIME
      self.active_monitoring_mode = True
    else:
      if self.active_monitoring_mode:
        self.awareness_active = self.awareness
        self.awareness = self.awareness_passive

      self.threshold_pre = _AWARENESS_PRE_TIME_TILL_TERMINAL / _AWARENESS_TIME
      self.threshold_prompt = _AWARENESS_PROMPT_TIME_TILL_TERMINAL / _AWARENESS_TIME
      self.step_change = DT_DMON / _AWARENESS_TIME
      self.active_monitoring_mode = False

  def _is_driver_distracted(self, pose, blink):
    if not self.pose_calibrated:
      pitch_error = pose.pitch - _PITCH_NATURAL_OFFSET
      yaw_error = pose.yaw - _YAW_NATURAL_OFFSET
    else:
      pitch_error = pose.pitch - self.pose.pitch_offseter.filtered_stat.mean()
      yaw_error = pose.yaw - self.pose.yaw_offseter.filtered_stat.mean()

    # positive pitch allowance
    if pitch_error > 0.:
      pitch_error = max(pitch_error - _PITCH_POS_ALLOWANCE, 0.)
    pitch_error *= _PITCH_WEIGHT
    pose_metric = sqrt(yaw_error**2 + pitch_error**2)

    if pose_metric > _METRIC_THRESHOLD*pose.cfactor:
      return DistractedType.BAD_POSE
    elif (blink.left_blink + blink.right_blink)*0.5 > _BLINK_THRESHOLD*blink.cfactor:
      return DistractedType.BAD_BLINK
    else:
      return DistractedType.NOT_DISTRACTED

  def set_policy(self, model_data):
    ep = min(model_data.meta.engagedProb, 0.8) / 0.8
    self.pose.cfactor = interp(ep, [0, 0.5, 1], [_METRIC_THRESHOLD_STRICT, _METRIC_THRESHOLD, _METRIC_THRESHOLD_SLACK])/_METRIC_THRESHOLD
    self.blink.cfactor = interp(ep, [0, 0.5, 1], [_BLINK_THRESHOLD_STRICT, _BLINK_THRESHOLD, _BLINK_THRESHOLD_SLACK])/_BLINK_THRESHOLD

  def get_pose(self, driver_state, cal_rpy, car_speed, op_engaged):
    # 10 Hz
    if len(driver_state.faceOrientation) == 0 or len(driver_state.facePosition) == 0 or len(driver_state.faceOrientationStd) == 0 or len(driver_state.facePositionStd) == 0:
      return

    self.pose.roll, self.pose.pitch, self.pose.yaw = face_orientation_from_net(driver_state.faceOrientation, driver_state.facePosition, cal_rpy)
    self.pose.pitch_std = driver_state.faceOrientationStd[0]
    self.pose.yaw_std = driver_state.faceOrientationStd[1]
    # self.pose.roll_std = driver_state.faceOrientationStd[2]
    model_std_max = max(self.pose.pitch_std, self.pose.yaw_std)
    self.pose.low_std = model_std_max < _POSESTD_THRESHOLD
    self.blink.left_blink = driver_state.leftBlinkProb * (driver_state.leftEyeProb>_EYE_THRESHOLD)
    self.blink.right_blink = driver_state.rightBlinkProb * (driver_state.rightEyeProb>_EYE_THRESHOLD)
    self.face_detected = driver_state.faceProb > _FACE_THRESHOLD and \
                          abs(driver_state.facePosition[0]) <= 0.4 and abs(driver_state.facePosition[1]) <= 0.45 and \
                          not self.is_rhd_region

    self.driver_distracted = self._is_driver_distracted(self.pose, self.blink) > 0
    # first order filters
    self.driver_distraction_filter.update(self.driver_distracted)

    # update offseter
    # only update when driver is actively driving the car above a certain speed
    if self.face_detected and car_speed>_POSE_CALIB_MIN_SPEED and self.pose.low_std and (not op_engaged or not self.driver_distracted):
      self.pose.pitch_offseter.push_and_update(self.pose.pitch)
      self.pose.yaw_offseter.push_and_update(self.pose.yaw)

    self.pose_calibrated = self.pose.pitch_offseter.filtered_stat.n > _POSE_OFFSET_MIN_COUNT and \
                            self.pose.yaw_offseter.filtered_stat.n > _POSE_OFFSET_MIN_COUNT

    is_model_uncertain = self.hi_stds * DT_DMON > _HI_STD_FALLBACK_TIME
    self._set_timers(self.face_detected and not is_model_uncertain)
    if self.face_detected and not self.pose.low_std:
      if not is_model_uncertain:
        self.step_change *= max(0, (model_std_max-0.5)*(model_std_max-2))
      self.hi_stds += 1
    elif self.face_detected and self.pose.low_std:
      self.hi_stds = 0

  def update(self, events, driver_engaged, ctrl_active, standstill):
    if (driver_engaged and self.awareness > 0) or not ctrl_active:
      # reset only when on disengagement if red reached
      self.awareness = 1.
      self.awareness_active = 1.
      self.awareness_passive = 1.
      return events

    driver_attentive = self.driver_distraction_filter.x < 0.37
    awareness_prev = self.awareness

    if self.face_detected and self.hi_stds * DT_DMON > _HI_STD_TIMEOUT:
      events.append(create_event('driverMonitorLowAcc', [ET.WARNING]))

    if (driver_attentive and self.face_detected and self.pose.low_std and self.awareness > 0):
      # only restore awareness when paying attention and alert is not red
      self.awareness = min(self.awareness + ((_RECOVERY_FACTOR_MAX-_RECOVERY_FACTOR_MIN)*(1.-self.awareness)+_RECOVERY_FACTOR_MIN)*self.step_change, 1.)
      if self.awareness == 1.:
        self.awareness_passive = min(self.awareness_passive + self.step_change, 1.)
      # don't display alert banner when awareness is recovering and has cleared orange
      if self.awareness > self.threshold_prompt:
        return events

    # should always be counting if distracted unless at standstill and reaching orange
    if (not (self.face_detected and self.hi_stds * DT_DMON <= _HI_STD_FALLBACK_TIME) or (self.driver_distraction_filter.x > 0.63 and self.driver_distracted and self.face_detected)) and \
       not (standstill and self.awareness - self.step_change <= self.threshold_prompt):
      self.awareness = max(self.awareness - self.step_change, -0.1)

    alert = None
    if self.awareness <= 0.:
      # terminal red alert: disengagement required
      alert = 'driverDistracted' if self.active_monitoring_mode else 'driverUnresponsive'
      self.terminal_time += 1
      if awareness_prev > 0.:
        self.terminal_alert_cnt += 1
    elif self.awareness <= self.threshold_prompt:
      # prompt orange alert
      alert = 'promptDriverDistracted' if self.active_monitoring_mode else 'promptDriverUnresponsive'
    elif self.awareness <= self.threshold_pre:
      # pre green alert
      alert = 'preDriverDistracted' if self.active_monitoring_mode else 'preDriverUnresponsive'

    if alert is not None:
      events.append(create_event(alert, [ET.WARNING]))

    return events
