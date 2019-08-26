import numpy as np
from common.realtime import sec_since_boot, DT_CTRL, DT_DMON
from selfdrive.controls.lib.drive_helpers import create_event, EventTypes as ET
from common.filter_simple import FirstOrderFilter

_AWARENESS_TIME = 90.        # 1.5 minutes limit without user touching steering wheels make the car enter a terminal status
_AWARENESS_PRE_TIME_TILL_TERMINAL = 20.    # a first alert is issued 20s before expiration
_AWARENESS_PROMPT_TIME_TILL_TERMINAL = 5.  # a second alert is issued 5s before start decelerating the car
_DISTRACTED_TIME = 10.
_DISTRACTED_PRE_TIME_TILL_TERMINAL = 7.
_DISTRACTED_PROMPT_TIME_TILL_TERMINAL = 5.

_FACE_THRESHOLD = 0.4
_EYE_THRESHOLD = 0.4
_BLINK_THRESHOLD = 0.2 # 0.225
_PITCH_WEIGHT = 1.35 # 1.5  # pitch matters a lot more
_METRIC_THRESHOLD = 0.4
_PITCH_POS_ALLOWANCE = 0.04 # 0.08  # rad, to not be too sensitive on positive pitch
_PITCH_NATURAL_OFFSET = 0.12  # 0.1   # people don't seem to look straight when they drive relaxed, rather a bit up
_YAW_NATURAL_OFFSET = 0.08  # people don't seem to look straight when they drive relaxed, rather a bit to the right (center of car)
_DISTRACTED_FILTER_TS = 0.25  # 0.6Hz
_VARIANCE_FILTER_TS = 20.     # 0.008Hz

MAX_TERMINAL_ALERTS = 3      # not allowed to engage after 3 terminal alerts

# model output refers to center of cropped image, so need to apply the x displacement offset
RESIZED_FOCAL = 320.0
H, W, FULL_W = 320, 160, 426

class DistractedType(object):
  NOT_DISTRACTED = 0
  BAD_POSE = 1
  BAD_BLINK = 2

def head_orientation_from_descriptor(angles_desc, pos_desc, rpy_calib):
  # the output of these angles are in device frame
  # so from driver's perspective, pitch is up and yaw is right

  # TODO: calibrate based on position
  pitch_prnet = angles_desc[0]
  yaw_prnet = angles_desc[1]
  roll_prnet = angles_desc[2]

  face_pixel_position = ((pos_desc[0] + .5)*W - W + FULL_W, (pos_desc[1]+.5)*H)
  yaw_focal_angle = np.arctan2(face_pixel_position[0] - FULL_W/2, RESIZED_FOCAL)
  pitch_focal_angle = np.arctan2(face_pixel_position[1] - H/2, RESIZED_FOCAL)

  roll = roll_prnet
  pitch = pitch_prnet + pitch_focal_angle
  yaw = -yaw_prnet + yaw_focal_angle

  # no calib for roll
  pitch -= rpy_calib[1]
  yaw -= rpy_calib[2]

  return np.array([roll, pitch, yaw])


class _DriverPose():
  def __init__(self):
    self.yaw = 0.
    self.pitch = 0.
    self.roll = 0.
    self.yaw_offset = 0.
    self.pitch_offset = 0.

class _DriverBlink():
  def __init__(self):
    self.left_blink = 0.
    self.right_blink = 0.



def _monitor_hysteresis(variance_level, monitor_valid_prev):
  var_thr = 0.63 if monitor_valid_prev else 0.37
  return variance_level < var_thr

class DriverStatus():
  def __init__(self, monitor_on=False):
    self.pose = _DriverPose()
    self.blink = _DriverBlink()
    self.monitor_on = monitor_on
    self.monitor_param_on = monitor_on
    self.monitor_valid = True   # variance needs to be low
    self.awareness = 1.
    self.driver_distracted = False
    self.driver_distraction_filter = FirstOrderFilter(0., _DISTRACTED_FILTER_TS, DT_DMON)
    self.variance_high = False
    self.variance_filter = FirstOrderFilter(0., _VARIANCE_FILTER_TS, DT_DMON)
    self.ts_last_check = 0.
    self.face_detected = False
    self.terminal_alert_cnt = 0
    self.step_change = 0.
    self._set_timers(self.monitor_on)

  def _reset_filters(self):
    self.driver_distraction_filter.x = 0.
    self.variance_filter.x = 0.
    self.monitor_valid = True

  def _set_timers(self, active_monitoring):
    if active_monitoring:
      # when falling back from passive mode to active mode, reset awareness to avoid false alert
      if self.step_change == DT_CTRL / _AWARENESS_TIME:
        self.awareness = 1.
      self.threshold_pre = _DISTRACTED_PRE_TIME_TILL_TERMINAL / _DISTRACTED_TIME
      self.threshold_prompt = _DISTRACTED_PROMPT_TIME_TILL_TERMINAL / _DISTRACTED_TIME
      self.step_change = DT_CTRL / _DISTRACTED_TIME
    else:
      self.threshold_pre = _AWARENESS_PRE_TIME_TILL_TERMINAL / _AWARENESS_TIME
      self.threshold_prompt = _AWARENESS_PROMPT_TIME_TILL_TERMINAL / _AWARENESS_TIME
      self.step_change = DT_CTRL / _AWARENESS_TIME

  def _is_driver_distracted(self, pose, blink):
    # TODO: natural pose calib of each driver
    pitch_error = pose.pitch - _PITCH_NATURAL_OFFSET
    yaw_error = pose.yaw - _YAW_NATURAL_OFFSET
    # add positive pitch allowance
    if pitch_error > 0.:
      pitch_error = max(pitch_error - _PITCH_POS_ALLOWANCE, 0.)
    pitch_error *= _PITCH_WEIGHT
    pose_metric = np.sqrt(yaw_error**2 + pitch_error**2)

    if pose_metric > _METRIC_THRESHOLD:
      return DistractedType.BAD_POSE 
    elif blink.left_blink>_BLINK_THRESHOLD and blink.right_blink>_BLINK_THRESHOLD:
      return DistractedType.BAD_BLINK
    else:
      return DistractedType.NOT_DISTRACTED


  def get_pose(self, driver_monitoring, params, cal_rpy):
    if len(driver_monitoring.faceOrientation) == 0 or len(driver_monitoring.facePosition) == 0:
      return

    self.pose.roll, self.pose.pitch, self.pose.yaw = head_orientation_from_descriptor(driver_monitoring.faceOrientation, driver_monitoring.facePosition, cal_rpy)
    self.blink.left_blink = driver_monitoring.leftBlinkProb * (driver_monitoring.leftEyeProb>_EYE_THRESHOLD)
    self.blink.right_blink = driver_monitoring.rightBlinkProb * (driver_monitoring.rightEyeProb>_EYE_THRESHOLD)
    self.face_detected = driver_monitoring.faceProb > _FACE_THRESHOLD

    self.driver_distracted = self._is_driver_distracted(self.pose, self.blink)>0
    # first order filters
    self.driver_distraction_filter.update(self.driver_distracted)

    monitor_param_on_prev = self.monitor_param_on

    # don't check for param too often as it's a kernel call
    ts = sec_since_boot()
    if ts - self.ts_last_check > 1.:
      self.monitor_param_on = params.get("IsDriverMonitoringEnabled") == "1"
      self.ts_last_check = ts

    self.monitor_on = self.monitor_valid and self.monitor_param_on
    if monitor_param_on_prev != self.monitor_param_on:
      self._reset_filters()
    self._set_timers(self.monitor_on and self.face_detected)


  def update(self, events, driver_engaged, ctrl_active, standstill):
    if driver_engaged:
      self.awareness = 1.
      return events

    driver_engaged |= (self.driver_distraction_filter.x < 0.37 and self.monitor_on)
    awareness_prev = self.awareness

    if (driver_engaged and self.awareness > 0) or not ctrl_active:
      # always reset if driver is in control (unless we are in red alert state) or op isn't active
      self.awareness = min(self.awareness + (2.75*(1.-self.awareness)+1.25)*self.step_change, 1.)

    # should always be counting if distracted unless at standstill and reaching orange
    if ((not self.monitor_on or (self.monitor_on and not self.face_detected)) or (self.driver_distraction_filter.x > 0.63 and self.driver_distracted and self.face_detected)) and \
       not (standstill and self.awareness - self.step_change <= self.threshold_prompt):
      self.awareness = max(self.awareness - self.step_change, -0.1)

    alert = None
    if self.awareness < 0.:
      # terminal red alert: disengagement required
      alert = 'driverDistracted' if self.monitor_on else 'driverUnresponsive'
      if awareness_prev >= 0.:
        self.terminal_alert_cnt += 1
    elif self.awareness <= self.threshold_prompt:
      # prompt orange alert
      alert = 'promptDriverDistracted' if self.monitor_on else 'promptDriverUnresponsive'
    elif self.awareness <= self.threshold_pre:
      # pre green alert
      alert = 'preDriverDistracted' if self.monitor_on else 'preDriverUnresponsive'

    if alert is not None:
      events.append(create_event(alert, [ET.WARNING]))

    return events


if __name__ == "__main__":
  ds = DriverStatus(True)
  ds.driver_distraction_filter.x = 0.
  ds.driver_distracted = 1
  for i in range(10):
    ds.update([], False, True, False)
    print(ds.awareness, ds.driver_distracted, ds.driver_distraction_filter.x)
  ds.update([], True, True, False)
  print(ds.awareness, ds.driver_distracted, ds.driver_distraction_filter.x)
