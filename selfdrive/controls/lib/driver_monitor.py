import numpy as np
from common.realtime import sec_since_boot
from selfdrive.controls.lib.drive_helpers import create_event, EventTypes as ET
from common.filter_simple import FirstOrderFilter

_DT = 0.01                  # update runs at 100Hz
_DTM = 0.1                  # DM runs at 10Hz
_AWARENESS_TIME = 1800       # 3 minutes limit without user touching steering wheels make the car enter a terminal status
_AWARENESS_PRE_TIME = 200.   # a first alert is issued 20s before expiration
_AWARENESS_PROMPT_TIME = 50. # a second alert is issued 5s before start decelerating the car
_DISTRACTED_TIME = 70.
_DISTRACTED_PRE_TIME = 40.
_DISTRACTED_PROMPT_TIME = 20.
# measured 1 rad in x FOV. 1152x864 is original image, 160x320 is a right crop for model
_CAMERA_FOV_X = 1.   # rad
_CAMERA_FOV_Y = 0.75 # 4/3 aspect ratio
# model output refers to center of cropped image, so need to apply the x displacement offset
_CAMERA_OFFSET_X = 0.3125   #(1152/2 - 0.5*(160*864/320))/1152
_CAMERA_X_CONV = 0.375      # 160*864/320/1152
_PITCH_WEIGHT = 1.5  # pitch matters a lot more
_METRIC_THRESHOLD = 0.4
_PITCH_POS_ALLOWANCE = 0.08  # rad, to not be too sensitive on positive pitch
_PITCH_NATURAL_OFFSET = 0.1  # people don't seem to look straight when they drive relaxed, rather a bit up
_STD_THRESHOLD = 0.1         # above this standard deviation consider the measurement invalid
_DISTRACTED_FILTER_TS = 0.25 # 0.6Hz
_VARIANCE_FILTER_TS = 20.    # 0.008Hz


class _DriverPose():
  def __init__(self):
    self.yaw = 0.
    self.pitch = 0.
    self.roll = 0.
    self.yaw_offset = 0.
    self.pitch_offset = 0.

def _monitor_hysteresys(variance_level, monitor_valid_prev):
  var_thr = 0.63 if monitor_valid_prev else 0.37
  return variance_level < var_thr

class DriverStatus():
  def __init__(self, monitor_on=False):
    self.pose = _DriverPose()
    self.monitor_on = monitor_on
    self.monitor_param_on = monitor_on
    self.monitor_valid = True   # variance needs to be low
    self.awareness = 1.
    self.driver_distracted = False
    self.driver_distraction_filter = FirstOrderFilter(0., _DISTRACTED_FILTER_TS, _DTM)
    self.variance_high = False
    self.variance_filter = FirstOrderFilter(0., _VARIANCE_FILTER_TS, _DTM)
    self.ts_last_check = 0.
    self._set_timers()

  def _reset_filters(self):
    self.driver_distraction_filter.x = 0.
    self.variance_filter.x = 0.
    self.monitor_valid = True

  def _set_timers(self):
    if self.monitor_on:
      self.threshold_pre = _DISTRACTED_PRE_TIME / _DISTRACTED_TIME
      self.threshold_prompt = _DISTRACTED_PROMPT_TIME / _DISTRACTED_TIME
      self.step_change = _DT / _DISTRACTED_TIME
    else:
      self.threshold_pre = _AWARENESS_PRE_TIME / _AWARENESS_TIME
      self.threshold_prompt = _AWARENESS_PROMPT_TIME / _AWARENESS_TIME
      self.step_change = _DT / _AWARENESS_TIME

  def _is_driver_distracted(self, pose):
    # to be tuned and to learn the driver's normal pose
    yaw_error = pose.yaw - pose.yaw_offset
    pitch_error = pose.pitch - pose.pitch_offset - _PITCH_NATURAL_OFFSET
    # add positive pitch allowance
    if pitch_error > 0.:
      pitch_error = max(pitch_error - _PITCH_POS_ALLOWANCE, 0.)
    pitch_error *= _PITCH_WEIGHT
    metric = np.sqrt(yaw_error**2 + pitch_error**2)
    #print "%02.4f" % np.degrees(pose.pitch), "%02.4f" % np.degrees(pitch_error), "%03.4f" % np.degrees(pose.pitch_offset), metric
    return 1 if metric > _METRIC_THRESHOLD else 0


  def get_pose(self, driver_monitoring, params):

    self.pose.pitch = driver_monitoring.descriptor[0]
    self.pose.yaw = driver_monitoring.descriptor[1]
    self.pose.roll = driver_monitoring.descriptor[2]
    self.pose.yaw_offset = (driver_monitoring.descriptor[3] * _CAMERA_X_CONV + _CAMERA_OFFSET_X) * _CAMERA_FOV_X
    self.pose.pitch_offset = -driver_monitoring.descriptor[4] * _CAMERA_FOV_Y  # positive y is down
    self.driver_distracted = self._is_driver_distracted(self.pose)
    # first order filters
    self.driver_distraction_filter.update(self.driver_distracted)
    self.variance_high = driver_monitoring.std > _STD_THRESHOLD
    self.variance_filter.update(self.variance_high)

    monitor_param_on_prev = self.monitor_param_on
    monitor_valid_prev = self.monitor_valid

    # don't check for param too often as it's a kernel call
    ts = sec_since_boot()
    if ts - self.ts_last_check > 1.:
      self.monitor_param_on = params.get("IsDriverMonitoringEnabled") == "1"
      self.ts_last_check = ts

    self.monitor_valid = _monitor_hysteresys(self.variance_filter.x, monitor_valid_prev)
    self.monitor_on = self.monitor_valid and self.monitor_param_on
    if monitor_param_on_prev != self.monitor_param_on:
      self._reset_filters()
    self._set_timers()


  def update(self, events, driver_engaged, ctrl_active, standstill):

    driver_engaged |= (self.driver_distraction_filter.x < 0.37 and self.monitor_on)

    if (driver_engaged and self.awareness > 0.) or not ctrl_active:
      # always reset if driver is in control (unless we are in red alert state) or op isn't active
      self.awareness = 1.

    if (not self.monitor_on or (self.driver_distraction_filter.x > 0.63 and self.driver_distracted)) and \
       not (standstill and self.awareness - self.step_change <= self.threshold_prompt):
      self.awareness = max(self.awareness - self.step_change, -0.1)

    alert = None
    if self.awareness <= 0.:
      # terminal red alert: disengagement required
      alert = 'driverDistracted' if self.monitor_on else 'driverUnresponsive'
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

