from math import atan2

from cereal import car
from common.numpy_fast import interp
from common.realtime import DT_DMON
from selfdrive.hardware import TICI
from common.filter_simple import FirstOrderFilter
from common.stat_live import RunningStatFilter

EventName = car.CarEvent.EventName

# ******************************************************************************************
#  NOTE: To fork maintainers.
#  Disabling or nerfing safety features will get you and your users banned from our servers.
#  We recommend that you do not change these numbers from the defaults.
# ******************************************************************************************

class DRIVER_MONITOR_SETTINGS():
  def __init__(self, TICI=TICI, DT_DMON=DT_DMON):
    self._DT_DMON = DT_DMON
    self._AWARENESS_TIME = 35. # passive wheeltouch total timeout
    self._AWARENESS_PRE_TIME_TILL_TERMINAL = 12.
    self._AWARENESS_PROMPT_TIME_TILL_TERMINAL = 6.
    self._DISTRACTED_TIME = 11. # active monitoring total timeout
    self._DISTRACTED_PRE_TIME_TILL_TERMINAL = 8.
    self._DISTRACTED_PROMPT_TIME_TILL_TERMINAL = 6.

    self._FACE_THRESHOLD = 0.5
    self._PARTIAL_FACE_THRESHOLD = 0.765 if TICI else 0.455
    self._EYE_THRESHOLD = 0.75 if TICI else 0.55
    self._SG_THRESHOLD = 0.888 if TICI else 0.85
    self._BLINK_THRESHOLD = 0.811 if TICI else 0.62
    self._BLINK_THRESHOLD_SLACK = 0.9 if TICI else 0.8
    self._BLINK_THRESHOLD_STRICT = self._BLINK_THRESHOLD
    self._PITCH_WEIGHT = 1.35  # pitch matters a lot more
    self._POSESTD_THRESHOLD = 0.2 if TICI else 0.175

    self._METRIC_THRESHOLD = 0.48
    self._METRIC_THRESHOLD_SLACK = 0.66
    self._METRIC_THRESHOLD_STRICT = self._METRIC_THRESHOLD
    self._PITCH_NATURAL_OFFSET = 0.02  # people don't seem to look straight when they drive relaxed, rather a bit up
    self._YAW_NATURAL_OFFSET = 0.08  # people don't seem to look straight when they drive relaxed, rather a bit to the right (center of car)

    self._HI_STD_FALLBACK_TIME = int(10  / self._DT_DMON)  # fall back to wheel touch if model is uncertain for 10s
    self._DISTRACTED_FILTER_TS = 0.25  # 0.6Hz

    self._POSE_CALIB_MIN_SPEED = 13  # 30 mph
    self._POSE_OFFSET_MIN_COUNT = int(60 / self._DT_DMON)  # valid data counts before calibration completes, 1min cumulative
    self._POSE_OFFSET_MAX_COUNT = int(360 / self._DT_DMON)  # stop deweighting new data after 6 min, aka "short term memory"

    self._RECOVERY_FACTOR_MAX = 5.  # relative to minus step change
    self._RECOVERY_FACTOR_MIN = 1.25  # relative to minus step change

    self._MAX_TERMINAL_ALERTS = 3  # not allowed to engage after 3 terminal alerts
    self._MAX_TERMINAL_DURATION = int(30 / self._DT_DMON)  # not allowed to engage after 30s of terminal alerts


# model output refers to center of cropped image, so need to apply the x displacement offset
RESIZED_FOCAL = 320.0
H, W, FULL_W = 320, 160, 426

class DistractedType:
  NOT_DISTRACTED = 0
  BAD_POSE = 1
  BAD_BLINK = 2

def face_orientation_from_net(angles_desc, pos_desc, rpy_calib, is_rhd):
  # the output of these angles are in device frame
  # so from driver's perspective, pitch is up and yaw is right

  pitch_net, yaw_net, roll_net = angles_desc

  face_pixel_position = ((pos_desc[0] + .5)*W - W + FULL_W, (pos_desc[1]+.5)*H)
  yaw_focal_angle = atan2(face_pixel_position[0] - FULL_W//2, RESIZED_FOCAL)
  pitch_focal_angle = atan2(face_pixel_position[1] - H//2, RESIZED_FOCAL)

  pitch = pitch_net + pitch_focal_angle
  yaw = -yaw_net + yaw_focal_angle

  # no calib for roll
  pitch -= rpy_calib[1]
  yaw -= rpy_calib[2] * (1 - 2 * int(is_rhd))  # lhd -> -=, rhd -> +=
  return roll_net, pitch, yaw

class DriverPose():
  def __init__(self, max_trackable):
    self.yaw = 0.
    self.pitch = 0.
    self.roll = 0.
    self.yaw_std = 0.
    self.pitch_std = 0.
    self.roll_std = 0.
    self.pitch_offseter = RunningStatFilter(max_trackable=max_trackable)
    self.yaw_offseter = RunningStatFilter(max_trackable=max_trackable)
    self.low_std = True
    self.cfactor = 1.

class DriverBlink():
  def __init__(self):
    self.left_blink = 0.
    self.right_blink = 0.
    self.cfactor = 1.

class DriverStatus():
  def __init__(self, rhd=False, settings=DRIVER_MONITOR_SETTINGS()):
    # init policy settings
    self.settings = settings

    # init driver status
    self.is_rhd_region = rhd
    self.pose = DriverPose(self.settings._POSE_OFFSET_MAX_COUNT)
    self.pose_calibrated = False
    self.blink = DriverBlink()
    self.awareness = 1.
    self.awareness_active = 1.
    self.awareness_passive = 1.
    self.driver_distracted = False
    self.driver_distraction_filter = FirstOrderFilter(0., self.settings._DISTRACTED_FILTER_TS, self.settings._DT_DMON)
    self.face_detected = False
    self.face_partial = False
    self.terminal_alert_cnt = 0
    self.terminal_time = 0
    self.step_change = 0.
    self.active_monitoring_mode = True
    self.is_model_uncertain = False
    self.hi_stds = 0
    self.threshold_pre = self.settings._DISTRACTED_PRE_TIME_TILL_TERMINAL / self.settings._DISTRACTED_TIME
    self.threshold_prompt = self.settings._DISTRACTED_PROMPT_TIME_TILL_TERMINAL / self.settings._DISTRACTED_TIME

    self._set_timers(active_monitoring=True)

  def _set_timers(self, active_monitoring):
    if self.active_monitoring_mode and self.awareness <= self.threshold_prompt:
      if active_monitoring:
        self.step_change = self.settings._DT_DMON / self.settings._DISTRACTED_TIME
      else:
        self.step_change = 0.
      return  # no exploit after orange alert
    elif self.awareness <= 0.:
      return

    if active_monitoring:
      # when falling back from passive mode to active mode, reset awareness to avoid false alert
      if not self.active_monitoring_mode:
        self.awareness_passive = self.awareness
        self.awareness = self.awareness_active

      self.threshold_pre = self.settings._DISTRACTED_PRE_TIME_TILL_TERMINAL / self.settings._DISTRACTED_TIME
      self.threshold_prompt = self.settings._DISTRACTED_PROMPT_TIME_TILL_TERMINAL / self.settings._DISTRACTED_TIME
      self.step_change = self.settings._DT_DMON / self.settings._DISTRACTED_TIME
      self.active_monitoring_mode = True
    else:
      if self.active_monitoring_mode:
        self.awareness_active = self.awareness
        self.awareness = self.awareness_passive

      self.threshold_pre = self.settings._AWARENESS_PRE_TIME_TILL_TERMINAL / self.settings._AWARENESS_TIME
      self.threshold_prompt = self.settings._AWARENESS_PROMPT_TIME_TILL_TERMINAL / self.settings._AWARENESS_TIME
      self.step_change = self.settings._DT_DMON / self.settings._AWARENESS_TIME
      self.active_monitoring_mode = False

  def _is_driver_distracted(self, pose, blink):
    if not self.pose_calibrated:
      pitch_error = pose.pitch - self.settings._PITCH_NATURAL_OFFSET
      yaw_error = pose.yaw - self.settings._YAW_NATURAL_OFFSET
    else:
      pitch_error = pose.pitch - self.pose.pitch_offseter.filtered_stat.mean()
      yaw_error = pose.yaw - self.pose.yaw_offseter.filtered_stat.mean()

    pitch_error = 0 if pitch_error > 0 else abs(pitch_error) # no positive pitch limit
    yaw_error = abs(yaw_error)

    if pitch_error*self.settings._PITCH_WEIGHT > self.settings._METRIC_THRESHOLD*pose.cfactor or \
       yaw_error > self.settings._METRIC_THRESHOLD*pose.cfactor:
      return DistractedType.BAD_POSE
    elif (blink.left_blink + blink.right_blink)*0.5 > self.settings._BLINK_THRESHOLD*blink.cfactor:
      return DistractedType.BAD_BLINK
    else:
      return DistractedType.NOT_DISTRACTED

  def set_policy(self, model_data):
    ep = min(model_data.meta.engagedProb, 0.8) / 0.8
    self.pose.cfactor = interp(ep, [0, 0.5, 1],
                                           [self.settings._METRIC_THRESHOLD_STRICT,
                                            self.settings. _METRIC_THRESHOLD,
                                            self.settings._METRIC_THRESHOLD_SLACK]) / self.settings._METRIC_THRESHOLD
    self.blink.cfactor = interp(ep, [0, 0.5, 1],
                                           [self.settings._BLINK_THRESHOLD_STRICT,
                                            self.settings._BLINK_THRESHOLD,
                                            self.settings._BLINK_THRESHOLD_SLACK]) / self.settings._BLINK_THRESHOLD

  def get_pose(self, driver_state, cal_rpy, car_speed, op_engaged):
    if not all(len(x) > 0 for x in [driver_state.faceOrientation, driver_state.facePosition,
                                    driver_state.faceOrientationStd, driver_state.facePositionStd]):
      return

    self.face_partial = driver_state.partialFace > self.settings._PARTIAL_FACE_THRESHOLD
    self.face_detected = driver_state.faceProb > self.settings._FACE_THRESHOLD or self.face_partial
    self.pose.roll, self.pose.pitch, self.pose.yaw = face_orientation_from_net(driver_state.faceOrientation, driver_state.facePosition, cal_rpy, self.is_rhd_region)
    self.pose.pitch_std = driver_state.faceOrientationStd[0]
    self.pose.yaw_std = driver_state.faceOrientationStd[1]
    # self.pose.roll_std = driver_state.faceOrientationStd[2]
    model_std_max = max(self.pose.pitch_std, self.pose.yaw_std)
    self.pose.low_std = model_std_max < self.settings._POSESTD_THRESHOLD and not self.face_partial
    self.blink.left_blink = driver_state.leftBlinkProb * (driver_state.leftEyeProb > self.settings._EYE_THRESHOLD) * (driver_state.sunglassesProb < self.settings._SG_THRESHOLD)
    self.blink.right_blink = driver_state.rightBlinkProb * (driver_state.rightEyeProb > self.settings._EYE_THRESHOLD) * (driver_state.sunglassesProb < self.settings._SG_THRESHOLD)

    self.driver_distracted = self._is_driver_distracted(self.pose, self.blink) > 0 and \
                                   driver_state.faceProb > self.settings._FACE_THRESHOLD and self.pose.low_std
    self.driver_distraction_filter.update(self.driver_distracted)

    # update offseter
    # only update when driver is actively driving the car above a certain speed
    if self.face_detected and car_speed > self.settings._POSE_CALIB_MIN_SPEED and self.pose.low_std and (not op_engaged or not self.driver_distracted):
      self.pose.pitch_offseter.push_and_update(self.pose.pitch)
      self.pose.yaw_offseter.push_and_update(self.pose.yaw)

    self.pose_calibrated = self.pose.pitch_offseter.filtered_stat.n > self.settings._POSE_OFFSET_MIN_COUNT and \
                                       self.pose.yaw_offseter.filtered_stat.n > self.settings._POSE_OFFSET_MIN_COUNT

    self.is_model_uncertain = self.hi_stds > self.settings._HI_STD_FALLBACK_TIME
    self._set_timers(self.face_detected and not self.is_model_uncertain)
    if self.face_detected and not self.pose.low_std and not self.driver_distracted:
      self.hi_stds += 1
    elif self.face_detected and self.pose.low_std:
      self.hi_stds = 0

  def update(self, events, driver_engaged, ctrl_active, standstill):
    if (driver_engaged and self.awareness > 0) or not ctrl_active:
      # reset only when on disengagement if red reached
      self.awareness = 1.
      self.awareness_active = 1.
      self.awareness_passive = 1.
      return

    driver_attentive = self.driver_distraction_filter.x < 0.37
    awareness_prev = self.awareness

    if (driver_attentive and self.face_detected and self.pose.low_std and self.awareness > 0):
      # only restore awareness when paying attention and alert is not red
      self.awareness = min(self.awareness + ((self.settings._RECOVERY_FACTOR_MAX-self.settings._RECOVERY_FACTOR_MIN)*(1.-self.awareness)+self.settings._RECOVERY_FACTOR_MIN)*self.step_change, 1.)
      if self.awareness == 1.:
        self.awareness_passive = min(self.awareness_passive + self.step_change, 1.)
      # don't display alert banner when awareness is recovering and has cleared orange
      if self.awareness > self.threshold_prompt:
        return

    standstill_exemption = standstill and self.awareness - self.step_change <= self.threshold_prompt
    certainly_distracted = self.driver_distraction_filter.x > 0.63 and self.driver_distracted and self.face_detected
    maybe_distracted = self.hi_stds > self.settings._HI_STD_FALLBACK_TIME or not self.face_detected
    if certainly_distracted or maybe_distracted:
      # should always be counting if distracted unless at standstill and reaching orange
      if not standstill_exemption:
        self.awareness = max(self.awareness - self.step_change, -0.1)

    alert = None
    if self.awareness <= 0.:
      # terminal red alert: disengagement required
      alert = EventName.driverDistracted if self.active_monitoring_mode else EventName.driverUnresponsive
      self.terminal_time += 1
      if awareness_prev > 0.:
        self.terminal_alert_cnt += 1
    elif self.awareness <= self.threshold_prompt:
      # prompt orange alert
      alert = EventName.promptDriverDistracted if self.active_monitoring_mode else EventName.promptDriverUnresponsive
    elif self.awareness <= self.threshold_pre:
      # pre green alert
      alert = EventName.preDriverDistracted if self.active_monitoring_mode else EventName.preDriverUnresponsive

    if alert is not None:
      events.add(alert)
