from collections import defaultdict
from math import atan2, radians
import numpy as np

from cereal import car, log
import cereal.messaging as messaging
from openpilot.common.realtime import DT_DMON
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.params import Params
from openpilot.common.stat_live import RunningStatFilter
from openpilot.common.transformations.camera import DEVICE_CAMERAS

AlertLevel = log.DriverMonitoringState.AlertLevel
MonitoringPolicy = log.DriverMonitoringState.MonitoringPolicy

def to_percent(v):
  return int(min(max(v * 100., 0.), 100.))

# ******************************************************************************************
#  NOTE: To fork maintainers.
#  Disabling or nerfing safety features will get you and your users banned from our servers.
#  We recommend that you do not change these numbers from the defaults.
# ******************************************************************************************

class DRIVER_MONITOR_SETTINGS:
  def __init__(self):
    # https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:42018X1947&rid=2
    self._WHEELTOUCH_POLICY_ALERT_1_TIMEOUT = 15.
    self._WHEELTOUCH_POLICY_ALERT_2_TIMEOUT = 24.
    self._WHEELTOUCH_POLICY_ALERT_3_TIMEOUT = 30.
    # https://cdn.euroncap.com/cars/assets/euro_ncap_protocol_safe_driving_driver_engagement_v11_a30e874152.pdf
    self._VISION_POLICY_ALERT_1_TIMEOUT = 3.
    self._VISION_POLICY_ALERT_2_TIMEOUT = 5.
    self._VISION_POLICY_ALERT_3_TIMEOUT = 11.

    self._TIMEOUT_RECOVERY_FACTOR_MAX = 5.
    self._TIMEOUT_RECOVERY_FACTOR_MIN = 1.25

    self._MAX_TERMINAL_ALERTS = 3  # not allowed to engage after 3 terminal alerts
    self._MAX_TERMINAL_DURATION = int(30 / DT_DMON)  # not allowed to engage after 30s of terminal alerts

    self._FACE_THRESHOLD = 0.7
    self._EYE_THRESHOLD = 0.65
    self._SG_THRESHOLD = 0.9
    self._BLINK_THRESHOLD = 0.865
    self._PHONE_THRESH = 0.5
    self._POSE_PITCH_THRESHOLD = 0.3133
    self._POSE_PITCH_THRESHOLD_SLACK = 0.3237
    self._POSE_PITCH_THRESHOLD_STRICT = self._POSE_PITCH_THRESHOLD
    self._POSE_YAW_THRESHOLD = 0.4020
    self._POSE_YAW_THRESHOLD_SLACK = 0.5042
    self._POSE_YAW_THRESHOLD_STRICT = self._POSE_YAW_THRESHOLD
    self._POSE_YAW_MIN_STEER_DEG = 30
    self._POSE_YAW_STEER_FACTOR = 0.15
    self._POSE_YAW_STEER_MAX_OFFSET = 0.3927
    self._PITCH_NATURAL_OFFSET = 0.011 # initial value before offset is learned
    self._PITCH_NATURAL_THRESHOLD = 0.449
    self._YAW_NATURAL_OFFSET = 0.075 # initial value before offset is learned
    self._PITCH_NATURAL_VAR = 3*0.01
    self._YAW_NATURAL_VAR = 3*0.05
    self._PITCH_MAX_OFFSET = 0.124
    self._PITCH_MIN_OFFSET = -0.0881
    self._YAW_MAX_OFFSET = 0.289
    self._YAW_MIN_OFFSET = -0.0246

    self._DCAM_UNCERTAIN_ALERT_THRESHOLD = 0.1
    self._DCAM_UNCERTAIN_ALERT_COUNT = int(60  / DT_DMON)
    self._DCAM_UNCERTAIN_RESET_COUNT = int(20  / DT_DMON)
    self._HI_STD_THRESHOLD = 0.3
    self._HI_STD_FALLBACK_TIME = int(10  / DT_DMON)  # fall back to wheel touch if model is uncertain for 10s
    self._DISTRACTED_FILTER_TS = 0.25  # 0.6Hz

    self._POSE_CALIB_MIN_SPEED = 13  # 30 mph
    self._POSE_OFFSET_MIN_COUNT = int(60 / DT_DMON)  # valid data counts before calibration completes, 1min cumulative
    self._POSE_OFFSET_MAX_COUNT = int(360 / DT_DMON)  # stop deweighting new data after 6 min, aka "short term memory"
    self._WHEELPOS_CALIB_MIN_SPEED = 11
    self._WHEELPOS_THRESHOLD = 0.5
    self._WHEELPOS_FILTER_MIN_COUNT = int(15 / DT_DMON) # allow 15 seconds to converge wheel side
    self._WHEELPOS_DATA_AVG = 0.03
    self._WHEELPOS_DATA_VAR = 3*5.5e-5
    self._WHEELPOS_MAX_COUNT = -1

class DriverPose:
  def __init__(self, settings):
    pitch_filter_raw_priors = (settings._PITCH_NATURAL_OFFSET, settings._PITCH_NATURAL_VAR, 2)
    yaw_filter_raw_priors = (settings._YAW_NATURAL_OFFSET, settings._YAW_NATURAL_VAR, 2)
    self.yaw = 0.
    self.pitch = 0.
    self.pitch_offsetter = RunningStatFilter(raw_priors=pitch_filter_raw_priors, max_trackable=settings._POSE_OFFSET_MAX_COUNT)
    self.yaw_offsetter = RunningStatFilter(raw_priors=yaw_filter_raw_priors, max_trackable=settings._POSE_OFFSET_MAX_COUNT)
    self.calibrated = False
    self.low_std = True
    self.cfactor_pitch = 1.
    self.cfactor_yaw = 1.
    self.steer_yaw_offset = 0.

class DriverBlink:
  def __init__(self):
    self.left = 0.
    self.right = 0.

# model output refers to center of undistorted+leveled image
ref_undistorted_cam = DEVICE_CAMERAS[("tici", "ar0231")].dcam
dcam_undistorted_FL = 598.0
dcam_undistorted_W, dcam_undistorted_H = (ref_undistorted_cam.width, ref_undistorted_cam.height)

def face_orientation_from_model(orient_model, pos_model, rpy_calib):
  pitch_model = orient_model[0]
  yaw_model = orient_model[1]

  face_pixel_position = ((pos_model[0]+0.5)*dcam_undistorted_W, (pos_model[1]+0.5)*dcam_undistorted_H)
  yaw_focal_angle = atan2(face_pixel_position[0] - dcam_undistorted_W//2, dcam_undistorted_FL)
  pitch_focal_angle = atan2(face_pixel_position[1] - dcam_undistorted_H//2, dcam_undistorted_FL)

  pitch = pitch_model + pitch_focal_angle
  yaw = -yaw_model + yaw_focal_angle

  pitch -= rpy_calib[1]
  yaw -= rpy_calib[2]
  return pitch, yaw


class DriverMonitoring:
  def __init__(self, rhd_saved=False, settings=None, always_on=False):
    # init policy settings
    self.settings = settings if settings is not None else DRIVER_MONITOR_SETTINGS()

    # init driver status
    wheelpos_filter_raw_priors = (self.settings._WHEELPOS_DATA_AVG, self.settings._WHEELPOS_DATA_VAR, 2)
    self.wheelpos_offsetter = RunningStatFilter(raw_priors=wheelpos_filter_raw_priors, max_trackable=self.settings._WHEELPOS_MAX_COUNT)
    self.pose = DriverPose(settings=self.settings)
    self.blink = DriverBlink()
    self.phone_prob = 0.

    self.alert_level = AlertLevel.none
    self.always_on = always_on
    self.distracted_types = defaultdict(bool)
    self.driver_distracted = False
    self.driver_distraction_filter = FirstOrderFilter(0., self.settings._DISTRACTED_FILTER_TS, DT_DMON)
    self.wheel_on_right = False
    self.wheel_on_right_last = None
    self.wheel_on_right_default = rhd_saved
    self.face_detected = False
    self.terminal_alert_cnt = 0
    self.terminal_time = 0
    self.step_change = 0.
    self.active_policy = MonitoringPolicy.vision
    self.driver_interacting = False
    self.is_model_uncertain = False
    self.hi_stds = 0
    self.model_std_max = 0.
    self.threshold_alert_1 = 0.
    self.threshold_alert_2 = 0.
    self.dcam_uncertain_cnt = 0
    self.dcam_reset_cnt = 0
    self.too_distracted = Params().get_bool("DriverTooDistracted")

    self._reset_awareness()
    self._set_policy(MonitoringPolicy.vision)

  def _reset_awareness(self):
    self.awareness = 1.
    self.last_vision_awareness = 1.
    self.last_wheeltouch_awareness = 1.

  def _set_policy(self, target_policy):
    if self.active_policy == MonitoringPolicy.vision and self.awareness <= self.threshold_alert_2:
      if target_policy == MonitoringPolicy.vision:
        self.step_change = DT_DMON / self.settings._VISION_POLICY_ALERT_3_TIMEOUT
      else:
        self.step_change = 0.
      return  # no exploit after orange alert
    elif self.awareness <= 0.:
      return

    if target_policy == MonitoringPolicy.vision:
      # when falling back from passive mode to active mode, reset awareness to avoid false alert
      if self.active_policy != MonitoringPolicy.vision:
        self.last_wheeltouch_awareness = self.awareness
        self.awareness = self.last_vision_awareness

      self.threshold_alert_1 = 1. - self.settings._VISION_POLICY_ALERT_1_TIMEOUT / self.settings._VISION_POLICY_ALERT_3_TIMEOUT
      self.threshold_alert_2 = 1. - self.settings._VISION_POLICY_ALERT_2_TIMEOUT / self.settings._VISION_POLICY_ALERT_3_TIMEOUT
      self.step_change = DT_DMON / self.settings._VISION_POLICY_ALERT_3_TIMEOUT
      self.active_policy = MonitoringPolicy.vision
    else:
      if self.active_policy == MonitoringPolicy.vision:
        self.last_vision_awareness = self.awareness
        self.awareness = self.last_wheeltouch_awareness

      self.threshold_alert_1 = 1. - self.settings._WHEELTOUCH_POLICY_ALERT_1_TIMEOUT / self.settings._WHEELTOUCH_POLICY_ALERT_3_TIMEOUT
      self.threshold_alert_2 = 1. - self.settings._WHEELTOUCH_POLICY_ALERT_2_TIMEOUT / self.settings._WHEELTOUCH_POLICY_ALERT_3_TIMEOUT
      self.step_change = DT_DMON / self.settings._WHEELTOUCH_POLICY_ALERT_3_TIMEOUT
      self.active_policy = MonitoringPolicy.wheeltouch

  def _set_pose_strictness(self, brake_disengage_prob, car_speed):
    bp = brake_disengage_prob
    k1 = max(-0.00156*((car_speed-16)**2)+0.6, 0.2)
    bp_normal = max(min(bp / k1, 0.5),0)
    self.pose.cfactor_pitch = np.interp(bp_normal, [0, 0.5],
                                           [self.settings._POSE_PITCH_THRESHOLD_SLACK,
                                            self.settings._POSE_PITCH_THRESHOLD_STRICT]) / self.settings._POSE_PITCH_THRESHOLD
    self.pose.cfactor_yaw = np.interp(bp_normal, [0, 0.5],
                                           [self.settings._POSE_YAW_THRESHOLD_SLACK,
                                            self.settings._POSE_YAW_THRESHOLD_STRICT]) / self.settings._POSE_YAW_THRESHOLD

  def _get_distracted_types(self):
    self.distracted_types = defaultdict(bool)

    if not self.pose.calibrated:
      pitch_error = self.pose.pitch - self.settings._PITCH_NATURAL_OFFSET
      yaw_error = self.pose.yaw - self.settings._YAW_NATURAL_OFFSET
    else:
      pitch_error = self.pose.pitch - min(max(self.pose.pitch_offsetter.filtered_stat.mean(),
                                                       self.settings._PITCH_MIN_OFFSET), self.settings._PITCH_MAX_OFFSET)
      yaw_error = self.pose.yaw - min(max(self.pose.yaw_offsetter.filtered_stat.mean(),
                                                    self.settings._YAW_MIN_OFFSET), self.settings._YAW_MAX_OFFSET)
    pitch_error = 0 if pitch_error > 0 else abs(pitch_error) # no positive pitch limit

    if yaw_error * self.pose.steer_yaw_offset > 0: # unidirectional
      yaw_error = max(abs(yaw_error) - min(abs(self.pose.steer_yaw_offset), self.settings._POSE_YAW_STEER_MAX_OFFSET), 0.)
    else:
      yaw_error = abs(yaw_error)

    pitch_threshold = self.settings._POSE_PITCH_THRESHOLD * self.pose.cfactor_pitch if self.pose.calibrated else self.settings._PITCH_NATURAL_THRESHOLD
    yaw_threshold = self.settings._POSE_YAW_THRESHOLD * self.pose.cfactor_yaw

    self.distracted_types['pose'] = bool((pitch_error > pitch_threshold) or (yaw_error > yaw_threshold))
    self.distracted_types['eye'] = bool((self.blink.left + self.blink.right)*0.5 > self.settings._BLINK_THRESHOLD)
    self.distracted_types['phone'] = bool(self.phone_prob > self.settings._PHONE_THRESH)

  def _update_states(self, driver_state, cal_rpy, car_speed, op_engaged, standstill, demo_mode=False, steering_angle_deg=0.):
    rhd_pred = driver_state.wheelOnRightProb
    # calibrates only when there's movement and either face detected
    if car_speed > self.settings._WHEELPOS_CALIB_MIN_SPEED and (driver_state.leftDriverData.faceProb > self.settings._FACE_THRESHOLD or
                                          driver_state.rightDriverData.faceProb > self.settings._FACE_THRESHOLD):
      self.wheelpos_offsetter.push_and_update(rhd_pred)

    wheelpos_calibrated = self.wheelpos_offsetter.filtered_stat.n >= self.settings._WHEELPOS_FILTER_MIN_COUNT

    if wheelpos_calibrated or demo_mode:
      self.wheel_on_right = self.wheelpos_offsetter.filtered_stat.M > self.settings._WHEELPOS_THRESHOLD
    else:
      self.wheel_on_right = self.wheel_on_right_default # use default/saved if calibration is unfinished
    # make sure no switching when engaged
    if op_engaged and self.wheel_on_right_last is not None and self.wheel_on_right_last != self.wheel_on_right and not demo_mode:
      self.wheel_on_right = self.wheel_on_right_last
    driver_data = driver_state.rightDriverData if self.wheel_on_right else driver_state.leftDriverData
    if not all(len(x) > 0 for x in (driver_data.faceOrientation, driver_data.facePosition,
                                    driver_data.faceOrientationStd, driver_data.facePositionStd)):
      return

    self.face_detected = driver_data.faceProb > self.settings._FACE_THRESHOLD
    self.pose.pitch, self.pose.yaw = face_orientation_from_model(driver_data.faceOrientation, driver_data.facePosition, cal_rpy)
    steer_d = max(abs(steering_angle_deg) - self.settings._POSE_YAW_MIN_STEER_DEG, 0.)
    self.pose.steer_yaw_offset = radians(steer_d) * -np.sign(steering_angle_deg) * self.settings._POSE_YAW_STEER_FACTOR
    if self.wheel_on_right:
      self.pose.yaw *= -1
      self.pose.steer_yaw_offset *= -1
    self.wheel_on_right_last = self.wheel_on_right
    self.model_std_max = max(driver_data.faceOrientationStd[0], driver_data.faceOrientationStd[1])
    self.pose.low_std = self.model_std_max < self.settings._HI_STD_THRESHOLD
    self.blink.left = driver_data.leftBlinkProb * (driver_data.leftEyeProb > self.settings._EYE_THRESHOLD) \
                      * (driver_data.sunglassesProb < self.settings._SG_THRESHOLD)
    self.blink.right = driver_data.rightBlinkProb * (driver_data.rightEyeProb > self.settings._EYE_THRESHOLD) \
                      * (driver_data.sunglassesProb < self.settings._SG_THRESHOLD)
    self.phone_prob = driver_data.phoneProb

    self._get_distracted_types()
    self.driver_distracted = any(self.distracted_types.values()) and driver_data.faceProb > self.settings._FACE_THRESHOLD and self.pose.low_std
    self.driver_distraction_filter.update(self.driver_distracted)

    # only update offsetter when driver is actively driving the car above a certain speed
    if self.face_detected and car_speed > self.settings._POSE_CALIB_MIN_SPEED and self.pose.low_std and (not op_engaged or not self.driver_distracted):
      self.pose.pitch_offsetter.push_and_update(self.pose.pitch)
      self.pose.yaw_offsetter.push_and_update(self.pose.yaw)

    self.pose.calibrated = self.pose.pitch_offsetter.filtered_stat.n >= self.settings._POSE_OFFSET_MIN_COUNT and \
                           self.pose.yaw_offsetter.filtered_stat.n >= self.settings._POSE_OFFSET_MIN_COUNT

    if self.face_detected and not self.driver_distracted:
      dcam_uncertain = self.model_std_max > self.settings._DCAM_UNCERTAIN_ALERT_THRESHOLD
      if dcam_uncertain and not standstill:
        self.dcam_uncertain_cnt += 1
        self.dcam_reset_cnt = 0
      else:
        self.dcam_reset_cnt += 1
        if self.dcam_reset_cnt > self.settings._DCAM_UNCERTAIN_RESET_COUNT:
          self.dcam_uncertain_cnt = 0

    self.is_model_uncertain = self.hi_stds >= self.settings._HI_STD_FALLBACK_TIME
    self._set_policy(MonitoringPolicy.vision if self.face_detected and not self.is_model_uncertain else MonitoringPolicy.wheeltouch)
    if self.face_detected and not self.pose.low_std and not self.driver_distracted:
      self.hi_stds += 1
    elif self.face_detected and self.pose.low_std:
      self.hi_stds = 0

  def _update_events(self, driver_engaged, op_engaged, standstill, wrong_gear):
    self.alert_level = AlertLevel.none
    self.driver_interacting = driver_engaged

    if self.terminal_alert_cnt >= self.settings._MAX_TERMINAL_ALERTS or \
       self.terminal_time >= self.settings._MAX_TERMINAL_DURATION:
      self.too_distracted = True

    always_on_valid = self.always_on and not wrong_gear
    if (self.driver_interacting and self.awareness > 0 and self.active_policy == MonitoringPolicy.wheeltouch) or \
       (not always_on_valid and not op_engaged) or \
       (always_on_valid and not op_engaged and self.awareness <= 0):
      # always reset on disengage with normal mode; disengage resets only on red if always on
      self._reset_awareness()
      return

    awareness_prev = self.awareness
    _reaching_alert_1 = self.awareness - self.step_change <= self.threshold_alert_1
    _reaching_alert_3 = self.awareness - self.step_change <= 0
    standstill_exemption = standstill and _reaching_alert_1
    always_on_exemption = always_on_valid and not op_engaged and _reaching_alert_3

    if self.awareness > 0 and \
       ((self.driver_distraction_filter.x < 0.37 and self.face_detected and self.pose.low_std) or standstill_exemption):
      if self.driver_interacting:
        self._reset_awareness()
        return
      # only restore awareness when paying attention and alert is not red
      self.awareness = min(self.awareness + ((self.settings._TIMEOUT_RECOVERY_FACTOR_MAX-self.settings._TIMEOUT_RECOVERY_FACTOR_MIN)*
                                             (1.-self.awareness)+self.settings._TIMEOUT_RECOVERY_FACTOR_MIN)*self.step_change, 1.)
      if self.awareness == 1.:
        self.last_wheeltouch_awareness = min(self.last_wheeltouch_awareness + self.step_change, 1.)
      # don't display alert banner when awareness is recovering and has cleared orange
      if self.awareness > self.threshold_alert_2:
        return

    certainly_distracted = self.driver_distraction_filter.x > 0.63 and self.driver_distracted and self.face_detected
    maybe_distracted = self.is_model_uncertain or not self.face_detected

    if certainly_distracted or maybe_distracted:
      # should always be counting if distracted unless at standstill and reaching green
      # also will not be reaching 0 if DM is active when not engaged
      if not (standstill_exemption or always_on_exemption):
        self.awareness = max(self.awareness - self.step_change, -0.1)

    if self.awareness <= 0.:
      # terminal alert: disengagement required
      self.alert_level = AlertLevel.three
      self.terminal_time += 1
      if awareness_prev > 0.:
        self.terminal_alert_cnt += 1
    elif self.awareness <= self.threshold_alert_2:
      self.alert_level = AlertLevel.two
    elif self.awareness <= self.threshold_alert_1:
      self.alert_level = AlertLevel.one

  def get_state_packet(self, valid=True):
    # build driverMonitoringState packet
    dat = messaging.new_message('driverMonitoringState', valid=valid)
    dm = dat.driverMonitoringState

    dm.lockout = self.too_distracted
    dm.alertCountLockoutPercent = to_percent(self.terminal_alert_cnt / self.settings._MAX_TERMINAL_ALERTS)
    dm.alertTimeLockoutPercent = to_percent(self.terminal_time / self.settings._MAX_TERMINAL_DURATION)
    dm.alwaysOn = self.always_on
    dm.alwaysOnLockout = self.always_on and self.awareness <= self.threshold_alert_2
    dm.alertLevel = self.alert_level
    dm.activePolicy = self.active_policy
    dm.isRHD = self.wheel_on_right
    dm.rhdCalibration.calibratedPercent = to_percent(self.wheelpos_offsetter.filtered_stat.n / self.settings._WHEELPOS_FILTER_MIN_COUNT)
    dm.rhdCalibration.offset = self.wheelpos_offsetter.filtered_stat.M

    dm.visionPolicyState.awarenessPercent = to_percent(self.last_vision_awareness if self.active_policy != MonitoringPolicy.vision else self.awareness)
    dm.visionPolicyState.awarenessStep = self.step_change if self.active_policy == MonitoringPolicy.vision else 0.
    dm.visionPolicyState.isDistracted = self.driver_distracted
    dm.visionPolicyState.distractedTypes.pose = self.distracted_types['pose']
    dm.visionPolicyState.distractedTypes.eye = self.distracted_types['eye']
    dm.visionPolicyState.distractedTypes.phone = self.distracted_types['phone']
    dm.visionPolicyState.faceDetected = self.face_detected
    dm.visionPolicyState.pose.pitch = self.pose.pitch
    dm.visionPolicyState.pose.yaw = self.pose.yaw
    dm.visionPolicyState.pose.calibrated = self.pose.calibrated
    dm.visionPolicyState.pose.pitchCalib.calibratedPercent = to_percent(self.pose.pitch_offsetter.filtered_stat.n / self.settings._POSE_OFFSET_MIN_COUNT)
    dm.visionPolicyState.pose.pitchCalib.offset = self.pose.pitch_offsetter.filtered_stat.M
    dm.visionPolicyState.pose.yawCalib.calibratedPercent = to_percent(self.pose.yaw_offsetter.filtered_stat.n / self.settings._POSE_OFFSET_MIN_COUNT)
    dm.visionPolicyState.pose.yawCalib.offset = self.pose.yaw_offsetter.filtered_stat.M
    dm.visionPolicyState.pose.uncertainty = self.model_std_max
    dm.visionPolicyState.wheeltouchFallbackPercent = to_percent(self.hi_stds / self.settings._HI_STD_FALLBACK_TIME)
    dm.visionPolicyState.uncertainOffroadAlertPercent = to_percent(self.dcam_uncertain_cnt / self.settings._DCAM_UNCERTAIN_ALERT_COUNT)

    dm.wheeltouchPolicyState.awarenessPercent = to_percent(self.last_wheeltouch_awareness if self.active_policy == MonitoringPolicy.vision else self.awareness)
    dm.wheeltouchPolicyState.awarenessStep = 0. if self.active_policy == MonitoringPolicy.vision else self.step_change
    dm.wheeltouchPolicyState.driverInteracting = self.driver_interacting
    return dat

  def run_step(self, sm, demo=False):
    if demo:
      car_speed = 30
      enabled = True
      wrong_gear = False
      standstill = False
      driver_engaged = False
      brake_disengage_prob = 1.0
      steering_angle_deg = 0.0
      rpyCalib = [0., 0., 0.]
    else:
      car_speed = sm['carState'].vEgo
      enabled = sm['selfdriveState'].enabled
      wrong_gear = sm['carState'].gearShifter not in (car.CarState.GearShifter.drive, car.CarState.GearShifter.low)
      standstill = sm['carState'].standstill
      driver_engaged = sm['carState'].steeringPressed or sm['carState'].gasPressed
      brake_disengage_prob = sm['modelV2'].meta.disengagePredictions.brakeDisengageProbs[0] # brake disengage prob in next 2s
      steering_angle_deg = sm['carState'].steeringAngleDeg
      rpyCalib = sm['liveCalibration'].rpyCalib

    self._set_pose_strictness(
      brake_disengage_prob=brake_disengage_prob,
      car_speed=car_speed,
    )

    # Parse data from dmonitoringmodeld
    self._update_states(
      driver_state=sm['driverStateV2'],
      cal_rpy=rpyCalib,
      car_speed=car_speed,
      op_engaged=enabled,
      standstill=standstill,
      demo_mode=demo,
      steering_angle_deg=steering_angle_deg,
    )

    # Update distraction events
    self._update_events(
      driver_engaged=driver_engaged,
      op_engaged=enabled,
      standstill=standstill,
      wrong_gear=wrong_gear,
    )
