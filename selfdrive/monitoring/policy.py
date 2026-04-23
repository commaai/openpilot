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
DISTRACTION_FILTER_ATTENTIVE_THRESHOLD = 0.37
DISTRACTION_FILTER_DISTRACTED_THRESHOLD = 0.63


def to_percent(v):
  return int(min(max(v * 100., 0.), 100.))


# ******************************************************************************************
#  NOTE: To fork maintainers.
#  Disabling or nerfing safety features will get you and your users banned from our servers.
#  We recommend that you do not change these numbers from the defaults.
# ******************************************************************************************

class DRIVER_MONITOR_SETTINGS:
  def __init__(self):
    self._AWARENESS_RECOVERY_FACTOR_MAX = 5.
    self._AWARENESS_RECOVERY_FACTOR_MIN = 1.25

    self._MAX_TERMINAL_ALERTS = 3  # not allowed to engage after 3 terminal alerts
    self._MAX_TERMINAL_DURATION = int(30 / DT_DMON)  # not allowed to engage after 30s of terminal alerts
    self.vision = VisionPolicy.Settings()
    self.wheeltouch = WheeltouchPolicy.Settings()


class DriverPose:
  def __init__(self, settings):
    pitch_filter_raw_priors = (settings._PITCH_NATURAL_OFFSET, settings._PITCH_NATURAL_VAR, 2)
    yaw_filter_raw_priors = (settings._YAW_NATURAL_OFFSET, settings._YAW_NATURAL_VAR, 2)
    self.yaw = 0.
    self.pitch = 0.
    self.pitch_offsetter = RunningStatFilter(raw_priors=pitch_filter_raw_priors, max_trackable=settings._POSE_OFFSET_MAX_COUNT)
    self.yaw_offsetter = RunningStatFilter(raw_priors=yaw_filter_raw_priors, max_trackable=settings._POSE_OFFSET_MAX_COUNT)
    self.calibrated = False
    self.low_uncertainty = True
    self.cfactor_pitch = 1.
    self.cfactor_yaw = 1.
    self.steer_yaw_offset = 0.


class DriverProb:
  def __init__(self, raw_priors, max_trackable):
    self.prob_offseter = RunningStatFilter(raw_priors=raw_priors, max_trackable=max_trackable)


class DriverMonitoringPolicy:
  """
    Common awareness countdown/recovery logic shared by both DM policies
  """
  def __init__(self, shared_settings, alert_1_interval, alert_2_interval, alert_3_interval):
    self.shared_settings = shared_settings
    self.alert_1_threshold = 1. - alert_1_interval / alert_3_interval
    self.alert_2_threshold = 1. - alert_2_interval / alert_3_interval
    self.awareness_step = DT_DMON / alert_3_interval
    self.reset_awareness()

  def reset_awareness(self):
    self.awareness = 1.

  def _reaching_alert_1(self, step):
    return self.awareness - step <= self.alert_1_threshold

  def _reaching_alert_3(self, step):
    return self.awareness - step <= 0.

  def _recovery_available(self, can_recover, standstill, step):
    return can_recover or (standstill and self._reaching_alert_1(step))

  def _recover_awareness(self, step):
    self.awareness = min(self.awareness + ((self.shared_settings._AWARENESS_RECOVERY_FACTOR_MAX - self.shared_settings._AWARENESS_RECOVERY_FACTOR_MIN) *
                                           (1. - self.awareness) + self.shared_settings._AWARENESS_RECOVERY_FACTOR_MIN) * step, 1.)

  def _decrease_awareness(self, step):
    self.awareness = max(self.awareness - step, -0.1)

  def _update_awareness(self, *, step, can_recover, should_count_down, standstill, always_on_exemption, driver_interacting=False):
    if self.awareness <= 0.:
      return False

    standstill_exemption = standstill and self._reaching_alert_1(step)
    if self.awareness > 0 and self._recovery_available(can_recover, standstill, step):
      if driver_interacting:
        self.reset_awareness()
        return True
      self._recover_awareness(step)
      if self.awareness > self.alert_2_threshold:
        return True

    if should_count_down and not (standstill_exemption or always_on_exemption):
      self._decrease_awareness(step)
    return False

  def get_alert_level(self):
    if self.awareness <= 0.:
      return AlertLevel.three
    if self.awareness <= self.alert_2_threshold:
      return AlertLevel.two
    if self.awareness <= self.alert_1_threshold:
      return AlertLevel.one
    return AlertLevel.none


class VisionPolicy(DriverMonitoringPolicy):
  class Settings:
    def __init__(self):
      # https://cdn.euroncap.com/cars/assets/euro_ncap_protocol_safe_driving_driver_engagement_v11_a30e874152.pdf
      self._ALERT_1_INTERVAL = 3.
      self._ALERT_2_INTERVAL = 5.
      self._ALERT_3_INTERVAL = 11.

      self._FACE_THRESHOLD = 0.7
      self._EYE_THRESHOLD = 0.5
      self._BLINK_THRESHOLD = 0.5
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
      self._DCAM_UNCERTAIN_ALERT_COUNT = int(60 / DT_DMON)
      self._DCAM_UNCERTAIN_RESET_COUNT = int(20 / DT_DMON)
      self._POSE_UNCERTAINTY_THRESHOLD = 0.3
      self._WHEELTOUCH_FALLBACK_TIME = int(10 / DT_DMON)  # fall back to wheel touch if pose uncertainty stays high for 10s
      self._DISTRACTED_FILTER_TS = 0.25  # 0.6Hz

      self._POSE_CALIB_MIN_SPEED = 13  # 30 mph
      self._POSE_OFFSET_MIN_COUNT = int(60 / DT_DMON)  # valid data counts before calibration completes, 1min cumulative
      self._POSE_OFFSET_MAX_COUNT = int(360 / DT_DMON)  # stop deweighting new data after 6 min, aka "short term memory"
      self._WHEELPOS_CALIB_MIN_SPEED = 11
      self._WHEELPOS_THRESHOLD = 0.5
      self._WHEELPOS_FILTER_MIN_COUNT = int(15 / DT_DMON)  # allow 15 seconds to converge wheel side
      self._WHEELPOS_DATA_AVG = 0.03
      self._WHEELPOS_DATA_VAR = 3*5.5e-5
      self._WHEELPOS_MAX_COUNT = -1

  """
    This is the primary DM policy that relies on active monitoring via driver-facing camera.
  """
  def __init__(self, shared_settings, rhd_saved=False):
    super().__init__(shared_settings,
                     shared_settings.vision._ALERT_1_INTERVAL,
                     shared_settings.vision._ALERT_2_INTERVAL,
                     shared_settings.vision._ALERT_3_INTERVAL)
    self.settings = shared_settings.vision

    wheelpos_filter_raw_priors = (self.settings._WHEELPOS_DATA_AVG, self.settings._WHEELPOS_DATA_VAR, 2)
    self.wheelpos = DriverProb(raw_priors=wheelpos_filter_raw_priors, max_trackable=self.settings._WHEELPOS_MAX_COUNT)
    self.pose = DriverPose(settings=self.settings)
    self.blink_prob = 0.
    self.phone_prob = 0.
    self.distracted_types = defaultdict(bool)
    self.driver_distracted = False
    self.driver_distraction_filter = FirstOrderFilter(0., self.settings._DISTRACTED_FILTER_TS, DT_DMON)
    self.wheel_on_right = False
    self.wheel_on_right_last = None
    self.wheel_on_right_default = rhd_saved
    self.face_detected = False
    self.wheeltouch_fallback_active = False
    self.wheeltouch_fallback_count = 0
    self.pose_uncertainty = 0.
    self.dcam_uncertain_cnt = 0
    self.dcam_reset_cnt = 0
    self.is_available = True

  @property
  def attentive(self):
    return self.driver_distraction_filter.x < DISTRACTION_FILTER_ATTENTIVE_THRESHOLD and self.face_detected and self.pose.low_uncertainty

  @property
  def certainly_distracted(self):
    return self.driver_distraction_filter.x > DISTRACTION_FILTER_DISTRACTED_THRESHOLD and self.driver_distracted and self.face_detected

  @property
  def maybe_distracted(self):
    return self.wheeltouch_fallback_active or not self.face_detected

  def get_state(self, awareness_step):
    state = log.DriverMonitoringState.VisionPolicyState.new_message()
    state.awarenessPercent = to_percent(self.awareness)
    state.awarenessStep = awareness_step
    state.isDistracted = self.driver_distracted
    state.distractedTypes.pose = self.distracted_types['pose']
    state.distractedTypes.eye = self.distracted_types['eye']
    state.distractedTypes.phone = self.distracted_types['phone']
    state.faceDetected = self.face_detected
    state.pose.pitch = self.pose.pitch
    state.pose.yaw = self.pose.yaw
    state.pose.pitchCalib.calibratedPercent = to_percent(self.pose.pitch_offsetter.filtered_stat.n / self.settings._POSE_OFFSET_MIN_COUNT)
    state.pose.pitchCalib.offset = self.pose.pitch_offsetter.filtered_stat.M
    state.pose.yawCalib.calibratedPercent = to_percent(self.pose.yaw_offsetter.filtered_stat.n / self.settings._POSE_OFFSET_MIN_COUNT)
    state.pose.yawCalib.offset = self.pose.yaw_offsetter.filtered_stat.M
    state.pose.calibrated = self.pose.calibrated
    state.pose.uncertainty = self.pose_uncertainty
    state.wheeltouchFallbackPercent = to_percent(self.wheeltouch_fallback_count / self.settings._WHEELTOUCH_FALLBACK_TIME)
    state.uncertainOffroadAlertPercent = to_percent(self.dcam_uncertain_cnt / self.settings._DCAM_UNCERTAIN_ALERT_COUNT)
    return state

  def set_pose_strictness(self, brake_disengage_prob, car_speed):
    bp = brake_disengage_prob
    k1 = max(-0.00156 * ((car_speed - 16) ** 2) + 0.6, 0.2)
    bp_normal = max(min(bp / k1, 0.5), 0)
    self.pose.cfactor_pitch = np.interp(bp_normal, [0, 0.5],
                                        [self.settings._POSE_PITCH_THRESHOLD_SLACK,
                                         self.settings._POSE_PITCH_THRESHOLD_STRICT]) / self.settings._POSE_PITCH_THRESHOLD
    self.pose.cfactor_yaw = np.interp(bp_normal, [0, 0.5],
                                      [self.settings._POSE_YAW_THRESHOLD_SLACK,
                                       self.settings._POSE_YAW_THRESHOLD_STRICT]) / self.settings._POSE_YAW_THRESHOLD

  def _update_distracted_types(self):
    self.distracted_types = defaultdict(bool)

    if not self.pose.calibrated:
      pitch_error = self.pose.pitch - self.settings._PITCH_NATURAL_OFFSET
      yaw_error = self.pose.yaw - self.settings._YAW_NATURAL_OFFSET
    else:
      pitch_error = self.pose.pitch - min(max(self.pose.pitch_offsetter.filtered_stat.mean(),
                                              self.settings._PITCH_MIN_OFFSET), self.settings._PITCH_MAX_OFFSET)
      yaw_error = self.pose.yaw - min(max(self.pose.yaw_offsetter.filtered_stat.mean(),
                                          self.settings._YAW_MIN_OFFSET), self.settings._YAW_MAX_OFFSET)
    pitch_error = 0 if pitch_error > 0 else abs(pitch_error)  # no positive pitch limit

    if yaw_error * self.pose.steer_yaw_offset > 0:  # unidirectional
      yaw_error = max(abs(yaw_error) - min(abs(self.pose.steer_yaw_offset), self.settings._POSE_YAW_STEER_MAX_OFFSET), 0.)
    else:
      yaw_error = abs(yaw_error)

    pitch_threshold = self.settings._POSE_PITCH_THRESHOLD * self.pose.cfactor_pitch if self.pose.calibrated else self.settings._PITCH_NATURAL_THRESHOLD
    yaw_threshold = self.settings._POSE_YAW_THRESHOLD * self.pose.cfactor_yaw

    self.distracted_types['pose'] = bool((pitch_error > pitch_threshold) or (yaw_error > yaw_threshold))
    self.distracted_types['eye'] = bool(self.blink_prob > self.settings._BLINK_THRESHOLD)
    self.distracted_types['phone'] = bool(self.phone_prob > self.settings._PHONE_THRESH)

  def update(self, driver_state, cal_rpy, car_speed, op_engaged, standstill, demo_mode=False, steering_angle_deg=0.):
    rhd_pred = driver_state.wheelOnRightProb
    # calibrates only when there's movement and either face detected
    if car_speed > self.settings._WHEELPOS_CALIB_MIN_SPEED and (driver_state.leftDriverData.faceProb > self.settings._FACE_THRESHOLD or
                                                                driver_state.rightDriverData.faceProb > self.settings._FACE_THRESHOLD):
      self.wheelpos.prob_offseter.push_and_update(rhd_pred)

    wheelpos_calibrated = self.wheelpos.prob_offseter.filtered_stat.n >= self.settings._WHEELPOS_FILTER_MIN_COUNT

    if wheelpos_calibrated or demo_mode:
      self.wheel_on_right = self.wheelpos.prob_offseter.filtered_stat.M > self.settings._WHEELPOS_THRESHOLD
    else:
      self.wheel_on_right = self.wheel_on_right_default  # use default/saved if calibration is unfinished
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
    self.pose_uncertainty = max(driver_data.faceOrientationStd[0], driver_data.faceOrientationStd[1])
    self.pose.low_uncertainty = self.pose_uncertainty < self.settings._POSE_UNCERTAINTY_THRESHOLD
    self.blink_prob = driver_data.eyesClosedProb * (driver_data.eyesVisibleProb > self.settings._EYE_THRESHOLD)
    self.phone_prob = driver_data.phoneProb

    self._update_distracted_types()
    self.driver_distracted = any(self.distracted_types.values()) and self.face_detected and self.pose.low_uncertainty
    self.driver_distraction_filter.update(self.driver_distracted)

    # only update when driver is actively driving the car above a certain speed
    if self.face_detected and car_speed > self.settings._POSE_CALIB_MIN_SPEED and self.pose.low_uncertainty and (not op_engaged or not self.driver_distracted):
      self.pose.pitch_offsetter.push_and_update(self.pose.pitch)
      self.pose.yaw_offsetter.push_and_update(self.pose.yaw)

    self.pose.calibrated = self.pose.pitch_offsetter.filtered_stat.n >= self.settings._POSE_OFFSET_MIN_COUNT and \
                           self.pose.yaw_offsetter.filtered_stat.n >= self.settings._POSE_OFFSET_MIN_COUNT

    if self.face_detected and not self.driver_distracted:
      dcam_uncertain = self.pose_uncertainty > self.settings._DCAM_UNCERTAIN_ALERT_THRESHOLD
      if dcam_uncertain and not standstill:
        self.dcam_uncertain_cnt += 1
        self.dcam_reset_cnt = 0
      else:
        self.dcam_reset_cnt += 1
        if self.dcam_reset_cnt > self.settings._DCAM_UNCERTAIN_RESET_COUNT:
          self.dcam_uncertain_cnt = 0

    self.wheeltouch_fallback_active = self.wheeltouch_fallback_count >= self.settings._WHEELTOUCH_FALLBACK_TIME
    self.is_available = self.face_detected and not self.wheeltouch_fallback_active

    if self.face_detected and not self.pose.low_uncertainty and not self.driver_distracted:
      self.wheeltouch_fallback_count += 1
    elif self.face_detected and self.pose.low_uncertainty:
      self.wheeltouch_fallback_count = 0

  def update_awareness(self, *, active, standstill, always_on_exemption, driver_interacting=False):
    if not active:
      return False

    step = self.awareness_step if self.is_available else 0.
    return self._update_awareness(step=step,
                                  can_recover=self.attentive,
                                  should_count_down=self.certainly_distracted or self.maybe_distracted,
                                  standstill=standstill,
                                  always_on_exemption=always_on_exemption,
                                  driver_interacting=driver_interacting)


class WheeltouchPolicy(DriverMonitoringPolicy):
  class Settings:
    def __init__(self):
      # https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:42018X1947&rid=2
      self._ALERT_1_INTERVAL = 15.
      self._ALERT_2_INTERVAL = 24.
      self._ALERT_3_INTERVAL = 30.

  """
    This is the fallback DM policy when vision is unavailable.
  """
  def __init__(self, shared_settings):
    super().__init__(shared_settings,
                     shared_settings.wheeltouch._ALERT_1_INTERVAL,
                     shared_settings.wheeltouch._ALERT_2_INTERVAL,
                     shared_settings.wheeltouch._ALERT_3_INTERVAL)
    self.settings = shared_settings.wheeltouch
    self.driver_interacting = False

  def update(self, *, active, vision_policy, driver_interacting, standstill, always_on_exemption):
    self.driver_interacting = driver_interacting

    if active:
      return self._update_awareness(step=self.awareness_step,
                                    can_recover=vision_policy.attentive,
                                    should_count_down=vision_policy.certainly_distracted or vision_policy.maybe_distracted,
                                    standstill=standstill,
                                    always_on_exemption=always_on_exemption,
                                    driver_interacting=driver_interacting)

    vision_step = vision_policy.awareness_step if vision_policy.is_available else 0.
    if vision_policy.awareness == 1. and vision_policy._recovery_available(vision_policy.attentive, standstill, vision_step):
      self.awareness = min(self.awareness + vision_step, 1.)
    return False

  def get_state(self, awareness_step):
    state = log.DriverMonitoringState.WheeltouchPolicyState.new_message()
    state.awarenessPercent = to_percent(self.awareness)
    state.awarenessStep = awareness_step
    state.driverInteracting = self.driver_interacting
    return state


# model output refers to center of undistorted+leveled image
ref_undistorted_cam = DEVICE_CAMERAS[("tici", "ar0231")].dcam
dcam_undistorted_FL = 598.0
dcam_undistorted_W, dcam_undistorted_H = (ref_undistorted_cam.width, ref_undistorted_cam.height)


def face_orientation_from_model(orient_model, pos_model, rpy_calib):
  pitch_model = orient_model[0]
  yaw_model = orient_model[1]

  face_pixel_position = ((pos_model[0] + 0.5) * dcam_undistorted_W, (pos_model[1] + 0.5) * dcam_undistorted_H)
  yaw_focal_angle = atan2(face_pixel_position[0] - dcam_undistorted_W // 2, dcam_undistorted_FL)
  pitch_focal_angle = atan2(face_pixel_position[1] - dcam_undistorted_H // 2, dcam_undistorted_FL)

  pitch = pitch_model + pitch_focal_angle
  yaw = -yaw_model + yaw_focal_angle

  pitch -= rpy_calib[1]
  yaw -= rpy_calib[2]
  return pitch, yaw


class DriverMonitoring:
  def __init__(self, rhd_saved=False, settings=None, always_on=False):
    self.settings = settings if settings is not None else DRIVER_MONITOR_SETTINGS()
    self.vision_policy = VisionPolicy(self.settings, rhd_saved=rhd_saved)
    self.wheeltouch_policy = WheeltouchPolicy(self.settings)
    self.alert_level = AlertLevel.none
    self.always_on = always_on
    self.terminal_alert_cnt = 0
    self.terminal_time = 0
    self.active_policy = MonitoringPolicy.vision
    self.too_distracted = Params().get_bool("DriverTooDistracted")

  def _reset_awareness(self):
    self.vision_policy.reset_awareness()
    self.wheeltouch_policy.reset_awareness()

  @property
  def current_policy(self):
    return self.vision_policy if self.active_policy == MonitoringPolicy.vision else self.wheeltouch_policy

  @property
  def awareness(self):
    return self.current_policy.awareness

  @property
  def wheelpos(self):
    return self.vision_policy.wheelpos

  @property
  def wheel_on_right(self):
    return self.vision_policy.wheel_on_right

  def _get_active_awareness_step(self):
    if self.active_policy == MonitoringPolicy.wheeltouch:
      return self.wheeltouch_policy.awareness_step
    return self.vision_policy.awareness_step if self.vision_policy.is_available else 0.

  def _update_active_policy(self):
    if self.active_policy == MonitoringPolicy.vision:
      if not self.vision_policy.is_available and self.vision_policy.awareness > self.vision_policy.alert_2_threshold:
        self.active_policy = MonitoringPolicy.wheeltouch
    elif self.vision_policy.is_available and self.wheeltouch_policy.awareness > self.wheeltouch_policy.alert_2_threshold:
      self.active_policy = MonitoringPolicy.vision

  def _update_states(self, driver_state, cal_rpy, car_speed, op_engaged, standstill, demo_mode=False, steering_angle_deg=0.):
    self.vision_policy.update(driver_state=driver_state,
                              cal_rpy=cal_rpy,
                              car_speed=car_speed,
                              op_engaged=op_engaged,
                              standstill=standstill,
                              demo_mode=demo_mode,
                              steering_angle_deg=steering_angle_deg)
    self._update_active_policy()

  def _update_events(self, driver_engaged, op_engaged, standstill, wrong_gear):
    self.alert_level = AlertLevel.none
    self.wheeltouch_policy.driver_interacting = driver_engaged

    if self.terminal_alert_cnt >= self.settings._MAX_TERMINAL_ALERTS or \
       self.terminal_time >= self.settings._MAX_TERMINAL_DURATION:
      self.too_distracted = True

    always_on_valid = self.always_on and not wrong_gear
    if (driver_engaged and self.awareness > 0 and self.active_policy == MonitoringPolicy.wheeltouch) or \
       (not always_on_valid and not op_engaged) or \
       (always_on_valid and not op_engaged and self.awareness <= 0):
      # always reset on disengage with normal mode; disengage resets only on alert level three if always on
      self._reset_awareness()
      return

    current_policy = self.current_policy
    awareness_prev = current_policy.awareness
    active_awareness_step = self._get_active_awareness_step()
    always_on_exemption = always_on_valid and not op_engaged and current_policy._reaching_alert_3(active_awareness_step)

    vision_alert_cleared = self.vision_policy.update_awareness(active=self.active_policy == MonitoringPolicy.vision,
                                                               standstill=standstill,
                                                               always_on_exemption=always_on_exemption,
                                                               driver_interacting=driver_engaged)
    wheeltouch_alert_cleared = self.wheeltouch_policy.update(active=self.active_policy == MonitoringPolicy.wheeltouch,
                                                             vision_policy=self.vision_policy,
                                                             driver_interacting=driver_engaged,
                                                             standstill=standstill,
                                                             always_on_exemption=always_on_exemption)

    if (self.active_policy == MonitoringPolicy.vision and vision_alert_cleared) or \
       (self.active_policy == MonitoringPolicy.wheeltouch and wheeltouch_alert_cleared):
      return

    self.alert_level = self.current_policy.get_alert_level()
    if self.alert_level == AlertLevel.three:
      self.terminal_time += 1
      if awareness_prev > 0.:
        self.terminal_alert_cnt += 1

  def get_state_packet(self, valid=True):
    # build driverMonitoringState packet
    dat = messaging.new_message('driverMonitoringState', valid=valid)
    dm = dat.driverMonitoringState

    current_policy = self.current_policy
    dm.lockout = self.too_distracted
    dm.alertCountLockoutPercent = to_percent(self.terminal_alert_cnt / self.settings._MAX_TERMINAL_ALERTS)
    dm.alertTimeLockoutPercent = to_percent(self.terminal_time / self.settings._MAX_TERMINAL_DURATION)
    dm.alwaysOn = self.always_on
    dm.alwaysOnLockout = self.always_on and current_policy.awareness <= current_policy.alert_2_threshold
    dm.alertLevel = self.alert_level
    dm.activePolicy = self.active_policy
    dm.isRHD = self.vision_policy.wheel_on_right
    dm.rhdCalibration.calibratedPercent = to_percent(
      self.vision_policy.wheelpos.prob_offseter.filtered_stat.n / self.vision_policy.settings._WHEELPOS_FILTER_MIN_COUNT
    )
    dm.rhdCalibration.offset = self.vision_policy.wheelpos.prob_offseter.filtered_stat.M

    dm.visionPolicyState = self.vision_policy.get_state(
      awareness_step=self._get_active_awareness_step() if self.active_policy == MonitoringPolicy.vision else 0.
    )
    dm.wheeltouchPolicyState = self.wheeltouch_policy.get_state(
      awareness_step=0. if self.active_policy == MonitoringPolicy.vision else self.wheeltouch_policy.awareness_step
    )
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
      brake_disengage_prob = sm['modelV2'].meta.disengagePredictions.brakeDisengageProbs[0]  # brake disengage prob in next 2s
      steering_angle_deg = sm['carState'].steeringAngleDeg
      rpyCalib = sm['liveCalibration'].rpyCalib

    self.vision_policy.set_pose_strictness(
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
