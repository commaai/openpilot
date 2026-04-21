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

def to_perc(v):
  return int(min(max(v * 100., 0.), 100.))

# ******************************************************************************************
#  NOTE: To fork maintainers.
#  Disabling or nerfing safety features will get you and your users banned from our servers.
#  We recommend that you do not change these numbers from the defaults.
# ******************************************************************************************

class DRIVER_MONITOR_SETTINGS:
  def __init__(self):
    # https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:42018X1947&rid=2
    self._WHEELTOUCH_POLICY_ALERT_1_INTERVAL = 15.
    self._WHEELTOUCH_POLICY_ALERT_2_INTERVAL = 24.
    self._WHEELTOUCH_POLICY_ALERT_3_INTERVAL = 30.
    # https://cdn.euroncap.com/cars/assets/euro_ncap_protocol_safe_driving_driver_engagement_v11_a30e874152.pdf
    self._VISION_POLICY_ALERT_1_INTERVAL = 3.
    self._VISION_POLICY_ALERT_2_INTERVAL = 5.
    self._VISION_POLICY_ALERT_3_INTERVAL = 11.

    self._AWARENESS_RECOVERY_FACTOR_MAX = 5.
    self._AWARENESS_RECOVERY_FACTOR_MIN = 1.25

    self._MAX_TERMINAL_ALERTS = 3  # not allowed to engage after 3 terminal alerts
    self._MAX_TERMINAL_DURATION = int(30 / DT_DMON)  # not allowed to engage after 30s of terminal alerts

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
    self.roll = 0.
    self.yaw_std = 0.
    self.pitch_std = 0.
    self.roll_std = 0.
    self.pitch_offseter = RunningStatFilter(raw_priors=pitch_filter_raw_priors, max_trackable=settings._POSE_OFFSET_MAX_COUNT)
    self.yaw_offseter = RunningStatFilter(raw_priors=yaw_filter_raw_priors, max_trackable=settings._POSE_OFFSET_MAX_COUNT)
    self.calibrated = False
    self.low_std = True
    self.cfactor_pitch = 1.
    self.cfactor_yaw = 1.
    self.steer_yaw_offset = 0.

class DriverProb:
  def __init__(self, raw_priors, max_trackable):
    self.prob = 0.
    self.prob_offseter = RunningStatFilter(raw_priors=raw_priors, max_trackable=max_trackable)
    self.prob_calibrated = False


class DriverMonitoringPolicy:
  def __init__(self, settings, alert_1_interval, alert_2_interval, alert_3_interval):
    self.settings = settings
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
    self.awareness = min(self.awareness + ((self.settings._AWARENESS_RECOVERY_FACTOR_MAX - self.settings._AWARENESS_RECOVERY_FACTOR_MIN) *
                                           (1. - self.awareness) + self.settings._AWARENESS_RECOVERY_FACTOR_MIN) * step, 1.)

  def _decrease_awareness(self, step):
    self.awareness = max(self.awareness - step, -0.1)

  def _update_awareness(self, *, step, can_recover, should_count_down, standstill, always_on_exemption, driver_interacting=False):
    if self.awareness <= 0.:
      return

    standstill_exemption = standstill and self._reaching_alert_1(step)
    if self.awareness > 0 and self._recovery_available(can_recover, standstill, step):
      if driver_interacting:
        self.reset_awareness()
        return
      self._recover_awareness(step)
      if self.awareness > self.alert_2_threshold:
        return

    if should_count_down and not (standstill_exemption or always_on_exemption):
      self._decrease_awareness(step)

  def get_alert_level(self):
    if self.awareness <= 0.:
      return AlertLevel.three
    if self.awareness <= self.alert_2_threshold:
      return AlertLevel.two
    if self.awareness <= self.alert_1_threshold:
      return AlertLevel.one
    return AlertLevel.none


class VisionPolicy(DriverMonitoringPolicy):
  def __init__(self, settings, rhd_saved=False):
    super().__init__(settings,
                     settings._VISION_POLICY_ALERT_1_INTERVAL,
                     settings._VISION_POLICY_ALERT_2_INTERVAL,
                     settings._VISION_POLICY_ALERT_3_INTERVAL)

    wheelpos_filter_raw_priors = (settings._WHEELPOS_DATA_AVG, settings._WHEELPOS_DATA_VAR, 2)
    self.wheelpos = DriverProb(raw_priors=wheelpos_filter_raw_priors, max_trackable=settings._WHEELPOS_MAX_COUNT)
    self.pose = DriverPose(settings=settings)
    self.blink_prob = 0.
    self.phone_prob = 0.
    self.distracted_types = defaultdict(bool)
    self.driver_distracted = False
    self.driver_distraction_filter = FirstOrderFilter(0., settings._DISTRACTED_FILTER_TS, DT_DMON)
    self.wheel_on_right = False
    self.wheel_on_right_last = None
    self.wheel_on_right_default = rhd_saved
    self.face_detected = False
    self.is_model_uncertain = False
    self.hi_stds = 0
    self.model_std_max = 0.
    self.dcam_uncertain = False
    self.dcam_uncertain_cnt = 0
    self.dcam_reset_cnt = 0
    self.is_available = True

  @property
  def attentive(self):
    return self.driver_distraction_filter.x < 0.37 and self.face_detected and self.pose.low_std

  @property
  def certainly_distracted(self):
    return self.driver_distraction_filter.x > 0.63 and self.driver_distracted and self.face_detected

  @property
  def maybe_distracted(self):
    return self.is_model_uncertain or not self.face_detected

  def set_pose_strictness(self, brake_disengage_prob, car_speed):
    bp = brake_disengage_prob
    k1 = max(-0.00156*((car_speed-16)**2)+0.6, 0.2)
    bp_normal = max(min(bp / k1, 0.5),0)
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
      pitch_error = self.pose.pitch - min(max(self.pose.pitch_offseter.filtered_stat.mean(),
                                              self.settings._PITCH_MIN_OFFSET), self.settings._PITCH_MAX_OFFSET)
      yaw_error = self.pose.yaw - min(max(self.pose.yaw_offseter.filtered_stat.mean(),
                                          self.settings._YAW_MIN_OFFSET), self.settings._YAW_MAX_OFFSET)
    pitch_error = 0 if pitch_error > 0 else abs(pitch_error) # no positive pitch limit

    if yaw_error * self.pose.steer_yaw_offset > 0: # unidirectional
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

    self.wheelpos.prob_calibrated = self.wheelpos.prob_offseter.filtered_stat.n >= self.settings._WHEELPOS_FILTER_MIN_COUNT

    if self.wheelpos.prob_calibrated or demo_mode:
      self.wheel_on_right = self.wheelpos.prob_offseter.filtered_stat.M > self.settings._WHEELPOS_THRESHOLD
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
    self.pose.roll, self.pose.pitch, self.pose.yaw = face_orientation_from_net(driver_data.faceOrientation, driver_data.facePosition, cal_rpy)
    steer_d = max(abs(steering_angle_deg) - self.settings._POSE_YAW_MIN_STEER_DEG, 0.)
    self.pose.steer_yaw_offset = radians(steer_d) * -np.sign(steering_angle_deg) * self.settings._POSE_YAW_STEER_FACTOR
    if self.wheel_on_right:
      self.pose.yaw *= -1
      self.pose.steer_yaw_offset *= -1
    self.wheel_on_right_last = self.wheel_on_right
    self.pose.pitch_std = driver_data.faceOrientationStd[0]
    self.pose.yaw_std = driver_data.faceOrientationStd[1]
    self.model_std_max = max(self.pose.pitch_std, self.pose.yaw_std)
    self.pose.low_std = self.model_std_max < self.settings._HI_STD_THRESHOLD
    self.blink_prob = driver_data.eyesClosedProb * (driver_data.eyesVisibleProb > self.settings._EYE_THRESHOLD)
    self.phone_prob = driver_data.phoneProb

    self._update_distracted_types()
    self.driver_distracted = any(self.distracted_types.values()) and driver_data.faceProb > self.settings._FACE_THRESHOLD and self.pose.low_std
    self.driver_distraction_filter.update(self.driver_distracted)

    # update offseter
    # only update when driver is actively driving the car above a certain speed
    if self.face_detected and car_speed > self.settings._POSE_CALIB_MIN_SPEED and self.pose.low_std and (not op_engaged or not self.driver_distracted):
      self.pose.pitch_offseter.push_and_update(self.pose.pitch)
      self.pose.yaw_offseter.push_and_update(self.pose.yaw)

    self.pose.calibrated = self.pose.pitch_offseter.filtered_stat.n >= self.settings._POSE_OFFSET_MIN_COUNT and \
                           self.pose.yaw_offseter.filtered_stat.n >= self.settings._POSE_OFFSET_MIN_COUNT

    if self.face_detected and not self.driver_distracted:
      self.dcam_uncertain = self.model_std_max > self.settings._DCAM_UNCERTAIN_ALERT_THRESHOLD
      if self.dcam_uncertain and not standstill:
        self.dcam_uncertain_cnt += 1
        self.dcam_reset_cnt = 0
      else:
        self.dcam_reset_cnt += 1
        if self.dcam_reset_cnt > self.settings._DCAM_UNCERTAIN_RESET_COUNT:
          self.dcam_uncertain_cnt = 0

    self.is_model_uncertain = self.hi_stds >= self.settings._HI_STD_FALLBACK_TIME
    self.is_available = self.face_detected and not self.is_model_uncertain

    if self.face_detected and not self.pose.low_std and not self.driver_distracted:
      self.hi_stds += 1
    elif self.face_detected and self.pose.low_std:
      self.hi_stds = 0

  def update_awareness(self, *, active, standstill, always_on_exemption):
    if not active:
      return

    step = self.awareness_step if self.is_available else 0.
    self._update_awareness(step=step,
                           can_recover=self.attentive,
                           should_count_down=self.certainly_distracted or self.maybe_distracted,
                           standstill=standstill,
                           always_on_exemption=always_on_exemption)


class WheeltouchPolicy(DriverMonitoringPolicy):
  def __init__(self, settings):
    super().__init__(settings,
                     settings._WHEELTOUCH_POLICY_ALERT_1_INTERVAL,
                     settings._WHEELTOUCH_POLICY_ALERT_2_INTERVAL,
                     settings._WHEELTOUCH_POLICY_ALERT_3_INTERVAL)
    self.driver_interacting = False

  def update(self, *, active, vision_policy, driver_interacting, standstill, always_on_exemption):
    self.driver_interacting = driver_interacting

    if active:
      self._update_awareness(step=self.awareness_step,
                             can_recover=vision_policy.attentive,
                             should_count_down=vision_policy.certainly_distracted or vision_policy.maybe_distracted,
                             standstill=standstill,
                             always_on_exemption=always_on_exemption,
                             driver_interacting=driver_interacting)
      return

    vision_step = vision_policy.awareness_step if vision_policy.is_available else 0.
    if vision_policy.awareness == 1. and vision_policy._recovery_available(vision_policy.attentive, standstill, vision_step):
      self.awareness = min(self.awareness + vision_step, 1.)


# model output refers to center of undistorted+leveled image
EFL = 598.0 # focal length in K
cam = DEVICE_CAMERAS[("tici", "ar0231")] # corrected image has same size as raw
W, H = (cam.dcam.width, cam.dcam.height)  # corrected image has same size as raw

def face_orientation_from_net(angles_desc, pos_desc, rpy_calib):
  # the output of these angles are in driver camera frame
  # so from driver's perspective, pitch is up and yaw is right

  pitch_net, yaw_net, roll_net = angles_desc

  face_pixel_position = ((pos_desc[0]+0.5)*W, (pos_desc[1]+0.5)*H)
  yaw_focal_angle = atan2(face_pixel_position[0] - W//2, EFL)
  pitch_focal_angle = atan2(face_pixel_position[1] - H//2, EFL)

  pitch = pitch_net + pitch_focal_angle
  yaw = -yaw_net + yaw_focal_angle

  # no calib for roll
  pitch -= rpy_calib[1]
  yaw -= rpy_calib[2]
  return roll_net, pitch, yaw


class DriverMonitoring:
  def __init__(self, rhd_saved=False, settings=None, always_on=False):
    self.settings = settings if settings is not None else DRIVER_MONITOR_SETTINGS()
    self.vision_policy = VisionPolicy(self.settings, rhd_saved=rhd_saved)
    self.wheeltouch_policy = WheeltouchPolicy(self.settings)
    self.alert_level = AlertLevel.none
    self.always_on = always_on
    self.terminal_alert_cnt = 0
    self.terminal_time = 0
    self.active_monitoring_mode = True
    self.too_distracted = Params().get_bool("DriverTooDistracted")

  def _reset_awareness(self):
    self.vision_policy.reset_awareness()
    self.wheeltouch_policy.reset_awareness()

  @property
  def active_policy(self):
    return self.vision_policy if self.active_monitoring_mode else self.wheeltouch_policy

  @property
  def awareness(self):
    return self.active_policy.awareness

  @property
  def wheelpos(self):
    return self.vision_policy.wheelpos

  @property
  def wheel_on_right(self):
    return self.vision_policy.wheel_on_right

  def _get_active_awareness_step(self):
    if not self.active_monitoring_mode:
      return self.wheeltouch_policy.awareness_step
    return self.vision_policy.awareness_step if self.vision_policy.is_available else 0.

  def _update_active_policy(self):
    if self.active_monitoring_mode:
      if not self.vision_policy.is_available and self.vision_policy.awareness > self.vision_policy.alert_2_threshold:
        self.active_monitoring_mode = False
    elif self.vision_policy.is_available:
      self.active_monitoring_mode = True

  def _update_states(self, driver_state, cal_rpy, car_speed, op_engaged, standstill, demo_mode=False, steering_angle_deg=0.):
    self.vision_policy.update(driver_state=driver_state,
                              cal_rpy=cal_rpy,
                              car_speed=car_speed,
                              op_engaged=op_engaged,
                              standstill=standstill,
                              demo_mode=demo_mode,
                              steering_angle_deg=steering_angle_deg)
    self._update_active_policy()

  def _update_events(self, driver_engaged, op_engaged, standstill, wrong_gear, car_speed):
    self.alert_level = AlertLevel.none
    self.wheeltouch_policy.driver_interacting = driver_engaged

    if self.terminal_alert_cnt >= self.settings._MAX_TERMINAL_ALERTS or \
       self.terminal_time >= self.settings._MAX_TERMINAL_DURATION:
      self.too_distracted = True

    always_on_valid = self.always_on and not wrong_gear
    if (driver_engaged and self.awareness > 0 and not self.active_monitoring_mode) or \
       (not always_on_valid and not op_engaged) or \
       (always_on_valid and not op_engaged and self.awareness <= 0):
      # always reset on disengage with normal mode; disengage resets only on alert level three if always on
      self._reset_awareness()
      return

    active_policy = self.active_policy
    awareness_prev = active_policy.awareness
    active_awareness_step = self._get_active_awareness_step()
    always_on_exemption = always_on_valid and not op_engaged and active_policy._reaching_alert_3(active_awareness_step)

    self.vision_policy.update_awareness(active=self.active_monitoring_mode,
                                        standstill=standstill,
                                        always_on_exemption=always_on_exemption)
    self.wheeltouch_policy.update(active=not self.active_monitoring_mode,
                                  vision_policy=self.vision_policy,
                                  driver_interacting=driver_engaged,
                                  standstill=standstill,
                                  always_on_exemption=always_on_exemption)

    self.alert_level = self.active_policy.get_alert_level()
    if self.alert_level == AlertLevel.three:
      self.terminal_time += 1
      if awareness_prev > 0.:
        self.terminal_alert_cnt += 1

  def get_state_packet(self, valid=True):
    # build driverMonitoringState packet
    dat = messaging.new_message('driverMonitoringState', valid=valid)
    dm = dat.driverMonitoringState

    active_policy = self.active_policy
    dm.lockout = self.too_distracted
    dm.alertCountLockoutPercent = to_perc(self.terminal_alert_cnt / self.settings._MAX_TERMINAL_ALERTS)
    dm.alertTimeLockoutPercent = to_perc(self.terminal_time / self.settings._MAX_TERMINAL_DURATION)
    dm.alwaysOn = self.always_on
    dm.alwaysOnLockout = self.always_on and active_policy.awareness <= active_policy.alert_2_threshold
    dm.alertLevel = self.alert_level
    dm.activePolicy = MonitoringPolicy.vision if self.active_monitoring_mode else MonitoringPolicy.wheeltouch
    dm.isRHD = self.vision_policy.wheel_on_right
    dm.rhdCalibration.calibratedPercent = to_perc(self.vision_policy.wheelpos.prob_offseter.filtered_stat.n / self.settings._WHEELPOS_FILTER_MIN_COUNT)
    dm.rhdCalibration.offset = self.vision_policy.wheelpos.prob_offseter.filtered_stat.M

    dm.visionPolicyState.awarenessPercent = to_perc(self.vision_policy.awareness)
    dm.visionPolicyState.awarenessStep = self._get_active_awareness_step() if self.active_monitoring_mode else 0.
    dm.visionPolicyState.isDistracted = self.vision_policy.driver_distracted
    dm.visionPolicyState.distractedTypes.pose = self.vision_policy.distracted_types['pose']
    dm.visionPolicyState.distractedTypes.eye = self.vision_policy.distracted_types['eye']
    dm.visionPolicyState.distractedTypes.phone = self.vision_policy.distracted_types['phone']
    dm.visionPolicyState.faceDetected = self.vision_policy.face_detected
    dm.visionPolicyState.pose.pitch = self.vision_policy.pose.pitch
    dm.visionPolicyState.pose.yaw = self.vision_policy.pose.yaw
    dm.visionPolicyState.pose.calibrated = self.vision_policy.pose.calibrated
    dm.visionPolicyState.pose.pitchCalib.calibratedPercent = to_perc(self.vision_policy.pose.pitch_offseter.filtered_stat.n / self.settings._POSE_OFFSET_MIN_COUNT)
    dm.visionPolicyState.pose.pitchCalib.offset = self.vision_policy.pose.pitch_offseter.filtered_stat.M
    dm.visionPolicyState.pose.yawCalib.calibratedPercent = to_perc(self.vision_policy.pose.yaw_offseter.filtered_stat.n / self.settings._POSE_OFFSET_MIN_COUNT)
    dm.visionPolicyState.pose.yawCalib.offset = self.vision_policy.pose.yaw_offseter.filtered_stat.M
    dm.visionPolicyState.pose.uncertainty = self.vision_policy.model_std_max
    dm.visionPolicyState.wheeltouchFallbackPercent = to_perc(self.vision_policy.hi_stds / self.settings._HI_STD_FALLBACK_TIME)
    dm.visionPolicyState.uncertainOffroadAlertPercent = to_perc(self.vision_policy.dcam_uncertain_cnt / self.settings._DCAM_UNCERTAIN_ALERT_COUNT)

    dm.wheeltouchPolicyState.awarenessPercent = to_perc(self.wheeltouch_policy.awareness)
    dm.wheeltouchPolicyState.awarenessStep = 0. if self.active_monitoring_mode else self.wheeltouch_policy.awareness_step
    dm.wheeltouchPolicyState.driverInteracting = self.wheeltouch_policy.driver_interacting
    return dat

  def run_step(self, sm, demo=False):
    if demo:
      car_speed = 30
      enabled = True
      wrong_gear = False
      standstill = False
      driver_engaged = False
      brake_disengage_prob = 1.0
      rpyCalib = [0., 0., 0.]
    else:
      car_speed = sm['carState'].vEgo
      enabled = sm['selfdriveState'].enabled
      wrong_gear = sm['carState'].gearShifter not in (car.CarState.GearShifter.drive, car.CarState.GearShifter.low)
      standstill = sm['carState'].standstill
      driver_engaged = sm['carState'].steeringPressed or sm['carState'].gasPressed
      brake_disengage_prob = sm['modelV2'].meta.disengagePredictions.brakeDisengageProbs[0] # brake disengage prob in next 2s
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
      steering_angle_deg=sm['carState'].steeringAngleDeg,
    )

    # Update distraction events
    self._update_events(
      driver_engaged=driver_engaged,
      op_engaged=enabled,
      standstill=standstill,
      wrong_gear=wrong_gear,
      car_speed=car_speed
    )
