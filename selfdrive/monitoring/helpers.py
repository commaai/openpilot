from math import atan2
import numpy as np

from cereal import car, log
import cereal.messaging as messaging
from openpilot.selfdrive.selfdrived.events import Events
from openpilot.selfdrive.selfdrived.alertmanager import set_offroad_alert
from openpilot.common.realtime import DT_DMON
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.params import Params
from openpilot.common.stat_live import RunningStatFilter
from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.system.hardware import HARDWARE

EventName = log.OnroadEvent.EventName

# ******************************************************************************************
#  NOTE: To fork maintainers.
#  Disabling or nerfing safety features will get you and your users banned from our servers.
#  We recommend that you do not change these numbers from the defaults.
# ******************************************************************************************

class DRIVER_MONITOR_SETTINGS:
  def __init__(self, device_type):
    self._DT_DMON = DT_DMON
    # ref (page15-16): https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:42018X1947&rid=2
    self._AWARENESS_TIME = 30. # passive wheeltouch total timeout
    self._AWARENESS_PRE_TIME_TILL_TERMINAL = 15.
    self._AWARENESS_PROMPT_TIME_TILL_TERMINAL = 6.
    self._DISTRACTED_TIME = 11. # active monitoring total timeout
    self._DISTRACTED_PRE_TIME_TILL_TERMINAL = 8.
    self._DISTRACTED_PROMPT_TIME_TILL_TERMINAL = 6.

    self._FACE_THRESHOLD = 0.7
    self._EYE_THRESHOLD = 0.65
    self._SG_THRESHOLD = 0.9
    self._BLINK_THRESHOLD = 0.865

    self._PHONE_THRESH = 0.75 if device_type == 'mici' else 0.4
    self._PHONE_THRESH2 = 15.0
    self._PHONE_MAX_OFFSET = 0.06
    self._PHONE_MIN_OFFSET = 0.025
    self._PHONE_DATA_AVG = 0.05
    self._PHONE_DATA_VAR = 3*0.005
    self._PHONE_MAX_COUNT = int(360 / self._DT_DMON)

    self._POSE_PITCH_THRESHOLD = 0.3133
    self._POSE_PITCH_THRESHOLD_SLACK = 0.3237
    self._POSE_PITCH_THRESHOLD_STRICT = self._POSE_PITCH_THRESHOLD
    self._POSE_YAW_THRESHOLD = 0.4020
    self._POSE_YAW_THRESHOLD_SLACK = 0.5042
    self._POSE_YAW_THRESHOLD_STRICT = self._POSE_YAW_THRESHOLD
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
    self._DCAM_UNCERTAIN_ALERT_COUNT = int(60  / self._DT_DMON)
    self._DCAM_UNCERTAIN_RESET_COUNT = int(20  / self._DT_DMON)
    self._POSESTD_THRESHOLD = 0.3
    self._HI_STD_FALLBACK_TIME = int(10  / self._DT_DMON)  # fall back to wheel touch if model is uncertain for 10s
    self._DISTRACTED_FILTER_TS = 0.25  # 0.6Hz
    self._ALWAYS_ON_ALERT_MIN_SPEED = 11

    self._POSE_CALIB_MIN_SPEED = 13  # 30 mph
    self._POSE_OFFSET_MIN_COUNT = int(60 / self._DT_DMON)  # valid data counts before calibration completes, 1min cumulative
    self._POSE_OFFSET_MAX_COUNT = int(360 / self._DT_DMON)  # stop deweighting new data after 6 min, aka "short term memory"

    self._WHEELPOS_CALIB_MIN_SPEED = 11
    self._WHEELPOS_THRESHOLD = 0.5
    self._WHEELPOS_FILTER_MIN_COUNT = int(15 / self._DT_DMON) # allow 15 seconds to converge wheel side
    self._WHEELPOS_DATA_AVG = 0.03
    self._WHEELPOS_DATA_VAR = 3*5.5e-5
    self._WHEELPOS_MAX_COUNT = -1

    self._RECOVERY_FACTOR_MAX = 5.  # relative to minus step change
    self._RECOVERY_FACTOR_MIN = 1.25  # relative to minus step change

    self._MAX_TERMINAL_ALERTS = 3  # not allowed to engage after 3 terminal alerts
    self._MAX_TERMINAL_DURATION = int(30 / self._DT_DMON)  # not allowed to engage after 30s of terminal alerts

class DistractedType:

  NOT_DISTRACTED = 0
  DISTRACTED_POSE = 1 << 0
  DISTRACTED_BLINK = 1 << 1
  DISTRACTED_PHONE = 1 << 2

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

class DriverProb:
  def __init__(self, raw_priors, max_trackable):
    self.prob = 0.
    self.prob_offseter = RunningStatFilter(raw_priors=raw_priors, max_trackable=max_trackable)
    self.prob_calibrated = False

class DriverBlink:
  def __init__(self):
    self.left = 0.
    self.right = 0.


# model output refers to center of undistorted+leveled image
EFL = 598.0 # focal length in K
cam = DEVICE_CAMERAS[("tici", "ar0231")] # corrected image has same size as raw
W, H = (cam.dcam.width, cam.dcam.height)  # corrected image has same size as raw

def face_orientation_from_net(angles_desc, pos_desc, rpy_calib):
  # the output of these angles are in device frame
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
    # init policy settings
    self.settings = settings if settings is not None else DRIVER_MONITOR_SETTINGS(device_type=HARDWARE.get_device_type())

    # init driver status
    wheelpos_filter_raw_priors = (self.settings._WHEELPOS_DATA_AVG, self.settings._WHEELPOS_DATA_VAR, 2)
    phone_filter_raw_priors = (self.settings._PHONE_DATA_AVG, self.settings._PHONE_DATA_VAR, 2)
    self.wheelpos = DriverProb(raw_priors=wheelpos_filter_raw_priors, max_trackable=self.settings._WHEELPOS_MAX_COUNT)
    self.phone = DriverProb(raw_priors=phone_filter_raw_priors, max_trackable=self.settings._PHONE_MAX_COUNT)
    self.pose = DriverPose(settings=self.settings)
    self.blink = DriverBlink()

    self.always_on = always_on
    self.distracted_types = []
    self.driver_distracted = False
    self.driver_distraction_filter = FirstOrderFilter(0., self.settings._DISTRACTED_FILTER_TS, self.settings._DT_DMON)
    self.wheel_on_right = False
    self.wheel_on_right_last = None
    self.wheel_on_right_default = rhd_saved
    self.face_detected = False
    self.terminal_alert_cnt = 0
    self.terminal_time = 0
    self.step_change = 0.
    self.active_monitoring_mode = True
    self.is_model_uncertain = False
    self.hi_stds = 0
    self.threshold_pre = self.settings._DISTRACTED_PRE_TIME_TILL_TERMINAL / self.settings._DISTRACTED_TIME
    self.threshold_prompt = self.settings._DISTRACTED_PROMPT_TIME_TILL_TERMINAL / self.settings._DISTRACTED_TIME
    self.dcam_uncertain_cnt = 0
    self.dcam_uncertain_alerted = False # once per drive
    self.dcam_reset_cnt = 0

    self.params = Params()
    self.too_distracted = self.params.get_bool("DriverTooDistracted")

    self._reset_awareness()
    self._set_timers(active_monitoring=True)
    self._reset_events()

  def _reset_awareness(self):
    self.awareness = 1.
    self.awareness_active = 1.
    self.awareness_passive = 1.

  def _reset_events(self):
    self.current_events = Events()

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

  def _set_policy(self, brake_disengage_prob, car_speed):
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
    distracted_types = []

    if not self.pose.calibrated:
      pitch_error = self.pose.pitch - self.settings._PITCH_NATURAL_OFFSET
      yaw_error = self.pose.yaw - self.settings._YAW_NATURAL_OFFSET
    else:
      pitch_error = self.pose.pitch - min(max(self.pose.pitch_offseter.filtered_stat.mean(),
                                                       self.settings._PITCH_MIN_OFFSET), self.settings._PITCH_MAX_OFFSET)
      yaw_error = self.pose.yaw - min(max(self.pose.yaw_offseter.filtered_stat.mean(),
                                                    self.settings._YAW_MIN_OFFSET), self.settings._YAW_MAX_OFFSET)
    pitch_error = 0 if pitch_error > 0 else abs(pitch_error) # no positive pitch limit
    yaw_error = abs(yaw_error)

    pitch_threshold = self.settings._POSE_PITCH_THRESHOLD * self.pose.cfactor_pitch if self.pose.calibrated else self.settings._PITCH_NATURAL_THRESHOLD
    yaw_threshold = self.settings._POSE_YAW_THRESHOLD * self.pose.cfactor_yaw

    if pitch_error > pitch_threshold or yaw_error > yaw_threshold:
      distracted_types.append(DistractedType.DISTRACTED_POSE)

    if (self.blink.left + self.blink.right)*0.5 > self.settings._BLINK_THRESHOLD:
      distracted_types.append(DistractedType.DISTRACTED_BLINK)

    if self.phone.prob_calibrated:
      using_phone = self.phone.prob > max(min(self.phone.prob_offseter.filtered_stat.M, self.settings._PHONE_MAX_OFFSET), self.settings._PHONE_MIN_OFFSET) \
                                      * self.settings._PHONE_THRESH2
    else:
      using_phone = self.phone.prob > self.settings._PHONE_THRESH
    if using_phone:
      distracted_types.append(DistractedType.DISTRACTED_PHONE)

    return distracted_types

  def _update_states(self, driver_state, cal_rpy, car_speed, op_engaged, standstill, demo_mode=False):
    rhd_pred = driver_state.wheelOnRightProb
    # calibrates only when there's movement and either face detected
    if car_speed > self.settings._WHEELPOS_CALIB_MIN_SPEED and (driver_state.leftDriverData.faceProb > self.settings._FACE_THRESHOLD or
                                          driver_state.rightDriverData.faceProb > self.settings._FACE_THRESHOLD):
      self.wheelpos.prob_offseter.push_and_update(rhd_pred)

    self.wheelpos.prob_calibrated = self.wheelpos.prob_offseter.filtered_stat.n > self.settings._WHEELPOS_FILTER_MIN_COUNT

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
    if self.wheel_on_right:
      self.pose.yaw *= -1
    self.wheel_on_right_last = self.wheel_on_right
    self.pose.pitch_std = driver_data.faceOrientationStd[0]
    self.pose.yaw_std = driver_data.faceOrientationStd[1]
    model_std_max = max(self.pose.pitch_std, self.pose.yaw_std)
    self.pose.low_std = model_std_max < self.settings._POSESTD_THRESHOLD
    self.blink.left = driver_data.leftBlinkProb * (driver_data.leftEyeProb > self.settings._EYE_THRESHOLD) \
                      * (driver_data.sunglassesProb < self.settings._SG_THRESHOLD)
    self.blink.right = driver_data.rightBlinkProb * (driver_data.rightEyeProb > self.settings._EYE_THRESHOLD) \
                      * (driver_data.sunglassesProb < self.settings._SG_THRESHOLD)
    self.phone.prob = driver_data.phoneProb

    self.distracted_types = self._get_distracted_types()
    self.driver_distracted = (DistractedType.DISTRACTED_PHONE in self.distracted_types
                              or DistractedType.DISTRACTED_POSE in self.distracted_types
                              or DistractedType.DISTRACTED_BLINK in self.distracted_types) \
                              and driver_data.faceProb > self.settings._FACE_THRESHOLD and self.pose.low_std
    self.driver_distraction_filter.update(self.driver_distracted)

    # update offseter
    # only update when driver is actively driving the car above a certain speed
    if self.face_detected and car_speed > self.settings._POSE_CALIB_MIN_SPEED and self.pose.low_std and (not op_engaged or not self.driver_distracted):
      self.pose.pitch_offseter.push_and_update(self.pose.pitch)
      self.pose.yaw_offseter.push_and_update(self.pose.yaw)
      self.phone.prob_offseter.push_and_update(self.phone.prob)

    self.pose.calibrated = self.pose.pitch_offseter.filtered_stat.n > self.settings._POSE_OFFSET_MIN_COUNT and \
                           self.pose.yaw_offseter.filtered_stat.n > self.settings._POSE_OFFSET_MIN_COUNT
    self.phone.prob_calibrated = self.phone.prob_offseter.filtered_stat.n > self.settings._POSE_OFFSET_MIN_COUNT

    if self.face_detected and not self.driver_distracted:
      if model_std_max > self.settings._DCAM_UNCERTAIN_ALERT_THRESHOLD:
        if not standstill:
          self.dcam_uncertain_cnt += 1
          self.dcam_reset_cnt = 0
      else:
        self.dcam_reset_cnt += 1
        if self.dcam_reset_cnt > self.settings._DCAM_UNCERTAIN_RESET_COUNT:
          self.dcam_uncertain_cnt = 0

    self.is_model_uncertain = self.hi_stds > self.settings._HI_STD_FALLBACK_TIME
    self._set_timers(self.face_detected and not self.is_model_uncertain)
    if self.face_detected and not self.pose.low_std and not self.driver_distracted:
      self.hi_stds += 1
    elif self.face_detected and self.pose.low_std:
      self.hi_stds = 0

  def _update_events(self, driver_engaged, op_engaged, standstill, wrong_gear, car_speed):
    self._reset_events()
    # Block engaging until ignition cycle after max number or time of distractions
    if self.terminal_alert_cnt >= self.settings._MAX_TERMINAL_ALERTS or \
       self.terminal_time >= self.settings._MAX_TERMINAL_DURATION:
      if not self.too_distracted:
        self.params.put_bool_nonblocking("DriverTooDistracted", True)
      self.too_distracted = True

    # Always-on distraction lockout is temporary
    if self.too_distracted or (self.always_on and self.awareness <= self.threshold_prompt):
      self.current_events.add(EventName.tooDistracted)

    always_on_valid = self.always_on and not wrong_gear
    if (driver_engaged and self.awareness > 0 and not self.active_monitoring_mode) or \
       (not always_on_valid and not op_engaged) or \
       (always_on_valid and not op_engaged and self.awareness <= 0):
      # always reset on disengage with normal mode; disengage resets only on red if always on
      self._reset_awareness()
      return

    driver_attentive = self.driver_distraction_filter.x < 0.37
    awareness_prev = self.awareness

    if (driver_attentive and self.face_detected and self.pose.low_std and self.awareness > 0):
      if driver_engaged:
        self._reset_awareness()
        return
      # only restore awareness when paying attention and alert is not red
      self.awareness = min(self.awareness + ((self.settings._RECOVERY_FACTOR_MAX-self.settings._RECOVERY_FACTOR_MIN)*
                                             (1.-self.awareness)+self.settings._RECOVERY_FACTOR_MIN)*self.step_change, 1.)
      if self.awareness == 1.:
        self.awareness_passive = min(self.awareness_passive + self.step_change, 1.)
      # don't display alert banner when awareness is recovering and has cleared orange
      if self.awareness > self.threshold_prompt:
        return

    _reaching_audible = self.awareness - self.step_change <= self.threshold_prompt
    _reaching_terminal = self.awareness - self.step_change <= 0
    standstill_orange_exemption = standstill and _reaching_audible
    always_on_red_exemption = always_on_valid and not op_engaged and _reaching_terminal
    always_on_lowspeed_exemption = always_on_valid and not op_engaged and car_speed < self.settings._ALWAYS_ON_ALERT_MIN_SPEED

    certainly_distracted = self.driver_distraction_filter.x > 0.63 and self.driver_distracted and self.face_detected
    maybe_distracted = self.hi_stds > self.settings._HI_STD_FALLBACK_TIME or not self.face_detected

    if certainly_distracted or maybe_distracted:
      # should always be counting if distracted unless at standstill (lowspeed for always-on) and reaching orange
      # also will not be reaching 0 if DM is active when not engaged
      if not (standstill_orange_exemption or always_on_red_exemption or (always_on_lowspeed_exemption and _reaching_audible)):
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
    elif self.awareness <= self.threshold_pre and not always_on_lowspeed_exemption:
      # pre green alert
      alert = EventName.preDriverDistracted if self.active_monitoring_mode else EventName.preDriverUnresponsive

    if alert is not None:
      self.current_events.add(alert)

    if self.dcam_uncertain_cnt > self.settings._DCAM_UNCERTAIN_ALERT_COUNT and not self.dcam_uncertain_alerted:
      set_offroad_alert("Offroad_DriverMonitoringUncertain", True)
      self.dcam_uncertain_alerted = True


  def get_state_packet(self, valid=True):
    # build driverMonitoringState packet
    dat = messaging.new_message('driverMonitoringState', valid=valid)
    dat.driverMonitoringState = {
      "events": self.current_events.to_msg(),
      "faceDetected": self.face_detected,
      "isDistracted": self.driver_distracted,
      "distractedType": sum(self.distracted_types),
      "awarenessStatus": self.awareness,
      "posePitchOffset": self.pose.pitch_offseter.filtered_stat.mean(),
      "posePitchValidCount": self.pose.pitch_offseter.filtered_stat.n,
      "poseYawOffset": self.pose.yaw_offseter.filtered_stat.mean(),
      "poseYawValidCount": self.pose.yaw_offseter.filtered_stat.n,
      "phoneProbOffset": self.phone.prob_offseter.filtered_stat.mean(),
      "phoneProbValidCount": self.phone.prob_offseter.filtered_stat.n,
      "stepChange": self.step_change,
      "awarenessActive": self.awareness_active,
      "awarenessPassive": self.awareness_passive,
      "isLowStd": self.pose.low_std,
      "hiStdCount": self.hi_stds,
      "isActiveMode": self.active_monitoring_mode,
      "isRHD": self.wheel_on_right,
      "uncertainCount": self.dcam_uncertain_cnt,
    }
    return dat

  def run_step(self, sm, demo=False):
    if demo:
      highway_speed = 30
      enabled = True
      wrong_gear = False
      standstill = False
      driver_engaged = False
      brake_disengage_prob = 1.0
      rpyCalib = [0., 0., 0.]
    else:
      highway_speed = sm['carState'].vEgo
      enabled = sm['selfdriveState'].enabled
      wrong_gear = sm['carState'].gearShifter not in (car.CarState.GearShifter.drive, car.CarState.GearShifter.low)
      standstill = sm['carState'].standstill
      driver_engaged = sm['carState'].steeringPressed or sm['carState'].gasPressed
      brake_disengage_prob = sm['modelV2'].meta.disengagePredictions.brakeDisengageProbs[0] # brake disengage prob in next 2s
      rpyCalib = sm['liveCalibration'].rpyCalib
    self._set_policy(
      brake_disengage_prob=brake_disengage_prob,
      car_speed=highway_speed,
    )

    # Parse data from dmonitoringmodeld
    self._update_states(
      driver_state=sm['driverStateV2'],
      cal_rpy=rpyCalib,
      car_speed=highway_speed,
      op_engaged=enabled,
      standstill=standstill,
      demo_mode=demo,
    )

    # Update distraction events
    self._update_events(
      driver_engaged=driver_engaged,
      op_engaged=enabled,
      standstill=standstill,
      wrong_gear=wrong_gear,
      car_speed=highway_speed
    )
