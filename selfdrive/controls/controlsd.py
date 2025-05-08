#!/usr/bin/env python3
import math
from typing import SupportsFloat

from cereal import car, log
import cereal.messaging as messaging
from openpilot.common.conversions import Conversions as CV
from openpilot.common.params import Params
from openpilot.common.realtime import config_realtime_process, Priority, Ratekeeper
from openpilot.common.swaglog import cloudlog

from opendbc.car.car_helpers import interfaces
from opendbc.car.vehicle_model import VehicleModel
from openpilot.selfdrive.controls.lib.drive_helpers import clip_curvature
from openpilot.selfdrive.controls.lib.latcontrol import LatControl
from openpilot.selfdrive.controls.lib.latcontrol_pid import LatControlPID
from openpilot.selfdrive.controls.lib.latcontrol_angle import LatControlAngle, STEER_ANGLE_SATURATION_THRESHOLD
from openpilot.selfdrive.controls.lib.latcontrol_torque import LatControlTorque
from openpilot.selfdrive.controls.lib.longcontrol import LongControl
from openpilot.selfdrive.locationd.helpers import PoseCalibrator, Pose

State = log.SelfdriveState.OpenpilotState
LaneChangeState = log.LaneChangeState
LaneChangeDirection = log.LaneChangeDirection

ACTUATOR_FIELDS = tuple(car.CarControl.Actuators.schema.fields.keys())


class Controls:
  def __init__(self) -> None:
    self.params = Params()
    cloudlog.info("controlsd is waiting for CarParams")
    self.CP = messaging.log_from_bytes(self.params.get("CarParams", block=True), car.CarParams)
    cloudlog.info("controlsd got CarParams")

    self.CI = interfaces[self.CP.carFingerprint](self.CP)

    self.sm = messaging.SubMaster(
      ['liveParameters', 'liveTorqueParameters', 'modelV2', 'selfdriveState',
       'liveCalibration', 'livePose', 'longitudinalPlan', 'carState', 'carOutput',
       'driverMonitoringState', 'onroadEvents', 'driverAssistance'], poll='selfdriveState')
    self.pm = messaging.PubMaster(['carControl', 'controlsState'])

    self.steer_limited_by_controls = False
    self.curvature = 0.0
    self.desired_curvature = 0.0

    self.pose_calibrator = PoseCalibrator()
    self.calibrated_pose: Pose | None = None

    # Sticky engagement and brake-release resume
    self.brake_pressed_prev = False
    self.sticky_enabled = False
    self._cancel_sent = False  # track cancel sent

    self.LoC = LongControl(self.CP)
    self.VM = VehicleModel(self.CP)
    if self.CP.steerControlType == car.CarParams.SteerControlType.angle:
      self.LaC = LatControlAngle(self.CP, self.CI)
    elif self.CP.lateralTuning.which() == 'pid':
      self.LaC = LatControlPID(self.CP, self.CI)
    else:
      self.LaC = LatControlTorque(self.CP, self.CI)

  def update(self):
    self.sm.update(15)
    if self.sm.updated["liveCalibration"]:
      self.pose_calibrator.feed_live_calib(self.sm['liveCalibration'])
    if self.sm.updated["livePose"]:
      device_pose = Pose.from_live_pose(self.sm['livePose'])
      self.calibrated_pose = self.pose_calibrator.build_calibrated_pose(device_pose)

  def state_control(self):
    CS = self.sm['carState']

    # Froggpilot workaround: pretend stock cruise is active in lateral-only mode to avoid mismatch
    if self.sticky_enabled and not self.CP.openpilotLongitudinalControl:
      CS.cruiseState.enabled = True
      self.CP.pcmCruise = True

    # Sticky engagement persists through brake
    CS = self.sm['carState']

    # Sticky engagement persists through brake
    if self.sm['selfdriveState'].enabled and not self.sticky_enabled:
      self.sticky_enabled = True
    if not self.sm['selfdriveState'].enabled and self.sticky_enabled and not CS.brakePressed:
      self.sticky_enabled = False

    # Vehicle model update
    lp = self.sm['liveParameters']
    self.VM.update_params(max(lp.stiffnessFactor, 0.1), max(lp.steerRatio, 0.1))

    steer_angle = math.radians(CS.steeringAngleDeg - lp.angleOffsetDeg)
    self.curvature = -self.VM.calc_curvature(steer_angle, CS.vEgo, lp.roll)

    # Live torque params
    if self.CP.lateralTuning.which() == 'torque':
      tp = self.sm['liveTorqueParameters']
      if self.sm.all_checks(['liveTorqueParameters']) and tp.useParams:
        self.LaC.update_live_torque_params(tp.latAccelFactorFiltered,
                                           tp.latAccelOffsetFiltered,
                                           tp.frictionCoefficientFiltered)

    model_v2 = self.sm['modelV2']
    CC = car.CarControl.new_message()
    CC.enabled = self.sticky_enabled

    standstill = abs(CS.vEgo) <= max(self.CP.minSteerSpeed, 0.3) or CS.standstill
    # Pause lateral on brake, resume on release
    if CS.brakePressed:
      CC.latActive = False
      self.brake_pressed_prev = True
    else:
      if self.brake_pressed_prev and CC.enabled:
        CC.latActive = True
      else:
        CC.latActive = (CC.enabled
                       and not CS.steerFaultTemporary
                       and not CS.steerFaultPermanent
                       and (not standstill or self.CP.steerAtStandstill))
      self.brake_pressed_prev = False

    # Longitudinal under demand (SET) and ctrl
    CC.longActive = (self.sticky_enabled
                     and not any(e.overrideLongitudinal for e in self.sm['onroadEvents'])
                     and self.CP.openpilotLongitudinalControl)
    CC.actuators.longControlState = self.LoC.long_control_state

    # Blinkers for lane change
    if model_v2.meta.laneChangeState != LaneChangeState.off:
      CC.leftBlinker = model_v2.meta.laneChangeDirection == LaneChangeDirection.left
      CC.rightBlinker = model_v2.meta.laneChangeDirection == LaneChangeDirection.right

    if not CC.latActive:
      self.LaC.reset()
    if not CC.longActive:
      self.LoC.reset()

    # Longitudinal accel
    pid_limits = self.CI.get_pid_accel_limits(self.CP, CS.vEgo, CS.vCruise * CV.KPH_TO_MS)
    CC.actuators.accel = float(self.LoC.update(CC.longActive, CS,
                                              self.sm['longitudinalPlan'].aTarget,
                                              self.sm['longitudinalPlan'].shouldStop,
                                              pid_limits))

    # Lateral steering
    new_curv = model_v2.action.desiredCurvature if CC.latActive else self.curvature
    self.desired_curvature, limited = clip_curvature(CS.vEgo, self.desired_curvature, new_curv, lp.roll)
    CC.actuators.curvature = self.desired_curvature
    steer, angleDeg, lac_log = self.LaC.update(CC.latActive, CS, self.VM, lp,
                                              self.steer_limited_by_controls,
                                              self.desired_curvature,
                                              self.calibrated_pose,
                                              limited)
    CC.actuators.torque = float(steer)
    CC.actuators.steeringAngleDeg = float(angleDeg)

    for p in ACTUATOR_FIELDS:
      v = getattr(CC.actuators, p)
      if isinstance(v, SupportsFloat) and not math.isfinite(v):
        cloudlog.error(f"actuators.{p} not finite {CC.actuators.to_dict()}")
        setattr(CC.actuators, p, 0.0)

    return CC, lac_log

  def publish(self, CC, lac_log):
    CS = self.sm['carState']
    CC.currentCurvature = self.curvature
    if self.calibrated_pose is not None:
      CC.orientationNED = self.calibrated_pose.orientation.xyz.tolist()
      CC.angularVelocity = self.calibrated_pose.angular_velocity.xyz.tolist()

    # Send cancel only once on engage
    if CC.enabled and not self._cancel_sent:
      CC.cruiseControl.cancel = True
      self._cancel_sent = True
    else:
      CC.cruiseControl.cancel = False
    CC.cruiseControl.override = False

    speeds = self.sm['longitudinalPlan'].speeds
    if speeds and CC.enabled and CS.cruiseState.standstill:
      CC.cruiseControl.resume = True
    else:
      CC.cruiseControl.resume = False

    hud = CC.hudControl
    hud.setSpeed = float(CS.vCruiseCluster * CV.KPH_TO_MS)
    hud.speedVisible = CC.enabled
    hud.lanesVisible = CC.enabled
    hud.leadVisible = self.sm['longitudinalPlan'].hasLead
    hud.leadDistanceBars = self.sm['selfdriveState'].personality.raw + 1
    hud.visualAlert = self.sm['selfdriveState'].alertHudVisual
    hud.leftLaneVisible = True
    hud.rightLaneVisible = True
    if self.sm.valid['driverAssistance']:
      hud.leftLaneDepart = self.sm['driverAssistance'].leftLaneDeparture
      hud.rightLaneDepart = self.sm['driverAssistance'].rightLaneDeparture

    if CC.enabled:
      CO = self.sm['carOutput']
      self.steer_limited_by_controls = abs(CC.actuators.torque - CO.actuatorsOutput.torque) > 1e-2

    # Publish messages
    dat = messaging.new_message('controlsState')
    dat.valid = CS.canValid
    cs = dat.controlsState
    cs.curvature = self.curvature
    cs.desiredCurvature = self.desired_curvature
    cs.longControlState = self.LoC.long_control_state
    cs.upAccelCmd = float(self.LoC.pid.p)
    cs.uiAccelCmd = float(self.LoC.pid.i)
    cs.ufAccelCmd = float(self.LoC.pid.f)
    if isinstance(lac_log, dict):
      cs.lateralControlState = lac_log
    else:
      setattr(cs.lateralControlState, 'pidState', lac_log)
    self.pm.send('controlsState', dat)

    cc_send = messaging.new_message('carControl')
    cc_send.valid = CS.canValid
    cc_send.carControl = CC
    self.pm.send('carControl', cc_send)

  def run(self):
    rk = Ratekeeper(100, print_delay_threshold=None)
    while True:
      self.update()
      CC, lac_log = self.state_control()
      self.publish(CC, lac_log)
      rk.monitor_time()


def main():
  config_realtime_process(4, Priority.CTRL_HIGH)
  controls = Controls()
  controls.run()

if __name__ == "__main__":
  main()
