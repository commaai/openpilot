#!/usr/bin/env python3
from cereal import car
from common.params import Params
import cereal.messaging as messaging
from selfdrive.controls.lib.events import Events
from selfdrive.monitoring.driver_monitor import DriverStatus, MAX_TERMINAL_ALERTS, MAX_TERMINAL_DURATION
from selfdrive.locationd.calibrationd import Calibration


def dmonitoringd_thread(sm=None, pm=None):
  if pm is None:
    pm = messaging.PubMaster(['dMonitoringState'])

  if sm is None:
    sm = messaging.SubMaster(['driverState', 'liveCalibration', 'carState', 'controlsState', 'model'], poll=['driverState'])

  driver_status = DriverStatus()
  driver_status.is_rhd_region = Params().get("IsRHD") == b"1"

  offroad = Params().get("IsOffroad") == b"1"

  sm['liveCalibration'].calStatus = Calibration.INVALID
  sm['liveCalibration'].rpyCalib = [0, 0, 0]
  sm['carState'].vEgo = 0.
  sm['carState'].cruiseState.speed = 0.
  sm['carState'].buttonEvents = []
  sm['carState'].steeringPressed = False
  sm['carState'].gasPressed = False
  sm['carState'].standstill = True

  v_cruise_last = 0
  driver_engaged = False

  # 10Hz <- dmonitoringmodeld
  while True:
    sm.update()

    if not sm.updated['driverState']:
      continue

    # Get interaction
    if sm.updated['carState']:
      v_cruise = sm['carState'].cruiseState.speed
      driver_engaged = len(sm['carState'].buttonEvents) > 0 or \
                        v_cruise != v_cruise_last or \
                        sm['carState'].steeringPressed or \
                        sm['carState'].gasPressed
      if driver_engaged:
        driver_status.update(Events(), True, sm['controlsState'].enabled, sm['carState'].standstill)
      v_cruise_last = v_cruise

    if sm.updated['model']:
      driver_status.set_policy(sm['model'])

    # Get data from dmonitoringmodeld
    events = Events()
    driver_status.get_pose(sm['driverState'], sm['liveCalibration'].rpyCalib, sm['carState'].vEgo, sm['controlsState'].enabled)

    # Block engaging after max number of distrations
    if driver_status.terminal_alert_cnt >= MAX_TERMINAL_ALERTS or driver_status.terminal_time >= MAX_TERMINAL_DURATION:
      events.add(car.CarEvent.EventName.tooDistracted)

    # Update events from driver state
    driver_status.update(events, driver_engaged, sm['controlsState'].enabled, sm['carState'].standstill)

    # build dMonitoringState packet
    dat = messaging.new_message('dMonitoringState')
    dat.dMonitoringState = {
      "events": events.to_msg(),
      "faceDetected": driver_status.face_detected,
      "isDistracted": driver_status.driver_distracted,
      "awarenessStatus": driver_status.awareness,
      "isRHD": driver_status.is_rhd_region,
      "posePitchOffset": driver_status.pose.pitch_offseter.filtered_stat.mean(),
      "posePitchValidCount": driver_status.pose.pitch_offseter.filtered_stat.n,
      "poseYawOffset": driver_status.pose.yaw_offseter.filtered_stat.mean(),
      "poseYawValidCount": driver_status.pose.yaw_offseter.filtered_stat.n,
      "stepChange": driver_status.step_change,
      "awarenessActive": driver_status.awareness_active,
      "awarenessPassive": driver_status.awareness_passive,
      "isLowStd": driver_status.pose.low_std,
      "hiStdCount": driver_status.hi_stds,
      "isPreview": offroad,
    }
    pm.send('dMonitoringState', dat)

def main(sm=None, pm=None):
  dmonitoringd_thread(sm, pm)

if __name__ == '__main__':
  main()
