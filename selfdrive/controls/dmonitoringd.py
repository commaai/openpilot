#!/usr/bin/env python3
import gc
from common.realtime import set_realtime_priority
from common.params import Params, put_nonblocking
import cereal.messaging as messaging
from selfdrive.controls.lib.drive_helpers import create_event, EventTypes as ET
from selfdrive.controls.lib.driver_monitor import DriverStatus, MAX_TERMINAL_ALERTS, MAX_TERMINAL_DURATION
from selfdrive.locationd.calibration_helpers import Calibration
from selfdrive.controls.lib.gps_helpers import is_rhd_region

def dmonitoringd_thread(sm=None, pm=None):
  gc.disable()

  # start the loop
  set_realtime_priority(3)

  params = Params()

  # Pub/Sub Sockets
  if pm is None:
    pm = messaging.PubMaster(['dMonitoringState'])

  if sm is None:
    sm = messaging.SubMaster(['driverState', 'liveCalibration', 'carState', 'model', 'gpsLocation'], ignore_alive=['gpsLocation'])

  driver_status = DriverStatus()
  is_rhd = params.get("IsRHD")
  if is_rhd is not None:
    driver_status.is_rhd_region = bool(int(is_rhd))

  sm['liveCalibration'].calStatus = Calibration.INVALID
  sm['carState'].vEgo = 0.
  sm['carState'].cruiseState.enabled = False
  sm['carState'].cruiseState.speed = 0.
  sm['carState'].buttonEvents = []
  sm['carState'].steeringPressed = False
  sm['carState'].standstill = True

  cal_rpy = [0,0,0]
  v_cruise_last = 0
  driver_engaged = False

  # 10Hz <- dmonitoringmodeld
  while True:
    sm.update()

    # GPS coords RHD parsing, once every restart
    if not driver_status.is_rhd_region_checked and sm.updated['gpsLocation']:
      is_rhd = is_rhd_region(sm['gpsLocation'].latitude, sm['gpsLocation'].longitude)
      driver_status.is_rhd_region = is_rhd
      driver_status.is_rhd_region_checked = True
      put_nonblocking("IsRHD", "1" if is_rhd else "0")

    # Handle calibration
    if sm.updated['liveCalibration']:
      if sm['liveCalibration'].calStatus == Calibration.CALIBRATED:
        if len(sm['liveCalibration'].rpyCalib) == 3:
          cal_rpy = sm['liveCalibration'].rpyCalib

    # Get interaction
    if sm.updated['carState']:
      v_cruise = sm['carState'].cruiseState.speed
      driver_engaged = len(sm['carState'].buttonEvents) > 0 or \
                        v_cruise != v_cruise_last or \
                        sm['carState'].steeringPressed
      v_cruise_last = v_cruise

    # Get model meta
    if sm.updated['model']:
      driver_status.set_policy(sm['model'])

    # Get data from dmonitoringmodeld
    if sm.updated['driverState']:
      events = []
      driver_status.get_pose(sm['driverState'], cal_rpy, sm['carState'].vEgo, sm['carState'].cruiseState.enabled)
      # Block any engage after certain distrations
      if driver_status.terminal_alert_cnt >= MAX_TERMINAL_ALERTS or driver_status.terminal_time >= MAX_TERMINAL_DURATION:
        events.append(create_event("tooDistracted", [ET.NO_ENTRY]))
      # Update events from driver state
      events = driver_status.update(events, driver_engaged, sm['carState'].cruiseState.enabled, sm['carState'].standstill)

      # dMonitoringState packet
      dat = messaging.new_message()
      dat.init('dMonitoringState')
      dat.dMonitoringState = {
        "events": events,
        "faceDetected": driver_status.face_detected,
        "isDistracted": driver_status.driver_distracted,
        "awarenessStatus": driver_status.awareness,
        "isRHD": driver_status.is_rhd_region,
        "rhdChecked": driver_status.is_rhd_region_checked,
        "posePitchOffset": driver_status.pose.pitch_offseter.filtered_stat.mean(),
        "posePitchValidCount": driver_status.pose.pitch_offseter.filtered_stat.n,
        "poseYawOffset": driver_status.pose.yaw_offseter.filtered_stat.mean(),
        "poseYawValidCount": driver_status.pose.yaw_offseter.filtered_stat.n,
        "stepChange": driver_status.step_change,
        "awarenessActive": driver_status.awareness_active,
        "awarenessPassive": driver_status.awareness_passive,
        "isLowStd": driver_status.pose.low_std,
        "hiStdCount": driver_status.hi_stds,
      }
      pm.send('dMonitoringState', dat)

def main(sm=None, pm=None):
  dmonitoringd_thread(sm, pm)

if __name__ == '__main__':
  main()