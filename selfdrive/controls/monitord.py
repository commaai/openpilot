#!/usr/bin/env python3
import gc
from common.realtime import sec_since_boot, set_realtime_priority
from common.params import Params, put_nonblocking
import cereal.messaging as messaging
from selfdrive.controls.lib.drive_helpers import create_event, EventTypes as ET
from selfdrive.controls.lib.driver_monitor import DriverStatus, MAX_TERMINAL_ALERTS, MAX_TERMINAL_DURATION
from selfdrive.locationd.calibration_helpers import Calibration
from selfdrive.controls.lib.gps_helpers import is_rhd_region

def monitord_thread(sm=None, pm=None):
  gc.disable()

  # start the loop
  set_realtime_priority(3)

  params = Params()

  # Pub/Sub Sockets
  if pm is None:
    pm = messaging.PubMaster(['monitorState'])

  if sm is None:
    sm = messaging.SubMaster(['driverMonitoring', 'liveCalibration', 'carState', 'model', 'gpsLocation'], ignore_alive=['gpsLocation'])

  driver_status = DriverStatus()
  is_rhd = params.get("IsRHD")
  if is_rhd is not None:
    driver_status.is_rhd = bool(int(is_rhd))

  sm['liveCalibration'].calStatus = Calibration.INVALID
  v_cruise_last = 0

  # wait until carState live
  CS_updated = False
  while(not CS_updated):
    sm.update(0)
    CS_updated = sm.updated['carState']

  # 10Hz <- monitoringd
  while(True):
    start_time = sec_since_boot()
    sm.update(0)

    # Get data from monitoringd
    if not sm.updated['driverMonitoring']:
      continue

    driver_status.get_pose(sm['driverMonitoring'], cal_rpy, CS.vEgo, CS.cruiseState.enabled)

    CS = sm['carState']
    events = []

    # GPS coords RHD parsing, once every restart
    if not driver_status.is_rhd_region_checked and sm.updated['gpsLocation']:
      is_rhd = is_rhd_region(sm['gpsLocation'].latitude, sm['gpsLocation'].longitude)
      driver_status.is_rhd_region = is_rhd
      driver_status.is_rhd_region_checked = True
      put_nonblocking("IsRHD", "1" if is_rhd else "0")

    # Handle calibration
    cal_status = sm['liveCalibration'].calStatus
    cal_rpy = [0,0,0]
    if cal_status == Calibration.CALIBRATED:
      rpy = sm['liveCalibration'].rpyCalib
      if len(rpy) == 3:
        cal_rpy = rpy

    # Get model meta
    if sm.updated['model']:
      driver_status.set_policy(sm['model'])

    # Block any engage after certain distrations
    if driver_status.terminal_alert_cnt >= MAX_TERMINAL_ALERTS or driver_status.terminal_time >= MAX_TERMINAL_DURATION:
      events.append(create_event("tooDistracted", [ET.NO_ENTRY]))

    # Update events from driver state
    v_cruise = CS.cruiseState.speed
    driver_engaged = len(CS.buttonEvents) > 0 or \
                   v_cruise != v_cruise_last or \
                   CS.steeringPressed
    v_cruise_last = v_cruise
    events = driver_status.update(events, driver_engaged, CS.cruiseState.enabled, CS.standstill)

    # monitorState packet
    dat = messaging.new_message()
    dat.init('monitorState')
    dat.monitorState.events = events
    dat.monitorState.driverState = {
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
    pm.send('monitorState', dat)

def main(sm=None, pm=None):
  monitord_thread(sm, pm)

if __name__ == '__main__':
  main()