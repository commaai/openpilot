#!/usr/bin/env python3
import gc

import cereal.messaging as messaging
from cereal import car
from openpilot.common.params import Params
from openpilot.common.realtime import set_realtime_priority
from openpilot.selfdrive.controls.lib.events import Events
from openpilot.selfdrive.monitoring.driver_monitor import DriverStatus


def dmonitoringd_thread():
  gc.disable()
  set_realtime_priority(2)

  params = Params()
  pm = messaging.PubMaster(['driverMonitoringState'])
  sm = messaging.SubMaster(['driverStateV2', 'liveCalibration', 'carState', 'controlsState', 'modelV2'], poll='driverStateV2')

  driver_status = DriverStatus(rhd_saved=params.get_bool("IsRhdDetected"), always_on=params.get_bool("AlwaysOnDM"))

  driver_engaged = False

  # 20Hz <- dmonitoringmodeld
  while True:
    sm.update()
    if not sm.updated['driverStateV2']:
      continue

    # Get interaction
    if sm.updated['carState']:
      driver_engaged = sm['carState'].steeringPressed or \
                       sm['carState'].gasPressed

    if sm.updated['modelV2']:
      driver_status.set_policy(sm['modelV2'], sm['carState'].vEgo)

    # Get data from dmonitoringmodeld
    events = Events()

    if sm.all_checks() and len(sm['liveCalibration'].rpyCalib):
      driver_status.update_states(sm['driverStateV2'], sm['liveCalibration'].rpyCalib, sm['carState'].vEgo, sm['controlsState'].enabled)

    # Block engaging after max number of distrations or when alert active
    if driver_status.terminal_alert_cnt >= driver_status.settings._MAX_TERMINAL_ALERTS or \
       driver_status.terminal_time >= driver_status.settings._MAX_TERMINAL_DURATION or \
       driver_status.always_on and driver_status.awareness <= driver_status.threshold_prompt:
      events.add(car.CarEvent.EventName.tooDistracted)

    # Update events from driver state
    driver_status.update_events(events, driver_engaged, sm['controlsState'].enabled,
      sm['carState'].standstill, sm['carState'].gearShifter in [car.CarState.GearShifter.reverse, car.CarState.GearShifter.park], sm['carState'].vEgo)

    # build driverMonitoringState packet
    dat = messaging.new_message('driverMonitoringState', valid=sm.all_checks())
    dat.driverMonitoringState = {
      "events": events.to_msg(),
      "faceDetected": driver_status.face_detected,
      "isDistracted": driver_status.driver_distracted,
      "distractedType": sum(driver_status.distracted_types),
      "awarenessStatus": driver_status.awareness,
      "posePitchOffset": driver_status.pose.pitch_offseter.filtered_stat.mean(),
      "posePitchValidCount": driver_status.pose.pitch_offseter.filtered_stat.n,
      "poseYawOffset": driver_status.pose.yaw_offseter.filtered_stat.mean(),
      "poseYawValidCount": driver_status.pose.yaw_offseter.filtered_stat.n,
      "stepChange": driver_status.step_change,
      "awarenessActive": driver_status.awareness_active,
      "awarenessPassive": driver_status.awareness_passive,
      "isLowStd": driver_status.pose.low_std,
      "hiStdCount": driver_status.hi_stds,
      "isActiveMode": driver_status.active_monitoring_mode,
      "isRHD": driver_status.wheel_on_right,
    }
    pm.send('driverMonitoringState', dat)

    if sm['driverStateV2'].frameId % 40 == 1:
      driver_status.always_on = params.get_bool("AlwaysOnDM")

    # save rhd virtual toggle every 5 mins
    if (sm['driverStateV2'].frameId % 6000 == 0 and
     driver_status.wheelpos_learner.filtered_stat.n > driver_status.settings._WHEELPOS_FILTER_MIN_COUNT and
     driver_status.wheel_on_right == (driver_status.wheelpos_learner.filtered_stat.M > driver_status.settings._WHEELPOS_THRESHOLD)):
      params.put_bool_nonblocking("IsRhdDetected", driver_status.wheel_on_right)

def main():
  dmonitoringd_thread()


if __name__ == '__main__':
  main()
