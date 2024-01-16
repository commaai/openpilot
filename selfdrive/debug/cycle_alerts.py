#!/usr/bin/env python3
import time
import random

from cereal import car, log
import cereal.messaging as messaging
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.car.honda.interface import CarInterface
from openpilot.selfdrive.controls.lib.events import ET, Events
from openpilot.selfdrive.controls.lib.alertmanager import AlertManager

EventName = car.CarEvent.EventName

def randperc() -> float:
  return 100. * random.random()

def cycle_alerts(duration=200, is_metric=False):
  # this plays each type of audible alert
  alerts = [
    (EventName.buttonEnable, ET.ENABLE),
    (EventName.buttonCancel, ET.USER_DISABLE),
    (EventName.wrongGear, ET.NO_ENTRY),

    # (EventName.locationdTemporaryError, ET.SOFT_DISABLE),
    # (EventName.paramsdTemporaryError, ET.SOFT_DISABLE),
    (EventName.accFaulted, ET.IMMEDIATE_DISABLE),

    # DM sequence
    (EventName.preDriverDistracted, ET.WARNING),
    (EventName.promptDriverDistracted, ET.WARNING),
    (EventName.driverDistracted, ET.WARNING),
  ]

  # debug alerts
  alerts += [
    #(EventName.highCpuUsage, ET.NO_ENTRY),
    #(EventName.lowMemory, ET.PERMANENT),
    #(EventName.overheat, ET.PERMANENT),
    #(EventName.outOfSpace, ET.PERMANENT),
    #(EventName.modeldLagging, ET.PERMANENT),
    #(EventName.processNotRunning, ET.NO_ENTRY),
    #(EventName.commIssue, ET.NO_ENTRY),
    #(EventName.calibrationInvalid, ET.PERMANENT),
    (EventName.cameraMalfunction, ET.PERMANENT),
    (EventName.cameraFrameRate, ET.PERMANENT),
  ]

  cameras = ['roadCameraState', 'wideRoadCameraState', 'driverCameraState']

  CS = car.CarState.new_message()
  CP = CarInterface.get_non_essential_params("HONDA CIVIC 2016")
  sm = messaging.SubMaster(['deviceState', 'pandaStates', 'roadCameraState', 'modelV2', 'liveCalibration',
                            'driverMonitoringState', 'longitudinalPlan', 'lateralPlan', 'liveLocationKalman',
                            'managerState'] + cameras)

  pm = messaging.PubMaster(['controlsState', 'pandaStates', 'deviceState'])

  events = Events()
  AM = AlertManager()

  frame = 0
  while True:
    for alert, et in alerts:
      events.clear()
      events.add(alert)

      a = events.create_alerts([et, ], [CP, CS, sm, is_metric, 0])
      AM.add_many(frame, a)
      alert = AM.process_alerts(frame, [])
      print(alert)
      for _ in range(duration):
        dat = messaging.new_message('controlsState')
        dat.controlsState.enabled = False

        if alert:
          dat.controlsState.alertText1 = alert.alert_text_1
          dat.controlsState.alertText2 = alert.alert_text_2
          dat.controlsState.alertSize = alert.alert_size
          dat.controlsState.alertStatus = alert.alert_status
          dat.controlsState.alertBlinkingRate = alert.alert_rate
          dat.controlsState.alertType = alert.alert_type
          dat.controlsState.alertSound = alert.audible_alert
        pm.send('controlsState', dat)

        dat = messaging.new_message('deviceState')
        dat.deviceState.started = True
        pm.send('deviceState', dat)

        dat = messaging.new_message('pandaStates', 1)
        dat.pandaStates[0].ignitionLine = True
        dat.pandaStates[0].pandaType = log.PandaState.PandaType.uno
        pm.send('pandaStates', dat)

        frame += 1
        time.sleep(DT_CTRL)

if __name__ == '__main__':
  cycle_alerts()
