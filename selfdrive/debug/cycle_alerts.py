#!/usr/bin/env python3
# flake8: noqa
# pylint: skip-file
# type: ignore

import time

from cereal import car
import cereal.messaging as messaging
from selfdrive.car.honda.interface import CarInterface
from selfdrive.controls.lib.events import ET, EVENTS, Events
from selfdrive.controls.lib.alertmanager import AlertManager

EventName = car.CarEvent.EventName

def cycle_alerts(duration=2000, is_metric=False):
  alerts = list(EVENTS.keys())
  print(alerts)

  #alerts = [EventName.preDriverDistracted, EventName.promptDriverDistracted, EventName.driverDistracted]
  alerts = [EventName.preLaneChangeLeft, EventName.preLaneChangeRight]

  CP = CarInterface.get_params("HONDA CIVIC 2016")
  sm = messaging.SubMaster(['deviceState', 'pandaState', 'roadCameraState', 'modelV2', 'liveCalibration',
                            'driverMonitoringState', 'longitudinalPlan', 'lateralPlan', 'liveLocationKalman'])

  pm = messaging.PubMaster(['controlsState', 'pandaState', 'deviceState'])

  events = Events()
  AM = AlertManager()

  frame = 0
  idx, last_alert_millis = 0, 0
  while 1:
    if frame % duration == 0:
      idx = (idx + 1) % len(alerts)
      events.clear()
      events.add(alerts[idx])


    current_alert_types = [ET.PERMANENT, ET.USER_DISABLE, ET.IMMEDIATE_DISABLE,
                           ET.SOFT_DISABLE, ET.PRE_ENABLE, ET.NO_ENTRY,
                           ET.ENABLE, ET.WARNING]
    a = events.create_alerts(current_alert_types, [CP, sm, is_metric])
    AM.add_many(frame, a)
    AM.process_alerts(frame)

    dat = messaging.new_message()
    dat.init('controlsState')
    dat.controlsState.alertText1 = AM.alert_text_1
    dat.controlsState.alertText2 = AM.alert_text_2
    dat.controlsState.alertSize = AM.alert_size
    dat.controlsState.alertStatus = AM.alert_status
    dat.controlsState.alertBlinkingRate = AM.alert_rate
    dat.controlsState.alertType = AM.alert_type
    dat.controlsState.alertSound = AM.audible_alert
    pm.send('controlsState', dat)

    dat = messaging.new_message()
    dat.init('deviceState')
    dat.deviceState.started = True
    pm.send('deviceState', dat)

    dat = messaging.new_message()
    dat.init('pandaState')
    dat.pandaState.ignitionLine = True
    pm.send('pandaState', dat)

    time.sleep(0.01)

if __name__ == '__main__':
  cycle_alerts()
