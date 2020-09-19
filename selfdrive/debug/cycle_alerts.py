#!/usr/bin/env python3
# flake8: noqa
# pylint: skip-file
# type: ignore

import time

import cereal.messaging as messaging
from selfdrive.car.honda.interface import CarInterface
from selfdrive.controls.lib.events import ET, EVENTS, Events
from selfdrive.controls.lib.alertmanager import AlertManager


def cycle_alerts(duration=200, is_metric=False):
  alerts = list(EVENTS.keys())
  print(alerts)

  CP = CarInterface.get_params("HONDA CIVIC 2016 TOURING")
  sm = messaging.SubMaster(['thermal', 'health', 'frame', 'model', 'liveCalibration',
                            'dMonitoringState', 'plan', 'pathPlan', 'liveLocationKalman'])

  controls_state = messaging.pub_sock('controlsState')
  thermal = messaging.pub_sock('thermal')

  idx, last_alert_millis = 0, 0

  events = Events()
  AM = AlertManager()

  frame = 0

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
    controls_state.send(dat.to_bytes())

    dat = messaging.new_message()
    dat.init('thermal')
    dat.thermal.started = True
    thermal.send(dat.to_bytes())

    frame += 1
    time.sleep(0.01)

if __name__ == '__main__':
  cycle_alerts()
