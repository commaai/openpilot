#!/usr/bin/env python3
# flake8: noqa
# pylint: skip-file
# type: ignore

import argparse
import time

import cereal.messaging as messaging
from selfdrive.controls.lib.events import EVENTS, Alert

def now_millis(): return time.time() * 1000

ALERTS = [a for _, et in EVENTS.items() for _, a in et.items() if isinstance(a, Alert)]

#from cereal import car
#ALERTS = [a for a in ALERTS if a.audible_alert == car.CarControl.HUDControl.AudibleAlert.chimeWarningRepeat]

default_alerts = sorted(ALERTS, key=lambda alert: (alert.alert_size, len(alert.alert_text_2)))

def cycle_alerts(duration_millis, alerts=None):
  if alerts is None:
    alerts = default_alerts

  controls_state = messaging.pub_sock('controlsState')

  idx, last_alert_millis = 0, 0
  alert = alerts[0]
  while 1:
    if (now_millis() - last_alert_millis) > duration_millis:
      alert = alerts[idx]
      idx = (idx + 1) % len(alerts)
      last_alert_millis = now_millis()
      print('sending {}'.format(str(alert)))

    dat = messaging.new_message()
    dat.init('controlsState')

    dat.controlsState.alertType = alert.alert_type
    dat.controlsState.alertText1 = alert.alert_text_1
    dat.controlsState.alertText2 = alert.alert_text_2
    dat.controlsState.alertSize = alert.alert_size
    #dat.controlsState.alertStatus = alert.alert_status
    dat.controlsState.alertSound = alert.audible_alert
    controls_state.send(dat.to_bytes())

    time.sleep(0.01)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--duration', type=int, default=1000)
  parser.add_argument('--alert-types', nargs='+')
  args = parser.parse_args()
  alerts = None
  if args.alert_types:
    alerts = [next(a for a in ALERTS if a.alert_type==alert_type) for alert_type in args.alert_types]

  cycle_alerts(args.duration, alerts=alerts)
