# USAGE: python cycle_alerts.py [duration_millis=1000]
# Then start manager

import argparse
import time
import zmq

import cereal.messaging as messaging
from cereal.services import service_list
from selfdrive.controls.lib.alerts import ALERTS

def now_millis(): return time.time() * 1000

default_alerts = sorted(ALERTS, key=lambda alert: (alert.alert_size, len(alert.alert_text_2)))

def cycle_alerts(duration_millis, alerts=None):
    if alerts is None:
      alerts = default_alerts

    controls_state = messaging.pub_sock('controlsState')

    last_pop_millis = now_millis()
    alert = alerts.pop()
    while 1:
        if (now_millis() - last_pop_millis) > duration_millis:
            alerts.insert(0, alert)
            alert = alerts.pop()
            last_pop_millis = now_millis()
            print('sending {}'.format(str(alert)))

        dat = messaging.new_message()
        dat.init('controlsState')

        dat.controlsState.alertType = alert.alert_type
        dat.controlsState.alertText1 = alert.alert_text_1
        dat.controlsState.alertText2 = alert.alert_text_2
        dat.controlsState.alertSize = alert.alert_size
        dat.controlsState.alertStatus = alert.alert_status
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
