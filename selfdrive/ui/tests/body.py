#!/usr/bin/env python3
import time
import cereal.messaging as messaging

if __name__ == "__main__":
  while True:
    pm = messaging.PubMaster(['carParams', 'carState'])
    batt = 1.
    while True:
      msg = messaging.new_message('carParams')
      msg.carParams.carName = "BODY"
      msg.carParams.notCar = True
      pm.send('carParams', msg)

      for b in range(100, 0, -1):
        msg = messaging.new_message('carState')
        msg.carState.charging = True
        msg.carState.fuelGauge = b / 100.
        pm.send('carState', msg)
        time.sleep(0.1)

      time.sleep(1)
