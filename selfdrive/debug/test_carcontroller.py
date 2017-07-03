#!/usr/bin/env python
from evdev import InputDevice
from select import select
import time
import numpy as np
import zmq

from cereal import car

import selfdrive.messaging as messaging
from selfdrive.services import service_list
from common.realtime import Ratekeeper

from common.fingerprints import fingerprint

if __name__ == "__main__":
  # ***** connect to joystick *****
  # we use a Mad Catz V.1
  dev = InputDevice("/dev/input/event8")
  print dev

  button_values = [0]*7
  axis_values = [0.0, 0.0, 0.0]

  # ***** connect to car *****
  context = zmq.Context()
  logcan = messaging.sub_sock(context, service_list['can'].port)
  sendcan = messaging.pub_sock(context, service_list['sendcan'].port)

  CP = fingerprint(logcan)
  exec('from selfdrive.car.'+CP.carName+'.interface import CarInterface')

  CI = CarInterface(CP, logcan, sendcan)

  rk = Ratekeeper(100)

  while 1:
    # **** handle joystick ****
    r, w, x = select([dev], [], [], 0.0)
    if dev in r:
      for event in dev.read():
        # button event
        if event.type == 1:
          btn = event.code - 288 
          if btn >= 0 and btn < 7:
            button_values[btn] = int(event.value)

        # axis move event
        if event.type == 3:
          if event.code < 3:
            if event.code == 2:
              axis_values[event.code] = np.clip((255-int(event.value))/250.0, 0.0, 1.0)
            else:
              DEADZONE = 5 
              if event.value-DEADZONE < 128 and event.value+DEADZONE > 128:
                event.value = 128 
              axis_values[event.code] = np.clip((int(event.value)-128)/120.0, -1.0, 1.0)

    print axis_values, button_values
    # **** handle car ****

    CS = CI.update()
    #print CS

    CC = car.CarControl.new_message()

    CC.enabled = True

    CC.gas = float(np.clip(-axis_values[1], 0, 1.0))
    CC.brake = float(np.clip(axis_values[1], 0, 1.0))
    CC.steeringTorque = float(-axis_values[0])

    CC.hudControl.speedVisible = bool(button_values[1])
    CC.hudControl.lanesVisible = bool(button_values[2])
    CC.hudControl.leadVisible = bool(button_values[3])

    CC.cruiseControl.override = bool(button_values[0])
    CC.cruiseControl.cancel = bool(button_values[-1])

    CC.hudControl.setSpeed = float(axis_values[2] * 100.0)

    # TODO: test alerts
    CC.hudControl.visualAlert = "none"
    CC.hudControl.audibleAlert = "none"

    #print CC

    if not CI.apply(CC):
      print "CONTROLS FAILED"

    rk.keep_time()



