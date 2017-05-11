#!/usr/bin/env python
import os
import zmq

import selfdrive.messaging as messaging
from selfdrive.services import service_list

from common.fingerprints import fingerprint

def bpressed(CS, btype):
  for b in CS.buttonEvents:
    if b.type == btype:
      return True
  return False

def test_loop():
  context = zmq.Context()
  logcan = messaging.sub_sock(context, service_list['can'].port)

  CP = fingerprint(logcan)
  exec('from selfdrive.car.'+CP.carName+'.interface import CarInterface')

  CI = CarInterface(CP, logcan, None)

  state = 0

  states = [
    "'seatbeltNotLatched' in CS.errors",
    "CS.gasPressed",
    "CS.brakePressed",
    "CS.steeringPressed",
    "bpressed(CS, 'leftBlinker')",
    "bpressed(CS, 'rightBlinker')",
    "bpressed(CS, 'cancel')",
    "bpressed(CS, 'accelCruise')",
    "bpressed(CS, 'decelCruise')",
    "bpressed(CS, 'altButton1')",
    "'doorOpen' in CS.errors",
    "False"]

  while 1:
    # read CAN
    CS = CI.update()

    while eval(states[state]) == True:
      state += 1

    print "IN STATE %d: waiting for %s" % (state, states[state])
    #print CS

if __name__ == "__main__":
  test_loop()

