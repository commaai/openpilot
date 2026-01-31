#!/usr/bin/env python
import time
from opendbc.car.structs import CarParams
from panda import Panda

p = Panda()

while True:
  p.set_safety_mode(CarParams.SafetyModel.toyota)
  p.send_heartbeat()
  print("ON")
  time.sleep(1)
  p.set_safety_mode(CarParams.SafetyModel.noOutput)
  p.send_heartbeat()
  print("OFF")
  time.sleep(1)

