#!/usr/bin/env python
import time
from opendbc.safety import Safety
from panda import Panda

p = Panda()

while True:
  p.set_safety_mode(Safety.SAFETY_TOYOTA)
  p.send_heartbeat()
  print("ON")
  time.sleep(1)
  p.set_safety_mode(Safety.SAFETY_NOOUTPUT)
  p.send_heartbeat()
  print("OFF")
  time.sleep(1)

