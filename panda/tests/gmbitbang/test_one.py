#!/usr/bin/env python
import time
from panda import Panda

p = Panda()
p.set_safety_mode(Panda.SAFETY_ALLOUTPUT)

# ack any crap on bus
p.set_gmlan(bus=2)
time.sleep(0.1)
while len(p.can_recv()) > 0:
  print "clearing"
  time.sleep(0.1)
print "cleared"
p.set_gmlan(bus=None)

iden = 18000
dat = "\x01\x02\x03\x04\x05\x06\x07\x08"
while 1:
  iden += 1
  p.can_send(iden, dat, bus=3)
  time.sleep(0.01)

