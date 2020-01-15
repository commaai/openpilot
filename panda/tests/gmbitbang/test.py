#!/usr/bin/env python3
import time
from panda import Panda

p1 = Panda('380016000551363338383037')
p2 = Panda('430026000951363338383037')

# this is a test, no safety
p1.set_safety_mode(Panda.SAFETY_ALLOUTPUT)
p2.set_safety_mode(Panda.SAFETY_ALLOUTPUT)

# get versions
print(p1.get_version())
print(p2.get_version())

# this sets bus 2 to actually be GMLAN
p2.set_gmlan(bus=2)

# send w bitbang then without
#iden = 123
iden = 18000
#dat = "\x01\x02"
dat = "\x01\x02\x03\x04\x05\x06\x07\x08"
while 1:
  iden += 1
  p1.set_gmlan(bus=None)
  p1.can_send(iden, dat, bus=3)
  #p1.set_gmlan(bus=2)
  #p1.can_send(iden, dat, bus=3)
  time.sleep(0.01)
  print(p2.can_recv())
  #exit(0)

