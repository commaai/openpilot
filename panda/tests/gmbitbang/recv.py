#!/usr/bin/env python3
from panda import Panda

p = Panda()
p.set_safety_mode(Panda.SAFETY_ALLOUTPUT)
p.set_gmlan(bus=2)
#p.can_send(0xaaa, b"\x00\x00", bus=3)
last_add = None
while 1:
  ret = p.can_recv()
  if len(ret) > 0:
    add = ret[0][0]
    if last_add is not None and add != last_add + 1:
      print("MISS: ", last_add, add)
    last_add = add
    print(ret)
