#!/usr/bin/env python3
import zmq
import time
from hexdump import hexdump
import cereal.messaging as messaging
from cereal.services import service_list
from cereal import log

def leadRange(start, end, step):
    x = start
    while x < end:
        yield x
        x += (x * step)

def mock_lead():
  radarState = messaging.pub_sock('radarState')
  while 1:
    m = messaging.new_message()
    m.init('radarState')
    m.radarState.leadOne.status = True
    for x in leadRange(3.0, 65.0, 0.005):
        m.radarState.leadOne.dRel = x
        radarState.send(m.to_bytes())
        time.sleep(0.01)

if __name__=="__main__":
  mock_lead()
