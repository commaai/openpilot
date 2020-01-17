#!/usr/bin/env python3
import zmq
import time
from hexdump import hexdump
import cereal.messaging as messaging
from cereal.services import service_list
from cereal import log

def mock_x():
  liveMpc = messaging.pub_sock('liveMpc')
  while 1:
    m = messaging.new_message()
    mx = []
    m.init('liveMpc')
    for x in range(0, 100):
        mx.append(x*1.0)
        m.liveMpc.x = mx

    liveMpc.send(m.to_bytes())

if __name__=="__main__":
  mock_x()
