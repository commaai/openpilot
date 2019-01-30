#!/usr/bin/env python
import os
import struct
import zmq
import time

import selfdrive.messaging as messaging
from selfdrive.services import service_list

health_sock = messaging.pub_sock(zmq.Context(), service_list['health'].port)
started = False

while 1:
    # health packet @ 1hz
    if not started:
      msg = messaging.new_message()
      msg.init('health')

      # store the health to be logged
      msg.health.voltage = 12
      msg.health.current = 1
      msg.health.started = True
      health_sock.send(msg.to_bytes())
    time.sleep(10)
