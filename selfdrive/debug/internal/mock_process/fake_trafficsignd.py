#!/usr/bin/env python3
import zmq
import time
from cereal.services import service_list
import cereal.messaging as messaging
from cereal import log

def mock():
  traffic_events = messaging.pub_sock('trafficEvents')

  while 1:
    m = messaging.new_message()
    m.init('trafficEvents', 1)
    m.trafficEvents[0].type = log.TrafficEvent.Type.stopSign
    m.trafficEvents[0].resuming = False
    m.trafficEvents[0].distance = 100.
    m.trafficEvents[0].action = log.TrafficEvent.Action.stop
    traffic_events.send(m.to_bytes())
    time.sleep(0.01)

if __name__=="__main__":
  mock()
