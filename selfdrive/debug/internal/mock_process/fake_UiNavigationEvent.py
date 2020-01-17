#!/usr/bin/env python3
import zmq
import time
from cereal.services import service_list
import cereal.messaging as messaging
from cereal import log 
  
def mock():
  traffic_events = messaging.pub_sock('uiNavigationEvent')

  while 1:
    m = messaging.new_message()
    m.init('uiNavigationEvent')
    m.uiNavigationEvent.type = log.UiNavigationEvent.Type.mergeRight
    m.uiNavigationEvent.status = log.UiNavigationEvent.Status.active
    m.uiNavigationEvent.distanceTo = 100.
    traffic_events.send(m.to_bytes())
    time.sleep(0.01)

if __name__=="__main__":
  mock()
