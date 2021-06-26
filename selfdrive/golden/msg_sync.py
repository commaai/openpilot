#!/usr/bin/env python3
#pylint: skip-file
#from cereal import car
#from common.params import Params
import cereal.messaging as messaging
import os
import cereal.messaging.messaging_pyx as messaging_pyx
import time
#from cereal import log
#import threading
#import numpy as np
#import math

def msg_sync_thread():

  #start_sec = round(time.time())
  sync_topics = ['modelV2', 'carState', 'liveTracks', 'radarState', 'controlsState']
  # dMonitoringState', 'gpsLocation', 'radarState', 'model', 'gpsLocationExternal',
  # 'pathPlan', 'liveCalibration', laneSpeed

  frame_counts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  SLEEP_TIME = 0.01
  frame_limits = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

  sub_list = []
  pub_list = []

  for topic in sync_topics:
    sub_list.append(messaging.sub_sock(topic, conflate=True, timeout=100))

  os.environ["ZMQ"] = "1"
  pub_context = messaging_pyx.Context()
  for topic in sync_topics:
    pub = messaging_pyx.PubSocket()
    pub.connect(pub_context, topic)
    pub_list.append(pub)
  del os.environ["ZMQ"]

  while True:
    for index, sub in enumerate(sub_list):
      try:
        data = sub.receive()
        if data:
          #print ('sending data ' + sync_topics[index])
          frame_limit = frame_limits[index]
          if frame_limit != 0 and (frame_counts[index] % frame_limit != 0):
            pass
          else:
            pub_list[index].send(data)
          frame_counts[index] += 1
      except messaging_pyx.MessagingError:
        print('msg_sync MessagingError error happens ' + sync_topics[index])

    time.sleep(SLEEP_TIME)

def main():
  msg_sync_thread()

if __name__ == "__main__":
  main()
