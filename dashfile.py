#!/usr/bin/env python
import csv
import zmq
import time
import numpy as np
import selfdrive.messaging as messaging
from selfdrive.services import service_list
from common.realtime import set_realtime_priority, Ratekeeper
import os, os.path

# Polling rate should be twice the data rate to prevent aliasing
def main(rate=100):
  set_realtime_priority(5)
  context = zmq.Context()
  poller = zmq.Poller()

  live100 = messaging.sub_sock(context, service_list['live100'].port, conflate=False, poller=poller)
  carState = messaging.sub_sock(context, service_list['carState'].port, conflate=True, poller=poller)
  can = None #messaging.sub_sock(context, service_list['can'].port, conflate=True, poller=poller)

  vEgo = 0.0
  _live100 = None
  _can = None

  frame_count = 0
  skipped_count = 0

  rk = Ratekeeper(rate, print_delay_threshold=np.inf)

  # simple version for working with CWD
  #print len([name for name in os.listdir('.') if os.path.isfile(name)])

  # path joining version for other paths
  DIR = '/sdcard/tuning'
  filenumber = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])

  kegman_counter = 0
  monoTimeOffset = 0
  receiveTime = 0
  angle_rate = 0.0

  print("start")
  with open(DIR + '/dashboard_file_%d.csv' % filenumber, mode='w') as dash_file:
    print("opened")
    dash_writer = csv.writer(dash_file, delimiter=',', quotechar='', quoting=csv.QUOTE_NONE)
    print("initialized")
    dash_writer.writerow(['ff_rate','ff_angle', 'angleGain','rateGain','actualNoise','angle_steers_des','angle_steers','dampened_angle_steers_des','v_ego','steer_override','p','i','f','time'])
    print("first row")

    while 1:
      for socket, event in poller.poll(0):
        if socket is can:
          _can = messaging.recv_one(socket)
          print(_can)

        if socket is carState:
          _carState = messaging.drain_sock(socket)
          for cs in _carState:
            angle_rate = cs.carState.steeringRate

        if socket is live100:
          _live100 = messaging.drain_sock(socket)
          for l100 in _live100:
            vEgo = l100.live100.vEgo
            if vEgo > 0: # and l100.live100.active:
              receiveTime = int(monoTimeOffset + l100.logMonoTime)
              if (abs(receiveTime - int(time.time() * 1000000000)) > 10000000000):
                monoTimeOffset = (time.time() * 1000000000) - l100.logMonoTime
                receiveTime = int(monoTimeOffset + l100.logMonoTime)
              frame_count += 1
              dash_writer.writerow([str(round(1.0 - l100.live100.angleFFRatio, 2)),
                                    str(round(l100.live100.angleFFRatio, 2)),
                                    str(round(l100.live100.angleFFGain, 2)),
                                    str(round(l100.live100.rateFFGain, 5)),
                                    str(round(l100.live100.angleSteersNoise, 2)),
                                    str(round(l100.live100.angleSteersDes, 2)),
                                    str(round(l100.live100.angleSteers, 2)),
                                    str(round(l100.live100.dampAngleSteersDes, 2)),
                                    str(round(l100.live100.vEgo, 1)),
                                    1 if l100.live100.steerOverride else 0,
                                    str(round(l100.live100.upSteer, 4)),
                                    str(round(l100.live100.uiSteer, 4)),
                                    str(round(l100.live100.ufSteer, 4)),
                                    str(receiveTime)])
          else:
            skipped_count += 1
        else:
          skipped_count += 1
      if frame_count % 200 == 0:
        print("captured = %d" % frame_count)
        frame_count += 1
      if skipped_count % 200 == 0:
        print("skipped = %d" % skipped_count)
        skipped_count += 1

      rk.keep_time()

if __name__ == "__main__":
  main()
