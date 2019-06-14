#!/usr/bin/env python
import zmq
import time
import os
import json
import selfdrive.messaging as messaging
from selfdrive.services import service_list
from common.params import Params

def dashboard_thread():

  context = zmq.Context()
  poller = zmq.Poller()
  ipaddress = "127.0.0.1"
  vEgo = 0.0
  controlsState = messaging.sub_sock(context, service_list['controlsState'].port, addr=ipaddress, conflate=False, poller=poller)
  pathPlan = messaging.sub_sock(context, service_list['pathPlan'].port, addr=ipaddress, conflate=False, poller=poller)
  frame_count = 0

  try:
    if os.path.isfile('/data/kegman.json'):
      with open('/data/kegman.json', 'r') as f:
        config = json.load(f)
        user_id = config['userID']
    else:
        params = Params()
        user_id = params.get("DongleId")
  except:
    params = Params()
    user_id = params.get("DongleId")

  context = zmq.Context()
  steerpub = context.socket(zmq.PUSH)
  #steerpub.connect("tcp://kevo.live:8594")
  steerpub.connect("tcp://gernstation.synology.me:8593")
  influxFormatString = user_id + ",sources=capnp ff_standard=%s,angle_steers_des=%s,angle_steers=%s,deadzone=%s,path_angle_error=%s,steer_override=%s,v_ego=%s,p=%s,i=%s,f=%s %s\n"
  influxDataString = ""
  mapDataString = ""
  sendString = ""

  monoTimeOffset = 0
  receiveTime = 0

  while 1:
    for socket, event in poller.poll(0):
      if socket is pathPlan:
        _pathPlan = messaging.drain_sock(socket)
        for pp in _pathPlan:
          deadzone = pp.pathPlan.deadzone
          angle_error = pp.pathPlan.angleError

      if socket is controlsState:
        _controlsState = messaging.drain_sock(socket)
        for l100 in _controlsState:
          vEgo = l100.controlsState.vEgo
          receiveTime = int((monoTimeOffset + l100.logMonoTime) * .0000002) * 5
          if (abs(receiveTime - int(time.time() * 1000)) > 10000):
            monoTimeOffset = (time.time() * 1000000000) - l100.logMonoTime
            receiveTime = int((monoTimeOffset + l100.logMonoTime) * 0.0000002) * 5
          if vEgo > 0:

            influxDataString += ("%d,%0.2f,%0.2f,%0.2f,%0.2f,%d,%1f,%0.4f,%0.4f,%0.4f,%d|" %
                (1.0, l100.controlsState.angleSteers, l100.controlsState.angleSteersDes, deadzone, angle_error, l100.controlsState.steerOverride, vEgo,
                l100.controlsState.lateralControlState.pidState.p, l100.controlsState.lateralControlState.pidState.i, l100.controlsState.lateralControlState.pidState.f, receiveTime))

            frame_count += 1

    if frame_count >= 100:
      sendString = influxFormatString + "~" + influxDataString
      if mapDataString != "":
        sendString += "!" + mapFormatString + "~" + mapDataString
      steerpub.send_string(sendString)
      print("frames: %d   Characters: %d" % (frame_count, len(influxDataString)))
      frame_count = 0
      influxDataString = ""
      mapDataString = ""
    else:
      time.sleep(0.1)

def main():
  dashboard_thread()

if __name__ == "__main__":
  main()
