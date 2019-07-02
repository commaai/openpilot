#!/usr/bin/env python
import zmq
import time
import os
import json
import selfdrive.messaging as messaging
from selfdrive.services import service_list
from common.params import Params

def dashboard_thread(rate=100):

  kegman_valid = True

  #url_string = 'http://192.168.1.61:8086/write?db=carDB'
  #url_string = 'http://192.168.43.221:8086/write?db=carDB'
  #url_string = 'http://192.168.137.1:8086/write?db=carDB'
  #url_string = 'http://kevo.live:8086/write?db=carDB'

  context = zmq.Context()
  poller = zmq.Poller()
  ipaddress = "127.0.0.1"
  vEgo = 0.0
  controlsState = messaging.sub_sock(service_list['controlsState'].port)
  #controlsState = messaging.sub_sock(context, service_list['controlsState'].port, addr=ipaddress, conflate=False, poller=poller)
  carState = None #messaging.sub_sock(context, service_list['carState'].port, addr=ipaddress, conflate=False, poller=poller)
  liveMap = None #messaging.sub_sock(context, service_list['liveMapData'].port, addr=ipaddress, conflate=False, poller=poller)
  liveStreamData = None #messaging.sub_sock(context, 8600, addr=ipaddress, conflate=False, poller=poller)
  osmData = None #messaging.sub_sock(context, 8601, addr=ipaddress, conflate=False, poller=poller)
  canData = None #messaging.sub_sock(context, 8602, addr=ipaddress, conflate=False, poller=poller)
  #pathPlan = messaging.sub_sock(context, service_list['pathPlan'].port, addr=ipaddress, conflate=False, poller=poller)
  pathPlan = None #messaging.sub_sock(context, service_list['plan'].port, addr=ipaddress, conflate=False, poller=poller)

  #gpsNMEA = messaging.sub_sock(context, service_list['gpsNMEA'].port, addr=ipaddress, conflate=True)

  #_controlsState = None

  frame_count = 0

  #server_address = "tcp://kevo.live"
  server_address = "tcp://gernstation.synology.me"
  #server_address = "tcp://192.168.137.1"
  #server_address = "tcp://192.168.1.3"

  context = zmq.Context()
  steerPush = context.socket(zmq.PUSH)
  steerPush.connect(server_address + ":8593")
  tunePush = context.socket(zmq.PUSH)
  tunePush.connect(server_address + ":8595")
  tuneSub = context.socket(zmq.SUB)
  tuneSub.connect(server_address + ":8596")
  poller.register(tuneSub, zmq.POLLIN)
  poller.register(controlsState, zmq.POLLIN)

  try:
    if os.path.isfile('/data/kegman.json'):
      with open('/data/kegman.json', 'r') as f:
        config = json.load(f)
        user_id = config['userID']
        tunePush.send_json(config)
        tunePush = None
    else:
        params = Params()
        user_id = params.get("DongleId")
  except:
    params = Params()
    user_id = params.get("DongleId")
    config['userID'] = user_id
    tunePush.send_json(config)
    tunePush = None

  lateral_type = ""
  tuneSub.setsockopt(zmq.SUBSCRIBE, str(user_id))
  #influxFormatString = user_id + ",sources=capnp angle_accel=%s,damp_angle_rate=%s,angle_rate=%s,damp_angle=%s,apply_steer=%s,noise_feedback=%s,ff_standard=%s,ff_rate=%s,ff_angle=%s,angle_steers_des=%s,angle_steers=%s,dampened_angle_steers_des=%s,steer_override=%s,v_ego=%s,p2=%s,p=%s,i=%s,f=%s %s\n"
  mapFormatString = "location,user=" + user_id + " latitude=%s,longitude=%s,altitude=%s,speed=%s,bearing=%s,accuracy=%s,speedLimitValid=%s,speedLimit=%s,curvatureValid=%s,curvature=%s,wayId=%s,distToTurn=%s,mapValid=%s,speedAdvisoryValid=%s,speedAdvisory=%s,speedAdvisoryValid=%s,speedAdvisory=%s,speedLimitAheadValid=%s,speedLimitAhead=%s,speedLimitAheadDistance=%s %s\n"
  canFormatString="CANData,user=" + user_id + ",src=%s,pid=%s d1=%si,d2=%si "
  liveStreamFormatString = "curvature,user=" + user_id + " l_curv=%s,p_curv=%s,r_curv=%s,map_curv=%s,map_rcurv=%s,map_rcurvx=%s,v_curv=%s,l_diverge=%s,r_diverge=%s %s\n"
  pathFormatString = "pathPlan,user=" + user_id + " d0=%s,d1=%s,d2=%s,d3=%s %s\n"
  pathDataFormatString = "%d|"
  polyDataString = "%.10f,%0.8f,%0.6f,%0.4f,"
  pathDataString = ""
  influxDataString = ""
  kegmanDataString = ""
  liveStreamDataString = ""
  mapDataString = ""
  insertString = ""
  canInsertString = ""

  lastGPStime = 0
  lastMaptime = 0

  monoTimeOffset = 0
  receiveTime = 0
  active = False

  while 1:
    for socket, event in poller.poll(0):
      if socket is osmData:
        if vEgo > 0 and active:
          _osmData = osmData.recv_multipart()
          #print(_osmData)

      if socket is tuneSub:
        config = json.loads(tuneSub.recv_multipart()[1])
        #print(config)
        with open('/data/kegman.json', 'w') as f:
          json.dump(config, f, indent=2, sort_keys=True)
          os.chmod("/data/kegman.json", 0o764)

      if socket is liveStreamData:
        livestream = liveStreamData.recv_string() + str(receiveTime) + "|"
        if vEgo > 0 and active: liveStreamDataString += livestream

      if socket is canData:
        canString = canData.recv_string()
        #print(canString)
        if vEgo > 0 and active: canInsertString += canFormatString + str(receiveTime) + "\n~" + canString + "!"

      if socket is liveMap:
        _liveMap = messaging.drain_sock(socket)
        for lmap in _liveMap:
          if vEgo > 0 and active:
            receiveTime = int((monoTimeOffset + lmap.logMonoTime) * .0000002) * 5
            if (abs(receiveTime - int(time.time() * 1000)) > 10000):
              monoTimeOffset = (time.time() * 1000000000) - lmap.logMonoTime
              receiveTime = int((monoTimeOffset + lmap.logMonoTime) * 0.0000002) * 5
            lm = lmap.liveMapData
            lg = lm.lastGps
            #print(lm)
            mapDataString += ("%f,%f,%f,%f,%f,%f,%d,%f,%d,%f,%f,%f,%d,%d,%f,%d,%f,%d,%f,%f,%d|" %
                  (lg.latitude ,lg.longitude ,lg.altitude ,lg.speed ,lg.bearing ,lg.accuracy ,lm.speedLimitValid ,lm.speedLimit ,lm.curvatureValid
                  ,lm.curvature ,lm.wayId ,lm.distToTurn ,lm.mapValid ,lm.speedAdvisoryValid ,lm.speedAdvisory ,lm.speedAdvisoryValid ,lm.speedAdvisory
                  ,lm.speedLimitAheadValid ,lm.speedLimitAhead , lm.speedLimitAheadDistance , receiveTime))

      if socket is pathPlan:
        _pathPlan = messaging.drain_sock(socket)
        for _pp in _pathPlan:
          pp = _pp.plan
          if vEgo > 0 and active and (carState == None or boolStockRcvd):
            boolStockRcvd = False
            #pathDataString += polyDataString % tuple(map(float, pp.lPoly))
            #pathDataString += polyDataString % tuple(map(float, pp.rPoly))
            #pathDataString += polyDataString % tuple(map(float, pp.cPoly))
            pathDataString += polyDataString % tuple(map(float, pp.dPoly))
            #pathDataString += polyDataString % tuple(map(float, pp.pPoly))
            pathDataString +=  (pathDataFormatString % (int((monoTimeOffset + _pp.logMonoTime) * .0000002) * 5))

      if socket is controlsState:
        _controlsState = messaging.drain_sock(socket)
        #print("controlsState")
        for l100 in _controlsState:
          if lateral_type == "":
            if l100.controlsState.lateralControlState.which == "pidState":
              lateral_type = "pid"
              influxFormatString = user_id + ",sources=capnp ff_angle=%s,damp_angle_steers_des=%s,angle_steers_des=%s,angle_steers=%s,steer_override=%s,v_ego=%s,p=%s,i=%s,f=%s,output=%s %s\n"
              kegmanFormatString = user_id + ",sources=kegman KpV=%s,KiV=%s,Kf=%s %s\n"
            else:
              lateral_type = "indi"
              influxFormatString = user_id + ",sources=capnp angle_steers_des=%s,damp_angle_steers_des=%s,angle_steers=%s,damp_angle_steers_des=%s,steer_override=%s,v_ego=%s,output=%s,indi_angle=%s,indi_rate=%s,indi_rate_des=%s,indi_accel=%s,indi_accel_des=%s,accel_error=%s,delayed_output=%s,indi_delta=%s %s\n"
              kegmanFormatString = user_id + ",sources=kegman time_const=%s,act_effect=%s,inner_gain=%s,outer_gain=%s %s\n"
          vEgo = l100.controlsState.vEgo
          active = l100.controlsState.active
          #active = True
          #vEgo = 1.
          #print(active)
          receiveTime = int((monoTimeOffset + l100.logMonoTime) * .0000002) * 5
          if (abs(receiveTime - int(time.time() * 1000)) > 10000):
            monoTimeOffset = (time.time() * 1000000000) - l100.logMonoTime
            receiveTime = int((monoTimeOffset + l100.logMonoTime) * 0.0000002) * 5
          if vEgo > 0 and active:
            dat = l100.controlsState
            #print(dat)

            if lateral_type == "pid":
              influxDataString += ("%0.3f,%0.3f,%0.3f,%0.2f,%d,%0.1f,%0.4f,%0.4f,%0.4f,%0.4f,%d|" %
                  (dat.lateralControlState.pidState.angleFFRatio, dat.dampAngleSteersDes, dat.angleSteersDes, dat.angleSteers,  dat.steerOverride, vEgo,
                  dat.lateralControlState.pidState.p, dat.lateralControlState.pidState.i, dat.lateralControlState.pidState.f,dat.lateralControlState.pidState.output, receiveTime))
            else:
              s = dat.lateralControlState.indiState
              influxDataString += ("%0.3f,%0.2f,%0.2f,%d,%0.1f,%0.4f,%0.4f,%0.4f,%0.4f,%0.4f,%0.4f,%0.4f,%0.4f,%0.4f,%d|" %
                  (dat.angleSteersDes, dat.dampAngleSteersDes, dat.angleSteers,  dat.steerOverride, vEgo,
                  s.output, s.steerAngle, s.steerRate, s.rateSetPoint, s.steerAccel, s.accelSetPoint, s.accelError, s.delayedOutput, s.delta, receiveTime))

            #print(dat.upFine, dat.uiFine)
            frame_count += 1

    #if lastGPStime + 2.0 <= time.time():
    #  lastGPStime = time.time()
    #  _gps = messaging.recv_one_or_none(gpsNMEA)
    #  print(_gps)
    #if lastMaptime + 2.0 <= time.time():
    #  lastMaptime = time.time()
    #  _map = messaging.recv_one_or_none(liveMap)

    '''liveMapData = (
    speedLimitValid = false,
    speedLimit = 0,
    curvatureValid = false,
    curvature = 0,
    wayId = 0,
    lastGps = (
      flags = 0,
      latitude = 44.7195573,
      longitude = -100.8218663,
      altitude = 10542.853000000001,
      speed = 0,
      bearing = 0,
      accuracy = 4294967.5,
      timestamp = 1556581592999,
      source = ublox,
      vNED = [0, 0, 0],
      verticalAccuracy = 3750000.2,
      bearingAccuracy = 180,
      speedAccuracy = 20.001 ),
    distToTurn = 0,
    mapValid = false,
    speedAdvisoryValid = false,
    speedAdvisory = 0,
    speedLimitAheadValid = false,
    speedLimitAhead = 0,
    speedLimitAheadDistance = 0 ) )
    '''
    #else:
    #  print(time.time())

    #print(frame_count)
    if frame_count >= 100:
      if kegman_valid:
        try:
          if os.path.isfile('/data/kegman.json'):
            with open('/data/kegman.json', 'r') as f:
              config = json.load(f)
              if lateral_type == "pid":
                steerKpV = config['Kp']
                steerKiV = config['Ki']
                steerKf = config['Kf']

                kegmanDataString += ("%s,%s,%s,%s|" % \
                      (steerKpV, steerKiV, steerKf, receiveTime))
              else:
                timeConst = config['timeConst']
                actEffect = config['actEffect']
                innerGain = config['innerGain']
                outerGain = config['outerGain']

                kegmanDataString += ("%s,%s,%s,%s,%s|" % \
                      (timeConst, actEffect, innerGain, outerGain, receiveTime))
                print(kegmanDataString, kegmanFormatString)
              insertString += kegmanFormatString + "~" + kegmanDataString + "!"

        except:
          kegman_valid = False

      if liveStreamDataString != "":
        insertString += liveStreamFormatString + "~" + liveStreamDataString + "!"
        #print(insertString)
        liveStreamDataString =""
      insertString += influxFormatString + "~" + influxDataString + "!"
      insertString += pathFormatString + "~" + pathDataString + "!"
      insertString += mapFormatString + "~" + mapDataString + "!"
      #insertString += canInsertString
      #print(canInsertString)
      steerPush.send_string(insertString)
      print(len(insertString))
      frame_count = 0
      influxDataString = ""
      kegmanDataString = ""
      mapDataString = ""
      pathDataString = ""
      insertString = ""
      canInsertString = ""
    else:
      time.sleep(0.01)

def main(rate=200):
  dashboard_thread(rate)

if __name__ == "__main__":
  main()
