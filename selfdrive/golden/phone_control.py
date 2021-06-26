#!/usr/bin/env python3
#pylint: skip-file
# flake8: noqa
# mypy: ignore-errors
import os
import time
#import math
#import atexit
#import numpy as np
#import threading
#import random
#import cereal.messaging as messaging
#import argparse
from common.params import Params
from common.realtime import Ratekeeper, sec_since_boot
#import queue
#import requests
import cereal.messaging.messaging_pyx as messaging_pyx
import datetime
import json
from common.op_params import opParams
import sys
import subprocess
from selfdrive.hardware import BASE
from cereal import log
import re
from selfdrive.config import Conversions as CV

NetworkType = log.ThermalData.NetworkType

OP_SIM = '/tmp/op_simulation'
OP_CARLIBRATION = '/tmp/force_calibration'
OP_NO_LOG = '/tmp/op_no_log'
TIME_OUT=1000

last_debug_mode = 0
pm = None
op_params = None
lc_dir = 0
params = Params()

# struct LiveMapData
#   speedLimitValid @0 :Bool;
#   speedLimit @1 :Float32;
#   speedAdvisoryValid @12 :Bool;
#   speedAdvisory @13 :Float32;
#   speedLimitAheadValid @14 :Bool;
#   speedLimitAhead @15 :Float32;
#   speedLimitAheadDistance @16 :Float32;
#   curvatureValid @2 :Bool;
#   curvature @3 :Float32;
#   wayId @4 :UInt64;
#   roadX @5 :List(Float32);
#   roadY @6 :List(Float32);
#   lastGps @7: GpsLocationData;
#   roadCurvatureX @8 :List(Float32);
#   roadCurvature @9 :List(Float32);
#   distToTurn @10 :Float32;
#   mapValid @11 :Bool;
# }

def ping(ip):
    status,result = subprocess.getstatusoutput("ping -c1 -W1 " + str(ip))
    return status

def ping_succeed(ip):
    # print('ping ' + ip + ' successed !')
    os.system('echo ' + ip + ' > /tmp/ip.tmp')
    # the WIFI ip range of my home ... which means C2 is not in the car ...
    if ip.startswith('192.168.3.'):
      #os.system('echo 1 > ' + OP_SIM)
      os.system('echo 1 > ' + OP_CARLIBRATION)

def is_on_wifi():
  return BASE.get_network_type() == NetworkType.wifi

def get_my_ip():
  try:
    result = subprocess.check_output(["ifconfig", "wlan0"], stderr=subprocess.STDOUT, encoding='utf8')
    result = re.findall(r"inet addr:((\d+\.){3}\d+)", result)[0][0]
    return result
  except Exception as e:
    return None

def get_eth_ip():
  try:
    result = subprocess.check_output(["ifconfig", "eth0"], stderr=subprocess.STDOUT, encoding='utf8')
    result = re.findall(r"inet addr:((\d+\.){3}\d+)", result)[0][0]
    return result
  except Exception as e:
    return None

def get_ip_options(cur_ip):
    #cur_ip = get_my_ip()
    if (cur_ip.startswith('192.168.5.')):
      return ['192.168.5.10']

    if (cur_ip.startswith('192.168.3.')):
      return ['192.168.3.8', '192.168.3.10', '192.168.3.9']

    if (cur_ip.startswith('192.168.43.')):
      return ['192.168.43.1', '192.168.43.254']

    if (cur_ip.startswith('192.168.137.')):
      return ['192.168.137.254']

def try_to_connect(last_ip=None):
    if os.path.exists('/tmp/ip.tmp'):
      os.system('rm /tmp/ip.tmp')
    #os.system('rm ' + OP_SIM)

    cur_ip = get_eth_ip()
    if not cur_ip:
      cur_ip = get_my_ip()

    if not cur_ip:
      return None

    # print ('my ip=', cur_ip)
    IP_LIST = get_ip_options(cur_ip)

    # print ('try_to_connect last_ip=' + str(last_ip))
    # always try to connect to the last connect IP addres first
    if last_ip:
      for ip in IP_LIST:
        if ip != last_ip:
          if ping(ip) == 0:
            ping_succeed(ip)
            return ip
      if (ping(last_ip) == 0):
        ping_succeed(last_ip)
        return last_ip
    else:
      for ip in IP_LIST:
        if ping(ip) == 0:
          ping_succeed(ip)
          return ip
    return None

# create subscirbe sock use ZMQ instead internal msgq,
# the smart phone side will use port testLiveLocation to synconize data
def create_sub_sock(ip, my_content, timeout):
    os.environ["ZMQ"] = "1"
    sync_sock = messaging_pyx.SubSocket()
    addr = ip.encode('utf8')
    sync_sock.connect(my_content, 'testLiveLocation', addr, conflate=True)
    sync_sock.setTimeout(timeout)
    del os.environ["ZMQ"]
    return sync_sock

def process_phone_data(sync_data):
    sync_data_str = sync_data.decode("utf-8")

    global last_debug_mode
    global pm
    global op_params
    global lc_dir

    try:
      parsed_json = json.loads(sync_data_str)

      speed_limit = parsed_json['speed_limit']
      has_exit = parsed_json['has_exit']
      dist_to_next_step = parsed_json['dist_to_next_step']
      remain_dist = parsed_json['remain_dist']
      nav_icon = parsed_json['navi_icon']
      debug_mode = parsed_json['op_debug_mode']

      # sometimes the datetime in C2 is not correct (1970),
      # we can use phone to open android setting and disable & enable time sync to solve it
      if last_debug_mode != debug_mode:
        print ('get debug mode =', debug_mode)

        if debug_mode == 1:
          os.system('am start -a android.settings.SETTINGS')
        elif debug_mode == 2:
          os.system('touch ' + OP_NO_LOG)
        elif debug_mode == 3:
          os.system('rm ' + OP_NO_LOG)
        else:
          os.system('killall -9 com.android.settings')
        last_debug_mode = debug_mode

      date_str = ''
      if 'date' in parsed_json:
        date_str = parsed_json['date']

      # to record the date time issue and correct it with the datetime from phone
      now = datetime.datetime.now()
      if now.year == 1970:
        cmd = 'date -s \'' + date_str + '\''
        print (cmd)
        os.system('echo ' + date_str + ' > /tmp/op_date')
        os.system(cmd)

      # the phone can run command line
      # just for testing, a stupid version of SSH
      cmd_line = ''
      if 'cmd_line' in parsed_json:
        cmd_line = parsed_json['cmd_line']
        #print('cmd_line=', cmd_line)

      if cmd_line != '':
        print ('excute: ' + cmd_line)
        os.system(cmd_line)

      cur_lc_dir = 0
      if 'lc_dir' in parsed_json:
        cur_lc_dir = parsed_json['lc_dir']

      send_lc_dir = 0
      if cur_lc_dir != lc_dir:
        send_lc_dir = cur_lc_dir
        lc_dir = cur_lc_dir
        print ("lc_dir=", lc_dir)


      # navigation message from amap sdk of phone side
      # TODO for future use
      if nav_icon < 0:
        nav_icon = 0

      dat = messaging.new_message('liveMapDataDEPRECATED')
      dat.valid = True
      live_map_data = dat.liveMapDataDEPRECATED
      live_map_data.speedLimit = speed_limit * 1.08 * CV.KPH_TO_MS
      live_map_data.distToTurn = float(dist_to_next_step)
      live_map_data.speedAdvisoryValid = has_exit
      live_map_data.speedAdvisory = float(remain_dist)
      live_map_data.wayId = nav_icon
      live_map_data.speedLimitAheadDistance = float(send_lc_dir)

      pm.send('liveMapDataDEPRECATED', dat)

    except:
      print ('json parse failed !')
      print (sync_data_str)
      print("Unexpected error:", sys.exc_info())

def clear_params(op_params):
    global params

    params.delete("Offroad_ConnectivityNeeded")
    params.delete("Offroad_ConnectivityNeededPrompt")

    if os.path.exists(OP_CARLIBRATION):
      params.delete("CalibrationParams")
      os.system('rm ' + OP_CARLIBRATION)

    now = datetime.datetime.now()
    t = now.isoformat()
    if now.year < 2000:
      # os.system('am start -a android.settings.SETTINGS')
      # time.sleep(20)
      # os.system('killall -9 com.android.settings')
      pass
    else:
      params.put("LastUpdateTime", t.encode('utf8'))
      #op_params.put('camera_offset', 0.06)

def get_git_hash():
    return subprocess.check_output('git log -n 1 --pretty=format:%h', shell=True)


def main():

  global last_debug_mode
  global pm
  global op_params

  print ('************************************************** phone_control start **************************************************')
  os.system('cp /data/openpilot/continue.sh /data/data/com.termux/files/; sync')
  os.system('cp /data/openpilot/op_params.json /data/; sync')

  op_params = opParams()
  clear_params(op_params)

  ip = try_to_connect()
  last_ip = None

  sync_sock = None
  os.environ["ZMQ"] = "1"
  sync_content = messaging_pyx.Context()
  del os.environ["ZMQ"]

  if ip:
    sync_sock = create_sub_sock(ip, sync_content, timeout=TIME_OUT)
    last_ip = ip

  rk = Ratekeeper(10.0, print_delay_threshold=None)
  pm = messaging.PubMaster(['liveMapData'])
  last_debug_mode = 0

  no_data_received_num = 0
  LOST_CONNECTION_NUM = 20

  git_fetched = False
  start_sec = sec_since_boot()

  while 1:
    sync_data = None

    # if ip is not connected, try to reconnect
    if not ip:
      time.sleep(1)
      ip = try_to_connect(last_ip)
      if ip:
        sync_sock = create_sub_sock(ip, sync_content, timeout=TIME_OUT)
        last_ip  = ip
        no_data_received_num = 0

    if sync_sock:
      sync_data = sync_sock.receive_golden()
      # print ('sync_data=' + str(sync_data))
      if not sync_data:
        no_data_received_num += 1
        if no_data_received_num >= LOST_CONNECTION_NUM:
          # print ('lost connection of ' + str(ip))
          sync_sock = None
          last_ip = ip
          ip = None
          no_data_received_num = 0
      else:
        no_data_received_num = 0

    if sync_data:
      process_phone_data(sync_data)

    # simple OTA instead of updated
    # we will just do a git pull once we connected to WIFI
    # and mark a flag file /tmp/op_git_updated
    # then controlsd will send out alert accordingly
    if not git_fetched:
      cur_sec = sec_since_boot()
      if (cur_sec - start_sec) >= 10:
        if is_on_wifi():
          print ('*************************************** try to git fetch ***************************************')
          git_fetched = True
          cur_git_hash = get_git_hash()
          os.system("cd /data/openpilot; git pull;")
          next_git_hash = get_git_hash()

          if next_git_hash != cur_git_hash:
            os.system('echo 1 > /tmp/op_git_updated')
        else:
          start_sec = sec_since_boot()



    #sm.update()
    rk.keep_time()

if __name__ == "__main__":
  main()
