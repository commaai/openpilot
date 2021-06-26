#!/usr/bin/env python3
#pylint: skip-file
import os
import time
#import math
#import atexit
#import numpy as np
#import threading
#import random
import cereal.messaging as messaging
#import argparse
from common.params import Params
from common.realtime import Ratekeeper
from selfdrive.golden.can import can_function, sendcan_function
from selfdrive.car.honda.values import CruiseButtons
#import subprocess
import sys
import signal
import threading
from queue import Queue
from selfdrive.golden.keyboard_ctrl import keyboard_poll_thread, keyboard_shutdown

params = Params()

def shutdown():
  global params
  global pm

  print('shutdown !')

  keyboard_shutdown()

  params.delete("CalibrationParams")

  dat = messaging.new_message('pandaState')
  dat.valid = True
  dat.pandaState = {
    'ignitionLine': False,
    'pandaType': "uno",
    'controlsAllowed': True,
    'safetyModel': "hondaNidec"
  }

  for seq in range(10):
    pm.send('pandaState', dat)
    time.sleep(0.1)

  print ("exiting")
  sys.exit(0)

def main():

  global params
  global pm

  params.delete("Offroad_ConnectivityNeeded")
  params.delete("CalibrationParams")
  params.put("CalibrationParams", '{"calib_radians": [0,0,0], "valid_blocks": 20}')

  os.system('rm /tmp/op_git_updated')
  os.system('touch /tmp/op_simulation')

  start_loggerd = False
  if len(sys.argv) > 1:
    start_loggerd = (sys.argv[1] == '1')

  print ('start_loggerd=', start_loggerd)

  if start_loggerd:
    os.system('cd /data/openpilot/; ./selfdrive/loggerd/loggerd &')

  os.system('echo 1 > /tmp/force_calibration')

  # make volume 0
  os.system('service call audio 3 i32 3 i32 0 i32 1')

  q = Queue()

  t = threading.Thread(target=keyboard_poll_thread, args=[q])
  t.start()

  pm = messaging.PubMaster(['can', 'pandaState'])

  # can loop
  sendcan = messaging.sub_sock('sendcan')
  rk = Ratekeeper(100, print_delay_threshold=None)
  steer_angle = 0.0
  speed = 50.0 / 3.6
  cruise_button = 0

  btn_list = []
  btn_hold_times = 2

  frames = 0

  while 1:
    # check keyboard input
    if not q.empty():
      message = q.get()
      print (message)

      if (message == 'quit'):
        shutdown()
        return

      m = message.split('_')
      if m[0] == "cruise":
        if m[1] == "down":
          cruise_button = CruiseButtons.DECEL_SET
          if len(btn_list) == 0:
            for x in range(btn_hold_times):
              btn_list.append(cruise_button)
        if m[1] == "up":
          cruise_button = CruiseButtons.RES_ACCEL
          if len(btn_list) == 0:
            for x in range(btn_hold_times):
              btn_list.append(cruise_button)
        if m[1] == "cancel":
          cruise_button = CruiseButtons.CANCEL
          if len(btn_list) == 0:
            for x in range(btn_hold_times):
              btn_list.append(cruise_button)

    btn = 0
    if len(btn_list) > 0:
      btn = btn_list[0]
      btn_list.pop(0)

    # print ('cruise_button=', cruise_button)
    can_function(pm, speed * 3.6, steer_angle, rk.frame, cruise_button=btn, is_engaged=1)
    if rk.frame%5 == 0:
      throttle, brake, steer = sendcan_function(sendcan)
      steer_angle += steer/10000.0 # torque
      #print(speed * 3.6, steer, throttle, brake)

    if frames % 20 == 0:
      dat = messaging.new_message('pandaState')
      dat.valid = True
      dat.pandaState = {
        'ignitionLine': True,
        'pandaType': "uno",
        'controlsAllowed': True,
        'safetyModel': "hondaNidec",
        'fanSpeedRpm' : 1000
      }
      pm.send('pandaState', dat)

    frames += 1

    rk.keep_time()

  shutdown()

def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    shutdown()

if __name__ == "__main__":
  signal.signal(signal.SIGINT, signal_handler)

  print (sys.argv)
  print ("input 1 to curse resume/+")
  print ("input 2 to curse set/-")
  print ("input 3 to curse cancel")
  print ("input q to quit")

  main()
