#!/usr/bin/env python3
#import os
import time
#import math
#import atexit
#import numpy as np
#import threading
#import random
import cereal.messaging as messaging
#import argparse
#from common.params import Params
from common.realtime import Ratekeeper
#import queue
#import requests
#import cereal.messaging.messaging_pyx as messaging_pyx
#import datetime
#import json


def main():

  rk = Ratekeeper(5.0, print_delay_threshold=None)
  pm = messaging.PubMaster(['liveMapDataDEPRECATED'])

  while 1:

    time.sleep(1)

    speed_limit = 60.0
    has_exit = True
    dist_to_next_step = 1000.0
    remain_dist = 2500.0
    nav_icon = 2

    dat = messaging.new_message('liveMapDataDEPRECATED')
    dat.valid = True

    # struct LiveMapData {
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

    live_map_data = dat.liveMapDataDEPRECATED
    live_map_data.speedLimit = speed_limit * 1.08
    live_map_data.distToTurn = dist_to_next_step
    live_map_data.speedAdvisoryValid = has_exit
    live_map_data.speedAdvisory = remain_dist
    live_map_data.wayId = nav_icon

    pm.send('liveMapDataDEPRECATED', dat)

    #sm.update()
    rk.keep_time()

if __name__ == "__main__":
  main()
