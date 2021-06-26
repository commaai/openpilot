#!/usr/bin/env python3
#import os
#import time
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
from common.op_params import opParams


def main():

    op_params = opParams()

    rk = Ratekeeper(1.0, print_delay_threshold=None)
    sm = messaging.SubMaster(['liveMapData'])

    while 1:
        sm.update()

        if sm.updated['liveMapData']:
            print (sm['liveMapData'])

        camera_offset = op_params.get('camera_offset')
        print (camera_offset)

    #sm.update()
    rk.keep_time()

if __name__ == "__main__":
    main()
