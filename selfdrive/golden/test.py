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
#from common.op_params import opParams


def main():

    #op_params = opParams()

    rk = Ratekeeper(1.0, print_delay_threshold=None)
    sm = messaging.SubMaster(['modelV2'])

    while 1:
        sm.update()

        log_text = 'not_valid:\n'
        service_list = sm.valid.keys()
        for s in service_list:
          if not sm.valid[s]:
            log_text += str(s)
            log_text += '\n'

        log_text += 'not_alive:\n'
        service_list = sm.alive.keys()
        for s in service_list:
          if not sm.alive[s]:
            if s not in sm.ignore_alive:
              log_text += str(s)
              log_text += '\n'

        text_file = open("/tmp/test.txt", "wt")
        text_file.write(log_text)
        text_file.close()

    #sm.update()
    rk.keep_time()

if __name__ == "__main__":
    main()
