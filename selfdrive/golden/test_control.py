#!/usr/bin/env python3
#pylint: skip-file
import os
import sys
#import time
#import math
#import atexit
#import numpy as np
#import threading
#import random
#import cereal.messaging as messaging
#import argparse
from common.realtime import Ratekeeper
#import queue
#import requests
import cereal.messaging.messaging_pyx as messaging_pyx
#import datetime
import json

def ping(ip):
    import subprocess
    status,result = subprocess.getstatusoutput("ping -c1 -W1 " + str(ip))
    return status

def create_sub_sock(ip, timeout):
    os.environ["ZMQ"] = "1"
    my_content = messaging_pyx.Context()
    sync_sock = messaging_pyx.SubSocket()
    addr = ip.encode('utf8')
    sync_sock.connect(my_content, 'testLiveLocation', addr, conflate=True)
    sync_sock.setTimeout(timeout)
    del os.environ["ZMQ"]
    return sync_sock, my_content

def main(s_ip):

  ip = '192.168.3.3'
  if s_ip:
    ip = s_ip

  sync_sock, sync_content = create_sub_sock(ip, timeout=1000)

  status = ping(ip)
  print('ping ' + ip + ' status=' + str(status))

  rk = Ratekeeper(5.0, print_delay_threshold=None)

  while 1:
    sync_data = None

    if sync_sock:
      sync_data = sync_sock.receive_golden()
      if sync_data:
        sync_data_str = sync_data.decode("utf-8")
        #print (sync_data_str)
        parsed_json = json.loads(sync_data_str)
        print (parsed_json)

    rk.keep_time()

if __name__ == "__main__":

  print ('Number of arguments:', len(sys.argv), 'arguments.')
  print ('Argument List:', str(sys.argv))

  main(sys.argv[1])
