#!/usr/bin/env python3
import time

import cereal.messaging as messaging
from common.params import Params
from common.realtime import DT_DMON

def main():
  pm = messaging.PubMaster(['controlsState', 'dMonitoringState'])

  rhd = Params().get("IsRHD") == b"1"

  while True:
    dat = messaging.new_message('dMonitoringState')
    dat.dMonitoringState = {
      "isRHD": rhd,
      "isPreview": True,
    }
    pm.send('dMonitoringState', dat)

    time.sleep(DT_DMON)

if __name__ == '__main__':
  main()
