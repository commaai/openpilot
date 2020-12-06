#!/usr/bin/env python3
import os
import sys
import time
import json

from common.basedir import BASEDIR
from common.params import Params
from selfdrive.controls.lib.alertmanager import set_offroad_alert

if __name__ == "__main__":
  params = Params()

  with open(os.path.join(BASEDIR, "selfdrive/controls/lib/alerts_offroad.json")) as f:
    offroad_alerts = json.load(f)

  t = 10 if len(sys.argv) < 2 else int(sys.argv[1])
  while True:
    print("setting alert update")
    params.put("UpdateAvailable", "1")
    params.put("ReleaseNotes", "this is a new version")
    time.sleep(t)
    params.put("UpdateAvailable", "0")

    # cycle through normal alerts
    for a in offroad_alerts:
      print("setting alert:", a)
      set_offroad_alert(a, True)
      time.sleep(t)
      set_offroad_alert(a, False)

    print("no alert")
    time.sleep(t)
