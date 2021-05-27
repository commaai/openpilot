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

  t = 15
  print("TRUE")
  params.put_bool("UpdateAvailable", True)
  r = open(os.path.join(BASEDIR, "RELEASES.md"), "r").read()
  r = r[:r.find('\n\n')]  # Slice latest release notes
  params.put("ReleaseNotes", r + "\n")

  time.sleep(t)
  print("FALSE")
  params.put_bool("UpdateAvailable", False)
