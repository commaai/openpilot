#!/usr/bin/env python3
import os
import sys
import time

from common.basedir import BASEDIR
from common.params import Params
from selfdrive.controls.lib.alertmanager import set_offroad_alert

if __name__ == "__main__":
  params = Params()

  t = 10 if len(sys.argv) < 2 else int(sys.argv[1])
  while True:
    print("setting alert update")
    params.put_bool("UpdateAvailable", True)
    r = open(os.path.join(BASEDIR, "RELEASES.md")).read()
    r = r[:r.find('\n\n')]  # Slice latest release notes
    params.put("UpdaterNewReleaseNotes", r + "\n")

    time.sleep(t)
    params.put_bool("UpdateAvailable", False)

    # cycle through normal alerts
    for a in [k for k in params.all_keys() if k.startswith(b'Offroad_')]:
      print("setting alert:", a)
      set_offroad_alert(a, True, '{extra text}' if a in (b'Offroad_UpdateFailed', b'Offroad_ConnectivityNeededPrompt') else None)
      time.sleep(t)
      set_offroad_alert(a, False)

    print("no alert")
    time.sleep(t)
