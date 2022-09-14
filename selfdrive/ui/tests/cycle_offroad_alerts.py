#!/usr/bin/env python3
import os
import sys
import time

from common.basedir import BASEDIR
from common.params import Params
from selfdrive.controls.lib.alertmanager import set_offroad_alert

if __name__ == "__main__":
  params = Params()

  offroad_alerts = (
    'Offroad_TemperatureTooHigh',
    'Offroad_ConnectivityNeeded',
    'Offroad_UpdateFailed',
    'Offroad_InvalidTime',
    'Offroad_UnofficialHardware',
    'Offroad_StorageMissing',
    'Offroad_BadNvme',
    'Offroad_CarUnrecognized',
    'Offroad_ConnectivityNeededPrompt',
    'Offroad_NoFirmware',
    'Offroad_IsTakingSnapshot',
    'Offroad_NeosUpdate',
  )

  t = 10 if len(sys.argv) < 2 else int(sys.argv[1])
  while True:
    print("setting alert update")
    params.put_bool("UpdateAvailable", True)
    r = open(os.path.join(BASEDIR, "RELEASES.md")).read()
    r = r[:r.find('\n\n')]  # Slice latest release notes
    params.put("ReleaseNotes", r + "\n")

    time.sleep(t)
    params.put_bool("UpdateAvailable", False)

    # cycle through normal alerts
    for a in offroad_alerts:
      print("setting alert:", a)
      set_offroad_alert(a, True, '{extra text}' if a in ('Offroad_UpdateFailed', 'Offroad_ConnectivityNeededPrompt') else None)
      time.sleep(t)
      set_offroad_alert(a, False)

    print("no alert")
    time.sleep(t)
