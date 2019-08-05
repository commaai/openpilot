#!/usr/bin/env python2

import os
import requests

from helpers import safety_modes
from replay_drive import replay_drive
from tools.lib.logreader import LogReader

BASE_URL = "https://commadataci.blob.core.windows.net/openpilotci/"

# (route, safety mode, param)
logs = [
  ("b0c9d2329ad1606b|2019-05-30--20-23-57.bz2", "HONDA", 0), # HONDA.CIVIC
  ("38bfd238edecbcd7|2019-06-07--10-15-25.bz2", "TOYOTA", 66), # TOYOTA.PRIUS
  ("f89c604cf653e2bf|2018-09-29--13-46-50.bz2", "GM", 0), # GM.VOLT
  ("0375fdf7b1ce594d|2019-05-21--20-10-33.bz2", "HONDA_BOSCH", 1), # HONDA.ACCORD
  ("02ec6bea180a4d36|2019-04-17--11-21-35.bz2", "HYUNDAI", 0), # HYUNDAI.SANTA_FE
  ("03efb1fda29e30fe|2019-02-21--18-03-45.bz2", "CHRYSLER", 0), # CHRYSLER.PACIFICA_2018_HYBRID
  ("791340bc01ed993d|2019-04-08--10-26-00.bz2", "SUBARU", 0), # SUBARU.IMPREZA
]

if __name__ == "__main__":
  for route, _, _ in logs:
    if not os.path.isfile(route):
      with open(route, "w") as f:
        f.write(requests.get(BASE_URL + route).content)

  failed = []
  for route, mode, param in logs:
    lr = LogReader(route)
    m = safety_modes.get(mode, mode)

    print "\nreplaying %s with safety mode %d and param %s" % (route, m, param)
    if not replay_drive(lr, m, int(param)):
      failed.append(route)

    for f in failed:
      print "\n**** failed on %s ****" % f
    assert len(failed) == 0, "\nfailed on %d logs" % len(failed)

