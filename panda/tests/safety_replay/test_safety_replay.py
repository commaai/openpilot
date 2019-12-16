#!/usr/bin/env python3

import os
import requests

from panda import Panda
from replay_drive import replay_drive
from tools.lib.logreader import LogReader  # pylint: disable=import-error

BASE_URL = "https://commadataci.blob.core.windows.net/openpilotci/"

# (route, safety mode, param)
logs = [
  ("b0c9d2329ad1606b|2019-05-30--20-23-57.bz2", Panda.SAFETY_HONDA, 0), # HONDA.CIVIC
  ("38bfd238edecbcd7|2019-06-07--10-15-25.bz2", Panda.SAFETY_TOYOTA, 66), # TOYOTA.PRIUS
  ("f89c604cf653e2bf|2018-09-29--13-46-50.bz2", Panda.SAFETY_GM, 0), # GM.VOLT
  ("0375fdf7b1ce594d|2019-05-21--20-10-33.bz2", Panda.SAFETY_HONDA_BOSCH, 1), # HONDA.ACCORD
  ("02ec6bea180a4d36|2019-04-17--11-21-35.bz2", Panda.SAFETY_HYUNDAI, 0), # HYUNDAI.SANTA_FE
  ("6fb4948a7ebe670e|2019-11-12--00-35-53.bz2", Panda.SAFETY_CHRYSLER, 0), # CHRYSLER.PACIFICA_2018_HYBRID
  ("791340bc01ed993d|2019-04-08--10-26-00.bz2", Panda.SAFETY_SUBARU, 0), # SUBARU.IMPREZA
  ("b0c9d2329ad1606b|2019-11-17--17-06-13.bz2", Panda.SAFETY_VOLKSWAGEN, 0), # VOLKSWAGEN.GOLF
]

if __name__ == "__main__":
  for route, _, _ in logs:
    if not os.path.isfile(route):
      with open(route, "wb") as f:
        f.write(requests.get(BASE_URL + route).content)

  failed = []
  for route, mode, param in logs:
    lr = LogReader(route)

    print("\nreplaying %s with safety mode %d and param %s" % (route, mode, param))
    if not replay_drive(lr, mode, int(param)):
      failed.append(route)

    for f in failed:
      print("\n**** failed on %s ****" % f)
    assert len(failed) == 0, "\nfailed on %d logs" % len(failed)

