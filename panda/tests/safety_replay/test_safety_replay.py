#!/usr/bin/env python3
from collections import namedtuple
import os
import requests

from panda import Panda
from panda.python import ALTERNATIVE_EXPERIENCE as ALT_EXP
from panda.tests.safety_replay.replay_drive import replay_drive
from tools.lib.logreader import LogReader  # pylint: disable=import-error

BASE_URL = "https://commadataci.blob.core.windows.net/openpilotci/"

ReplayRoute = namedtuple("ReplayRoute", ("route", "safety_mode", "param", "alternative_experience"), defaults=(0, 0))

logs = [
  ReplayRoute("2425568437959f9d|2019-12-22--16-24-37.bz2", Panda.SAFETY_HONDA_NIDEC),       # HONDA.CIVIC (fcw presents: 0x1FA blocked as expected)
  ReplayRoute("38bfd238edecbcd7|2019-06-07--10-15-25.bz2", Panda.SAFETY_TOYOTA, 66),        # TOYOTA.PRIUS
  ReplayRoute("f89c604cf653e2bf|2018-09-29--13-46-50.bz2", Panda.SAFETY_GM),                # GM.VOLT
  ReplayRoute("6fb4948a7ebe670e|2019-11-12--00-35-53.bz2", Panda.SAFETY_CHRYSLER),          # CHRYSLER.PACIFICA_2018_HYBRID
  ReplayRoute("791340bc01ed993d|2019-04-08--10-26-00.bz2", Panda.SAFETY_SUBARU),            # SUBARU.IMPREZA
  ReplayRoute("76b83eb0245de90e|2020-03-05--19-16-05.bz2", Panda.SAFETY_VOLKSWAGEN_MQB),    # VOLKSWAGEN.GOLF (MK7)
  ReplayRoute("d12cd943127f267b|2020-03-27--15-57-18.bz2", Panda.SAFETY_VOLKSWAGEN_PQ),     # 2009 VW Passat R36 (B6), supporting OP port not yet upstreamed
  ReplayRoute("fbbfa6af821552b9|2020-03-03--08-09-43.bz2", Panda.SAFETY_NISSAN),            # NISSAN.XTRAIL
  ReplayRoute("5b7c365c50084530_2020-04-15--16-13-24--3--rlog.bz2", Panda.SAFETY_HYUNDAI),  # HYUNDAI.SONATA
  ReplayRoute("610ebb9faaad6b43|2020-06-13--15-28-36.bz2", Panda.SAFETY_HYUNDAI_LEGACY),    # HYUNDAI.IONIQ_EV_LTD
  ReplayRoute("5ab784f361e19b78_2020-06-08--16-30-41.bz2", Panda.SAFETY_SUBARU_LEGACY),     # SUBARU.OUTBACK
  ReplayRoute("bb50caf5f0945ab1|2021-06-19--17-20-18.bz2", Panda.SAFETY_TESLA),             # TESLA.AP2_MODELS
  ReplayRoute("bd6a637565e91581_2021-10-29--22-18-31--1--rlog.bz2", Panda.SAFETY_MAZDA),    # MAZDA.CX9_2021
  # HONDA.CIVIC_2022
  ReplayRoute("1a5d045d2c531a6d_2022-06-07--22-03-00--1--rlog.bz2", Panda.SAFETY_HONDA_BOSCH, Panda.FLAG_HONDA_RADARLESS, ALT_EXP.DISABLE_DISENGAGE_ON_GAS),
]


if __name__ == "__main__":

  # get all the routes
  for route, _, _, _ in logs:
    if not os.path.isfile(route):
      with open(route, "wb") as f:
        f.write(requests.get(BASE_URL + route).content)

  failed = []
  for route, mode, param, alt_exp in logs:
    lr = LogReader(route)

    print("\nreplaying %s with safety mode %d, param %s, alternative experience %s" % (route, mode, param, alt_exp))
    if not replay_drive(lr, mode, param, alt_exp):
      failed.append(route)

    for f in failed:  # type: ignore
      print(f"\n**** failed on {f} ****")
    assert len(failed) == 0, "\nfailed on %d logs" % len(failed)
