#!/usr/bin/env python3

import time
import atexit

from cereal import messaging
from openpilot.selfdrive.manager.process_config import managed_processes

TIMEOUT = 10*60

def kill():
  for proc in ['ubloxd', 'pigeond']:
    managed_processes[proc].stop(retry=True, block=True)

if __name__ == "__main__":
  # start ubloxd
  managed_processes['ubloxd'].start()
  atexit.register(kill)

  sm = messaging.SubMaster(['ubloxGnss'])

  times = []
  for i in range(20):
    # start pigeond
    st = time.monotonic()
    managed_processes['pigeond'].start()

    # wait for a >4 satellite fix
    while True:
      sm.update(0)
      msg = sm['ubloxGnss']
      if msg.which() == 'measurementReport' and sm.updated["ubloxGnss"]:
        report = msg.measurementReport
        if report.numMeas > 4:
          times.append(time.monotonic() - st)
          print(f"\033[94m{i}: Got a fix in {round(times[-1], 2)} seconds\033[0m")
          break

      if time.monotonic() - st > TIMEOUT:
        raise TimeoutError("\033[91mFailed to get a fix in {TIMEOUT} seconds!\033[0m")

      time.sleep(0.1)

    # stop pigeond
    managed_processes['pigeond'].stop(retry=True, block=True)
    time.sleep(20)

  print(f"\033[92mAverage TTFF: {round(sum(times) / len(times), 2)}s\033[0m")
