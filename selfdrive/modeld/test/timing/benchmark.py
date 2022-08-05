#!/usr/bin/env python3
# type: ignore
# pylint: skip-file

import os
import time
import numpy as np

import cereal.messaging as messaging
from selfdrive.manager.process_config import managed_processes


N = int(os.getenv("N", "5"))
TIME = int(os.getenv("TIME", "30"))

if __name__ == "__main__":
  sock = messaging.sub_sock('modelV2', conflate=False, timeout=1000)

  execution_times = []

  for _ in range(N):
    os.environ['LOGPRINT'] = 'debug'
    managed_processes['modeld'].start()
    time.sleep(5)

    t = []
    start = time.monotonic()
    while time.monotonic() - start < TIME:
      msgs = messaging.drain_sock(sock, wait_for_one=True)
      for m in msgs:
        t.append(m.modelV2.modelExecutionTime)

    execution_times.append(np.array(t[10:]) * 1000)
    managed_processes['modeld'].stop()

  print("\n\n")
  print(f"ran modeld {N} times for {TIME}s each")
  for n, t in enumerate(execution_times):
    print(f"\tavg: {sum(t)/len(t):0.2f}ms, min: {min(t):0.2f}ms, max: {max(t):0.2f}ms")
  print("\n\n")
