#!/usr/bin/env python3
import os
import time
import numpy as np
import cereal.messaging as messaging
import selfdrive.manager as manager


N = int(os.getenv("N", "5"))
TIME = int(os.getenv("TIME", "30"))

if __name__ == "__main__":
  sock = messaging.sub_sock('modelV2', conflate=False, timeout=1000)

  execution_times = []

  for _ in range(N):
    os.environ['LOGPRINT'] = 'debug'
    manager.start_managed_process('modeld')
    time.sleep(5)

    t = []
    start = time.monotonic()
    while time.monotonic() - start < TIME:
      msgs = messaging.drain_sock(sock, wait_for_one=True)
      for m in msgs:
        t.append(m.modelV2.modelExecutionTime)

    execution_times.append(np.array(t[10:]) * 1000)
    manager.kill_managed_process('modeld')

  print("\n\n")
  print(f"ran modeld {N} times for {TIME}s each")
  for n, t in enumerate(execution_times):
    print(f"\tavg: {sum(t)/len(t):0.2f}ms, min: {min(t):0.2f}ms, max: {max(t):0.2f}ms")
  print("\n\n")
