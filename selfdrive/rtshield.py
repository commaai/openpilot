#!/usr/bin/env python3
import os
import time
from typing import NoReturn

from common.realtime import set_core_affinity, set_realtime_priority

# RT shield - ensure CPU 3 always remains available for RT processes
#   runs as SCHED_FIFO with minimum priority to ensure kthreads don't
#   get scheduled onto CPU 3, but it's always preemptible by realtime
#   openpilot processes

def main() -> NoReturn:
  set_core_affinity(int(os.getenv("CORE", "3")))
  set_realtime_priority(1)

  while True:
    time.sleep(0.000001)

if __name__ == "__main__":
  main()
