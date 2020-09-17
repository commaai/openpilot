#!/usr/bin/env python3
import os
from common.realtime import set_core_affinity, set_realtime_priority

# a shield process to ensure CPU 3 always remains available for RT processes

def main():
  set_core_affinity(3)
  set_realtime_priority(os.sched_get_priority_min(os.SCHED_FIFO))

  while True:
    pass
