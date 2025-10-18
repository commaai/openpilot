#!/usr/bin/env python3
import os
import time
import numpy as np
from multiprocessing import Process
from setproctitle import setproctitle

def waste(core):
  os.sched_setaffinity(0, [core,])

  m1 = np.zeros((200, 200)) + 0.8
  m2 = np.zeros((200, 200)) + 1.2

  i = 1
  st = time.monotonic()
  j = 0
  while 1:
    if (i % 100) == 0:
      setproctitle(f"{core:3d}: {i:8d}")
      lt = time.monotonic()
      print(f"{core:3d}: {i:8d} {lt-st:f}  {j:.2f}")
      st = lt
    i += 1
    j = np.sum(np.matmul(m1, m2))

def main(gctx=None):
  print("1-2 seconds is baseline")
  for i in range(os.cpu_count()):
    p = Process(target=waste, args=(i,))
    p.start()

if __name__ == "__main__":
  main()
