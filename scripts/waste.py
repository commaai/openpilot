#!/usr/bin/env python3
from multiprocessing import Process
from setproctitle import setproctitle
import os
import numpy as np
from common.realtime import sec_since_boot

def waste(pid):
  # set affinity
  os.system("taskset -p %d %d" % (1 << pid, os.getpid()))

  m1 = np.zeros((200,200)) + 0.8
  m2 = np.zeros((200,200)) + 1.2

  i = 1
  st = sec_since_boot()
  j = 0
  while 1:
    if (i % 100) == 0:
      setproctitle("%3d: %8d" % (pid, i))
      lt = sec_since_boot()
      print("%3d: %8d %f  %.2f" % (pid, i, lt-st, j))
      st = lt
    i += 1
    j = np.sum(np.matmul(m1, m2))

def main(gctx=None):
  print("1-2 seconds is baseline")
  for i in range(4):
    p = Process(target=waste, args=(i,))
    p.start()

if __name__ == "__main__":
  main()

