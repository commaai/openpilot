#!/usr/bin/env python3
import io
import os
import time
import pstats
import cProfile
from contextlib import contextmanager

from panda import Panda, PandaDFU
from panda.tests.hitl.helpers import get_random_can_messages


PROFILE = "PROFILE" in os.environ

@contextmanager
def print_time(desc):
  if PROFILE:
    pr = cProfile.Profile()
    pr.enable()
  start = time.perf_counter()
  yield
  end = time.perf_counter()
  print(f"{end - start:.3f}s - {desc}")
  if PROFILE:
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumtime")
    ps.print_stats()
    print(s.getvalue())


if __name__ == "__main__":
  with print_time("Panda()"):
    p = Panda()

  with print_time("PandaDFU.list()"):
    PandaDFU.list()

  fxn = [
    'reset',
    'reconnect',
    'up_to_date',
    'health',
    #'flash',
  ]
  for f in fxn:
    with print_time(f"Panda.{f}()"):
      getattr(p, f)()

  p.set_can_loopback(True)

  for n in range(6):
    msgs = get_random_can_messages(int(10**n))
    with print_time(f"Panda.can_send_many() - {len(msgs)} msgs"):
      p.can_send_many(msgs)

  with print_time("Panda.can_recv()"):
    m = p.can_recv()
