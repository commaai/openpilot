#!/usr/bin/env python3

import os
import sys
import bz2
import struct
from panda import Panda
from panda.tests.safety_replay.replay_drive import replay_drive
from tools.lib.logreader import LogReader
from xx.chffr.lib.route import Route

# get a complete canlog (sendcan and can) for a drive
def get_canlog(route):
  if os.path.isfile(route + ".bz2"):
    return

  r = Route(route)
  log_msgs = []
  for i, segment in enumerate(r.log_paths()):
    print("downloading segment %d/%d" % (i+1, len(r.log_paths())))
    log = LogReader(segment)
    log_msgs.extend(filter(lambda msg: msg.which() in ('can', 'sendcan'), log))
  log_msgs.sort(key=lambda msg: msg.logMonoTime)

  dat = b"".join(m.as_builder().to_bytes() for m in log_msgs)
  dat = bz2.compress(dat)
  with open(route + ".bz2", "wb") as f:
    f.write(dat)


def get_logreader(route):
  try:
    lr = LogReader(route + ".bz2")
  except IOError:
    print("downloading can log")
    get_canlog(route)
    lr = LogReader(route + ".bz2")

  return lr

if __name__ == "__main__":
  route = sys.argv[1]
  mode = int(sys.argv[2])
  param = 0 if len(sys.argv) < 4 else int(sys.argv[3])

  lr = get_logreader(route)
  print("replaying drive %s with safety model %d and param %d" % (route, mode, param))

  replay_drive(lr, mode, param)
