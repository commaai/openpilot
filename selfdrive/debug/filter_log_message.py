#!/usr/bin/env python3
import os
import argparse
import json

import cereal.messaging as messaging


LEVELS = {
  "DEBUG": 10,
  "INFO": 20,
  "WARNING": 30,
  "ERROR": 40,
  "CRITICAL": 50,
}


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('--level', default='DEBUG')
  parser.add_argument('--addr', default='127.0.0.1')
  parser.add_argument("socket", type=str, nargs='*', help="socket name")
  args = parser.parse_args()

  sm = messaging.SubMaster(['logMessage', 'androidLogEntry'], addr=args.addr)

  min_level = LEVELS[args.level]

  while True:
    sm.update()

    if sm.updated['logMessage']:
      t = sm.logMonoTime['logMessage']
      log = json.loads(sm['logMessage'])
      if log['levelnum'] >= min_level:
        print(f"[{t / 1e9:.6f}] {log['filename']}:{log.get('lineno', '')} - {log.get('funcname', '')}: {log['msg']}")
    if sm.updated['androidLogEntry']:
      t = sm.logMonoTime['androidLogEntry']
      print(f"[{t / 1e9:.6f}] - ")
      print(sm['androidLogEntry'])
