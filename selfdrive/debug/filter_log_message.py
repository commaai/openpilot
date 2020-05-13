#!/usr/bin/env python3
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

ANDROID_LOG_SOURCE = {
  0: "MAIN",
  1: "RADIO",
  2: "EVENTS",
  3: "SYSTEM",
  4: "CRASH",
  5: "KERNEL",
}


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('--level', default='DEBUG')
  parser.add_argument('--addr', default='127.0.0.1')
  parser.add_argument("socket", type=str, nargs='*', help="socket name")
  args = parser.parse_args()

  sm = messaging.SubMaster(['logMessage', 'androidLog'], addr=args.addr)

  min_level = LEVELS[args.level]

  while True:
    sm.update()

    if sm.updated['logMessage']:
      t = sm.logMonoTime['logMessage']
      try:
        log = json.loads(sm['logMessage'])
        if log['levelnum'] >= min_level:
          print(f"[{t / 1e9:.6f}] {log['filename']}:{log.get('lineno', '')} - {log.get('funcname', '')}: {log['msg']}")
      except json.decoder.JSONDecodeError:
        print(f"[{t / 1e9:.6f}] decode error: {sm['logMessage']}")

    if sm.updated['androidLog']:
      t = sm.logMonoTime['androidLog']
      m = sm['androidLog']
      source = ANDROID_LOG_SOURCE[m.id]
      print(f"[{t / 1e9:.6f}] {source} {m.pid} {m.tag} - {m.message}")
