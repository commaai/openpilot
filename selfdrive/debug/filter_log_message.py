#!/usr/bin/env python3
import argparse
import json

import cereal.messaging as messaging
from openpilot.tools.lib.logreader import LogReader

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


def print_logmessage(t, msg, min_level):
  try:
    log = json.loads(msg)
    if log['levelnum'] >= min_level:
      print(f"[{t / 1e9:.6f}] {log['filename']}:{log.get('lineno', '')} - {log.get('funcname', '')}: {log['msg']}")
      if 'exc_info' in log:
        print(log['exc_info'])
  except json.decoder.JSONDecodeError:
    print(f"[{t / 1e9:.6f}] decode error: {msg}")


def print_androidlog(t, msg):
  source = ANDROID_LOG_SOURCE[msg.id]
  try:
    m = json.loads(msg.message)['MESSAGE']
  except Exception:
    m = msg.message

  print(f"[{t / 1e9:.6f}] {source} {msg.pid} {msg.tag} - {m}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--level', default='DEBUG')
  parser.add_argument('--addr', default='127.0.0.1')
  parser.add_argument("route", type=str, nargs='*', help="route name + segment number for offline usage")
  args = parser.parse_args()

  min_level = LEVELS[args.level]

  if args.route:
    for route in args.route:
      lr = LogReader(route)
      for m in lr:
        if m.which() == 'logMessage':
          print_logmessage(m.logMonoTime, m.logMessage, min_level)
        elif m.which() == 'errorLogMessage':
          print_logmessage(m.logMonoTime, m.errorLogMessage, min_level)
        elif m.which() == 'androidLog':
          print_androidlog(m.logMonoTime, m.androidLog)
  else:
    sm = messaging.SubMaster(['logMessage', 'androidLog'], addr=args.addr)
    while True:
      sm.update()

      if sm.updated['logMessage']:
        print_logmessage(sm.logMonoTime['logMessage'], sm['logMessage'], min_level)

      if sm.updated['androidLog']:
        print_androidlog(sm.logMonoTime['androidLog'], sm['androidLog'])
