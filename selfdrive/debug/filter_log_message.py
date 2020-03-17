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

  if args.addr != "127.0.0.1":
    os.environ["ZMQ"] = "1"
    messaging.context = messaging.Context()

  poller = messaging.Poller()
  sock = messaging.sub_sock("logMessage", poller, addr=args.addr)

  min_level = LEVELS[args.level]

  while True:
    polld = poller.poll(1000)
    for sock in polld:
      evt = messaging.recv_one(sock)
      log = json.loads(evt.logMessage)

      if log['levelnum'] >= min_level:
        print(f"[{evt.logMonoTime / 1e9:.6f}] {log['filename']}:{log.get('lineno', '')} - {log.get('funcname', '')}: {log['msg']}")
