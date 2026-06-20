#!/usr/bin/env python3
import json
import subprocess

import cereal.messaging as messaging
from openpilot.common.swaglog import cloudlog


def main():
  pm = messaging.PubMaster(['androidLog'])
  cmd = ['journalctl', '-f', '-o', 'json']
  proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
  assert proc.stdout is not None
  try:
    for line in proc.stdout:
      line = line.strip()
      if not line:
        continue
      try:
        kv = json.loads(line)
      except json.JSONDecodeError:
        cloudlog.exception("failed to parse journalctl output")
        continue

      msg = messaging.new_message('androidLog')
      entry = msg.androidLog
      entry.ts = int(kv.get('__REALTIME_TIMESTAMP', 0))
      entry.message = json.dumps(kv)
      if '_PID' in kv:
        entry.pid = int(kv['_PID'])
      if 'PRIORITY' in kv:
        entry.priority = int(kv['PRIORITY'])
      if 'SYSLOG_IDENTIFIER' in kv:
        entry.tag = kv['SYSLOG_IDENTIFIER']

      pm.send('androidLog', msg)
  finally:
    proc.terminate()
    proc.wait()


if __name__ == '__main__':
  main()
