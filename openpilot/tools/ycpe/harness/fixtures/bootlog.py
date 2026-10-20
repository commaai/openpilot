#!/usr/bin/env python3
"""writes bootlog.zst, a made-up boot log shaped like system/loggerd/bootlog.cc's (openpilot venv, PYTHONPATH=openpilot)"""
from pathlib import Path

import zstandard
from cereal import log

JOURNAL = '[ -x "$(command -v journalctl)" ] && journalctl -b -n 2000 -o short-monotonic --no-pager'


def event(**kw):
  ev = log.Event.new_message(**kw)
  ev.valid = True
  return ev.to_bytes()


init = event(initData={
  "version": "0.11.2", "gitBranch": "master", "gitCommit": "7e84a6160000000000000000000000000000000", "dirty": True,
  "deviceType": "mici", "osVersion": "16", "kernelVersion": "Linux 4.9.103+", "kernelArgs": ["console=ttyMSM0,115200n8", "quiet"],
  "params": {"entries": [{"key": "IsMetric", "value": b"1"}, {"key": "CarParamsCache", "value": bytes(range(256))}]},
  "commands": {"entries": [{"key": "df -h", "value": b"Filesystem  Size  Used Avail Use% Mounted on\n/dev/sda9  10G  6G  4G  60% /\n"}]},
})
journal = "\n".join(f"[ {k * 0.05:10.6f}] comma-fixture kernel: {'thermal zone error' if k % 25 == 0 else 'probe ok'} {k}" for k in range(600))
boot = event(boot={
  "wallTimeNanos": 1785225913000000000,  # 2026-07-28
  "launchLog": "launch_chffrplus.sh: starting\nlaunch_chffrplus.sh: manager started\n",
  "commands": {"entries": [{"key": JOURNAL, "value": journal.encode()}]},
  "pstore": {"entries": [{"key": "console-ramoops-0", "value": b"Kernel panic - not syncing: fixture\n\0\0"}]},
})
Path(__file__).with_name("bootlog.zst").write_bytes(zstandard.ZstdCompressor().compress(init + boot))
