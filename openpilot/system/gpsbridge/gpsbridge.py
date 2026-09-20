#!/usr/bin/env python3
"""Relay GPS between two commas on the same network.

A metallized windshield can leave the main device with no sky view at all, so a
second device in a better spot supplies the fix. Two params decide the role:

  GpsPublish  bool    on the GPS device: offer its msgq over ZMQ
  GpsSource   string  on the main device: IP of the GPS device

Both are PERSISTENT, so they survive reboots and this needs setting only once.

This wraps cereal's own bridge, which already does both directions:
  bridge                       msgq -> ZMQ, every service
  bridge <ip> <whitelist>      ZMQ  -> msgq, whitelisted services only

The receiving side needs the local ubloxd stopped, or two publishers fight over
the gpsLocationExternal endpoint; process_config gates that on GpsSource.
"""
import os
import signal
import subprocess
import time

from openpilot.common.basedir import BASEDIR
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog

BRIDGE = os.path.join(BASEDIR, "openpilot/cereal/messaging/bridge")
SERVICE = "gpsLocationExternal"
POLL_SECONDS = 2.0


def desired_cmd(params: Params) -> list[str] | None:
  source = (params.get("GpsSource") or "").strip()
  if source:
    return [BRIDGE, source, SERVICE]
  if params.get_bool("GpsPublish"):
    return [BRIDGE]
  return None


def main() -> None:
  params = Params()
  proc: subprocess.Popen | None = None
  running: list[str] | None = None

  def stop() -> None:
    nonlocal proc, running
    if proc is not None:
      proc.send_signal(signal.SIGINT)
      try:
        proc.wait(timeout=5)
      except subprocess.TimeoutExpired:
        proc.kill()
      proc = None
      running = None

  signal.signal(signal.SIGTERM, lambda *_: (stop(), os._exit(0)))

  while True:
    cmd = desired_cmd(params)

    if cmd != running:
      stop()
      if cmd is not None:
        cloudlog.warning(f"gpsbridge: {' '.join(cmd)}")
        proc = subprocess.Popen(cmd)
        running = cmd
      else:
        cloudlog.warning("gpsbridge: idle, neither GpsSource nor GpsPublish set")

    # the bridge exits on a bind conflict; restart it rather than sit dead
    if proc is not None and proc.poll() is not None:
      cloudlog.warning(f"gpsbridge: bridge exited {proc.returncode}, restarting")
      proc = None
      running = None

    time.sleep(POLL_SECONDS)


if __name__ == "__main__":
  main()
