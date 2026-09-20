#!/usr/bin/env python3
"""Relay GPS between two commas on the same network.

A metallized windshield can leave the car's device tracking zero satellites. Put a
second comma where it can see sky and relay its fix over the network. Two params:

  GpsPublish  bool    offer this device's msgq over ZMQ
  GpsSource   string  IP of the device to take gpsLocationExternal from

Both are PERSISTENT, so this is a one-time setup that survives reboots. They are
independent: a receiving device usually wants GpsPublish too, so tooling on a
laptop can still subscribe to it.

This wraps cereal's own bridge, which relays in both directions:
  bridge                       msgq -> ZMQ, every service   (publish)
  bridge <ip> <whitelist>      ZMQ  -> msgq, whitelisted     (receive)

Receive mode only connects outward, publish mode only binds, so both can run at
once on the same device.
"""
import os
import signal
import subprocess
import time

import openpilot.cereal.messaging as messaging
from openpilot.common.basedir import BASEDIR
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog

BRIDGE = os.path.join(BASEDIR, "openpilot/cereal/messaging/bridge")
SERVICE = "gpsLocationExternal"
POLL_SECONDS = 2.0
PUBLISHER_WAIT_SECONDS = 60.0


def wait_for_local_gps() -> bool:
  """Block until something publishes SERVICE locally.

  The bridge attaches its msgq subscriber the first time a ZMQ client connects. If
  no publisher has ever existed for that endpoint the subscriber ends up on a dead
  shared-memory segment and forwards nothing, with every port still listening. So
  do not offer the socket until the data behind it is real.
  """
  sock = messaging.sub_sock(SERVICE, timeout=1000)
  deadline = time.monotonic() + PUBLISHER_WAIT_SECONDS
  while time.monotonic() < deadline:
    if messaging.recv_one_or_none(sock) is not None:
      return True
    time.sleep(0.2)
  return False


def desired(params: Params) -> dict[str, list[str]]:
  out = {}
  source = (params.get("GpsSource") or "").strip()
  if source:
    out["receive"] = [BRIDGE, source, SERVICE]
  if params.get_bool("GpsPublish"):
    out["publish"] = [BRIDGE]
  return out


def main() -> None:
  params = Params()
  procs: dict[str, subprocess.Popen] = {}
  cmds: dict[str, list[str]] = {}

  def stop(name: str) -> None:
    proc = procs.pop(name, None)
    cmds.pop(name, None)
    if proc is None:
      return
    proc.send_signal(signal.SIGINT)
    try:
      proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
      proc.kill()

  def stop_all(*_) -> None:
    for name in list(procs):
      stop(name)
    os._exit(0)

  signal.signal(signal.SIGTERM, stop_all)
  signal.signal(signal.SIGINT, stop_all)

  while True:
    want = desired(params)

    for name in list(procs):
      if name not in want or cmds.get(name) != want[name]:
        cloudlog.warning(f"gpsbridge: stopping {name}")
        stop(name)

    for name, cmd in want.items():
      if name in procs and procs[name].poll() is None:
        continue
      if name in procs:
        cloudlog.warning(f"gpsbridge: {name} exited {procs[name].returncode}, restarting")
        stop(name)

      # Only a device sourcing its own GPS has to wait; on a receiving device the
      # data arrives from the receive bridge, which is started in the same pass.
      if name == "publish" and "receive" not in want and not wait_for_local_gps():
        cloudlog.warning(f"gpsbridge: no local {SERVICE} yet, not publishing")
        continue

      cloudlog.warning(f"gpsbridge: starting {name}: {' '.join(cmd)}")
      procs[name] = subprocess.Popen(cmd)
      cmds[name] = cmd

    time.sleep(POLL_SECONDS)


if __name__ == "__main__":
  main()
