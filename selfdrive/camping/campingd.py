#!/usr/bin/env python3
import os
import time
import subprocess

from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog

"""
Camping mode Miracast receiver launcher.
- Runs offroad only (manager gating), behind Params key `CampingMode`.
- Launches Miracast receiver for screen mirroring from phones/devices.
"""

def main():
  params = Params()
  cloudlog.event("campingd.start")

  # Launch receiver (best-effort priority order)
  proc = None
  try:
    local_bin = "/data/camping/bin"
    candidates: list[tuple[str, list[str]]] = []
    def add(bin_path: str, args: list[str] | None = None):
      if os.path.exists(bin_path) and os.access(bin_path, os.X_OK):
        candidates.append((os.path.basename(bin_path), [bin_path] + (args or [])))

    # Preferred: Open Screen cast receiver
    add(os.path.join(local_bin, "openscreen-cast-receiver"))
    add("/usr/bin/openscreen-cast-receiver")
    # DLNA audio fallback
    add(os.path.join(local_bin, "gmediarender"), ["-f", "openpilot-camping"])
    add("/usr/bin/gmediarender", ["-f", "openpilot-camping"])
    # mkchromecast as last resort (sender-like, may not work as receiver)
    add("/usr/bin/mkchromecast")
    # Miraclecast daemon (requires Wi-Fi P2P support and privileges)
    add(os.path.join(local_bin, "miracle-wifid"))

    if candidates:
      name, cmd = candidates[0]
      proc = subprocess.Popen(cmd)
      cloudlog.event("campingd.receiver", name=name, cmd=" ".join(cmd))
    else:
      cloudlog.event("campingd.receiver", name="none_found")
      proc = None

    # heartbeat loop
    while True:
      time.sleep(1.0)
      # allow runtime disable
      if not params.get_bool("CampingMode"):
        cloudlog.event("campingd.stop_param")
        break
  except Exception as e:
    cloudlog.exception("campingd.exception", error=True)
  finally:
    if proc and proc.poll() is None:
      proc.terminate()
      try:
        proc.wait(timeout=2)
      except Exception:
        proc.kill()
    cloudlog.event("campingd.exit")

if __name__ == "__main__":
  main()
