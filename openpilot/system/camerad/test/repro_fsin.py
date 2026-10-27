#!/usr/bin/env python3
# does disturbing the panda's FSIN while camerad starts wedge a camera?
import random
import subprocess
import sys
import time
from collections import Counter

from openpilot.cereal import messaging
from panda import Panda
from openpilot.selfdrive.pandad.pandad import flash_panda, panda_ready
from openpilot.system.manager.process_config import managed_processes

CAMS = ('narrowRoadCameraState', 'cabinCameraState', 'wideRoadCameraState')


def disturb(mode):
  if mode == "reset":
    assert Panda.wait_for_panda(None, timeout=5)
    with Panda(cli=False) as p:
      p.reset(reconnect=False)
  elif mode == "flash":
    flash_panda()
  elif mode == "jump":
    # one 75ms FSIN period, a 25ms phase jump like the panda restarting its timers
    with Panda(cli=False) as p:
      p.set_clock_source_timer_params(75, 2, 2, 2)
      time.sleep(0.06)
      p.set_clock_source_timer_params(50, 2, 2, 2)


def trial(mode):
  socks = {c: messaging.sub_sock(c, conflate=False, timeout=10) for c in CAMS}
  managed_processes['camerad'].start()
  try:
    delay = random.uniform(0, 3)
    time.sleep(delay)
    disturb(mode)
    time.sleep(4)
    for s in socks.values():
      messaging.drain_sock(s)
    time.sleep(2)
    n = {c: len(messaging.drain_sock(s)) for c, s in socks.items()}
    if any(k < 30 for k in n.values()):
      # the kernel's view while it's still wedged
      out = subprocess.run("sudo dmesg", shell=True, capture_output=True, text=True).stdout.splitlines()
      print("DMESG\n" + "\n".join([l for l in out if "CAM" in l][-80:]), flush=True)
    return delay, n
  finally:
    managed_processes['camerad'].stop()
    Panda.wait_for_panda(None, timeout=10)


if __name__ == "__main__":
  if not panda_ready():
    flash_panda()
  assert panda_ready()
  minutes = float(sys.argv[1]) if len(sys.argv) > 1 else 10
  modes = sys.argv[2].split(",") if len(sys.argv) > 2 else ["jump", "jump", "jump", "none"]
  tally = Counter()
  end = time.monotonic() + minutes*60
  i = 0
  while time.monotonic() < end:
    mode = modes[i % len(modes)]
    i += 1
    try:
      delay, n = trial(mode)
    except Exception as e:
      print(f"REPRO {i} mode={mode} ERROR {e!r}", flush=True)
      tally[mode, "error"] += 1
      if not panda_ready():
        flash_panda()
      continue
    # 2s at 20Hz is 40 frames
    bad = [c for c, k in n.items() if k < 30]
    tally[mode, bool(bad)] += 1
    print(f"REPRO {i} mode={mode} delay={delay:.2f} frames={n} {'WEDGED ' + ','.join(bad) if bad else 'ok'}", flush=True)
  print("REPRO tally", dict(tally), flush=True)
  sys.exit(1 if any(bad is True for (_, bad) in tally) else 0)
