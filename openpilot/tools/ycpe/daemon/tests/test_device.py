#!/usr/bin/env python3
"""Device pull over real ssh to localhost, against a stand-in realdata directory (never a real device)."""
import getpass
import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from openpilot.tools.ycpe.daemon import cache, jobs
from openpilot.tools.ycpe.daemon.core import config
from openpilot.tools.ycpe.daemon.tests.common import DaemonCase

DONGLE, LOG = "fedcba9876543210", "0000002a--0123456789"
FAKE = Path(tempfile.mkdtemp())
REALDATA = FAKE / "data" / "media" / "0" / "realdata"  # the device's layout, DongleId included
FILES = {0: {"rlog.zst": 200_000, "qlog.zst": 20_000, "qcamera.ts": 5_000}, 1: {"rlog.zst": 150_000, "qlog.zst": 15_000},
         2: {"rlog.zst": 1_000, "rlog.lock": 0}}  # segment 2 is still being recorded
for seg, files in FILES.items():
  d = REALDATA / f"{LOG}--{seg}"
  d.mkdir(parents=True)
  for name, size in files.items():
    (d / name).write_bytes(os.urandom(size))
(REALDATA / "boot").mkdir()
(FAKE / "data" / "params" / "d").mkdir(parents=True)
(FAKE / "data" / "params" / "d" / "DongleId").write_text(DONGLE + "\n")


def tree() -> list:
  return sorted((str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in FAKE.rglob("*"))


def ssh_ok() -> bool:
  try:
    return subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=3", "localhost", "true"], timeout=10).returncode == 0
  except (OSError, subprocess.TimeoutExpired):
    return False


@unittest.skipUnless(ssh_ok(), "needs passwordless ssh to localhost")
class TestDevice(DaemonCase):
  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    config.cache_root = tempfile.mkdtemp() + "/"
    config.device, config.device_root = f"{getpass.getuser()}@localhost", str(REALDATA)
    cls.before = tree()

  def test_list_pull_resolve(self):
    info = self.json("/local/device/routes")
    self.assertEqual(info["dongle"], DONGLE)
    [r] = info["routes"]
    self.assertEqual((r["route"], r["active"]), (f"{DONGLE}|{LOG}", [2]))
    self.assertEqual(r["segments"]["0"], FILES[0])

    t = time.monotonic()
    job = jobs.get(self.json("/local/device/pull", {"route": LOG})["id"])
    while job.state == "running":
      time.sleep(0.02)
    self.assertEqual(job.state, "done", job.error)
    self.assertEqual(job.result["files"], 4)  # rlog + qlog of segments 0 and 1; 2 is still recording
    self.assertEqual(job.result["skipped"], [2])
    self.assertEqual(job.meta["segs"], {"0": "disk", "1": "disk"})
    print(f"pulled {job.result['bytes']} bytes in {time.monotonic() - t:.2f} s")
    for seg in (0, 1):
      for name in ("rlog.zst", "qlog.zst"):
        self.assertEqual(cache.device_path(cache.Key(DONGLE, LOG, seg, name)).read_bytes(), (REALDATA / f"{LOG}--{seg}" / name).read_bytes())

    # the pulled logs are the "device" source, served from the cache
    res = self.json("/local/sources/resolve", {"route": f"{DONGLE}|{LOG}", "segs": [0, 1], "kinds": ["log"], "sources": ["device"]})
    self.assertEqual({n: e["log"]["source"] for n, e in res["segments"].items()}, {"0": "device", "1": "device"})
    r, data = self.req(res["segments"]["1"]["log"]["url"], token=False)
    self.assertEqual(data, (REALDATA / f"{LOG}--1" / "rlog.zst").read_bytes())
    # pulling again copies nothing
    job = jobs.get(self.json("/local/device/pull", {"route": f"{DONGLE}|{LOG}", "segs": [0], "kinds": ["qcamera"]})["id"])
    while job.state == "running":
      time.sleep(0.02)
    self.assertEqual((job.state, job.result["files"]), ("done", 1))
    self.assertEqual(tree(), self.before)  # read-only: nothing on the "device" changed

  def test_bad_host(self):
    for h in ("-oProxyCommand=touch /tmp/pwned", "a b", "x;rm", "", "user@-p"):
      self.json("/local/device", {"host": h}, status=400)

  def test_missing_route(self):
    self.json("/local/device/pull", {"route": "0000ffff--ffffffffff"}, status=404)


if __name__ == "__main__":
  unittest.main()
