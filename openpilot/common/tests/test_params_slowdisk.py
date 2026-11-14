import json
import os
import subprocess
import sys
import tempfile

from openpilot.common.basedir import BASEDIR
from openpilot.common.params import Params
from openpilot.common.test import OpenpilotTestCase

SLOWDISK_SRC = os.path.join(BASEDIR, "openpilot/selfdrive/test/slowdisk.c")
EXT_MAGIC = "ef53"  # slowdisk.c only delays ext2/3/4
MAX_LATENCY_MS = 50  # every disk operation takes 100ms under the shim
ITERATIONS = 200

# runs in a subprocess with the shim preloaded. touching the disk is fine until the flag exists
WORKER = r"""
import json
import os
import time
from pathlib import Path

from openpilot.common.params import Params

params = Params()
params.put("DongleId", "startup", block=True)
params.put_bool("IsMetric", True, block=True)
params.put("CarParams", b"startup", block=True)
for k in ("DongleId", "IsMetric", "CarParams", "IsOffroad", "LongitudinalPersonality"):
  params.get(k)

Path(os.environ["SLOWDISK_FLAG"]).touch()
time.sleep(1.1)  # the shim checks for the flag once a second

worst = {}
def timed(name, fn, *args, **kwargs):
  t = time.perf_counter()
  ret = fn(*args, **kwargs)
  worst[name] = max(worst.get(name, 0.), (time.perf_counter() - t) * 1e3)
  return ret

for i in range(int(os.environ["ITERATIONS"])):
  timed("put", params.put, "DongleId", f"v{i}")
  assert timed("get", params.get, "DongleId") == f"v{i}"

  timed("put_bool", params.put_bool, "IsMetric", bool(i % 2))
  assert timed("get_bool", params.get_bool, "IsMetric") == bool(i % 2)

  timed("put", params.put, "CarParams", os.urandom(4096))
  timed("remove", params.remove, "CarParams")
  assert timed("get", params.get, "CarParams") is None

  assert timed("get", params.get, "IsOffroad") is None
  assert timed("get", params.get, "LongitudinalPersonality", return_default=True) is not None
  assert timed("new", Params).get("DongleId") == f"v{i}"
  time.sleep(0.002)

os.unlink(os.environ["SLOWDISK_FLAG"])
print(json.dumps(worst))
"""


class TestParamsSlowDisk(OpenpilotTestCase):
  def test_no_blocking_on_disk(self):
    params = Params()
    fs_type = subprocess.check_output(["stat", "-f", "-c", "%t", params.get_param_path()], text=True).strip()
    if fs_type != EXT_MAGIC:
      self.skipTest("params aren't on ext4")

    with tempfile.TemporaryDirectory(dir="/dev/shm") as d:
      shim = os.path.join(d, "slowdisk.so")
      subprocess.check_call(["cc", "-O2", "-shared", "-fPIC", "-o", shim, SLOWDISK_SRC, "-ldl"])
      env = {**os.environ, "LD_PRELOAD": shim, "SLOWDISK_FLAG": os.path.join(d, "flag"), "ITERATIONS": str(ITERATIONS)}
      out = subprocess.check_output([sys.executable, "-c", WORKER], env=env, text=True, timeout=120)

    worst = json.loads(out.strip().splitlines()[-1])
    for name, ms in worst.items():
      assert ms < MAX_LATENCY_MS, f"{name} blocked on disk: {worst}"

    # the worker has exited, so its writes are on disk
    with open(params.get_param_path("DongleId"), "rb") as f:
      assert f.read() == f"v{ITERATIONS - 1}".encode()
    with open(params.get_param_path("IsMetric"), "rb") as f:
      assert f.read() == (b"1" if (ITERATIONS - 1) % 2 else b"0")
    assert not os.path.exists(params.get_param_path("CarParams"))
