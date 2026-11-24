import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time

from openpilot.common.params import Params, ParamKeyFlag
from openpilot.common.test import OpenpilotTestCase


def read(path):
  try:
    with open(path, "rb") as f:
      return f.read()
  except FileNotFoundError:
    return None


class TestParamsDisk(OpenpilotTestCase):
  # like on the device: params are in RAM, and saved to the disk

  def setup_method(self):
    self.ram = tempfile.mkdtemp(dir="/dev/shm")
    self.disk = os.path.join(tempfile.mkdtemp(), os.environ["OPENPILOT_PREFIX"])
    os.environ["PARAMS_ROOT"] = self.ram
    os.environ["PARAMS_DISK_ROOT"] = os.path.dirname(self.disk)

  def teardown_method(self):
    # the prefix cleanup makes a Params after this
    del os.environ["PARAMS_ROOT"], os.environ["PARAMS_DISK_ROOT"]
    shutil.rmtree(self.ram, ignore_errors=True)
    shutil.rmtree(os.path.dirname(self.disk))

  def reboot(self):
    shutil.rmtree(self.ram)
    os.mkdir(self.ram)

  def on_disk(self, key):
    return read(os.path.join(self.disk, key))

  def test_save(self):
    params = Params()
    assert params.get_param_path().startswith(self.ram)

    params.put("DongleId", "a")
    params.put("CarParams", b"b")
    assert self.on_disk("DongleId") is None
    assert params.save() == 0
    assert self.on_disk("DongleId") == b"a" and self.on_disk("CarParams") == b"b"

    params.put("DongleId", "c")
    params.remove("CarParams")
    assert params.get("CarParams") is None
    assert params.save() == 0
    assert self.on_disk("DongleId") == b"c" and self.on_disk("CarParams") is None

    # removed and written again
    params.remove("DongleId")
    params.put("DongleId", "d")
    assert params.save() == 0
    assert self.on_disk("DongleId") == b"d"

    # others list and copy this directory, so nothing but params is left in either
    assert os.listdir(self.disk) == ["DongleId"]
    assert os.listdir(params.get_param_path()) == ["DongleId"]

    # empty isn't the same as not there
    params.put("GitBranch", "")
    assert params.save() == 0
    assert self.on_disk("GitBranch") == b""

  def test_wait_for_write(self):
    params = Params()
    t = time.monotonic()
    params.wait_for_write(5)  # the first one returns at once, for what was written before it
    params.wait_for_write(0.2)
    assert 0.15 < time.monotonic() - t < 1

    # manager saves as soon as something is written
    for write in (lambda: Params().put("DongleId", "a"), lambda: Params().remove("DongleId")):
      threading.Timer(0.1, write).start()
      t = time.monotonic()
      params.wait_for_write(5)
      assert time.monotonic() - t < 1
      assert params.save() == 0
    assert self.on_disk("DongleId") is None

  def test_block(self):
    params = Params()
    params.put("DongleId", "a", block=True)
    params.put_bool("IsMetric", True, block=True)
    assert self.on_disk("DongleId") == b"a" and self.on_disk("IsMetric") == b"1"

  def test_wait_for_disk(self):
    # for some params a write isn't done until it's on the disk, whoever makes it
    params = Params()
    assert params.put_bool("RecordFront", False) == 0
    assert params.put("GithubSshKeys", "key") == 0
    assert self.on_disk("RecordFront") == b"0" and self.on_disk("GithubSshKeys") == b"key"

    assert params.remove("GithubSshKeys") == 0
    assert self.on_disk("GithubSshKeys") is None

    # and it fails if it can't get there
    if os.getuid() != 0:
      os.chmod(self.disk, 0o500)
      try:
        assert params.put_bool("RecordFront", True) != 0
      finally:
        os.chmod(self.disk, 0o775)

  def test_reboot(self):
    params = Params()
    params.put("DongleId", "a", block=True)
    params.put("GitBranch", "b", block=True)
    params.put("DongleId", "not saved")
    params.remove("GitBranch")
    with open(os.path.join(self.disk, ".tmp_value_abc123"), "w") as f:  # a save that was cut short
      f.write("half")

    self.reboot()
    params = Params()
    assert params.get("DongleId") == "a"
    assert params.get("GitBranch") == "b"
    assert sorted(os.listdir(self.disk)) == ["DongleId", "GitBranch"]

    params.clear_all(ParamKeyFlag.ALL)
    assert params.save() == 0
    self.reboot()
    assert Params().get("DongleId") is None

  def test_written_to_the_disk_by_others(self):
    # AGNOS writes a few params straight to the disk. they're read at the next boot
    Params().put("DongleId", "a", block=True)
    with open(os.path.join(self.disk, "LastPowerDropDetected"), "w") as f:
      f.write("agnos")
    with open(os.path.join(self.disk, "not_a_param"), "w") as f:
      f.write("junk")

    params = Params()
    assert params.get("LastPowerDropDetected") is None
    assert params.save() == 0
    assert self.on_disk("LastPowerDropDetected") == b"agnos"

    self.reboot()
    params = Params()
    assert params.get("LastPowerDropDetected") == "agnos"

    # and cleared like any other
    params.clear_all(ParamKeyFlag.CLEAR_ON_MANAGER_START)
    assert params.save() == 0
    assert params.get("DongleId") == "a"
    assert os.listdir(self.disk) == ["DongleId"]

  def test_first_use(self):
    Params().put("DongleId", "a", block=True)
    self.reboot()

    # everyone starts at once after boot. the copy from the disk is made once, and nobody's write is lost
    code = "import sys; from openpilot.common.params import Params; p = Params(); assert p.get('DongleId') == 'a'; p.put(sys.argv[1], 'x')"
    keys = ["GitBranch", "GitCommit", "GitRemote", "GitDiff", "Version", "GsmApn", "HardwareSerial", "GithubUsername"]
    procs = [subprocess.Popen([sys.executable, "-c", code, k]) for k in keys]
    assert [p.wait() for p in procs] == [0] * len(keys)

    params = Params()
    assert all(params.get(k) == "x" for k in keys)
    assert len(os.listdir(self.ram)) == 2  # the directory and its symlink

  def test_disk_errors(self):
    if os.getuid() == 0:
      self.skipTest("root can always write")
    params = Params()
    params.put("DongleId", "a", block=True)
    os.chmod(self.disk, 0o500)
    try:
      params.put("DongleId", "b")
      assert params.save() != 0
      assert params.get("DongleId") == "b"
    finally:
      os.chmod(self.disk, 0o775)
    assert params.save() == 0
    assert self.on_disk("DongleId") == b"b"

    # what can't be read isn't taken for empty
    os.chmod(params.get_param_path("DongleId"), 0)
    assert params.save() == 0
    assert self.on_disk("DongleId") == b"b"
