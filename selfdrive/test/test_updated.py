#!/usr/bin/env python3
import datetime
import os
import time
import tempfile
import unittest
import shutil
import signal
import subprocess
import random

from common.basedir import BASEDIR
from common.params import Params
from common.timeout import Timeout
from system.hardware import AGNOS


class TestUpdated(unittest.TestCase):
  """Changes to updated.py need to be committed before they are applied in the test environment."""

  def setUp(self):
    self.updated_proc = None
    os.environ["GIT_LFS_SKIP_SMUDGE"] = "1"

    self.tmp_dir = tempfile.TemporaryDirectory(dir=os.path.abspath(os.path.join(BASEDIR, "..")) if AGNOS else None)
    org_dir = os.path.join(self.tmp_dir.name, "commaai")

    self.basedir = os.path.join(org_dir, "openpilot")
    self.params_path = os.path.join(self.basedir, "persist/params")
    self.git_remote_dir = os.path.join(org_dir, "openpilot_remote")
    self.staging_root = os.path.join(org_dir, "safe_staging")
    for d in [org_dir, self.basedir, self.git_remote_dir, self.staging_root]:
      os.mkdir(d)

    self.upper_dir = os.path.join(self.staging_root, "upper")
    self.merged_dir = os.path.join(self.staging_root, "merged")
    self.finalized_dir = os.path.join(self.staging_root, "finalized")

    # setup local submodule remotes
    submodules = subprocess.check_output("git submodule --quiet foreach 'echo $path'",
                                         shell=True, cwd=BASEDIR, encoding='utf8').split()
    for s in submodules:
      sub_path = os.path.join(org_dir, s.split("_repo")[0])
      self._run(f"git clone {s} {sub_path}.git", cwd=BASEDIR)

    # setup two git repos, a remote and one we'll run updated in
    self._run([
      f"git clone {BASEDIR} {self.git_remote_dir}",
      f"git clone --recurse-submodules {self.git_remote_dir} {self.basedir}",
      f"cd {self.basedir} && scons -j{os.cpu_count()} cereal/ common/"
    ])

    self.params = Params(self.params_path)
    self.params.clear_all()
    os.sync()

  def tearDown(self):
    try:
      if self.updated_proc is not None:
        self.updated_proc.terminate()
        self.updated_proc.wait(30)
    except Exception as e:
      print(e)
    finally:
      self._run(f"sudo umount -l {self.merged_dir}")
    exception = self._read_param("LastUpdateException")
    self.assertIsNone(exception, f"LastUpdateException: {exception}")
    self.tmp_dir.cleanup()


  # *** test helpers ***


  def _run(self, cmd, cwd=None):
    if not isinstance(cmd, list):
      cmd = (cmd,)

    for c in cmd:
      subprocess.check_output(c, cwd=cwd, shell=True)

  def _get_updated_proc(self):
    os.environ["PYTHONPATH"] = self.basedir
    os.environ["GIT_AUTHOR_NAME"] = "testy tester"
    os.environ["GIT_COMMITTER_NAME"] = "testy tester"
    os.environ["GIT_AUTHOR_EMAIL"] = "testy@tester.test"
    os.environ["GIT_COMMITTER_EMAIL"] = "testy@tester.test"
    os.environ["UPDATER_PARAMS_PATH"] = self.params_path
    os.environ["UPDATER_LOCK_FILE"] = os.path.join(self.tmp_dir.name, "updater.lock")
    os.environ["UPDATER_STAGING_ROOT"] = self.staging_root
    updated_path = os.path.join(self.basedir, "selfdrive/updated.py")
    return subprocess.Popen(updated_path, env=os.environ)

  def _start_updater(self, offroad=True):
    self.params.put_bool("IsOffroad", offroad)
    self.updated_proc = self._get_updated_proc()
    self._wait_for_idle()

  def _update_now(self):
    self.updated_proc.send_signal(signal.SIGHUP)

  # TODO: this should be implemented in params
  def _read_param(self, key: str, timeout=1):
    ret = None
    start_time = time.monotonic()
    while ret is None:
      ret = self.params.get(key, encoding='utf8')
      if time.monotonic() - start_time > timeout:
        break
      time.sleep(0.01)
    return ret

  def _wait_for_idle(self, timeout=120):
    with Timeout(timeout, "timed out waiting for updated to be idle"):
      while True:
        state = self._read_param("UpdaterState")
        if state == "idle":
          break
        time.sleep(0.1)

  def _wait_for_update(self, timeout=120, clear_param=False):
    if clear_param:
      self.params.remove("UpdaterState")

    self._update_now()
    self._wait_for_idle(timeout)

  def _make_commit(self):
    all_dirs, all_files = [], []
    for root, dirs, files in os.walk(self.git_remote_dir):
      if ".git" in root:
        continue
      for d in dirs:
        all_dirs.append(os.path.join(root, d))
      for f in files:
        all_files.append(os.path.join(root, f))

    # make a new dir and some new files
    new_dir = os.path.join(self.git_remote_dir, "this_is_a_new_dir")
    os.mkdir(new_dir)
    for _ in range(random.randrange(5, 30)):
      for d in (new_dir, random.choice(all_dirs)):
        with tempfile.NamedTemporaryFile(dir=d, delete=False) as f:
          f.write(os.urandom(random.randrange(1, 1000000)))

    # modify some files
    for f in random.sample(all_files, random.randrange(5, 50)):
      with open(f, "w+") as ff:
        txt = ff.readlines()
        ff.seek(0)
        for line in txt:
          ff.write(line[::-1])

    # remove some files
    for f in random.sample(all_files, random.randrange(5, 50)):
      os.remove(f)

    # remove some dirs
    for d in random.sample(all_dirs, random.randrange(1, 10)):
      shutil.rmtree(d)

    # commit the changes
    self._run([
      "git add -A",
      "git commit -m 'an update'",
    ], cwd=self.git_remote_dir)

  def _check_update_state(self, update_available: bool):
    # make sure LastUpdateTime is recent
    t = self._read_param("LastUpdateTime")
    last_update_time = datetime.datetime.fromisoformat(t)
    td = datetime.datetime.utcnow() - last_update_time
    self.assertLess(td.total_seconds(), 10)
    self.params.remove("LastUpdateTime")

    # wait a bit for the rest of the params to be written
    time.sleep(1)

    # check params
    update = self._read_param("UpdateAvailable")
    self.assertEqual(update == "1", update_available, f"UpdateAvailable: {repr(update)}")
    self.assertEqual(self._read_param("UpdateFailedCount"), "0")

    # TODO: check that the finalized update actually matches remote
    # check the .overlay_init and .overlay_consistent flags
    self.assertTrue(os.path.isfile(os.path.join(self.basedir, ".overlay_init")))
    self.assertTrue(os.path.isfile(os.path.join(self.finalized_dir, ".overlay_consistent")))


  # *** test cases ***


  # Run updated for 10 cycles with no update
  def test_no_update(self):
    self._start_updater()
    for _ in range(10):
      self._wait_for_update(clear_param=True)
      self._check_update_state(False)


if __name__ == "__main__":
  unittest.main()
