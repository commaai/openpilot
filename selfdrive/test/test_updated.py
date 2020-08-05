#!/usr/bin/env python3
import datetime
import os
import time
import tempfile
import unittest
import signal
import subprocess

from common.basedir import BASEDIR
from common.params import Params


class TestUpdater(unittest.TestCase):

  def setUp(self):
    self.updated_proc = None

    self.tmp_dir = tempfile.TemporaryDirectory()
    org_dir = os.path.join(self.tmp_dir.name, "commaai")

    self.basedir = os.path.join(org_dir, "openpilot")
    self.git_remote_dir = os.path.join(org_dir, "openpilot_remote")
    self.staging_dir = os.path.join(org_dir, "safe_staging")
    for d in [org_dir, self.basedir, self.git_remote_dir, self.staging_dir]:
      os.mkdir(d)

    # setup local submodule remotes
    submodules = subprocess.check_output("git submodule --quiet foreach 'echo $name'",
                                         shell=True, cwd=BASEDIR, encoding='utf8').split()
    for s in submodules:
      sub_path = os.path.join(org_dir, s.split("_repo")[0])
      self._run(f"git clone {s} {sub_path}.git", cwd=BASEDIR)

    # setup two git repos, a remote and one we'll run updated in
    self._run([
      f"git clone {BASEDIR} {self.git_remote_dir}",
      f"git clone {self.git_remote_dir} {self.basedir}",
      f"cd {self.basedir} && git submodule init && git submodule update",
      f"cd {self.basedir} && scons -j4 cereal"
    ])

    self.params = Params(db=os.path.join(self.basedir, "persist/params"))
    self.params.clear_all()
    os.sync()

  def tearDown(self):
    try:
      if self.updated_proc is not None:
        self.updated_proc.terminate()
        self.updated_proc.wait(30)
    except Exception as e:
      print(e)
    self.tmp_dir.cleanup()

  def _run(self, cmd, cwd=None):
    if not isinstance(cmd, list):
      cmd = (cmd,)

    for c in cmd:
      subprocess.check_output(c, cwd=cwd, shell=True)

  def _get_updated_proc(self):
    os.environ["PYTHONPATH"] = self.basedir
    os.environ["UPDATER_TESTING"] = "1"
    os.environ["UPDATER_LOCK_FILE"] = os.path.join(self.tmp_dir.name, "updater.lock")
    os.environ["UPDATER_STAGING_ROOT"] = self.staging_dir
    updated_path = os.path.join(self.basedir, "selfdrive/updated.py")
    return subprocess.Popen(updated_path, env=os.environ)

  def _start_updater(self, offroad=True):
    self.params.put("IsOffroad", "1" if offroad else "0")
    self.updated_proc = self._get_updated_proc()

  def _update_now(self):
    self.updated_proc.send_signal(signal.SIGHUP)

  def _wait_for_update(self, timeout=30, clear_param=False):
    if clear_param:
      self.params.delete("LastUpdateTime")

    if os.getenv("CI") is not None:
      timeout = timeout*2

    self._update_now()
    start_time = time.monotonic()
    while self.params.get("LastUpdateTime") is None:
      if time.monotonic() - start_time > timeout:
        raise Exception("timed out waiting for update to complate")
      time.sleep(0.05)

  def _make_commit(self):
    self._run([
      "git config user.email tester@testing.com",
      "git config user.name Testy Tester",
      "git commit --allow-empty -m 'an update'",
    ], cwd=self.git_remote_dir)

  def _check_update_time(self):
    # make sure LastUpdateTime is recent
    t = self.params.get("LastUpdateTime", encoding='utf8')
    last_update_time = datetime.datetime.fromisoformat(t)
    td = datetime.datetime.utcnow() - last_update_time
    self.assertLess(td.total_seconds(), 10)

  def _check_failed_updates(self, expected_count=0):
    failed_updates = int(self.params.get("UpdateFailedCount", encoding='utf8'))
    self.assertEqual(failed_updates, expected_count)

  def _assert_update_available(self, available):
    update = self.params.get("UpdateAvailable")
    self.assertEqual(update == b"1", available, f"UpdateAvailable: '{repr(update)}'")

  # Run updated for 50 cycles with no update
  def test_no_update(self):
    self._start_updater()
    time.sleep(2)

    for _ in range(50):
      self._wait_for_update(clear_param=True)

      # give a bit of time to write all the params
      time.sleep(0.1)
      self._check_update_time()
      self._assert_update_available(False)
      self._check_failed_updates()

  # Let the updater run with no update for a cycle, then write an update
  def test_update(self):
    self._start_updater()
    time.sleep(2)

    # run for a cycle with no update
    self._wait_for_update(clear_param=True)

    # give a bit of time to write all the params
    time.sleep(0.1)
    self._check_update_time()
    self._assert_update_available(False)
    self._check_failed_updates()

    self.params.delete("LastUpdateTime")

    # write an update to our remote
    self._make_commit()

    self._wait_for_update(timeout=60, clear_param=True)
    time.sleep(0.1)
    self._check_update_time()
    self._assert_update_available(True)
    self._check_failed_updates()

  # Let the updater run for 10 cycle, and write an update every cycle
  def test_update_loop(self):
    self._start_updater()
    time.sleep(2)

    # run for a cycle with no update
    self._wait_for_update(clear_param=True)

    # TODO: make this faster, so we can do more loops
    for _ in range(5):
      time.sleep(0.5)
      self._make_commit()
      self._wait_for_update(timeout=90, clear_param=True)

      # give a bit of time to write all the params
      time.sleep(0.2)
      self._check_update_time()
      self._assert_update_available(True)
      self._check_failed_updates()
      self.params.delete("UpdateAvailable")

  # Test overlay re-creation after touching basedir's git
  def test_overlay_reinit(self):
    self._start_updater()

    time.sleep(2)

    overlay_init_fn = os.path.join(self.basedir, ".overlay_init")

    # run for a cycle with no update
    self._wait_for_update(clear_param=True)
    self.params.delete("LastUpdateTime")
    first_mtime = os.path.getmtime(overlay_init_fn)

    # touch a file in the basedir
    self._run("touch new_file && git add new_file", cwd=self.basedir)

    # run another cycle and check mtime
    self._wait_for_update(clear_param=True)
    new_mtime = os.path.getmtime(overlay_init_fn)
    self.assertTrue(first_mtime != new_mtime)

  # Make sure updated exits if another instance is running
  def test_multiple_instances(self):
    # start updated and let it run for a cycle
    self._start_updater()
    time.sleep(1)
    self._wait_for_update(clear_param=True)

    # start another instance
    second_updated = self._get_updated_proc()
    ret_code = second_updated.wait(timeout=5)
    self.assertTrue(ret_code is not None)

if __name__ == "__main__":
  unittest.main()
