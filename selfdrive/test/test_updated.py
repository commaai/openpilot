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


class TestUpdated(unittest.TestCase):

  def setUp(self):
    self.updated_proc = None

    self.tmp_dir = tempfile.TemporaryDirectory()
    org_dir = os.path.join(self.tmp_dir.name, "commaai")

    self.basedir = os.path.join(org_dir, "openpilot")
    self.git_remote_dir = os.path.join(org_dir, "openpilot_remote")
    self.staging_dir = os.path.join(org_dir, "safe_staging")
    for d in [org_dir, self.basedir, self.git_remote_dir, self.staging_dir]:
      os.mkdir(d)

    self.neos_version = os.path.join(org_dir, "neos_version")
    self.neosupdate_dir = os.path.join(org_dir, "neosupdate")
    with open(self.neos_version, "w") as f:
      v = subprocess.check_output(r"bash -c 'source launch_env.sh && echo $REQUIRED_NEOS_VERSION'",
                                  cwd=BASEDIR, shell=True, encoding='utf8').strip()
      f.write(v)

    self.upper_dir = os.path.join(self.staging_dir, "upper")
    self.merged_dir = os.path.join(self.staging_dir, "merged")
    self.finalized_dir = os.path.join(self.staging_dir, "finalized")

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
      f"cd {self.basedir} && scons -j{os.cpu_count()} cereal/ common/"
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
    os.environ["UPDATER_TEST_IP"] = "localhost"
    os.environ["UPDATER_LOCK_FILE"] = os.path.join(self.tmp_dir.name, "updater.lock")
    os.environ["UPDATER_STAGING_ROOT"] = self.staging_dir
    os.environ["UPDATER_NEOS_VERSION"] = self.neos_version
    os.environ["UPDATER_NEOSUPDATE_DIR"] = self.neosupdate_dir
    updated_path = os.path.join(self.basedir, "selfdrive/updated.py")
    return subprocess.Popen(updated_path, env=os.environ)

  def _start_updater(self, offroad=True, nosleep=False):
    self.params.put_bool("IsOffroad", offroad)
    self.updated_proc = self._get_updated_proc()
    if not nosleep:
      time.sleep(1)

  def _update_now(self):
    self.updated_proc.send_signal(signal.SIGHUP)

  # TODO: this should be implemented in params
  def _read_param(self, key, timeout=1):
    ret = None
    start_time = time.monotonic()
    while ret is None:
      ret = self.params.get(key, encoding='utf8')
      if time.monotonic() - start_time > timeout:
        break
      time.sleep(0.01)
    return ret

  def _wait_for_update(self, timeout=30, clear_param=False):
    if clear_param:
      self.params.delete("LastUpdateTime")

    self._update_now()
    t = self._read_param("LastUpdateTime", timeout=timeout)
    if t is None:
      raise Exception("timed out waiting for update to complete")

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

  def _check_update_state(self, update_available):
    # make sure LastUpdateTime is recent
    t = self._read_param("LastUpdateTime")
    last_update_time = datetime.datetime.fromisoformat(t)
    td = datetime.datetime.utcnow() - last_update_time
    self.assertLess(td.total_seconds(), 10)
    self.params.delete("LastUpdateTime")

    # wait a bit for the rest of the params to be written
    time.sleep(0.1)

    # check params
    update = self._read_param("UpdateAvailable")
    self.assertEqual(update == "1", update_available, f"UpdateAvailable: {repr(update)}")
    self.assertEqual(self._read_param("UpdateFailedCount"), "0")

    # TODO: check that the finalized update actually matches remote
    # check the .overlay_init and .overlay_consistent flags
    self.assertTrue(os.path.isfile(os.path.join(self.basedir, ".overlay_init")))
    self.assertEqual(os.path.isfile(os.path.join(self.finalized_dir, ".overlay_consistent")), update_available)


  # *** test cases ***


  # Run updated for 100 cycles with no update
  def test_no_update(self):
    self._start_updater()
    for _ in range(100):
      self._wait_for_update(clear_param=True)
      self._check_update_state(False)

  # Let the updater run with no update for a cycle, then write an update
  def test_update(self):
    self._start_updater()

    # run for a cycle with no update
    self._wait_for_update(clear_param=True)
    self._check_update_state(False)

    # write an update to our remote
    self._make_commit()

    # run for a cycle to get the update
    self._wait_for_update(timeout=60, clear_param=True)
    self._check_update_state(True)

    # run another cycle with no update
    self._wait_for_update(clear_param=True)
    self._check_update_state(True)

  # Let the updater run for 10 cycles, and write an update every cycle
  @unittest.skip("need to make this faster")
  def test_update_loop(self):
    self._start_updater()

    # run for a cycle with no update
    self._wait_for_update(clear_param=True)
    for _ in range(10):
      time.sleep(0.5)
      self._make_commit()
      self._wait_for_update(timeout=90, clear_param=True)
      self._check_update_state(True)

  # Test overlay re-creation after tracking a new file in basedir's git
  def test_overlay_reinit(self):
    self._start_updater()

    overlay_init_fn = os.path.join(self.basedir, ".overlay_init")

    # run for a cycle with no update
    self._wait_for_update(clear_param=True)
    self.params.delete("LastUpdateTime")
    first_mtime = os.path.getmtime(overlay_init_fn)

    # touch a file in the basedir
    self._run("touch new_file && git add new_file", cwd=self.basedir)

    # run another cycle, should have a new mtime
    self._wait_for_update(clear_param=True)
    second_mtime = os.path.getmtime(overlay_init_fn)
    self.assertTrue(first_mtime != second_mtime)

    # run another cycle, mtime should be same as last cycle
    self._wait_for_update(clear_param=True)
    new_mtime = os.path.getmtime(overlay_init_fn)
    self.assertTrue(second_mtime == new_mtime)

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


  # *** test cases with NEOS updates ***


  # Run updated with no update, make sure it clears the old NEOS update
  def test_clear_neos_cache(self):
    # make the dir and some junk files
    os.mkdir(self.neosupdate_dir)
    for _ in range(15):
      with tempfile.NamedTemporaryFile(dir=self.neosupdate_dir, delete=False) as f:
        f.write(os.urandom(random.randrange(1, 1000000)))

    self._start_updater()
    self._wait_for_update(clear_param=True)
    self._check_update_state(False)
    self.assertFalse(os.path.isdir(self.neosupdate_dir))

  # Let the updater run with no update for a cycle, then write an update
  @unittest.skip("TODO: only runs on device")
  def test_update_with_neos_update(self):
    # bump the NEOS version and commit it
    self._run([
      "echo 'export REQUIRED_NEOS_VERSION=3' >> launch_env.sh",
      "git -c user.name='testy' -c user.email='testy@tester.test' \
       commit -am 'a neos update'",
    ], cwd=self.git_remote_dir)

    # run for a cycle to get the update
    self._start_updater()
    self._wait_for_update(timeout=60, clear_param=True)
    self._check_update_state(True)

    # TODO: more comprehensive check
    self.assertTrue(os.path.isdir(self.neosupdate_dir))


if __name__ == "__main__":
  unittest.main()
