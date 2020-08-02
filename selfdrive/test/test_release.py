#!/usr/bin/env python3
import os
import re
import unittest
import subprocess

from parameterized import parameterized

from common.basedir import BASEDIR
from selfdrive.car.fingerprints import all_known_cars
from selfdrive.car.car_helpers import interfaces

# TODO: tests for devel/master-ci

def run(cmd):
  return subprocess.call(cmd, cwd=BASEDIR, shell=True)

class TestReleaseCommon(unittest.TestCase):

  BRANCH = ""

  @classmethod
  def setUpClass(cls):
    if cls.__name__ == "TestReleaseCommon":
      raise unittest.SkipTest

    r = run(f"git checkout {cls.BRANCH}")
    assert r == 0, f"failed to checkout branch '{cls.BRANCH}'"

    r = run("git clean -xdf")
    assert r == 0, "git clean failed"

  #def test_build_dry_run(self):
  #  r = run("scons -j3 --question")
  #  assert r == 0, "scons dry run failed"

  # run manager for a minute and ensure there's no diff
  def test_no_build(self):
    pass

  # TODO: remove a build product and ensure scons isn't called
  def test_no_build_should_fail(self):
    pass

class TestDashcam(TestReleaseCommon):

  BRANCH = "dashcam-staging"

  @parameterized.expand(all_known_cars)
  def test_no_carcontroller(self, car_model):
    _, CarController, _ = interfaces[car_model]
    self.assertTrue(CarController is None)


class TestRelease2(TestReleaseCommon):

  BRANCH = "release2-staging"

  def test_git_history(self):
    with open(os.path.join(BASEDIR, "selfdrive/common/version.h")) as f:
      version_h = f.read()
    version = re.findall('"([^"]*)"', version_h)[0].split("-release")[0]
    commit_msg = f"openpilot v{version} release"

    email = "user@comma.ai"
    name = "Vehicle Researcher"
    attrs = [
      ("s", commit_msg),
      ("ae", email),
      ("ce", email),
      ("an", name),
      ("cn", name),
    ]
    for fmt, expected_val in attrs:
      out = subprocess.check_output(f"git log --pretty='{fmt}'", cwd=BASEDIR, shell=True, encoding='utf8').strip()
      assert len(out.split("\n")) == 1, "release2 should only have one commit"
      assert expected_val == out, f"wrong output from git, expected '{expected_val}' but got '{out}'"

if __name__ == "__main__":
  unittest.main()
