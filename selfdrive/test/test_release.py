#!/usr/bin/env python3
import os
import unittest


class TestReleaseCommon(unittest.TestCase):

  BRANCH = ""

  @classmethod
  def setUpClass(cls):
    if cls.__name__ == "TestReleaseCommon":
      raise unittest.SkipTest

    r = os.system(f"git checkout {cls.BRANCH}")
    assert r == 0, f"failed to checkout branch '{cls.BRANCH}'"

  def test_build_dry_run(self):
    r = os.system("scons -j3 --question")
    assert r == 0, "scons dry run failed"

  # run manager for a minute and ensure there's no diff
  def test_no_build(self):
    pass

  # TODO: remove a build product and ensure scons isn't called
  def test_no_build_should_fail(self):
    pass

  # test commit msg version
  def test_version(self):
    pass

class TestDashcam(TestReleaseCommon):

  BRANCH = "dashcam-staging"

  def test_no_carcontroller(self):
    pass


class TestRelease2(TestReleaseCommon):

  BRANCH = "dashcam-staging"

  def test_history(self):
    pass


if __name__ == "__main__":
  unittest.main()
