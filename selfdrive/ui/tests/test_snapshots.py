#!/usr/bin/env python3
import os
import subprocess
import unittest
from pathlib import Path
from tempfile import NamedTemporaryFile

from common.params import Params


def take_snapshot(output_path: str) -> None:
  env = os.environ.copy()
  env["QT_QPA_PLATFORM"] = "offscreen"
  subprocess.run(["selfdrive/ui/tests/ui_snapshot", "--output", output_path], env=env, check=True, stderr=subprocess.DEVNULL)


def check_images_identical(image_path_a: str, image_path_b: str) -> bool:
  try:
    subprocess.run(["compare", "-metric", "AE", image_path_a, image_path_b, "null:"], check=True, stderr=subprocess.DEVNULL)
    return True
  except subprocess.CalledProcessError:
    return False


def check_snapshot_identical(snapshot_path: str) -> bool:
  with NamedTemporaryFile(suffix=".png", delete=False) as f:
    take_snapshot(f.name)
    if check_images_identical(snapshot_path, f.name):
      os.unlink(f.name)
      return True
    else:
      print(f"Snapshot {snapshot_path} does not match {f.name}")
      return False


class TestSnapshots(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.params = Params()
    cls.snapshot_dir = Path(__file__).parent / "snapshots"

  def setUp(self):
    self.params.put_bool("ExperimentalMode", False)
    self.params.put_bool("PrimeType", False)

  def test_base(self):
    self.assertTrue(check_snapshot_identical(self.snapshot_dir / "base.png"))

  def test_prime(self):
    self.params.put_bool("PrimeType", True)
    self.assertTrue(check_snapshot_identical(self.snapshot_dir / "prime.png"))

  def test_experimental_mode(self):
    self.params.put_bool("ExperimentalMode", True)
    self.assertTrue(check_snapshot_identical(self.snapshot_dir / "experimental_mode.png"))


if __name__ == "__main__":
  unittest.main()
