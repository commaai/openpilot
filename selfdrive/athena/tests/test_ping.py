#!/usr/bin/env python3
import subprocess
import time
import unittest
from typing import Optional

from common.timeout import Timeout
from selfdrive.athena.tests.test_athenad_ping import wifi_radio


def ping() -> None:
  # https://serverfault.com/a/42382
  subprocess.run(["bash", "-c", "until ping -c1 www.google.com >/dev/null 2>&1; do :; done"], check=True)


class Timer(Timeout):
  start_time: Optional[float]
  end_time: Optional[float]
  elapsed_time: Optional[float]

  def handle_timeout(self, signume, frame):
    self.end_time = time.monotonic()
    super().handle_timeout(signume, frame)

  def __enter__(self):
    self.start_time = time.monotonic()
    self.end_time = None
    self.elapsed_time = None
    return super().__enter__()

  def __exit__(self, exc_type, exc_val, exc_tb):
    if self.end_time is None:
      self.end_time = time.monotonic()
      self.elapsed_time = self.end_time - self.start_time
    return super().__exit__(exc_type, exc_val, exc_tb)


class TestPing(unittest.TestCase):
  @classmethod
  def tearDownClass(cls):
    wifi_radio(True)

  @unittest.skip("only run on desk")
  def test_ping(self):
    """Measure how long it takes for connectivity after disabling Wi-Fi"""
    timer = Timer(seconds=20)

    with self.subTest("Wi-Fi"):
      with timer:
        ping()
      print(f"Wi-Fi ping took {timer.elapsed_time:.2f} seconds")

    with self.subTest("LTE"):
      wifi_radio(False)
      with timer:
        ping()
      print(f"LTE ping took {timer.elapsed_time:.2f} seconds")

    with self.subTest("Wi-Fi"):
      wifi_radio(True)
      with timer:
        ping()
      print(f"Wi-Fi ping took {timer.elapsed_time:.2f} seconds")


if __name__ == "__main__":
  unittest.main()
