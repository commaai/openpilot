#!/usr/bin/env python3
import subprocess
import unittest

from selfdrive.athena.tests.test_athenad_ping import Timer, wifi_radio


def ping() -> None:
  # https://serverfault.com/a/42382
  subprocess.run(["bash", "-c", "until ping -c1 www.google.com >/dev/null 2>&1; do :; done"], check=True)


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
