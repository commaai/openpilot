#!/usr/bin/env python3
import subprocess
import time
import unittest
from typing import Optional

from common.timeout import Timeout


def connect_lte() -> None:
  subprocess.run(["nmcli", "connection", "modify", "--temporary", "lte", "ipv4.route-metric", "1", "ipv6.route-metric", "1"], check=True)
  subprocess.run(["nmcli", "connection", "up", "lte"], check=True)


def restart_network_manager() -> None:
  subprocess.run(["sudo", "systemctl", "restart", "NetworkManager"], check=True)


def ping() -> None:
  """attempt to ping google.com, return on success"""
  subprocess.run(["bash", "-c", "until ping -c1 www.google.com; do :; done"], check=True)


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
    restart_network_manager()

  # Measure how long it takes to switch from Wi-Fi to LTE (time to first ping after disconnect)
  def test_ping(self):
    timer = Timer(60)

    with self.subTest("Wi-Fi"):
      with timer:
        ping()
      print(f"Wi-Fi ping took {timer.elapsed_time:.2f} seconds")

    with self.subTest("LTE"):
      try:
        connect_lte()
        with timer:
          ping()
        print(f"LTE ping took {timer.elapsed_time:.2f} seconds")
      finally:
        restart_network_manager()

    with self.subTest("Wi-Fi"):
      with timer:
        ping()
      print(f"Wi-Fi ping took {timer.elapsed_time:.2f} seconds")


if __name__ == "__main__":
  unittest.main()
