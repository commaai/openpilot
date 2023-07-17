#!/usr/bin/env python3
import subprocess
import threading
import time
import unittest
from typing import Callable, cast, Optional
from unittest.mock import MagicMock

from common.params import Params
from common.timeout import Timeout
from selfdrive.athena import athenad
from system.hardware import TICI


def wifi_radio(on: bool) -> None:
  if not TICI:
    return
  print(f"wifi {'on' if on else 'off'}")
  subprocess.run(["nmcli", "radio", "wifi", "on" if on else "off"], check=True)


class TestAthenadPing(unittest.TestCase):
  params: Params
  dongle_id: str

  athenad: threading.Thread
  exit_event: threading.Event

  _create_connection: Callable

  def _get_ping_time(self) -> Optional[str]:
    return cast(Optional[str], self.params.get("LastAthenaPingTime", encoding="utf-8"))

  def _clear_ping_time(self) -> None:
    self.params.remove("LastAthenaPingTime")

  def _received_ping(self) -> bool:
    return self._get_ping_time() is not None

  @classmethod
  def setUpClass(cls) -> None:
    cls.params = Params()
    cls.dongle_id = cls.params.get("DongleId", encoding="utf-8")
    cls._create_connection = athenad.create_connection
    athenad.create_connection = MagicMock(wraps=cls._create_connection)

  @classmethod
  def tearDownClass(cls) -> None:
    wifi_radio(True)
    athenad.create_connection = cls._create_connection

  def setUp(self) -> None:
    wifi_radio(True)
    self._clear_ping_time()

    self.exit_event = threading.Event()
    self.athenad = threading.Thread(target=athenad.main, args=(self.exit_event,))

    athenad.create_connection.reset_mock()

  def tearDown(self) -> None:
    if self.athenad.is_alive():
      self.exit_event.set()
      self.athenad.join()

  @unittest.skipIf(not TICI, "only run on desk")
  def test_timeout(self) -> None:
    self.athenad.start()

    time.sleep(1)
    athenad.create_connection.assert_called_once()
    athenad.create_connection.reset_mock()

    # check normal behaviour
    with self.subTest("Wi-Fi: receives ping"), Timeout(70, "no ping received"):
      while not self._received_ping():
        time.sleep(0.1)
      print("ping received")

    athenad.create_connection.assert_not_called()

    # websocket should attempt reconnect after short time
    with self.subTest("LTE: attempt reconnect"):
      wifi_radio(False)
      print("waiting for reconnect attempt")
      start_time = time.monotonic()
      with Timeout(180, "no reconnect attempt"):
        while not athenad.create_connection.called:
          time.sleep(0.1)
        print(f"reconnect attempt after {time.monotonic() - start_time:.2f}s")

    self._clear_ping_time()

    # check ping received after reconnect
    with self.subTest("LTE: receives ping"), Timeout(70, "no ping received"):
      while not self._received_ping():
        time.sleep(0.1)
      print("ping received")


if __name__ == "__main__":
  unittest.main()
