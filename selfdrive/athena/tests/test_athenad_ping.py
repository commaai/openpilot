#!/usr/bin/env python3
import subprocess
import threading
import time
import unittest
from typing import cast, Optional
from unittest.mock import MagicMock

from websocket import create_connection

from common import realtime
from common.params import Params
from common.timeout import Timeout
from selfdrive.athena import athenad


realtime.set_core_affinity = MagicMock()
athenad.create_connection = MagicMock(wraps=create_connection)
athenad.upload_handler = MagicMock()
# athenad.ws_recv = MagicMock()
# athenad.ws_send = MagicMock()
athenad.upload_handler = MagicMock()
athenad.log_handler = MagicMock()
athenad.stat_handler = MagicMock()
athenad.jsonrpc_handler = MagicMock()


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


def wifi_radio(on: bool) -> None:
  print(f"wifi_radio({'on' if on else 'off'})")
  subprocess.run(["nmcli", "radio", "wifi", "on" if on else "off"], check=True)


class TestException(BaseException):
  pass


class TestAthenadPing(unittest.TestCase):
  params: Params
  dongle_id: str

  athenad: threading.Thread
  exit_event: threading.Event

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

  @classmethod
  def tearDownClass(cls) -> None:
    wifi_radio(True)

  def setUp(self) -> None:
    wifi_radio(True)
    self._clear_ping_time()

    self.exit_event = threading.Event()
    self.exit_event.set()
    self.athenad = threading.Thread(target=athenad.main, args=(self.exit_event,))

    athenad.create_connection.reset_mock()

  def tearDown(self) -> None:
    if self.athenad.is_alive():
      self.exit_event.set()
      self.athenad.join()

  @unittest.skip("only run on desk")
  def test_timeout(self) -> None:
    self.athenad.start()

    time.sleep(1)
    athenad.create_connection.assert_called_once()
    athenad.create_connection.reset_mock()

    # check normal behaviour
    with self.subTest("Wi-Fi: receives ping"), Timeout(70, "no ping received"):
      while not self._received_ping():
        time.sleep(0.1)

    athenad.create_connection.assert_not_called()

    # websocket should attempt reconnect after short time
    timer = Timer(180, "no reconnect attempt")
    with self.subTest(f"LTE: attempt reconnect within {timer.seconds}s"):
      wifi_radio(False)
      print("waiting for reconnect attempt")
      with timer:
        while not athenad.create_connection.called:
          time.sleep(0.1)
      print(f"reconnect attempt after {timer.elapsed_time:.2f}s")

    self._clear_ping_time()

    # check ping received after reconnect
    with self.subTest("LTE: receives ping"), Timeout(70, "no ping received"):
      while not self._received_ping():
        time.sleep(0.1)


if __name__ == "__main__":
  unittest.main()
