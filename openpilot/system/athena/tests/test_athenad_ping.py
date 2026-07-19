import subprocess
import threading
import time
import unittest
from typing import cast
from unittest import mock

from openpilot.common.params import Params
from openpilot.common.timeout import Timeout
from openpilot.selfdrive.test.helpers import OpenpilotTestCase
from openpilot.system.athena import athenad
from openpilot.common.hardware import TICI

TIMEOUT_TOLERANCE = 20  # seconds


def wifi_radio(on: bool) -> None:
  if not TICI:
    return
  print(f"wifi {'on' if on else 'off'}")
  subprocess.run(["nmcli", "radio", "wifi", "on" if on else "off"], check=True)


class TestAthenadPing(OpenpilotTestCase):
  params: Params
  dongle_id: str

  athenad: threading.Thread
  exit_event: threading.Event

  def _get_ping_time(self) -> str | None:
    return cast(str | None, self.params.get("LastAthenaPingTime"))

  def _clear_ping_time(self) -> None:
    self.params.remove("LastAthenaPingTime")

  def _received_ping(self) -> bool:
    return self._get_ping_time() is not None

  @classmethod
  def tearDownClass(cls) -> None:
    wifi_radio(True)

  def setUp(self) -> None:
    super().setUp()
    self.params = Params()
    self.dongle_id = self.params.get("DongleId")

    wifi_radio(True)
    self._clear_ping_time()

    self.exit_event = threading.Event()
    self.athenad = threading.Thread(target=athenad.main, args=(self.exit_event,))

  def tearDown(self) -> None:
    if self.athenad.is_alive():
      self.exit_event.set()
      self.athenad.join()

  def assertTimeout(self, reconnect_time: float) -> None:
    self.athenad.start()

    mock_create_connection = self.enterContext(mock.patch('openpilot.system.athena.athenad.create_connection',
                                                          new_callable=lambda: mock.MagicMock(wraps=athenad.create_connection)))

    time.sleep(1)
    mock_create_connection.assert_called_once()
    mock_create_connection.reset_mock()

    # check normal behavior, server pings on connection
    with self.subTest("Wi-Fi: receives ping"), Timeout(70, "no ping received"):
      while not self._received_ping():
        time.sleep(0.1)
      print("ping received")

    mock_create_connection.assert_not_called()

    # websocket should attempt reconnect after short time
    with self.subTest("LTE: attempt reconnect"):
      wifi_radio(False)
      print("waiting for reconnect attempt")
      start_time = time.monotonic()
      with Timeout(reconnect_time, "no reconnect attempt"):
        while not mock_create_connection.called:
          time.sleep(0.1)
        print(f"reconnect attempt after {time.monotonic() - start_time:.2f}s")

    self._clear_ping_time()

    # check ping received after reconnect
    with self.subTest("LTE: receives ping"), Timeout(70, "no ping received"):
      while not self._received_ping():
        time.sleep(0.1)
      print("ping received")

  @unittest.skipIf(not TICI, "only run on desk")
  def test_offroad(self) -> None:
    self.params.put_bool("IsOffroad", True, block=True)
    self.assertTimeout(60 + TIMEOUT_TOLERANCE)  # based using TCP keepalive settings

  @unittest.skipIf(not TICI, "only run on desk")
  def test_onroad(self) -> None:
    self.params.put_bool("IsOffroad", False, block=True)
    self.assertTimeout(21 + TIMEOUT_TOLERANCE)
