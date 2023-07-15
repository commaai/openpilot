#!/usr/bin/env python3
import subprocess
import threading
import time
import unittest
from typing import Callable, cast, Optional

from websocket import create_connection, WebSocketException

from common.api import Api
from common.params import Params
from common.timeout import Timeout
from selfdrive.athena.athenad import ATHENA_HOST, backoff, ws_recv, ws_send


def wifi_radio(on: bool) -> None:
  subprocess.run(["nmcli", "radio", "wifi", "on" if on else "off"], check=True)


def athena_main(dongle_id: str, stop_condition: Callable[[], bool], reconnect: bool = False) -> None:
  start = None
  conn_retries = 0
  while not stop_condition():
    try:
      if start is None:
        start = time.monotonic()

      print(f"[WS] connecting ({conn_retries=})")
      ws = create_connection(ATHENA_HOST + "/ws/v2/" + dongle_id,
                             cookie="jwt=" + Api(dongle_id).get_token(),
                             enable_multithread=True,
                             timeout=30.0)

      duration = time.monotonic() - start
      print(f"[WS] connected in {duration:.2f}s")
      start = None

      conn_retries = 0

      end_event = threading.Event()
      threads = [
        threading.Thread(target=ws_recv, args=(ws, end_event), name="ws_recv"),
        threading.Thread(target=ws_send, args=(ws, end_event), name="ws_send"),
      ]

      for t in threads:
        t.start()
      try:
        while not stop_condition() and not end_event.is_set():
          time.sleep(0.1)
        end_event.set()
      except (KeyboardInterrupt, SystemExit):
        end_event.set()
        raise
      finally:
        for t in threads:
          print("[WS] joining", t.name)
          t.join()
    except (KeyboardInterrupt, SystemExit):
      break
    except (ConnectionError, TimeoutError, WebSocketException) as e:
      print("[WS] connection error")
      print(e)
      conn_retries += 1
    except Exception as e:
      print("[WS] exception")
      print(e)
      conn_retries += 1

    if not reconnect:
      break
    time.sleep(backoff(conn_retries))


class TestAthenadPing(unittest.TestCase):
  params: Params
  dongle_id: str

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
    wifi_radio(True)

  @classmethod
  def tearDownClass(cls) -> None:
    wifi_radio(True)

  def setUp(self) -> None:
    wifi_radio(True)
    self._clear_ping_time()

  @unittest.skip("only run on desk")
  def test_ws_timeout(self) -> None:
    # start athena_main in a thread
    stop_event = threading.Event()
    t = threading.Thread(target=athena_main, args=(self.dongle_id, lambda: stop_event.is_set()))  # pylint: disable=unnecessary-lambda
    t.start()

    try:
      # check normal behaviour
      with self.subTest("Wi-Fi"), Timeout(120, "no ping received"):
        while not self._received_ping():
          time.sleep(0.1)

      # check that websocket disconnects in less than 60 seconds
      with self.subTest("Switch to LTE"), Timeout(60, "did not disconnect"):
        athena_main(self.dongle_id, stop_condition=lambda: False, reconnect=False)
    finally:
      stop_event.set()
      t.join()

  @unittest.skip("only run on desk")
  def test_recover_ping_wifi_to_lte(self) -> None:
    # check normal behaviour
    with self.subTest("Wi-Fi"), Timeout(120, "no ping received"):
      athena_main(self.dongle_id, stop_condition=self._received_ping)

    self._clear_ping_time()

    # check that we continue to update ping after switching to LTE
    with self.subTest("Switch to LTE"):
      wifi_radio(False)
      with Timeout(120, "no ping received"):
        athena_main(self.dongle_id, stop_condition=self._received_ping)


if __name__ == "__main__":
  unittest.main()
