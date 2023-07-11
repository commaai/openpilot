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


def athena_main(dongle_id: str, stop_condition: Callable[[], bool]) -> None:
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

    time.sleep(backoff(conn_retries))


def connect_lte() -> None:
  subprocess.run(["nmcli", "connection", "modify", "--temporary", "lte", "ipv4.route-metric", "1", "ipv6.route-metric", "1"], check=True)
  subprocess.run(["nmcli", "connection", "up", "lte"], check=True)


def restart_network_manager() -> None:
  subprocess.run(["sudo", "systemctl", "restart", "NetworkManager"], check=True)


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

  def setUp(self) -> None:
    self._clear_ping_time()

  @unittest.skip("TODO")
  def test_ping_wifi_to_lte(self) -> None:
    with self.subTest("wifi"):
      with Timeout(120, "no ping received"):
        athena_main(self.dongle_id, stop_condition=self._received_ping)
      self.assertIsNotNone(self._get_ping_time())

    self._clear_ping_time()

    with self.subTest("lte"):
      try:
        connect_lte()

        with Timeout(120, "no ping received"):
          athena_main(self.dongle_id, stop_condition=self._received_ping)
      finally:
        restart_network_manager()
      self.assertIsNotNone(self._get_ping_time())


if __name__ == "__main__":
  try:
    unittest.main()
  finally:
    restart_network_manager()
