#!/usr/bin/env python3
import threading
import time
import unittest
from typing import Callable, cast, Optional

from websocket import create_connection

from common.api import Api
from common.params import Params
from common.timeout import Timeout
from selfdrive.athena.athenad import ATHENA_HOST, ws_recv, ws_send


def athena_main(ws_uri: str, cookie: str, stop_condition: Callable[[], bool]) -> None:
  while 1:
    ws = create_connection(ws_uri,
                           cookie=cookie,
                           enable_multithread=True,
                           timeout=30.0)
    end_event = threading.Event()
    threads = [
      threading.Thread(target=ws_recv, args=(ws, end_event), name="ws_recv"),
      threading.Thread(target=ws_send, args=(ws, end_event), name="ws_send"),
    ]

    for t in threads:
      print("starting", t.name)
      t.start()
    try:
      while not stop_condition() and not end_event.is_set():
        time.sleep(0.1)
      print("goodbye threads")
      end_event.set()
    except (KeyboardInterrupt, SystemExit):
      end_event.set()
      raise
    finally:
      for t in threads:
        print("joining", t.name)
        t.join()


class TestAthenadPing(unittest.TestCase):
  params: Params
  ws_uri: str
  cookie: str

  def _get_ping_time(self) -> Optional[str]:
    return cast(Optional[str], self.params.get("LastAthenaPingTime", encoding="utf-8"))

  def _clear_ping_time(self) -> None:
    self.params.remove("LastAthenaPingTime")

  def _received_ping(self) -> bool:
    return self._get_ping_time() is not None

  @classmethod
  def setUpClass(cls) -> None:
    cls.params = Params()

    dongle_id = cls.params.get("DongleId", encoding="utf-8")
    cls.ws_uri = ATHENA_HOST + "/ws/v2/" + dongle_id
    cls.cookie = "jwt=" + Api(dongle_id).get_token()

  def setUp(self) -> None:
    self._clear_ping_time()

  def test_ping(self) -> None:
    with Timeout(70, "no ping received"):
      athena_main(self.ws_uri, self.cookie, stop_condition=self._received_ping)
    self.assertIsNotNone(self._get_ping_time())


if __name__ == "__main__":
  unittest.main()
