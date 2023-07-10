#!/usr/bin/env python3
import threading
import time
import unittest
from typing import cast, Optional

from websocket import create_connection

from common.api import Api
from common.params import Params
from selfdrive.athena.athenad import ATHENA_HOST, ws_recv, ws_send


class TestAthenadPing(unittest.TestCase):
  def setUp(self) -> None:
    self.params = Params()
    dongle_id = self.params.get("DongleId", encoding="utf-8")
    api = Api(dongle_id)

    self.ws_uri = ATHENA_HOST + "/ws/v2/" + dongle_id
    self.cookie = "jwt=" + api.get_token()

  def _get_ping_time(self) -> Optional[str]:
    return cast(Optional[str], self.params.get("LastAthenaPingTime", encoding="utf-8"))

  def _clear_ping_time(self) -> None:
    self.params.remove("LastAthenaPingTime")

  def test_ping(self) -> None:
    self._clear_ping_time()

    ws = create_connection(self.ws_uri,
                           cookie=self.cookie,
                           enable_multithread=True,
                           timeout=30.0)

    end_event = threading.Event()
    threads = [
      threading.Thread(target=ws_recv, args=(ws, end_event), name="ws_recv"),
      threading.Thread(target=ws_send, args=(ws, end_event), name="ws_send"),
    ]

    for t in threads:
      t.start()

    try:
      while not end_event.is_set() and self._get_ping_time() is None:
        time.sleep(0.1)
    except (KeyboardInterrupt, SystemExit):
      end_event.set()
      raise
    finally:
      end_event.set()
      for t in threads:
        t.join()

    self.assertIsNotNone(self._get_ping_time())


if __name__ == "__main__":
  unittest.main()
