import json
import queue
import threading

from openpilot.cereal import log
from openpilot.common.params import Params
from openpilot.common.test import OpenpilotTestCase
from openpilot.system.athena import athenad


class TestSleepKeepalive(OpenpilotTestCase):
  def setup_method(self):
    self.params = Params()
    self.params.put_bool("ModemSleep", True, block=True)
    athenad.send_queue = queue.PriorityQueue()
    self.modem_awake = True
    self.ka = athenad.SleepKeepalive(self.params, lambda: self.modem_awake)

  def _sent(self) -> dict:
    _, _, data = athenad.send_queue.get(timeout=1)
    return json.loads(data)

  def _ack(self, msg: dict, result=None) -> bool:
    if result is None:
      result = {"sleeping": msg["params"]["sleeping"]}
    return self.ka.handle_response({"jsonrpc": "2.0", "id": msg["id"], "result": result})

  def _modem_sleep(self) -> bool:
    return self.params.get_bool("ModemSleep")

  def test_starts_awake(self):
    assert not self._modem_sleep()

  def test_sleep_after_ack(self):
    assert self.ka.update(want_sleep=True)
    msg = self._sent()
    assert msg["method"] == "keepalive"
    assert msg["params"] == {"sleeping": True, "interval": athenad.KEEPALIVE_INTERVAL_S}
    assert not self._modem_sleep()  # not before athena knows

    assert self._ack(msg)
    assert self.ka.sleeping
    assert self._modem_sleep()
    assert self.params.get("LastAthenaPingTime") is not None

  def test_keepalive_every_interval(self, mocker):
    now = [100.0]
    mocker.patch.object(athenad.time, "monotonic", side_effect=lambda: now[0])
    self.ka.update(want_sleep=True)
    self._ack(self._sent())

    now[0] += athenad.KEEPALIVE_INTERVAL_S - 1
    self.ka.update(want_sleep=True)
    assert athenad.send_queue.empty()
    assert self.ka.next_due() == 1

    now[0] += 1
    self.ka.update(want_sleep=True)
    assert self._sent()["params"]["sleeping"]

  def test_wake(self):
    self.ka.update(want_sleep=True)
    self._ack(self._sent())

    self.ka.update(want_sleep=False)
    assert not self._modem_sleep()  # before athena is told
    msg = self._sent()
    assert msg["params"]["sleeping"] is False
    self._ack(msg)
    assert not self.ka.sleeping

    # awake, athena pings again: nothing more to send
    self.ka.update(want_sleep=False)
    assert athenad.send_queue.empty()

  def test_wake_waits_for_agnos(self, mocker):
    now = [100.0]
    mocker.patch.object(athenad.time, "monotonic", side_effect=lambda: now[0])
    self.ka.update(want_sleep=True)
    self._ack(self._sent())

    self.modem_awake = False
    self.ka.update(want_sleep=False)
    assert not self._modem_sleep()
    assert athenad.send_queue.empty()  # athena would send into a link that can still sleep
    assert self.ka.next_due() < 1

    self.modem_awake = True
    now[0] += 1
    self.ka.update(want_sleep=False)
    assert self._sent()["params"]["sleeping"] is False

  def test_wake_agnos_never_answers(self, mocker):
    now = [100.0]
    mocker.patch.object(athenad.time, "monotonic", side_effect=lambda: now[0])
    self.ka.update(want_sleep=True)
    self._ack(self._sent())

    self.modem_awake = False
    self.ka.update(want_sleep=False)
    now[0] += athenad.KEEPALIVE_TIMEOUT_S + 1
    self.ka.update(want_sleep=False)
    assert self._sent()["params"]["sleeping"] is False

  def test_wake_with_keepalive_in_flight(self):
    self.ka.update(want_sleep=True)
    self._ack(self._sent())
    self.ka.sent_at -= athenad.KEEPALIVE_INTERVAL_S
    self.ka.update(want_sleep=True)
    msg = self._sent()  # periodic keepalive, not answered yet

    self.ka.update(want_sleep=False)
    assert not self._modem_sleep()
    self._ack(msg)
    assert not self._modem_sleep()

  def test_wake_while_sleep_pending(self):
    self.ka.update(want_sleep=True)
    msg = self._sent()
    self.ka.update(want_sleep=False)
    self._ack(msg)
    assert not self._modem_sleep()

    # athena thinks we're sleeping now, tell it otherwise
    self.ka.update(want_sleep=False)
    assert self._sent()["params"]["sleeping"] is False

  def test_unanswered_keepalive(self, mocker):
    now = [100.0]
    mocker.patch.object(athenad.time, "monotonic", side_effect=lambda: now[0])
    self.ka.update(want_sleep=True)
    self._sent()
    now[0] += athenad.KEEPALIVE_TIMEOUT_S + 1
    assert not self.ka.update(want_sleep=True)

  def test_old_athena(self):
    # athena without keepalives answers with neither result nor error
    self.ka.update(want_sleep=True)
    msg = self._sent()
    assert self.ka.handle_response({"jsonrpc": "2.0", "id": msg["id"]})
    assert self.ka.unsupported
    assert not self._modem_sleep()
    assert self.ka.update(want_sleep=True)
    assert athenad.send_queue.empty()

  def test_unexpected_ack_while_sleeping(self):
    self.ka.update(want_sleep=True)
    self._ack(self._sent())
    self.ka.sent_at -= athenad.KEEPALIVE_INTERVAL_S
    self.ka.update(want_sleep=True)
    assert self.ka.handle_response({"jsonrpc": "2.0", "id": self._sent()["id"], "result": {}})
    assert self.ka.unsupported
    assert not self.ka.sleeping
    assert not self._modem_sleep()

  def test_other_responses(self):
    assert not self.ka.handle_response({"jsonrpc": "2.0", "id": "1700000000--log", "result": {"success": 1}})
    assert not self.ka.handle_response({"jsonrpc": "2.0", "id": 0, "result": {"success": 1}})

  def test_stop(self):
    self.ka.update(want_sleep=True)
    self._ack(self._sent())
    self.ka.stop()
    assert not self._modem_sleep()

  def test_jsonrpc_handler_routes_ack(self, mocker):
    for q in ('recv_queue', 'log_recv_queue'):
      mocker.patch.object(athenad, q, type(getattr(athenad, q))())
    mocker.patch.object(athenad, "keepalive", self.ka)
    self.ka.update(want_sleep=True)
    msg = self._sent()

    end_event = threading.Event()
    thread = threading.Thread(target=athenad.jsonrpc_handler, args=(end_event,), daemon=True)
    thread.start()
    try:
      # the old athena's answer: neither a valid response nor answered with an error
      athenad.recv_queue.put_nowait(json.dumps({"jsonrpc": "2.0", "id": msg["id"]}))
      for _ in range(100):
        if self.ka.unsupported:
          break
        threading.Event().wait(0.01)
      assert self.ka.unsupported
      assert athenad.send_queue.empty()
      assert athenad.log_recv_queue.empty()
    finally:
      end_event.set()
      thread.join()

  def _run_ws_manage(self, mocker, modem_state: dict, network_type=log.DeviceState.NetworkType.cell4G) -> None:
    mocker.patch.object(athenad, "keepalive", self.ka)
    mocker.patch.object(athenad.HARDWARE, "get_modem_state", return_value=modem_state)
    mocker.patch.object(athenad.HARDWARE, "get_network_type", return_value=network_type)
    self.params.put_bool("IsOffroad", True, block=True)
    ws = mocker.MagicMock()
    ws.sock = None
    end_event = threading.Event()
    thread = threading.Thread(target=athenad.ws_manage, args=(ws, end_event), daemon=True)
    thread.start()
    threading.Event().wait(0.3)
    end_event.set()
    thread.join()

  def test_ws_manage_sleeps_capable_modem(self, mocker):
    self._run_ws_manage(mocker, {"sleep_capable": True})
    assert self._sent()["params"]["sleeping"]

  def test_ws_manage_other_modems_stay_awake(self, mocker):
    # comma 3X, or AGNOS that can't sleep the modem: athena's pings are cheaper than keepalives
    self._run_ws_manage(mocker, {})
    assert athenad.send_queue.empty()

  def test_ws_manage_wifi_stays_awake(self, mocker):
    # nothing to save, and athena's pings keep its RPCs prompt
    self._run_ws_manage(mocker, {"sleep_capable": True}, log.DeviceState.NetworkType.wifi)
    assert athenad.send_queue.empty()
