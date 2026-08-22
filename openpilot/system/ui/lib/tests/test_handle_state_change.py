import threading
import time
import uuid
from typing import cast
from unittest import TestCase
from unittest.mock import MagicMock, call, mock_open, patch

from openpilot.system.ui.lib import wifi_manager as wifi_manager_module
from openpilot.system.ui.lib.wifi_manager import (
  CONNECTING_STALE_TIMEOUT_SECONDS,
  ConnectStatus,
  MeteredType,
  SCAN_PERIOD_SECONDS,
  SecurityType,
  WifiManager,
  WifiState,
)


def build_wifi_manager() -> WifiManager:
  store = MagicMock()
  store.get_metered.return_value = 0
  store.contains.return_value = False
  dhcp = MagicMock()
  with (
    patch.object(wifi_manager_module, "NetworkStore", return_value=store),
    patch.object(wifi_manager_module, "DhcpClient", return_value=dhcp),
    patch.object(wifi_manager_module, "Params", None),
    patch.object(WifiManager, "_initialize"),
    patch.object(wifi_manager_module.atexit, "register"),
  ):
    manager = WifiManager()

  manager._store = store
  manager._exit = True
  manager._ctrl = MagicMock()
  manager._ipv4_forward = True
  manager._apply_ipv4_forward = MagicMock()
  manager._tethering_ssid = "Hotspot"
  manager._tethering_psk = "hotspot-password"
  manager._last_connecting_at = time.monotonic() - CONNECTING_STALE_TIMEOUT_SECONDS - 1
  manager._update_active_connection_info = MagicMock()
  manager._poll_for_ip = MagicMock()
  manager._wifi_default_route_ready = MagicMock(return_value=True)
  manager._ctrl.request.return_value = "wpa_state=COMPLETED\nssid=TestNet\n"
  return manager
def complete_station_connection(manager: WifiManager, ssid: str):
  manager._ipv4_address = "192.168.1.20"
  manager._complete_station_connection(ssid, manager._user_epoch)




class TestConnectionState(TestCase):
  def setUp(self):
    self.manager = build_wifi_manager()

  def test_connected_persists_after_auth_and_is_idempotent(self):
    activated = MagicMock()
    self.manager.add_callbacks(activated=activated)
    self.manager._set_connecting("TestNet")
    self.manager._set_pending_connection("TestNet", "password123", False, SecurityType.WPA)
    profile_uuid = self.manager._pending_connection.profile_uuid

    with patch.object(wifi_manager_module, "generate_wpa_conf"):
      self.manager._handle_connected("TestNet")
      complete_station_connection(self.manager, "TestNet")
      self.manager._handle_connected("TestNet")

    self.manager.process_callbacks()
    assert self.manager.wifi_state == WifiState("TestNet", ConnectStatus.CONNECTED)
    self.manager._store.save_network.assert_called_once_with(
      "TestNet", psk="password123", hidden=False, security=SecurityType.WPA, profile_uuid=profile_uuid,
    )
    self.manager._dhcp.start.assert_called_once()
    activated.assert_called_once()
    assert call("ENABLE_NETWORK all") in self.manager._ctrl.request.call_args_list

  def test_connected_waits_for_ip_before_activation(self):
    activated = MagicMock()
    self.manager.add_callbacks(activated=activated)
    self.manager._set_connecting("TestNet")
    self.manager._set_pending_connection("TestNet", "password123", False, SecurityType.WPA)
    epoch = self.manager._user_epoch

    with patch.object(wifi_manager_module, "generate_wpa_conf"):
      self.manager._handle_connected("TestNet", expected_epoch=epoch)

    self.manager.process_callbacks()
    assert self.manager.wifi_state == WifiState("TestNet", ConnectStatus.CONNECTING)
    assert self.manager.connected_ssid is None
    self.manager._dhcp.start.assert_called_once()
    activated.assert_not_called()

    self.manager._ipv4_address = "192.168.1.20"
    self.manager._complete_station_connection("TestNet", epoch)
    self.manager.process_callbacks()

    assert self.manager.wifi_state == WifiState("TestNet", ConnectStatus.CONNECTED)
    assert self.manager.connected_ssid == "TestNet"
    activated.assert_called_once()

  def test_connected_applies_active_profile_ipv6_policy(self):
    for method, enabled in (("auto", True), ("ignore", False)):
      with self.subTest(method=method):
        manager = build_wifi_manager()
        manager._set_connecting("TestNet")
        profile_uuid = str(uuid.uuid4())
        manager._store.get_ipv6_method.return_value = method

        manager._handle_connected("TestNet", profile_uuid=profile_uuid)

        manager._store.get_ipv6_method.assert_called_once_with("TestNet", profile_uuid)
        manager._dhcp.set_ipv6_enabled.assert_called_once_with(enabled)
        manager._dhcp.start.assert_called_once()

  def test_connected_retries_after_ipv6_policy_failure(self):
    self.manager._set_connecting("TestNet")
    self.manager._dhcp.set_ipv6_enabled.side_effect = OSError("sysctl failed")

    self.manager._handle_connected("TestNet")

    assert self.manager._associated_ssid is None
    assert self.manager.wifi_state == WifiState("TestNet", ConnectStatus.CONNECTING)
    self.manager._dhcp.start.assert_not_called()

  def test_connected_waits_for_metric_600_default_route(self):
    activated = MagicMock()
    self.manager.add_callbacks(activated=activated)
    self.manager._set_connecting("TestNet")
    epoch = self.manager._user_epoch
    self.manager._associated_ssid = "TestNet"
    self.manager._associated_epoch = epoch
    self.manager._ipv4_address = "192.168.1.20"
    self.manager._wifi_default_route_ready.return_value = False

    self.manager._complete_station_connection("TestNet", epoch)
    self.manager.process_callbacks()

    assert self.manager.wifi_state == WifiState("TestNet", ConnectStatus.CONNECTING)
    activated.assert_not_called()

    self.manager._wifi_default_route_ready.return_value = True
    self.manager._complete_station_connection("TestNet", epoch)
    self.manager.process_callbacks()

    assert self.manager.wifi_state == WifiState("TestNet", ConnectStatus.CONNECTED)
    activated.assert_called_once()

  def test_ip_poll_continues_until_default_route_is_ready(self):
    self.manager._set_connecting("TestNet")
    epoch = self.manager._user_epoch
    self.manager._associated_ssid = "TestNet"
    self.manager._associated_epoch = epoch
    self.manager._ipv4_address = "192.168.1.20"
    self.manager._wifi_default_route_ready.side_effect = [False, True]

    with (
      patch.object(wifi_manager_module.threading, "Thread") as thread,
      patch.object(wifi_manager_module.time, "sleep"),
    ):
      WifiManager._poll_for_ip(self.manager, "TestNet", epoch)
      thread.call_args.kwargs["target"]()

    assert self.manager._update_active_connection_info.call_count == 2
    assert self.manager.wifi_state == WifiState("TestNet", ConnectStatus.CONNECTED)

  def test_wifi_default_route_ready_requires_one_metric_600_gateway(self):
    cases = (
      ("default via 192.168.1.1 dev wlan0 metric 600\n", True),
      ("", False),
      ("default via 192.168.1.1 dev wlan0 metric 0\n", False),
      ("default dev wlan0 metric 600\n", False),
      (
        "default via 192.168.1.1 dev wlan0 metric 600\ndefault via 192.168.1.2 dev wlan0 metric 600\n",
        False,
      ),
    )
    for output, expected in cases:
      with self.subTest(output=output):
        result = MagicMock(returncode=0, stdout=output)
        with patch.object(wifi_manager_module.subprocess, "run", return_value=result) as run:
          assert WifiManager._wifi_default_route_ready(self.manager) is expected
        run.assert_called_once_with(
          ["ip", "-4", "route", "show", "default", "dev", "wlan0"],
          capture_output=True,
          check=False,
          text=True,
          timeout=2,
        )


  def test_connected_transitions_are_serialized(self):
    self.manager._set_connecting("TestNet")
    self.manager._set_pending_connection("TestNet", "password123", False, SecurityType.WPA)
    first_persist_started = threading.Event()
    concurrent_persist_started = threading.Event()
    release_persist = threading.Event()
    active_persists = 0
    active_persists_lock = threading.Lock()

    def persist(_ssid):
      nonlocal active_persists
      with active_persists_lock:
        active_persists += 1
        if active_persists == 1:
          first_persist_started.set()
        else:
          concurrent_persist_started.set()
      assert release_persist.wait(1)
      with active_persists_lock:
        active_persists -= 1

    with patch.object(self.manager, "_persist_pending_connection", side_effect=persist):
      first = threading.Thread(target=self.manager._handle_connected, args=("TestNet",))
      second = threading.Thread(target=self.manager._handle_connected, args=("TestNet",))
      first.start()
      assert first_persist_started.wait(1)
      second.start()
      try:
        assert not concurrent_persist_started.wait(0.1)
      finally:
        release_persist.set()
        first.join(1)
        second.join(1)

    assert not first.is_alive()
    assert not second.is_alive()

  def test_connect_tap_does_not_wait_for_connected_transition(self):
    self.manager._set_connecting("CurrentNet")
    self.manager._set_pending_connection("CurrentNet", "current-password", False, SecurityType.WPA)
    current_epoch = self.manager._user_epoch
    persist_started = threading.Event()
    release_persist = threading.Event()
    connect_returned = threading.Event()

    def save_network(*_args, **_kwargs):
      persist_started.set()
      assert release_persist.wait(1)

    self.manager._store.save_network.side_effect = save_network
    connected = threading.Thread(target=self.manager._handle_connected, args=("CurrentNet",), kwargs={"expected_epoch": current_epoch})
    connected.start()
    assert persist_started.wait(1)

    def connect():
      self.manager.connect_to_network("NextNet", "next-password")
      connect_returned.set()

    thread_class = threading.Thread
    connector = thread_class(target=connect)
    with patch.object(wifi_manager_module.threading, "Thread"):
      connector.start()
      returned_during_persist = connect_returned.wait(0.1)
      release_persist.set()
      connector.join(1)
      connected.join(1)

    assert returned_during_persist
    assert not connector.is_alive()
    assert not connected.is_alive()
    assert self.manager.wifi_state == WifiState("NextNet", ConnectStatus.CONNECTING)
    assert self.manager._pending_connection is not None
    assert self.manager._pending_connection.ssid == "NextNet"
    self.manager._dhcp.start.assert_called_once()

  def test_connect_cleans_dhcp_when_superseding_an_association(self):
    self.manager._wifi_state = WifiState("CurrentNet", ConnectStatus.CONNECTING)
    self.manager._associated_ssid = "CurrentNet"
    self.manager._associated_epoch = self.manager._user_epoch

    with (
      patch.object(wifi_manager_module.threading, "Thread") as thread,
      patch.object(self.manager, "_list_network_ids", return_value=[]),
      patch.object(self.manager, "_add_and_select_network", return_value="1"),
    ):
      self.manager.connect_to_network("NextNet", "next-password")
      thread.call_args.kwargs["target"]()

    self.manager._dhcp.stop.assert_called_once()

  def test_activate_cleans_dhcp_when_superseding_an_association(self):
    self.manager._wifi_state = WifiState("CurrentNet", ConnectStatus.CONNECTING)
    self.manager._associated_ssid = "CurrentNet"
    self.manager._associated_epoch = self.manager._user_epoch

    with (
      patch.object(self.manager, "_list_network_ids", return_value=["1"]),
      patch.object(self.manager, "_select_network_ids"),
    ):
      self.manager.activate_connection("NextNet", block=True)

    self.manager._dhcp.stop.assert_called_once()

  def test_pending_persistence_is_retried_without_restarting_dhcp(self):
    for retry in ("connected", "reconcile"):
      with self.subTest(retry=retry):
        manager = build_wifi_manager()
        manager._set_connecting("TestNet")
        manager._set_pending_connection("TestNet", "password123", False, SecurityType.WPA)
        manager._store.save_network.side_effect = [OSError("read-only"), None]

        with patch.object(wifi_manager_module, "generate_wpa_conf"):
          manager._handle_connected("TestNet")
          if retry == "connected":
            manager._handle_connected("TestNet")
          else:
            manager._last_connecting_at = time.monotonic() - CONNECTING_STALE_TIMEOUT_SECONDS - 1
            manager._reconcile_connecting_state()

        assert manager._store.save_network.call_count == 2
        assert manager._pending_connection is None
        manager._dhcp.start.assert_called_once()

  def test_reconcile_completed_association_does_not_spawn_more_ip_pollers(self):
    self.manager._set_connecting("TestNet")
    self.manager._handle_connected("TestNet")
    self.manager._last_connecting_at = time.monotonic() - CONNECTING_STALE_TIMEOUT_SECONDS - 1

    for _ in range(3):
      self.manager._reconcile_connecting_state()

    self.manager._dhcp.start.assert_called_once()
    self.manager._poll_for_ip.assert_called_once()

  def test_disconnected_event_defers_station_cleanup(self):
    self.manager._wifi_state = WifiState("TestNet", ConnectStatus.CONNECTED)
    self.manager._ipv4_address = "192.168.1.20"
    self.manager._current_network_metered = MeteredType.YES

    self.manager._handle_event("CTRL-EVENT-DISCONNECTED reason=3")

    assert self.manager.wifi_state == WifiState("TestNet", ConnectStatus.CONNECTING)
    assert self.manager.ipv4_address == "192.168.1.20"
    assert self.manager.current_network_metered == MeteredType.YES
    self.manager._dhcp.stop.assert_not_called()

  def test_reconnect_after_disconnected_event_adopts_dhcp(self):
    self.manager._wifi_state = WifiState("TestNet", ConnectStatus.CONNECTED)
    self.manager._ipv4_address = "192.168.1.20"
    self.manager._dhcp.adopt.return_value = True

    self.manager._handle_event("CTRL-EVENT-DISCONNECTED reason=3")
    self.manager._ctrl.request.return_value = "wpa_state=COMPLETED\nssid=TestNet\n"
    self.manager._handle_event("CTRL-EVENT-CONNECTED")

    complete_station_connection(self.manager, "TestNet")
    assert self.manager.wifi_state == WifiState("TestNet", ConnectStatus.CONNECTED)
    assert self.manager.ipv4_address == "192.168.1.20"
    self.manager._dhcp.adopt.assert_called_once()
    self.manager._dhcp.stop.assert_not_called()
    self.manager._dhcp.start.assert_not_called()

  def test_disconnected_event_allows_fallback_to_different_saved_network(self):
    self.manager._wifi_state = WifiState("TestNet", ConnectStatus.CONNECTED)

    self.manager._handle_event("CTRL-EVENT-DISCONNECTED reason=3")
    self.manager._ctrl.request.return_value = "wpa_state=COMPLETED\nssid=NextNet\n"
    self.manager._handle_event("CTRL-EVENT-CONNECTED")

    complete_station_connection(self.manager, "NextNet")
    assert self.manager.wifi_state == WifiState("NextNet", ConnectStatus.CONNECTED)
    self.manager._dhcp.adopt.assert_not_called()
    self.manager._dhcp.clear_ipv6_state.assert_called_once()
    self.manager._dhcp.start.assert_called_once()

  def test_disconnected_event_cleans_station_after_timeout(self):
    self.manager._wifi_state = WifiState("TestNet", ConnectStatus.CONNECTED)
    self.manager._ipv4_address = "192.168.1.20"
    self.manager._current_network_metered = MeteredType.YES

    self.manager._handle_event("CTRL-EVENT-DISCONNECTED reason=3")
    self.manager._last_connecting_at = time.monotonic() - CONNECTING_STALE_TIMEOUT_SECONDS - 1
    self.manager._ctrl.request.return_value = "wpa_state=DISCONNECTED\n"
    self.manager._reconcile_connecting_state()

    assert self.manager.wifi_state == WifiState()
    assert self.manager.ipv4_address == ""
    assert self.manager.current_network_metered == MeteredType.UNKNOWN
    self.manager._dhcp.stop.assert_called_once()
    self.manager._dhcp.clear_ipv6_state.assert_called_once()

  def test_missed_disconnect_clears_association_before_reconnect(self):
    self.manager._set_connecting("TestNet")
    self.manager._handle_connected("TestNet")
    complete_station_connection(self.manager, "TestNet")
    self.manager._dhcp.start.reset_mock()

    self.manager._last_connected_recheck = 0.0
    self.manager._ctrl.request.return_value = "wpa_state=DISCONNECTED\n"
    self.manager._reconcile_connecting_state()

    self.assertIsNone(self.manager._associated_ssid)
    self.assertIsNone(self.manager._associated_epoch)

    self.manager._last_connected_recheck = 0.0
    self.manager._ctrl.request.return_value = "wpa_state=COMPLETED\nssid=TestNet\n"
    self.manager._reconcile_connecting_state()

    self.manager._dhcp.start.assert_called_once()

  def test_disconnected_event_does_not_override_user_connection(self):
    self.manager._set_connecting("NextNet")

    self.manager._handle_event("CTRL-EVENT-DISCONNECTED reason=3")

    assert self.manager.wifi_state == WifiState("NextNet", ConnectStatus.CONNECTING)
    self.manager._dhcp.stop.assert_not_called()

  def test_connected_event_rejects_unconfirmed_or_previous_network(self):
    cases = (
      ("wpa_state=ASSOCIATING\nssid=NextNet\n", "wrong-password"),
      ("wpa_state=COMPLETED\nssid=PreviousNet\n", "password123"),
    )
    for status, password in cases:
      with self.subTest(status=status):
        manager = build_wifi_manager()
        manager._set_connecting("NextNet")
        manager._set_pending_connection("NextNet", password, False, SecurityType.WPA)
        manager._ctrl.request.return_value = status

        manager._handle_event("CTRL-EVENT-CONNECTED")

        assert manager.wifi_state == WifiState("NextNet", ConnectStatus.CONNECTING)
        assert manager._pending_connection is not None
        manager._store.save_network.assert_not_called()
        manager._dhcp.start.assert_not_called()

  def test_connected_event_rechecks_epoch_inside_transition_lock(self):
    self.manager._set_connecting("FirstNet")
    self.manager._ctrl.request.return_value = "wpa_state=COMPLETED\nssid=FirstNet\n"
    lock = threading.Lock()
    waiting_for_lock = threading.Event()
    lock.acquire()

    class SignalingLock:
      def __enter__(self):
        waiting_for_lock.set()
        lock.acquire()

      def __exit__(self, *_):
        lock.release()

    self.manager.__dict__["_radio_lock"] = SignalingLock()
    worker = threading.Thread(target=self.manager._handle_event, args=("CTRL-EVENT-CONNECTED",))
    worker.start()
    assert waiting_for_lock.wait(1)

    self.manager._set_connecting("NextNet")
    lock.release()
    worker.join(1)

    assert not worker.is_alive()
    assert self.manager.wifi_state == WifiState("NextNet", ConnectStatus.CONNECTING)
    self.manager._dhcp.start.assert_not_called()

  def test_activate_enables_every_profile_sharing_ssid(self):
    self.manager._ctrl.request.return_value = "OK"

    with patch.object(self.manager, "_list_network_ids", return_value=["1", "2"]):
      self.manager.activate_connection("Pinned", block=True)

    assert self.manager._ctrl.request.call_args_list == [
      call("DISABLE_NETWORK all"),
      call("ENABLE_NETWORK 1"),
      call("ENABLE_NETWORK 2"),
      call("REASSOCIATE"),
    ]

  def test_activate_restores_every_saved_profile(self):
    profiles = (
      {"psk": "first-password", "hidden": False, "priority": 1, "bssid": "00:11:22:33:44:55", "uuid": "first-uuid", "security": SecurityType.WPA},
      {"psk": "second-password", "hidden": True, "priority": 2, "bssid": "66:77:88:99:aa:bb", "uuid": "second-uuid", "security": SecurityType.WPA},
    )
    self.manager._store.get_profiles.return_value = [("Pinned", profile) for profile in profiles]

    with (
      patch.object(self.manager, "_list_network_ids", return_value=[]),
      patch.object(self.manager, "_add_and_select_network", side_effect=["1", "2"]) as add_and_select_network,
      patch.object(self.manager, "_select_network_ids") as select_network_ids,
    ):
      self.manager.activate_connection("Pinned", block=True)

    assert add_and_select_network.call_args_list == [
      call("Pinned", "first-password", False, 1, bssid="00:11:22:33:44:55", profile_uuid="first-uuid", security=SecurityType.WPA),
      call("Pinned", "second-password", True, 2, bssid="66:77:88:99:aa:bb", profile_uuid="second-uuid", security=SecurityType.WPA),
    ]
    select_network_ids.assert_called_once_with(["1", "2"])

  def test_metered_worker_updates_requested_network_only(self):
    self.manager._wifi_state = WifiState("FirstNet", ConnectStatus.CONNECTED)
    self.manager._current_network_metered = MeteredType.NO

    with patch.object(wifi_manager_module.threading, "Thread") as thread:
      self.manager.set_current_network_metered(MeteredType.YES)

    self.manager._wifi_state = WifiState("NextNet", ConnectStatus.CONNECTED)
    thread.call_args.kwargs["target"]()

    self.manager._store.set_metered.assert_called_once_with("FirstNet", int(MeteredType.YES))
    assert self.manager.current_network_metered == MeteredType.NO

  def test_metered_worker_reports_persistence_failure(self):
    self.manager._wifi_state = WifiState("TestNet", ConnectStatus.CONNECTED)
    self.manager._current_network_metered = MeteredType.NO
    self.manager._store.set_metered.side_effect = OSError("read-only")

    with (
      patch.object(wifi_manager_module.threading, "Thread") as thread,
      patch.object(wifi_manager_module.cloudlog, "exception") as exception,
    ):
      self.manager.set_current_network_metered(MeteredType.YES)
      thread.call_args.kwargs["target"]()

    exception.assert_called_once_with("Failed to update metered state for TestNet")
    assert self.manager.current_network_metered == MeteredType.NO

  def test_active_profile_sets_metered_state(self):
    self.manager._wifi_state = WifiState("Duplicate", ConnectStatus.CONNECTED)
    self.manager._ctrl.request.return_value = "wpa_state=COMPLETED\nssid=Duplicate\nid_str=second-uuid\nip_address=192.168.1.20\n"
    self.manager._store.get_metered.return_value = MeteredType.NO

    WifiManager._update_active_connection_info(self.manager)

    self.manager._store.get_metered.assert_called_once_with("Duplicate", "second-uuid")
    assert self.manager.current_network_metered == MeteredType.NO

  def test_activate_restores_saved_profile_constraints(self):
    cases = (
      ("Preferred", {"psk": "password123", "hidden": False, "priority": 42, "uuid": "preferred-uuid", "security": SecurityType.WPA}, 42, None),
      (
        "Pinned",
        {"psk": "password123", "hidden": False, "bssid": "00:11:22:33:44:55", "uuid": "pinned-uuid", "security": SecurityType.WPA},
        0,
        "00:11:22:33:44:55",
      ),
    )
    for ssid, profile, priority, bssid in cases:
      with self.subTest(ssid=ssid):
        manager = build_wifi_manager()
        manager._store.get_profiles.return_value = [(ssid, profile)]

        with (
          patch.object(manager, "_list_network_ids", return_value=[]),
          patch.object(manager, "_add_and_select_network") as add_and_select_network,
        ):
          manager.activate_connection(ssid, block=True)

        add_and_select_network.assert_called_once_with(
          ssid, "password123", False, priority, bssid=bssid, profile_uuid=profile["uuid"], security=SecurityType.WPA,
        )

  def test_connection_changes_defer_dhcp_cleanup_to_worker(self):
    for action in ("connect", "activate"):
      with self.subTest(action=action):
        manager = build_wifi_manager()
        manager._wifi_state = WifiState("CurrentNet", ConnectStatus.CONNECTED)
        with (
          patch.object(wifi_manager_module.threading, "Thread") as thread,
          patch.object(manager, "_list_network_ids", return_value=["1"] if action == "activate" else []),
          patch.object(manager, "_add_and_select_network"),
          patch.object(manager, "_select_network_ids"),
        ):
          if action == "connect":
            manager.connect_to_network("NextNet", "password123")
          else:
            manager.activate_connection("NextNet")
          manager._dhcp.stop.assert_not_called()
          thread.call_args.kwargs["target"]()

        manager._dhcp.stop.assert_called_once()
        manager._dhcp.clear_ipv6_state.assert_called_once()

  def test_latest_connect_worker_owns_deferred_dhcp_cleanup(self):
    self.manager._wifi_state = WifiState("CurrentNet", ConnectStatus.CONNECTED)
    events = []
    self.manager._dhcp.stop.side_effect = lambda: events.append("cleanup")

    with (
      patch.object(wifi_manager_module.threading, "Thread") as thread,
      patch.object(self.manager, "_list_network_ids", return_value=[]),
      patch.object(self.manager, "_add_and_select_network", side_effect=lambda *_, **__: events.append("select") or "1"),
    ):
      self.manager.connect_to_network("FirstNet", "password123")
      first_worker = thread.call_args.kwargs["target"]
      self.manager.connect_to_network("NextNet", "password123")
      second_worker = thread.call_args.kwargs["target"]

      second_worker()
      first_worker()

    assert events == ["cleanup", "select"]

  def test_superseded_after_select_removes_exact_runtime_network(self):
    first_selected = threading.Event()
    release_first = threading.Event()
    second_committed = threading.Event()
    real_set_pending_network_id = self.manager._set_pending_network_id

    def add_network(ssid, *_, **__):
      if ssid == "FirstNet":
        first_selected.set()
        assert release_first.wait(1)
        return "1"
      return "2"

    def set_pending_network_id(net_id, epoch):
      real_set_pending_network_id(net_id, epoch)
      if net_id == "2":
        second_committed.set()

    self.manager._ctrl.request.return_value = "OK"
    with (
      patch.object(self.manager, "_list_network_ids", return_value=[]),
      patch.object(self.manager, "_add_and_select_network", side_effect=add_network),
      patch.object(self.manager, "_set_pending_network_id", side_effect=set_pending_network_id),
      patch.object(wifi_manager_module, "generate_wpa_conf"),
    ):
      self.manager.connect_to_network("FirstNet", "first-password")
      assert first_selected.wait(1)
      self.manager.connect_to_network("SecondNet", "second-password")
      release_first.set()
      assert second_committed.wait(1)

    assert call("REMOVE_NETWORK 1") in self.manager._ctrl.request.call_args_list
    assert self.manager._pending_connection is not None
    assert self.manager._pending_connection.ssid == "SecondNet"
    assert self.manager._pending_connection.network_id == "2"

  def test_runtime_network_encodes_control_characters_in_ssid(self):
    self.manager._ctrl.request.side_effect = ["0", "OK", "OK", "OK", "OK"]

    self.manager._add_and_select_network("Line\nBreak\r")

    ssid_hex = b"Line\nBreak\r".hex()
    assert call(f"SET_NETWORK 0 ssid {ssid_hex}") in self.manager._ctrl.request.call_args_list

  def test_runtime_network_sets_saved_profile_identifier(self):
    self.manager._ctrl.request.side_effect = ["0", "OK", "OK", "OK", "OK", "OK"]

    self.manager._add_and_select_network("TestNet", profile_uuid="profile-uuid")

    assert call('SET_NETWORK 0 id_str "profile-uuid"') in self.manager._ctrl.request.call_args_list

  def test_runtime_open_network_ignores_stale_psk(self):
    self.manager._ctrl.request.side_effect = ["0", "OK", "OK", "OK", "OK"]

    self.manager._add_and_select_network(
      "TestNet", psk="stale-password", security=SecurityType.OPEN,
    )

    requests = self.manager._ctrl.request.call_args_list
    assert call("SET_NETWORK 0 key_mgmt NONE") in requests
    assert not any(" psk " in item.args[0] for item in requests)

  def test_scan_only_reselects_when_disconnected(self):
    cases = (
      (WifiState("TestNet", ConnectStatus.CONNECTED), "SCAN TYPE=ONLY"),
      (WifiState(), "SCAN"),
    )
    for state, command in cases:
      with self.subTest(state=state):
        manager = build_wifi_manager()
        manager._wifi_state = state

        manager._request_scan()

        manager._ctrl.request.assert_called_once_with(command)

  def test_scan_rejects_conflicting_security_variants(self):
    self.manager._ctrl.request.return_value = "\n".join((
      "bssid / frequency / signal level / flags / ssid",
      "00:11:22:33:44:55\t2437\t-40\t[ESS]\tMixed",
      "66:77:88:99:aa:bb\t2437\t-60\t[WPA2-PSK-CCMP][ESS]\tMixed",
    ))

    self.manager._update_networks()

    assert len(self.manager.networks) == 1
    assert self.manager.networks[0].ssid == "Mixed"
    assert self.manager.networks[0].security_type == SecurityType.UNSUPPORTED

  def test_scan_accepts_psk_bss_among_unsupported_variants(self):
    self.manager._ctrl.request.return_value = "\n".join((
      "bssid / frequency / signal level / flags / ssid",
      "00:11:22:33:44:55\t2437\t-40\t[RSN-SAE-CCMP][ESS]\tMixed",
      "66:77:88:99:aa:bb\t2437\t-60\t[WPA2-PSK-CCMP][ESS]\tMixed",
    ))

    self.manager._update_networks()

    assert len(self.manager.networks) == 1
    assert self.manager.networks[0].ssid == "Mixed"
    assert self.manager.networks[0].security_type == SecurityType.WPA

  def test_wrong_key_removes_runtime_credentials_and_clears_station_state(self):
    need_auth = MagicMock()
    self.manager.add_callbacks(need_auth=need_auth)
    self.manager._set_connecting("TestNet")
    self.manager._set_pending_connection("TestNet", "wrongpass", False, SecurityType.WPA)
    self.manager._set_pending_network_id("0", self.manager._user_epoch)
    self.manager._dhcp_adoption_ssid = "TestNet"
    self.manager._last_wrong_key_dispatch[("OldNet", None)] = 0.0
    self.manager._ctrl.request.return_value = "OK"

    with (
      patch.object(self.manager, "_list_network_ids", return_value=["0"]),
      patch.object(wifi_manager_module.time, "monotonic", return_value=100),
    ):
      self.manager._handle_event('CTRL-EVENT-SSID-TEMP-DISABLED id=0 ssid="TestNet" reason=WRONG_KEY')

    self.manager.process_callbacks()
    assert self.manager._pending_connection is None
    assert self.manager.wifi_state == WifiState()
    assert call("REMOVE_NETWORK 0") in self.manager._ctrl.request.call_args_list
    assert ("OldNet", None) not in self.manager._last_wrong_key_dispatch
    self.manager._dhcp.stop.assert_called_once()
    self.manager._dhcp.clear_ipv6_state.assert_called_once()
    need_auth.assert_called_once_with("TestNet")

  def test_wrong_key_ignores_same_ssid_event_for_other_profile(self):
    need_auth = MagicMock()
    self.manager.add_callbacks(need_auth=need_auth)
    self.manager._set_connecting("TestNet")
    self.manager._set_pending_connection("TestNet", "correct-password", False, SecurityType.WPA)
    self.manager._set_pending_network_id("1", self.manager._user_epoch)

    self.manager._handle_event('CTRL-EVENT-SSID-TEMP-DISABLED id=0 ssid="TestNet" reason=WRONG_KEY')
    self.manager.process_callbacks()

    assert self.manager._pending_connection is not None
    assert self.manager._pending_connection.password == "correct-password"
    assert self.manager.wifi_state == WifiState("TestNet", ConnectStatus.CONNECTING)
    self.manager._ctrl.request.assert_not_called()
    self.manager._dhcp.stop.assert_not_called()
    need_auth.assert_not_called()

  def test_wrong_key_exhausts_same_ssid_profiles_before_auth_failure(self):
    need_auth = MagicMock()
    self.manager.add_callbacks(need_auth=need_auth)
    self.manager._set_connecting("TestNet")

    with (
      patch.object(self.manager, "_list_network_ids", side_effect=[["0", "1"], ["1"]]),
      patch.object(self.manager, "_remove_wpa_network_id") as remove_network,
      patch.object(self.manager, "_select_network_ids") as select_networks,
      patch.object(wifi_manager_module.time, "monotonic", return_value=100),
    ):
      self.manager._handle_event('CTRL-EVENT-SSID-TEMP-DISABLED id=0 ssid="TestNet" reason=WRONG_KEY')
      self.manager.process_callbacks()

      assert self.manager.wifi_state == WifiState("TestNet", ConnectStatus.CONNECTING)
      select_networks.assert_called_once_with(["1"])
      need_auth.assert_not_called()

      self.manager._handle_event('CTRL-EVENT-SSID-TEMP-DISABLED id=1 ssid="TestNet" reason=WRONG_KEY')

    self.manager.process_callbacks()
    assert remove_network.call_args_list == [call("0"), call("1")]
    assert self.manager.wifi_state == WifiState()
    self.manager._dhcp.stop.assert_called_once()
    need_auth.assert_called_once_with("TestNet")

  def test_connect_allocates_profile_before_removing_matching_ids(self):
    requests = []

    def request(command):
      requests.append(command)
      if command == "LIST_NETWORKS":
        return "network id / ssid / bssid / flags\n0\tTestNet\tany\n"
      if command == "ADD_NETWORK":
        return "1"
      return "OK"

    self.manager._ctrl.request.side_effect = request
    real_thread = threading.Thread
    worker_threads = []

    class CapturingThread:
      def __init__(self, *args, **kwargs):
        self.thread = real_thread(*args, **kwargs)
        worker_threads.append(self.thread)

      def start(self):
        self.thread.start()

    with patch.object(wifi_manager_module.threading, "Thread", CapturingThread):
      self.manager.connect_to_network("TestNet", "correct-password")
      worker_threads[0].join()

    assert requests.index("ADD_NETWORK") < requests.index("REMOVE_NETWORK 0")
    assert self.manager._pending_connection is not None
    assert self.manager._pending_connection.network_id == "1"

  def test_connect_rejects_invalid_passphrases(self):
    invalid_passwords = ("short", "x" * 64, "é" * 32)

    for password in invalid_passwords:
      with self.subTest(password=password):
        manager = build_wifi_manager()
        need_auth = MagicMock()
        manager.add_callbacks(need_auth=need_auth)

        with patch.object(wifi_manager_module.threading.Thread, "start"):
          manager.connect_to_network("TestNet", password)

        manager.process_callbacks()
        assert manager.wifi_state == WifiState()
        assert manager._pending_connection is None
        need_auth.assert_called_once_with("TestNet")

  def test_connect_rejects_oversized_hidden_ssid(self):
    ssid = "é" * 17

    with patch.object(wifi_manager_module.threading.Thread, "start") as start:
      self.manager.connect_to_network(ssid, "password123", hidden=True)

    assert self.manager.wifi_state == WifiState()
    assert self.manager._pending_connection is None
    start.assert_not_called()

  def test_network_not_found_clears_connecting_state_after_reconciliation(self):
    disconnected = MagicMock()
    self.manager.add_callbacks(disconnected=disconnected)
    self.manager._set_connecting("MissingNet")
    self.manager._set_pending_connection("MissingNet", "password123", True, SecurityType.WPA)
    self.manager._set_pending_network_id("7", self.manager._user_epoch)
    self.manager._ctrl.request.side_effect = lambda command: "wpa_state=SCANNING\n" if command == "STATUS" else "OK"

    with patch.object(self.manager, "_remove_wpa_network") as remove_wpa_network:
      self.manager._last_connecting_at = time.monotonic() - CONNECTING_STALE_TIMEOUT_SECONDS - 1
      self.manager._handle_event("CTRL-EVENT-NETWORK-NOT-FOUND")
      self.manager._handle_event("CTRL-EVENT-NETWORK-NOT-FOUND")
      assert self.manager.wifi_state == WifiState("MissingNet", ConnectStatus.CONNECTING)
      assert self.manager._pending_connection is not None
      remove_wpa_network.assert_not_called()

      self.manager._reconcile_connecting_state()

    self.manager.process_callbacks()
    assert self.manager.wifi_state == WifiState()
    assert self.manager._pending_connection is None
    remove_wpa_network.assert_not_called()
    assert call("REMOVE_NETWORK 7") in self.manager._ctrl.request.call_args_list
    assert call("RECONFIGURE") in self.manager._ctrl.request.call_args_list
    assert call("ENABLE_NETWORK all") not in self.manager._ctrl.request.call_args_list
    self.manager._dhcp.stop.assert_called_once()
    disconnected.assert_called_once()

  def test_network_not_found_ends_scanning_after_reconciliation_deferred(self):
    self.manager._set_connecting("MissingNet")
    self.manager._set_pending_connection("MissingNet", "password123", True, SecurityType.WPA)
    self.manager._ctrl.request.return_value = "wpa_state=SCANNING\n"
    self.manager._last_connecting_at = time.monotonic() - CONNECTING_STALE_TIMEOUT_SECONDS - 1

    self.manager._reconcile_connecting_state()
    self.manager._handle_event("CTRL-EVENT-NETWORK-NOT-FOUND")
    self.manager._handle_event("CTRL-EVENT-NETWORK-NOT-FOUND")
    self.manager._reconcile_connecting_state()

    assert self.manager.wifi_state == WifiState()
    assert self.manager._pending_connection is None

  def test_delayed_network_not_found_does_not_bind_to_fresh_attempt(self):
    self.manager._set_connecting("PreviousNet")
    self.manager._last_connecting_at = time.monotonic() - CONNECTING_STALE_TIMEOUT_SECONDS - 1
    self.manager._set_connecting("HiddenNet")
    self.manager._set_pending_connection("HiddenNet", "password123", True, SecurityType.WPA)
    self.manager._ctrl.request.return_value = "wpa_state=SCANNING\n"

    self.manager._last_connecting_at = time.monotonic() - CONNECTING_STALE_TIMEOUT_SECONDS - 1
    self.manager._handle_event("CTRL-EVENT-NETWORK-NOT-FOUND")
    self.manager._reconcile_connecting_state()

    assert self.manager.wifi_state == WifiState("HiddenNet", ConnectStatus.CONNECTING)
    assert self.manager._pending_connection is not None
    self.manager._dhcp.stop.assert_not_called()

  def test_reconcile_keeps_saved_runtime_network_after_transient_failure(self):
    self.manager._store.contains.return_value = True
    self.manager._set_connecting("SavedNet")
    self.manager._last_connecting_at = time.monotonic() - CONNECTING_STALE_TIMEOUT_SECONDS - 1
    self.manager._ctrl.request.side_effect = lambda command: "wpa_state=DISCONNECTED\n" if command == "STATUS" else "OK"

    with patch.object(self.manager, "_remove_wpa_network") as remove_wpa_network:
      self.manager._reconcile_connecting_state()

    remove_wpa_network.assert_not_called()
    assert call("RECONFIGURE") in self.manager._ctrl.request.call_args_list
    assert call("ENABLE_NETWORK all") not in self.manager._ctrl.request.call_args_list
    assert self.manager.wifi_state == WifiState()

  def test_reconcile_rechecks_epoch_before_cancelling_stale_connection(self):
    self.manager._set_connecting("PreviousNet")
    self.manager._set_pending_connection("PreviousNet", "password123", False, SecurityType.WPA)
    self.manager._last_connecting_at = time.monotonic() - CONNECTING_STALE_TIMEOUT_SECONDS - 1
    self.manager._ctrl.request.return_value = "wpa_state=DISCONNECTED\n"
    restore_started = threading.Event()
    release_restore = threading.Event()

    def restore(*_):
      restore_started.set()
      assert release_restore.wait(1)

    self.manager._restore_station_runtime = MagicMock(side_effect=restore)
    worker = threading.Thread(target=self.manager._reconcile_connecting_state)
    worker.start()
    assert restore_started.wait(1)

    with patch.object(wifi_manager_module.threading.Thread, "start"):
      self.manager.connect_to_network("NextNet", "next-password")
    release_restore.set()
    worker.join(1)

    assert not worker.is_alive()
    assert self.manager.wifi_state == WifiState("NextNet", ConnectStatus.CONNECTING)
    assert self.manager._pending_connection is not None
    assert self.manager._pending_connection.ssid == "NextNet"
    self.manager._dhcp.stop.assert_not_called()

  def test_reconcile_does_not_timeout_connection_associated_after_status(self):
    self.manager._set_connecting("TestNet")
    self.manager._set_pending_connection("TestNet", "password123", False, SecurityType.WPA)
    self.manager._last_connecting_at = time.monotonic() - CONNECTING_STALE_TIMEOUT_SECONDS - 1
    epoch = self.manager._user_epoch
    status_started = threading.Event()
    release_status = threading.Event()

    def request(command):
      if command == "STATUS":
        status_started.set()
        assert release_status.wait(1)
        return "wpa_state=DISCONNECTED\n"
      return "OK"

    self.manager._request = MagicMock(side_effect=request)
    self.manager._restore_station_runtime = MagicMock()
    worker = threading.Thread(target=self.manager._reconcile_connecting_state)
    worker.start()
    assert status_started.wait(1)

    with patch.object(wifi_manager_module, "generate_wpa_conf"):
      self.manager._handle_connected("TestNet", expected_epoch=epoch)
    release_status.set()
    worker.join(1)

    assert not worker.is_alive()
    assert self.manager._associated_ssid == "TestNet"
    assert self.manager._associated_epoch == epoch
    assert self.manager.wifi_state == WifiState("TestNet", ConnectStatus.CONNECTING)
    self.manager._dhcp.start.assert_called_once()
    self.manager._dhcp.stop.assert_not_called()
    self.manager._restore_station_runtime.assert_not_called()

  def test_reconcile_times_out_stalled_handshake(self):
    for wpa_state in ("AUTHENTICATING", "ASSOCIATING", "ASSOCIATED", "4WAY_HANDSHAKE", "GROUP_HANDSHAKE"):
      with self.subTest(wpa_state=wpa_state):
        manager = build_wifi_manager()
        manager._store.contains.return_value = True
        manager._set_connecting("StalledNet")
        manager._set_pending_connection("StalledNet", "password123", False, SecurityType.WPA)
        manager._set_pending_network_id("7", manager._user_epoch)
        manager._last_connecting_at = time.monotonic() - CONNECTING_STALE_TIMEOUT_SECONDS - 1
        def request(command, state=wpa_state):
          return f"wpa_state={state}\nssid=StalledNet\n" if command == "STATUS" else "OK"
        manager._ctrl.request.side_effect = request

        manager._reconcile_connecting_state()

        assert manager.wifi_state == WifiState()
        assert manager._pending_connection is None
        assert call("REMOVE_NETWORK 7") in manager._ctrl.request.call_args_list
        assert call("RECONFIGURE") in manager._ctrl.request.call_args_list
        assert call("ENABLE_NETWORK all") not in manager._ctrl.request.call_args_list
        manager._dhcp.stop.assert_called_once()

  def test_reconcile_does_not_report_generic_disconnect_as_auth_failure(self):
    need_auth = MagicMock()
    self.manager.add_callbacks(need_auth=need_auth)
    self.manager._store.contains.return_value = True
    self.manager._networks = [wifi_manager_module.Network("SavedNet", 100, SecurityType.WPA, False)]
    self.manager._set_connecting("SavedNet")
    self.manager._last_connecting_at = time.monotonic() - CONNECTING_STALE_TIMEOUT_SECONDS - 1
    self.manager._ctrl.request.return_value = "wpa_state=DISCONNECTED\n"

    self.manager._reconcile_connecting_state()
    self.manager.process_callbacks()

    assert self.manager.wifi_state == WifiState()
    need_auth.assert_not_called()

  def test_reconcile_clears_ipv6_state_before_adopting_another_network(self):
    previous_state = WifiState("PreviousNet", ConnectStatus.CONNECTED)
    self.manager._wifi_state = previous_state
    self.manager._ctrl.request.return_value = "wpa_state=COMPLETED\nmode=station\nssid=TestNet\n"
    self.manager._dhcp.clear_ipv6_state.side_effect = lambda: self.assertEqual(self.manager.wifi_state, previous_state)

    self.manager._reconcile_connecting_state()

    complete_station_connection(self.manager, "TestNet")
    assert self.manager.wifi_state == WifiState("TestNet", ConnectStatus.CONNECTED)
    self.manager._dhcp.clear_ipv6_state.assert_called_once()
    self.manager._dhcp.start.assert_called_once()

  def test_stale_network_not_found_does_not_clear_fresh_connection(self):
    self.manager._set_connecting("TestNet")
    self.manager._set_pending_connection("TestNet", "password123", False, SecurityType.WPA)
    profile_uuid = self.manager._pending_connection.profile_uuid

    with (
      patch.object(self.manager, "_remove_wpa_network") as remove_wpa_network,
      patch.object(wifi_manager_module, "generate_wpa_conf"),
    ):
      self.manager._handle_event("CTRL-EVENT-NETWORK-NOT-FOUND")
      self.manager._handle_connected("TestNet")

    complete_station_connection(self.manager, "TestNet")
    assert self.manager.wifi_state == WifiState("TestNet", ConnectStatus.CONNECTED)
    self.manager._store.save_network.assert_called_once_with(
      "TestNet", psk="password123", hidden=False, security=SecurityType.WPA, profile_uuid=profile_uuid,
    )
    remove_wpa_network.assert_not_called()

  def test_forget_cancels_in_flight_connection(self):
    runtime_networks = set()
    connect_started = threading.Event()
    release_connect = threading.Event()
    connect_added = threading.Event()
    forget_removed = threading.Event()

    def remove_network(ssid):
      runtime_networks.discard(ssid)
      if connect_added.is_set():
        forget_removed.set()

    def add_network(ssid, *_, **__):
      connect_started.set()
      assert release_connect.wait(1)
      runtime_networks.add(ssid)
      connect_added.set()
      return "1"

    self.manager._store.contains.return_value = True
    self.manager._store.remove.return_value = True

    with (
      patch.object(self.manager, "_list_network_ids", return_value=[]),
      patch.object(self.manager, "_remove_wpa_network", side_effect=remove_network),
      patch.object(self.manager, "_add_and_select_network", side_effect=add_network),
      patch.object(wifi_manager_module, "generate_wpa_conf"),
    ):
      self.manager.connect_to_network("TestNet", "password123")
      assert connect_started.wait(1)

      self.manager.forget_connection("TestNet")
      assert self.manager.wifi_state == WifiState("TestNet", ConnectStatus.CONNECTING)

      release_connect.set()
      assert connect_added.wait(1)
      assert forget_removed.wait(1)

    assert runtime_networks == set()
    assert self.manager.wifi_state == WifiState()

  def test_forget_does_not_disconnect_fresh_connection(self):
    forget_removing = threading.Event()
    release_forget = threading.Event()
    new_network_selected = threading.Event()
    forget_finished = threading.Event()

    def remove_saved_network(ssid):
      assert ssid == "TestNet"
      forget_removing.set()
      assert release_forget.wait(1)
      return True

    def select_network(*_, **__):
      new_network_selected.set()
      return "1"

    self.manager._wifi_state = WifiState("TestNet", ConnectStatus.CONNECTED)
    self.manager._store.contains.return_value = True
    self.manager._store.remove.side_effect = remove_saved_network

    with (
      patch.object(self.manager, "_list_network_ids", return_value=[]),
      patch.object(self.manager, "_remove_wpa_network"),
      patch.object(self.manager, "_add_and_select_network", side_effect=select_network),
      patch.object(self.manager, "_enqueue_callbacks", side_effect=lambda *_: forget_finished.set()),
      patch.object(wifi_manager_module, "generate_wpa_conf"),
    ):
      self.manager.forget_connection("TestNet")
      assert forget_removing.wait(1)

      self.manager.connect_to_network("NextNet", "password123")
      assert not new_network_selected.wait(0.1)

      release_forget.set()
      assert forget_finished.wait(1)
      assert new_network_selected.wait(1)

    assert self.manager.wifi_state == WifiState("NextNet", ConnectStatus.CONNECTING)
    assert call("ENABLE_NETWORK all") not in self.manager._ctrl.request.call_args_list
    assert call("DISCONNECT") not in self.manager._ctrl.request.call_args_list
    assert call("REASSOCIATE") not in self.manager._ctrl.request.call_args_list

  def test_forget_reports_persistent_success_when_runtime_removal_fails(self):
    forgotten = MagicMock()
    forget_failed = MagicMock()
    self.manager.add_callbacks(forgotten=forgotten, forget_failed=forget_failed)
    self.manager._store.contains.return_value = True
    self.manager._store.remove.return_value = True

    def request(command):
      if command == "LIST_NETWORKS":
        return "network id / ssid / bssid / flags\n0\tSavedNet\tany\t\n"
      if command == "REMOVE_NETWORK 0":
        return "FAIL\n"
      return "OK\n"

    self.manager._ctrl.request.side_effect = request
    with patch.object(wifi_manager_module, "generate_wpa_conf"):
      self.manager.forget_connection("SavedNet", block=True)

    self.manager.process_callbacks()
    forgotten.assert_called_once_with("SavedNet")
    forget_failed.assert_not_called()

  def test_forget_removes_runtime_when_config_generation_fails(self):
    forgotten = MagicMock()
    self.manager.add_callbacks(forgotten=forgotten)
    self.manager._store.contains.return_value = True
    self.manager._store.remove.return_value = True

    with (
      patch.object(wifi_manager_module, "generate_wpa_conf", side_effect=OSError("read-only")),
      patch.object(self.manager, "_remove_wpa_network") as remove_wpa_network,
    ):
      self.manager.forget_connection("SavedNet", block=True)

    self.manager.process_callbacks()
    remove_wpa_network.assert_called_once_with("SavedNet")
    forgotten.assert_called_once_with("SavedNet")

  def test_forget_allows_fallback_connection_after_disconnect_event(self):
    self.manager._wifi_state = WifiState("TestNet", ConnectStatus.CONNECTED)
    self.manager._store.contains.return_value = True
    self.manager._store.remove.return_value = True

    def request(command):
      if command == "DISCONNECT":
        self.manager._handle_event("CTRL-EVENT-DISCONNECTED reason=3")
      return "OK"

    self.manager._ctrl.request.side_effect = request
    with (
      patch.object(wifi_manager_module, "generate_wpa_conf"),
      patch.object(self.manager, "_remove_wpa_network"),
    ):
      self.manager.forget_connection("TestNet", block=True)

    self.manager._ctrl.request.side_effect = None
    self.manager._ctrl.request.return_value = "wpa_state=COMPLETED\nssid=NextNet\n"
    self.manager._handle_event("CTRL-EVENT-CONNECTED")

    complete_station_connection(self.manager, "NextNet")
    assert self.manager.wifi_state == WifiState("NextNet", ConnectStatus.CONNECTED)

  def test_forget_during_reconnect_clears_retained_station_state(self):
    self.manager._store.contains.return_value = True
    self.manager._store.remove.return_value = True
    self.manager._set_connecting("TestNet")
    self.manager._handle_connected("TestNet")
    complete_station_connection(self.manager, "TestNet")
    self.manager._handle_event("CTRL-EVENT-DISCONNECTED reason=3")
    self.manager._dhcp.stop.reset_mock()
    self.manager._dhcp.clear_ipv6_state.reset_mock()

    with (
      patch.object(wifi_manager_module, "generate_wpa_conf"),
      patch.object(self.manager, "_remove_wpa_network"),
    ):
      self.manager.forget_connection("TestNet", block=True)

    assert self.manager.wifi_state == WifiState()
    assert self.manager.ipv4_address == ""
    self.manager._dhcp.stop.assert_called_once()
    self.manager._dhcp.clear_ipv6_state.assert_called_once()

  def test_forget_during_reconnect_cleans_before_fresh_connection(self):
    real_thread = threading.Thread
    worker_threads = []
    forget_removing = threading.Event()
    release_forget = threading.Event()

    class CapturingThread:
      def __init__(self, *args, **kwargs):
        self.thread = real_thread(*args, **kwargs)
        worker_threads.append(self.thread)

      def start(self):
        self.thread.start()

    def remove_saved_network(ssid):
      assert ssid == "TestNet"
      forget_removing.set()
      assert release_forget.wait(1)
      return True

    def select_network(*_, **__):
      self.manager._dhcp.stop.assert_called_once()
      self.manager._dhcp.clear_ipv6_state.assert_called_once()
      return "2"

    self.manager._store.contains.return_value = True
    self.manager._store.remove.side_effect = remove_saved_network
    self.manager._set_connecting("TestNet")
    self.manager._handle_connected("TestNet")
    complete_station_connection(self.manager, "TestNet")
    self.manager._handle_event("CTRL-EVENT-DISCONNECTED reason=3")
    self.manager._dhcp.stop.reset_mock()
    self.manager._dhcp.clear_ipv6_state.reset_mock()
    self.manager._ctrl.request.reset_mock()

    with (
      patch.object(wifi_manager_module.threading, "Thread", CapturingThread),
      patch.object(self.manager, "_list_network_ids", return_value=[]),
      patch.object(self.manager, "_remove_wpa_network"),
      patch.object(self.manager, "_add_and_select_network", side_effect=select_network),
      patch.object(wifi_manager_module, "generate_wpa_conf"),
    ):
      self.manager.forget_connection("TestNet")
      assert forget_removing.wait(1)
      self.manager.connect_to_network("NextNet", "password123")
      release_forget.set()
      for worker_thread in worker_threads:
        worker_thread.join(1)
        assert not worker_thread.is_alive()

    assert self.manager.wifi_state == WifiState("NextNet", ConnectStatus.CONNECTING)
    self.manager._dhcp.stop.assert_called_once()
    self.manager._dhcp.clear_ipv6_state.assert_called_once()
    assert call("DISCONNECT") not in self.manager._ctrl.request.call_args_list
    assert call("REASSOCIATE") not in self.manager._ctrl.request.call_args_list

  def test_forget_failure_releases_caller_without_reporting_success(self):
    forgotten = MagicMock()
    forget_failed = MagicMock()
    self.manager.add_callbacks(forgotten=forgotten, forget_failed=forget_failed)
    self.manager._store.contains.return_value = True
    self.manager._store.remove.return_value = False

    self.manager.forget_connection("SavedNet", block=True)
    self.manager.process_callbacks()

    forgotten.assert_not_called()
    forget_failed.assert_called_once_with("SavedNet")

  def test_forget_failure_preserves_connected_station_state(self):
    self.manager._set_connecting("TestNet")
    self.manager._handle_connected("TestNet")
    complete_station_connection(self.manager, "TestNet")
    epoch = self.manager._user_epoch
    operation = self.manager._station_operation
    self.manager._store.contains.return_value = True
    self.manager._store.remove.return_value = False
    self.manager._dhcp.stop.reset_mock()
    self.manager._dhcp.clear_ipv6_state.reset_mock()

    self.manager.forget_connection("TestNet", block=True)

    assert self.manager.wifi_state == WifiState("TestNet", ConnectStatus.CONNECTED)
    assert self.manager._user_epoch == epoch
    assert self.manager._associated_ssid == "TestNet"
    assert self.manager._associated_epoch == epoch
    assert self.manager._station_operation is operation
    assert not self.manager._station_cleanup_pending
    self.manager._dhcp.stop.assert_not_called()
    self.manager._dhcp.clear_ipv6_state.assert_not_called()

  def test_forget_failure_preserves_retained_reconnect_state(self):
    self.manager._set_connecting("TestNet")
    self.manager._handle_connected("TestNet")
    complete_station_connection(self.manager, "TestNet")
    self.manager._handle_event("CTRL-EVENT-DISCONNECTED reason=3")
    epoch = self.manager._user_epoch
    operation = self.manager._station_operation
    self.manager._store.contains.return_value = True
    self.manager._store.remove.return_value = False
    self.manager._dhcp.stop.reset_mock()
    self.manager._dhcp.clear_ipv6_state.reset_mock()

    self.manager.forget_connection("TestNet", block=True)

    assert self.manager.wifi_state == WifiState("TestNet", ConnectStatus.CONNECTING)
    assert self.manager._user_epoch == epoch
    assert self.manager._associated_ssid == "TestNet"
    assert self.manager._associated_epoch == epoch
    assert self.manager._dhcp_adoption_ssid == "TestNet"
    assert self.manager._station_operation is operation
    assert not self.manager._station_cleanup_pending
    self.manager._dhcp.stop.assert_not_called()
    self.manager._dhcp.clear_ipv6_state.assert_not_called()

  def test_failed_connect_worker_does_not_reset_fresh_selection(self):
    real_thread = threading.Thread
    worker_threads = []

    class CapturingThread:
      def __init__(self, *args, **kwargs):
        self.thread = real_thread(*args, **kwargs)
        worker_threads.append(self.thread)

      def start(self):
        self.thread.start()

    def fail_after_fresh_selection(*_, **__):
      self.manager._set_connecting("NextNet")
      self.manager._set_pending_connection("NextNet", "new-password", False, SecurityType.WPA)
      raise OSError("stale request failed")

    with (
      patch.object(wifi_manager_module.threading, "Thread", CapturingThread),
      patch.object(self.manager, "_list_network_ids", return_value=[]),
      patch.object(self.manager, "_remove_wpa_network"),
      patch.object(self.manager, "_add_and_select_network", side_effect=fail_after_fresh_selection),
    ):
      self.manager.connect_to_network("OldNet", "old-password")
      worker_threads[0].join()

    assert self.manager.wifi_state == WifiState("NextNet", ConnectStatus.CONNECTING)
    assert self.manager._pending_connection is not None
    assert self.manager._pending_connection.ssid == "NextNet"

  def test_failed_activate_worker_does_not_reset_fresh_selection(self):
    def fail_after_fresh_selection(*_):
      self.manager._set_connecting("NextNet")
      raise OSError("stale request failed")

    with patch.object(self.manager, "_list_network_ids", side_effect=fail_after_fresh_selection):
      self.manager.activate_connection("OldNet", block=True)

    assert self.manager.wifi_state == WifiState("NextNet", ConnectStatus.CONNECTING)

  def test_failed_connect_restores_previous_selection_exactly(self):
    class ImmediateThread:
      def __init__(self, target, **_):
        self._target = target

      def start(self):
        self._target()

    def request(command):
      if command == "LIST_NETWORKS":
        return "network id / ssid / bssid / flags\n1\tTestNet\tany\t\n"
      if command == "ADD_NETWORK":
        return "2"
      if command == "REMOVE_NETWORK 1":
        return "FAIL"
      return "OK"

    self.manager._ctrl.request.side_effect = request
    with patch.object(wifi_manager_module.threading, "Thread", ImmediateThread):
      self.manager.connect_to_network("TestNet", "password123")

    requests = self.manager._ctrl.request.call_args_list
    assert call("REMOVE_NETWORK 2") in requests
    assert call("ENABLE_NETWORK 1") in requests
    assert call("REASSOCIATE") in requests
    assert call("ENABLE_NETWORK all") not in requests
    assert self.manager.wifi_state == WifiState()

  def test_failed_replacement_removes_selected_network_before_restoring_profiles(self):
    class ImmediateThread:
      def __init__(self, target, **_):
        self._target = target

      def start(self):
        self._target()

    def request(command):
      if command == "LIST_NETWORKS":
        return "network id / ssid / bssid / flags\n1\tTestNet\tany\t\n2\tTestNet\tany\t\n"
      if command == "ADD_NETWORK":
        return "3"
      if command == "REMOVE_NETWORK 1":
        return "FAIL"
      return "OK"

    self.manager._ctrl.request.side_effect = request
    with patch.object(wifi_manager_module.threading, "Thread", ImmediateThread):
      self.manager.connect_to_network("TestNet", "password123")

    requests = self.manager._ctrl.request.call_args_list
    assert call("REMOVE_NETWORK 3") in requests
    assert call("DISABLE_NETWORK all") in requests
    assert call("ENABLE_NETWORK 1") in requests
    assert call("ENABLE_NETWORK 2") in requests
    assert call("REASSOCIATE") in requests
    assert call("ENABLE_NETWORK all") not in requests

  def test_failed_second_old_id_removal_reconfigures_durable_profiles(self):
    class ImmediateThread:
      def __init__(self, target, **_):
        self._target = target

      def start(self):
        self._target()

    def request(command):
      if command == "LIST_NETWORKS":
        return "network id / ssid / bssid / flags\n1\tTestNet\tany\t\n2\tTestNet\tany\t\n"
      if command == "ADD_NETWORK":
        return "3"
      if command == "REMOVE_NETWORK 2":
        return "FAIL"
      return "OK"

    self.manager._ctrl.request.side_effect = request
    with patch.object(wifi_manager_module.threading, "Thread", ImmediateThread):
      self.manager.connect_to_network("TestNet", "password123")

    requests = self.manager._ctrl.request.call_args_list
    assert call("REMOVE_NETWORK 1") in requests
    assert call("REMOVE_NETWORK 3") in requests
    assert call("RECONFIGURE") in requests
    assert call("ENABLE_NETWORK all") not in requests

  def test_saved_replacement_timeout_removes_exact_pending_network(self):
    self.manager._store.contains.return_value = True
    self.manager._set_connecting("TestNet")
    self.manager._set_pending_connection("TestNet", "new-password", False, SecurityType.WPA)
    epoch = self.manager._user_epoch
    self.manager._set_pending_network_id("7", epoch)
    self.manager._ctrl.request.side_effect = lambda command: "wpa_state=DISCONNECTED\n" if command == "STATUS" else "OK"
    self.manager._last_connecting_at = time.monotonic() - CONNECTING_STALE_TIMEOUT_SECONDS - 1

    self.manager._reconcile_connecting_state()

    assert call("REMOVE_NETWORK 7") in self.manager._ctrl.request.call_args_list
    assert call("RECONFIGURE") in self.manager._ctrl.request.call_args_list
    assert call("ENABLE_NETWORK all") not in self.manager._ctrl.request.call_args_list


  def test_failed_activate_reconfigures_saved_networks(self):
    self.manager._ctrl.request.side_effect = lambda command: "FAIL" if command == "REASSOCIATE" else "OK"
    with patch.object(self.manager, "_list_network_ids", return_value=["1"]):
      self.manager.activate_connection("TestNet", block=True)

    assert call("RECONFIGURE") in self.manager._ctrl.request.call_args_list
    assert call("ENABLE_NETWORK all") not in self.manager._ctrl.request.call_args_list
    assert self.manager.wifi_state == WifiState()

  def test_request_error_invalidates_control_socket(self):
    self.manager._ctrl.request.side_effect = OSError("socket closed")
    epoch = self.manager._monitor_epoch

    with self.assertRaises(OSError):
      self.manager._request("SCAN")

    assert self.manager._ctrl is None
    assert self.manager._monitor_epoch == epoch + 1


class TestStartupAdoption(TestCase):
  def setUp(self):
    self.manager = build_wifi_manager()

  def test_station_dhcp_adoption(self):
    cases = (
      ("connected", "TestNet", True, True, False, False),
      ("missing-client", "TestNet", False, True, True, False),
      ("reconnecting", "TestNet", True, True, False, False),
      ("different-network", "PreviousNet", False, False, True, True),
    )
    for state, adoption_ssid, adoption_result, expect_adopt, expect_start, expect_clear_ipv6 in cases:
      with self.subTest(state=state):
        manager = build_wifi_manager()
        manager._dhcp_adoption_ssid = adoption_ssid
        manager._dhcp.adopt.return_value = adoption_result
        if state in ("reconnecting", "different-network"):
          manager._ctrl.request.side_effect = (
            "wpa_state=ASSOCIATING\nmode=station\nssid=TestNet\n",
            "wpa_state=COMPLETED\nmode=station\nssid=TestNet\n",
            "OK",
          )
        else:
          manager._ctrl.request.return_value = "wpa_state=COMPLETED\nmode=station\nssid=TestNet\n"

        manager._init_wifi_state()
        if state in ("reconnecting", "different-network"):
          manager._handle_event("CTRL-EVENT-CONNECTED")

        complete_station_connection(manager, "TestNet")
        assert manager.wifi_state == WifiState("TestNet", ConnectStatus.CONNECTED)
        assert bool(manager._dhcp.adopt.call_count) == expect_adopt
        assert bool(manager._dhcp.start.call_count) == expect_start
        assert bool(manager._dhcp.clear_ipv6_state.call_count) == expect_clear_ipv6

  def test_mid_association_adoption_starts_fresh_timeout(self):
    self.manager._dhcp_adoption_ssid = "TestNet"
    self.manager._last_connecting_at = 0.0
    self.manager._ctrl.request.return_value = "wpa_state=ASSOCIATING\nmode=station\nssid=TestNet\n"

    with patch.object(wifi_manager_module.time, "monotonic", return_value=100.0):
      self.manager._init_wifi_state()

    assert self.manager.wifi_state == WifiState("TestNet", ConnectStatus.CONNECTING)
    assert self.manager._last_connecting_at == 100.0
    assert self.manager._dhcp_adoption_ssid == "TestNet"

    self.manager._ctrl.request.reset_mock()
    with patch.object(wifi_manager_module.time, "monotonic", return_value=104.0):
      self.manager._reconcile_connecting_state()

    self.manager._ctrl.request.assert_not_called()
    assert self.manager.wifi_state == WifiState("TestNet", ConnectStatus.CONNECTING)

  def test_disconnected_startup_cleans_station_state(self):
    self.manager._dhcp_adoption_ssid = "TestNet"
    self.manager._ctrl.request.return_value = "wpa_state=DISCONNECTED\nmode=station\n"

    self.manager._init_wifi_state()

    assert self.manager.wifi_state == WifiState()
    assert self.manager._dhcp_adoption_ssid is None
    self.manager._dhcp.stop.assert_called_once()
    self.manager._dhcp.clear_ipv6_state.assert_called_once()

  def test_hotspot_adopts_with_dhcp_and_nat(self):
    self.manager._ctrl.request.return_value = "wpa_state=COMPLETED\nmode=AP\nssid=Hotspot\n"

    with (
      patch.object(wifi_manager_module, "tethering_dnsmasq_running", return_value=True),
      patch.object(wifi_manager_module, "_tethering_firewall_ready", return_value=True),
      patch("builtins.open", mock_open(read_data='  psk="hotspot-password"\n')),
    ):
      self.manager._init_wifi_state()

    assert self.manager.is_tethering_active()
    assert self.manager.wifi_state == WifiState("Hotspot", ConnectStatus.CONNECTED)
    assert self.manager.ipv4_address == "192.168.43.1"
    self.manager._dhcp.start.assert_not_called()

  def test_hotspot_adoption_notifies_callback_registered_after_startup(self):
    with (
      patch.object(wifi_manager_module, "tethering_dnsmasq_running", return_value=True),
      patch.object(wifi_manager_module, "_tethering_firewall_ready", return_value=True),
      patch("builtins.open", mock_open(read_data='  psk="hotspot-password"\n')),
    ):
      assert self.manager._adopt_ap_state("Hotspot")

    activated = MagicMock()
    self.manager.add_callbacks(activated=activated)
    self.manager.process_callbacks()

    activated.assert_called_once()

  def test_hotspot_adoption_preserves_forwarding_until_policy_is_known(self):
    self.manager._ipv4_forward = None
    self.manager._apply_ipv4_forward.reset_mock()

    with (
      patch.object(wifi_manager_module, "tethering_dnsmasq_running", return_value=True),
      patch.object(wifi_manager_module, "_tethering_firewall_ready", return_value=True),
      patch("builtins.open", mock_open(read_data='  psk="hotspot-password"\n')),
    ):
      assert self.manager._adopt_ap_state("Hotspot")

    self.manager._apply_ipv4_forward.assert_not_called()

    self.manager.set_ipv4_forward(True)

    self.manager._apply_ipv4_forward.assert_called_once_with(True)

  def test_hotspot_password_mismatch_rebuilds_ap(self):
    with (
      patch.object(wifi_manager_module, "tethering_dnsmasq_running", return_value=True),
      patch.object(wifi_manager_module, "_tethering_firewall_ready", return_value=True),
      patch("builtins.open", mock_open(read_data='  psk="old-password"\n')),
      patch.object(self.manager, "_start_tethering") as start_tethering,
    ):
      assert self.manager._adopt_ap_state("Hotspot")

    start_tethering.assert_called_once()
    assert self.manager.is_tethering_active()

  def test_incomplete_hotspot_is_removed(self):
    for dnsmasq_running, nat_ready in ((False, True), (True, False)):
      with self.subTest(dnsmasq_running=dnsmasq_running, nat_ready=nat_ready):
        manager = build_wifi_manager()
        manager._ctrl.request.return_value = "wpa_state=COMPLETED\nmode=AP\nssid=Hotspot\n"
        with (
          patch.object(wifi_manager_module, "tethering_dnsmasq_running", return_value=dnsmasq_running),
          patch.object(wifi_manager_module, "_tethering_firewall_ready", return_value=nat_ready),
          patch.object(manager, "_stop_tethering") as stop_tethering,
        ):
          manager._init_wifi_state()

        assert manager.wifi_state == WifiState()
        assert not manager.is_tethering_active()
        stop_tethering.assert_called_once()

  def test_reconcile_adopts_missed_connection(self):
    self.manager._ctrl.request.return_value = "wpa_state=COMPLETED\nmode=station\nssid=TestNet\n"

    self.manager._reconcile_connecting_state()

    complete_station_connection(self.manager, "TestNet")
    assert self.manager.wifi_state == WifiState("TestNet", ConnectStatus.CONNECTED)
    self.manager._dhcp.start.assert_called_once()

  def test_reconcile_stops_hotspot_without_responsive_control_socket(self):
    for ctrl in ("unresponsive", None):
      with self.subTest(ctrl=ctrl):
        manager = build_wifi_manager()
        manager._tethering_active = True
        manager._wifi_state = WifiState("Hotspot", ConnectStatus.CONNECTED)
        if ctrl is None:
          manager._ctrl = None
        else:
          manager._ctrl.request.side_effect = OSError("socket closed")

        with patch.object(manager, "_stop_tethering") as stop_tethering:
          manager._reconcile_connecting_state()

        stop_tethering.assert_called_once()

  def test_reconcile_keeps_healthy_hotspot(self):
    self.manager._tethering_active = True
    self.manager._wifi_state = WifiState("Hotspot", ConnectStatus.CONNECTED)
    self.manager._ctrl.request.return_value = "wpa_state=COMPLETED\nmode=AP\nssid=Hotspot\n"

    with (
      patch.object(wifi_manager_module, "tethering_dnsmasq_running", return_value=True),
      patch.object(wifi_manager_module, "_tethering_firewall_ready", return_value=True),
      patch.object(self.manager, "_stop_tethering") as stop_tethering,
    ):
      self.manager._reconcile_connecting_state()

    stop_tethering.assert_not_called()

  def test_reconcile_stops_hotspot_without_nat(self):
    self.manager._tethering_active = True
    self.manager._wifi_state = WifiState("Hotspot", ConnectStatus.CONNECTED)
    self.manager._ctrl.request.return_value = "wpa_state=COMPLETED\nmode=AP\nssid=Hotspot\n"

    with (
      patch.object(wifi_manager_module, "tethering_dnsmasq_running", return_value=True),
      patch.object(wifi_manager_module.subprocess, "run", return_value=MagicMock(returncode=1)),
      patch.object(self.manager, "_stop_tethering") as stop_tethering,
    ):
      self.manager._reconcile_connecting_state()

    stop_tethering.assert_called_once()

  def test_reconcile_clears_state_when_hotspot_cleanup_fails(self):
    self.manager._tethering_active = True
    self.manager._wifi_state = WifiState("Hotspot", ConnectStatus.CONNECTED)
    self.manager._ctrl.request.side_effect = OSError("socket closed")
    disconnected = MagicMock()
    self.manager.add_callbacks(disconnected=disconnected)

    with patch.object(self.manager, "_stop_tethering", side_effect=OSError("cleanup failed")):
      self.manager._reconcile_connecting_state()

    self.manager.process_callbacks()
    assert not self.manager.is_tethering_active()
    assert self.manager.wifi_state == WifiState()
    disconnected.assert_called_once()


class TestLifecycle(TestCase):
  def test_manager_starts_inactive_until_ui_is_shown(self):
    with (
      patch.object(wifi_manager_module, "NetworkStore") as network_store,
      patch.object(wifi_manager_module, "DhcpClient"),
      patch.object(wifi_manager_module, "Params", None),
      patch.object(WifiManager, "_initialize"),
    ):
      manager = WifiManager()

    assert not manager._active
    assert manager._store is None
    assert manager._ipv4_forward is None
    network_store.assert_not_called()

  def test_initialization_loads_network_store_in_worker(self):
    manager = build_wifi_manager()
    manager._store = None
    manager._scan_thread = MagicMock()
    manager._state_thread = MagicMock()
    store = MagicMock()
    store.get_tethering_password.return_value = "custom-password"

    with (
      patch.object(wifi_manager_module, "NetworkStore", return_value=store) as network_store,
      patch("builtins.open", side_effect=FileNotFoundError),
      patch.object(wifi_manager_module, "generate_wpa_conf"),
      patch.object(manager, "_ensure_wpa_supplicant"),
      patch.object(manager, "_update_networks"),
      patch.object(manager, "_init_wifi_state"),
      patch.object(wifi_manager_module.threading, "Thread") as thread,
    ):
      manager._initialize()
      thread.call_args.kwargs["target"]()

    network_store.assert_called_once()
    assert manager._store is store
    assert manager.tethering_password == "custom-password"
    store.ensure_tethering_profile.assert_called_once_with("Hotspot", "custom-password")

  def test_initial_config_failure_recovers_without_restart(self):
    manager = build_wifi_manager()
    manager._tethering_ssid = "weedle"
    manager._scan_thread = MagicMock()
    manager._state_thread = MagicMock()
    ctrl = MagicMock()

    with (
      patch("builtins.open", side_effect=FileNotFoundError),
      patch.object(wifi_manager_module, "generate_wpa_conf", side_effect=[OSError("read-only"), None]) as generate,
      patch.object(wifi_manager_module, "wpa_supplicant_running", return_value=False),
      patch.object(wifi_manager_module, "ensure_wpa_supplicant", return_value=ctrl) as ensure,
      patch.object(wifi_manager_module.threading, "Thread") as thread,
    ):
      manager._initialize()
      thread.call_args.kwargs["target"]()
      manager._ensure_wpa_supplicant()

    manager._scan_thread.start.assert_called_once()
    manager._state_thread.start.assert_called_once()
    assert generate.call_count == 2
    ensure.assert_called_once()
    assert manager._ctrl is ctrl

  def test_station_recovery_cleans_abandoned_ap_services(self):
    manager = build_wifi_manager()
    ctrl = MagicMock()

    def ensure(_should_exit, _station_reconfigured, on_abandoned_ap):
      on_abandoned_ap()
      return ctrl

    with (
      patch.object(wifi_manager_module, "wpa_supplicant_running", return_value=True),
      patch.object(wifi_manager_module, "ensure_wpa_supplicant", side_effect=ensure),
      patch.object(wifi_manager_module, "stop_tethering_dnsmasq") as stop_dnsmasq,
      patch.object(wifi_manager_module, "_delete_tethering_firewall_rules") as delete_firewall,
      patch.object(wifi_manager_module.subprocess, "run"),
    ):
      manager._ensure_wpa_supplicant()

    manager._apply_ipv4_forward.assert_called_once_with(False)
    stop_dnsmasq.assert_called_once()
    delete_firewall.assert_called_once()
    assert manager._ctrl is ctrl

  def test_hidden_manager_reconciles_without_scanning(self):
    manager = build_wifi_manager()
    manager._exit = False
    manager._active = False
    manager._last_network_scan = 0.0

    with (
      patch.object(manager, "_reconcile_connecting_state") as reconcile,
      patch.object(manager, "_request_scan") as request_scan,
      patch.object(wifi_manager_module.time, "sleep", side_effect=lambda _: setattr(manager, "_exit", True)),
    ):
      manager._network_scanner()

    reconcile.assert_called_once()
    request_scan.assert_not_called()

  def test_failed_station_bringup_uses_scan_period_retry(self):
    manager = build_wifi_manager()
    manager._exit = False
    manager._ctrl = None
    sleeps = []

    def wait(duration):
      sleeps.append(duration)
      manager._exit = True

    with (
      patch.object(wifi_manager_module, "wpa_supplicant_running", return_value=False),
      patch.object(manager, "_ensure_wpa_supplicant"),
      patch.object(manager._exit_event, "wait", side_effect=wait),
    ):
      manager._monitor_state()

    assert sleeps == [SCAN_PERIOD_SECONDS]

  def test_monitor_exit_skips_retry_sleep(self):
    manager = build_wifi_manager()
    manager._exit = False
    monitor = MagicMock()
    monitor.recv.side_effect = lambda **_: setattr(manager, "_exit", True)

    with (
      patch.object(wifi_manager_module, "WpaCtrlMonitor", return_value=monitor),
      patch.object(wifi_manager_module.time, "sleep") as sleep,
    ):
      manager._monitor_state()

    sleep.assert_not_called()

  def test_disconnected_reconciliation_is_rate_limited(self):
    manager = build_wifi_manager()
    manager._wifi_state = WifiState()
    manager._last_connected_recheck = 0.0
    manager._ctrl.request.return_value = "wpa_state=DISCONNECTED\n"

    with patch.object(wifi_manager_module.time, "monotonic", return_value=100.0):
      manager._reconcile_connecting_state()
      manager._reconcile_connecting_state()

    manager._ctrl.request.assert_called_once_with("STATUS")

  def test_activating_manager_refreshes_state_and_networks(self):
    manager = build_wifi_manager()
    manager._active = False

    with (
      patch.object(manager, "_init_wifi_state") as init_wifi_state,
      patch.object(manager, "_update_networks") as update_networks,
    ):
      manager.set_active(True)

    assert manager._active
    init_wifi_state.assert_called_once_with(block=False)
    update_networks.assert_called_once_with(block=False)

  def test_stop_leaves_network_data_plane_running(self):
    manager = build_wifi_manager()
    manager._exit = False
    manager._tethering_active = True
    ctrl = manager._ctrl

    with patch.object(manager, "_stop_tethering") as stop_tethering:
      manager.stop()

    assert manager._exit
    assert manager._exit_event.is_set()
    assert ctrl is not None
    ctrl.interrupt.assert_called_once()
    ctrl.close.assert_called_once()
    manager._dhcp.stop.assert_not_called()
    stop_tethering.assert_not_called()

  def test_stop_interrupts_control_request_before_join(self):
    manager = build_wifi_manager()
    manager._exit = False
    request_started = threading.Event()
    interrupted = threading.Event()
    request_finished = threading.Event()
    closed = threading.Event()

    class BlockingCtrl:
      def request(self, _command):
        request_started.set()
        interrupted.wait(0.2)
        request_finished.set()

      def interrupt(self):
        interrupted.set()

      def close(self):
        closed.set()

    manager._ctrl = cast(wifi_manager_module.WpaCtrl, BlockingCtrl())
    manager._scan_thread = threading.Thread(target=manager._ctrl.request, args=("STATUS",))
    manager._state_thread = MagicMock()
    manager._state_thread.is_alive.return_value = False
    manager._scan_thread.start()
    assert request_started.wait(1)

    manager.stop()

    assert interrupted.is_set()
    assert request_finished.is_set()
    assert closed.is_set()
    assert not manager._scan_thread.is_alive()

  def test_callbacks_coalesce_network_updates(self):
    manager = build_wifi_manager()
    updated = MagicMock()
    manager.add_callbacks(networks_updated=updated)

    for _ in range(100):
      manager._mark_networks_updated()
    manager.process_callbacks()

    updated.assert_called_once_with(manager.networks)


class TestTetheringTransitions(TestCase):
  def test_station_transition_does_not_overlap_tethering_start(self):
    manager = build_wifi_manager()
    station_entered = threading.Event()
    release_station = threading.Event()
    tethering_entered = threading.Event()

    def add_station_network(*_args, **_kwargs):
      station_entered.set()
      assert release_station.wait(1)
      return "1"

    with (
      patch.object(manager, "_list_network_ids", return_value=[]),
      patch.object(manager, "_add_and_select_network", side_effect=add_station_network),
      patch.object(manager, "_start_tethering", side_effect=lambda: tethering_entered.set()),
    ):
      manager.connect_to_network("Station", "station-password")
      assert station_entered.wait(1)

      manager.set_tethering_active(True)
      try:
        assert not tethering_entered.wait(0.1)
      finally:
        release_station.set()
      assert tethering_entered.wait(1)

  def test_hotspot_adoption_does_not_overlap_tethering_transition(self):
    manager = build_wifi_manager()
    start_entered = threading.Event()
    release_start = threading.Event()
    adoption_entered = threading.Event()
    starts = 0

    def start_tethering():
      nonlocal starts
      starts += 1
      if starts == 1:
        start_entered.set()
        assert release_start.wait(1)
      else:
        adoption_entered.set()

    with (
      patch.object(manager, "_start_tethering", side_effect=start_tethering),
      patch.object(manager, "_ap_config_matches_password", return_value=False),
      patch.object(wifi_manager_module, "tethering_dnsmasq_running", return_value=True),
      patch.object(wifi_manager_module, "_tethering_firewall_ready", return_value=True),
    ):
      manager.set_tethering_active(True)
      assert start_entered.wait(1)

      adoption = threading.Thread(target=manager._adopt_ap_state, args=("Hotspot",))
      adoption.start()
      try:
        assert not adoption_entered.wait(0.1)
      finally:
        release_start.set()

      adoption.join(1)
      assert not adoption.is_alive()
      assert adoption_entered.is_set()

  def test_reconcile_waits_for_pending_tethering_start(self):
    manager = build_wifi_manager()
    manager._last_connected_recheck = 0.0

    with (
      patch.object(manager, "_start_tethering") as start_tethering,
      patch.object(manager, "_stop_tethering") as stop_tethering,
      patch.object(wifi_manager_module.threading, "Thread") as thread,
    ):
      manager.set_tethering_active(True)
      manager._reconcile_tethering_state()

      stop_tethering.assert_not_called()
      thread.call_args.kwargs["target"]()

    start_tethering.assert_called_once()
    assert manager.is_tethering_active()

  def test_startup_station_bringup_does_not_overlap_tethering(self):
    manager = build_wifi_manager()
    manager._tethering_ssid = "weedle"
    station_entered = threading.Event()
    release_station = threading.Event()
    tethering_entered = threading.Event()

    def ensure_station():
      station_entered.set()
      assert release_station.wait(1)

    with (
      patch.object(wifi_manager_module, "NetworkStore", return_value=manager._store),
      patch("builtins.open", side_effect=FileNotFoundError),
      patch.object(wifi_manager_module, "generate_wpa_conf"),
      patch.object(manager, "_ensure_wpa_supplicant", side_effect=ensure_station),
      patch.object(manager, "_update_networks"),
      patch.object(manager, "_init_wifi_state"),
      patch.object(manager, "_start_tethering", side_effect=lambda: tethering_entered.set()),
    ):
      with patch.object(wifi_manager_module.threading, "Thread") as thread:
        manager._initialize()
      initialize = thread.call_args.kwargs["target"]

      initialize_thread = threading.Thread(target=initialize)
      initialize_thread.start()
      assert station_entered.wait(1)

      manager.set_tethering_active(True)
      try:
        assert not tethering_entered.wait(0.1)
      finally:
        release_station.set()

      initialize_thread.join(1)
      assert not initialize_thread.is_alive()
      assert tethering_entered.wait(1)

  def test_tethering_transitions_do_not_overlap(self):
    manager = build_wifi_manager()
    start_entered = threading.Event()
    release_start = threading.Event()
    stop_entered = threading.Event()

    def start_tethering():
      manager._tethering_started = True
      start_entered.set()
      assert release_start.wait(1)

    with (
      patch.object(manager, "_start_tethering", side_effect=start_tethering),
      patch.object(manager, "_stop_tethering", side_effect=lambda: stop_entered.set()),
    ):
      manager.set_tethering_active(True)
      assert start_entered.wait(1)

      manager.set_tethering_active(False)
      try:
        assert not stop_entered.wait(0.1)
      finally:
        release_start.set()
      assert stop_entered.wait(1)

  def test_latest_tethering_request_wins(self):
    manager = build_wifi_manager()
    manager._wifi_state = WifiState("Station", ConnectStatus.CONNECTED)
    manager._ipv4_address = "192.168.1.20"
    station_ctrl = manager._ctrl
    transitions = []
    disconnected = MagicMock()
    manager.add_callbacks(disconnected=disconnected)

    def start_tethering():
      transitions.append(True)
      manager._tethering_active = True

    def stop_tethering():
      transitions.append(False)
      manager._tethering_active = False

    with (
      patch.object(manager, "_start_tethering", side_effect=start_tethering),
      patch.object(manager, "_stop_tethering", side_effect=stop_tethering),
      patch.object(wifi_manager_module.threading, "Thread") as thread,
    ):
      manager.set_tethering_active(True)
      manager.set_tethering_active(False)

      workers = [item.kwargs["target"] for item in thread.call_args_list]
      workers[1]()
      workers[0]()

    manager.process_callbacks()
    assert transitions == []
    assert not manager.is_tethering_active()
    assert manager.wifi_state == WifiState("Station", ConnectStatus.CONNECTED)
    assert manager.ipv4_address == "192.168.1.20"
    assert manager._ctrl is station_ctrl
    disconnected.assert_called_once()

  def test_failed_tethering_stop_notifies_disconnected(self):
    manager = build_wifi_manager()
    manager._tethering_active = True
    manager._tethering_started = True
    disconnected = MagicMock()
    manager.add_callbacks(disconnected=disconnected)

    with (
      patch.object(manager, "_stop_tethering", side_effect=OSError("cleanup failed")),
      patch.object(wifi_manager_module.threading, "Thread") as thread,
    ):
      manager.set_tethering_active(False)
      thread.call_args.kwargs["target"]()

    manager.process_callbacks()
    assert not manager.is_tethering_active()
    assert manager.wifi_state == WifiState()
    assert manager.ipv4_address == ""
    disconnected.assert_called_once()


class TestTetheringPassword(TestCase):
  def test_latest_password_request_wins(self):
    manager = build_wifi_manager()
    manager._store.set_tethering_password.return_value = True

    with (
      patch.object(wifi_manager_module, "atomic_write") as legacy_write,
      patch.object(wifi_manager_module.threading, "Thread") as thread,
    ):
      manager.set_tethering_password("first-password")
      manager.set_tethering_password("second-password")

      workers = [item.kwargs["target"] for item in thread.call_args_list]
      workers[1]()
      workers[0]()

    assert manager.tethering_password == "second-password"
    legacy_write.assert_not_called()
    manager._store.set_tethering_password.assert_called_once_with("Hotspot", "second-password")

  def test_startup_prefers_keyfile_without_reading_legacy_password(self):
    manager = build_wifi_manager()
    manager._tethering_ssid = "weedle"
    store = manager._store
    assert store is not None
    store.get_tethering_password.return_value = "custom-password"
    manager._scan_thread = MagicMock()
    manager._state_thread = MagicMock()

    with (
      patch.object(wifi_manager_module, "NetworkStore", return_value=store),
      patch("builtins.open") as legacy_open,
      patch.object(manager, "_ensure_wpa_supplicant"),
      patch.object(manager, "_update_networks"),
      patch.object(manager, "_init_wifi_state"),
      patch.object(wifi_manager_module.threading, "Thread") as thread,
    ):
      manager._initialize()
      thread.call_args.kwargs["target"]()

    assert manager.tethering_password == "custom-password"
    legacy_open.assert_not_called()

  def test_startup_migrates_legacy_password_into_keyfile(self):
    manager = build_wifi_manager()
    manager._tethering_ssid = "weedle"
    store = manager._store
    assert store is not None
    store.get_tethering_password.return_value = None
    store.set_tethering_password.return_value = True
    manager._scan_thread = MagicMock()
    manager._state_thread = MagicMock()

    with (
      patch.object(wifi_manager_module, "NetworkStore", return_value=store),
      patch("builtins.open", mock_open(read_data="legacy-password\n")),
      patch.object(manager, "_ensure_wpa_supplicant"),
      patch.object(manager, "_update_networks"),
      patch.object(manager, "_init_wifi_state"),
      patch.object(wifi_manager_module.threading, "Thread") as thread,
    ):
      manager._initialize()
      thread.call_args.kwargs["target"]()

    assert manager.tethering_password == "legacy-password"
    store.set_tethering_password.assert_called_once_with("weedle", "legacy-password")

  def test_persist_failure_reenables_active_tethering_controls(self):
    manager = build_wifi_manager()
    manager._tethering_active = True
    manager._tethering_psk = "old-password"
    activated = MagicMock()
    manager.add_callbacks(activated=activated)
    manager.process_callbacks()
    activated.reset_mock()

    manager._store.set_tethering_password.return_value = False
    with patch.object(wifi_manager_module.threading, "Thread") as thread:
      manager.set_tethering_password("replacement-password")
      thread.call_args.kwargs["target"]()

    manager.process_callbacks()
    assert manager.tethering_password == "old-password"
    activated.assert_called_once()

  def test_teardown_failure_reenables_active_tethering_controls(self):
    manager = build_wifi_manager()
    manager._tethering_active = True
    manager._wifi_state = WifiState(manager._tethering_ssid, ConnectStatus.CONNECTED)
    disconnected = MagicMock()
    manager.add_callbacks(disconnected=disconnected)
    manager._store.set_tethering_password.return_value = True

    with (
      patch.object(manager, "_stop_tethering", side_effect=OSError("cleanup failed")) as stop_tethering,
      patch.object(manager, "_start_tethering") as start_tethering,
      patch.object(wifi_manager_module.threading, "Thread") as thread,
    ):
      manager.set_tethering_password("replacement-password")
      thread.call_args.kwargs["target"]()

    manager.process_callbacks()
    assert manager.tethering_password == "replacement-password"
    assert not manager.is_tethering_active()
    assert manager.wifi_state == WifiState()
    assert stop_tethering.call_count == 2
    start_tethering.assert_not_called()
    disconnected.assert_called_once()

  def test_restart_failure_keeps_committed_password(self):
    manager = build_wifi_manager()
    manager._tethering_active = True
    manager._wifi_state = WifiState(manager._tethering_ssid, ConnectStatus.CONNECTED)
    manager._store.set_tethering_password.return_value = True

    def stop_tethering():
      manager._tethering_active = False
      manager._wifi_state = WifiState()

    with (
      patch.object(manager, "_stop_tethering", side_effect=stop_tethering) as stop_tethering,
      patch.object(manager, "_start_tethering", side_effect=OSError("restart failed")) as start_tethering,
      patch.object(wifi_manager_module.threading, "Thread") as thread,
    ):
      manager.set_tethering_password("replacement-password")
      thread.call_args.kwargs["target"]()

    assert manager.tethering_password == "replacement-password"
    manager._store.set_tethering_password.assert_called_once_with("Hotspot", "replacement-password")
    start_tethering.assert_called_once()
    assert stop_tethering.call_count == 2
    assert not manager.is_tethering_active()
