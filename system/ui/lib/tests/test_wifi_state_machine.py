"""Tests for WifiManager._handle_state_change state machine.

Focuses on the two race conditions identified during network switching:
1. DISCONNECTED+CONNECTION_REMOVED: forgetting network A while connecting to B
   would incorrectly clear B's connecting state.
2. PREPARE/CONFIG ssid lookup: switching A->B, stale DBus returns A's connection,
   overwriting the user-set B ssid.
"""
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from openpilot.system.ui.lib.networkmanager import NMDeviceState, NMDeviceStateReason
from openpilot.system.ui.lib.wifi_manager import WifiManager, WifiState, ConnectStatus


@pytest.fixture
def wm():
  """Create a WifiManager with DBus/threading disabled, fields set directly."""
  with patch.object(WifiManager, '__init__', lambda self: None):
    mgr = WifiManager.__new__(WifiManager)

  mgr._wifi_state = WifiState()
  mgr._connections = {}
  mgr._callback_queue = []
  mgr._need_auth = []
  mgr._activated = []
  mgr._forgotten = []
  mgr._exit = True
  mgr._router_main = MagicMock()
  mgr._update_networks = MagicMock()
  mgr._update_active_connection_info = MagicMock()
  mgr._get_active_wifi_connection = MagicMock(return_value=(None, None))
  return mgr


def _fire(wm, new_state, reason=NMDeviceStateReason.NONE, prev_state=NMDeviceState.UNKNOWN):
  """Shorthand: feed a state change into the handler."""
  router = MagicMock()
  wm._handle_state_change(new_state, prev_state, reason, router)
  return router


# ---------------------------------------------------------------------------
# Race Condition 1: DISCONNECTED + CONNECTION_REMOVED
# Scenario: user forgets network A while connecting to network B.
# NM fires DISCONNECTED+CONNECTION_REMOVED for A's removal, which must NOT
# clear B's connecting state.
# ---------------------------------------------------------------------------

class TestRaceCondition1:
  def test_forget_A_while_connecting_to_B__keeps_B(self, wm):
    """Guard: B is in _connections, so CONNECTION_REMOVED must not clear state."""
    wm._connections = {"B": "/path/B"}
    wm._wifi_state = WifiState("B", ConnectStatus.CONNECTING)

    _fire(wm, NMDeviceState.DISCONNECTED, NMDeviceStateReason.CONNECTION_REMOVED)

    assert wm._wifi_state == WifiState("B", ConnectStatus.CONNECTING)

  def test_forget_B_while_connecting_to_B__clears(self, wm):
    """Inverse: B was already removed from _connections, so state should clear."""
    wm._connections = {}
    wm._wifi_state = WifiState("B", ConnectStatus.CONNECTING)

    _fire(wm, NMDeviceState.DISCONNECTED, NMDeviceStateReason.CONNECTION_REMOVED)

    assert wm._wifi_state == WifiState(None, ConnectStatus.DISCONNECTED)

  def test_forget_A_while_connected_to_B__keeps_B(self, wm):
    """Also protects already-connected state from stale removal signal."""
    wm._connections = {"B": "/path/B"}
    wm._wifi_state = WifiState("B", ConnectStatus.CONNECTED)

    _fire(wm, NMDeviceState.DISCONNECTED, NMDeviceStateReason.CONNECTION_REMOVED)

    assert wm._wifi_state == WifiState("B", ConnectStatus.CONNECTED)

  def test_forget_A_no_active_connection__clears(self, wm):
    """If disconnected and ssid not in connections, state clears."""
    wm._connections = {"C": "/path/C"}
    wm._wifi_state = WifiState("A", ConnectStatus.CONNECTING)

    _fire(wm, NMDeviceState.DISCONNECTED, NMDeviceStateReason.CONNECTION_REMOVED)

    assert wm._wifi_state == WifiState(None, ConnectStatus.DISCONNECTED)


# ---------------------------------------------------------------------------
# Race Condition 2: PREPARE/CONFIG stale ssid lookup
# Scenario: user switches from A to B. NM fires PREPARE for the new activation,
# but DBus still returns A's connection path. The guard must NOT overwrite B.
# ---------------------------------------------------------------------------

class TestRaceCondition2:
  def test_prepare_with_user_ssid_set__skips_dbus_lookup(self, wm):
    """User-initiated: ssid already set via _set_connecting, DBus lookup skipped."""
    wm._wifi_state = WifiState("B", ConnectStatus.CONNECTING)
    wm._connections = {"A": "/path/A", "B": "/path/B"}

    _fire(wm, NMDeviceState.PREPARE)

    assert wm._wifi_state.ssid == "B"
    assert wm._wifi_state.status == ConnectStatus.CONNECTING
    wm._get_active_wifi_connection.assert_not_called()

  def test_config_with_user_ssid_set__skips_dbus_lookup(self, wm):
    """Same guard applies to CONFIG state."""
    wm._wifi_state = WifiState("B", ConnectStatus.CONNECTING)

    _fire(wm, NMDeviceState.CONFIG)

    assert wm._wifi_state.ssid == "B"
    wm._get_active_wifi_connection.assert_not_called()

  def test_prepare_auto_connection__looks_up_ssid(self, wm):
    """Auto-connection (ssid is None): must look up ssid from DBus."""
    wm._wifi_state = WifiState(None, ConnectStatus.DISCONNECTED)
    wm._connections = {"AutoNet": "/path/auto"}
    wm._get_active_wifi_connection = MagicMock(return_value=("/path/auto", {}))

    router = _fire(wm, NMDeviceState.PREPARE)

    assert wm._wifi_state.ssid == "AutoNet"
    assert wm._wifi_state.status == ConnectStatus.CONNECTING

  def test_prepare_auto_connection__dbus_fails(self, wm):
    """Auto-connection but DBus returns None: ssid stays None, status still CONNECTING."""
    wm._wifi_state = WifiState(None, ConnectStatus.DISCONNECTED)
    wm._get_active_wifi_connection = MagicMock(return_value=(None, None))

    _fire(wm, NMDeviceState.PREPARE)

    assert wm._wifi_state.ssid is None
    assert wm._wifi_state.status == ConnectStatus.CONNECTING

  def test_prepare_auto_connection__conn_path_not_in_connections(self, wm):
    """DBus returns a conn_path that doesn't match any known connection."""
    wm._wifi_state = WifiState(None, ConnectStatus.DISCONNECTED)
    wm._connections = {"Other": "/path/other"}
    wm._get_active_wifi_connection = MagicMock(return_value=("/path/unknown", {}))

    _fire(wm, NMDeviceState.PREPARE)

    assert wm._wifi_state.ssid is None
    assert wm._wifi_state.status == ConnectStatus.CONNECTING


# ---------------------------------------------------------------------------
# Normal state transitions
# ---------------------------------------------------------------------------

class TestNormalFlows:
  def test_disconnected_generic__clears_state(self, wm):
    """Normal disconnect (not CONNECTION_REMOVED, not NEW_ACTIVATION) clears state."""
    wm._wifi_state = WifiState("MyNet", ConnectStatus.CONNECTED)

    _fire(wm, NMDeviceState.DISCONNECTED, NMDeviceStateReason.UNKNOWN)

    assert wm._wifi_state == WifiState(None, ConnectStatus.DISCONNECTED)

  def test_disconnected_new_activation__noop(self, wm):
    """NEW_ACTIVATION reason: NM is about to connect to another network, don't clear."""
    wm._wifi_state = WifiState("OldNet", ConnectStatus.CONNECTED)

    _fire(wm, NMDeviceState.DISCONNECTED, NMDeviceStateReason.NEW_ACTIVATION)

    assert wm._wifi_state == WifiState("OldNet", ConnectStatus.CONNECTED)

  def test_deactivating__noop(self, wm):
    """DEACTIVATING is always a no-op (DISCONNECTED follows with correct state)."""
    wm._wifi_state = WifiState("MyNet", ConnectStatus.CONNECTED)

    _fire(wm, NMDeviceState.DEACTIVATING)

    assert wm._wifi_state == WifiState("MyNet", ConnectStatus.CONNECTED)

  def test_need_auth_supplicant_disconnect__fires_callback_and_clears(self, wm):
    """Wrong password (strong network): fires need_auth callback and clears state."""
    cb = MagicMock()
    wm._need_auth = [cb]
    wm._wifi_state = WifiState("SecureNet", ConnectStatus.CONNECTING)

    _fire(wm, NMDeviceState.NEED_AUTH, NMDeviceStateReason.SUPPLICANT_DISCONNECT)

    assert wm._wifi_state == WifiState(None, ConnectStatus.DISCONNECTED)
    assert len(wm._callback_queue) == 1
    wm._callback_queue[0]()
    cb.assert_called_once_with("SecureNet")

  def test_failed_no_secrets__fires_callback_and_clears(self, wm):
    """Wrong password (weak/gone network): fires need_auth callback and clears state."""
    cb = MagicMock()
    wm._need_auth = [cb]
    wm._wifi_state = WifiState("WeakNet", ConnectStatus.CONNECTING)

    _fire(wm, NMDeviceState.FAILED, NMDeviceStateReason.NO_SECRETS)

    assert wm._wifi_state == WifiState(None, ConnectStatus.DISCONNECTED)
    assert len(wm._callback_queue) == 1
    wm._callback_queue[0]()
    cb.assert_called_once_with("WeakNet")

  def test_need_auth_no_ssid__no_callback(self, wm):
    """If ssid is already None when NEED_AUTH fires, no callback is enqueued."""
    cb = MagicMock()
    wm._need_auth = [cb]
    wm._wifi_state = WifiState(None, ConnectStatus.DISCONNECTED)

    _fire(wm, NMDeviceState.NEED_AUTH, NMDeviceStateReason.SUPPLICANT_DISCONNECT)

    assert len(wm._callback_queue) == 0

  def test_passthrough_states__noop(self, wm):
    """NEED_AUTH (generic), IP_CONFIG, IP_CHECK, SECONDARIES, FAILED (generic) are no-ops."""
    for state in (NMDeviceState.NEED_AUTH, NMDeviceState.IP_CONFIG, NMDeviceState.IP_CHECK,
                  NMDeviceState.SECONDARIES, NMDeviceState.FAILED):
      wm._wifi_state = WifiState("Net", ConnectStatus.CONNECTING)
      _fire(wm, state, NMDeviceStateReason.NONE)
      assert wm._wifi_state == WifiState("Net", ConnectStatus.CONNECTING), f"State {state} should be no-op"

  def test_activated__sets_connected_and_ssid(self, wm):
    """ACTIVATED: sets CONNECTED, looks up ssid, fires callback, persists connection."""
    cb = MagicMock()
    wm._activated = [cb]
    wm._wifi_state = WifiState("MyNet", ConnectStatus.CONNECTING)
    wm._connections = {"MyNet": "/path/mynet"}
    wm._get_active_wifi_connection = MagicMock(return_value=("/path/mynet", {}))

    save_reply = MagicMock()
    save_reply.header.message_type = 1  # METHOD_RETURN (not MessageType.error)

    router = MagicMock()
    router.send_and_get_reply = MagicMock(return_value=save_reply)
    wm._handle_state_change(NMDeviceState.ACTIVATED, NMDeviceState.IP_CHECK, NMDeviceStateReason.NONE, router)

    assert wm._wifi_state.status == ConnectStatus.CONNECTED
    assert wm._wifi_state.ssid == "MyNet"
    wm._update_active_connection_info.assert_called_once()
    wm._update_networks.assert_called_once()
    assert len(wm._callback_queue) == 1
    router.send_and_get_reply.assert_called_once()

  def test_activated__conn_path_none__still_connected(self, wm):
    """ACTIVATED but DBus returns None: status is CONNECTED, ssid unchanged, callback fires."""
    cb = MagicMock()
    wm._activated = [cb]
    wm._wifi_state = WifiState("MyNet", ConnectStatus.CONNECTING)
    wm._get_active_wifi_connection = MagicMock(return_value=(None, None))

    _fire(wm, NMDeviceState.ACTIVATED)

    assert wm._wifi_state.status == ConnectStatus.CONNECTED
    assert wm._wifi_state.ssid == "MyNet"
    assert len(wm._callback_queue) == 1


# ---------------------------------------------------------------------------
# Full sequence simulations
# ---------------------------------------------------------------------------

class TestFullSequences:
  def test_forget_A_connect_B__new_connection_before_disconnected(self, wm):
    """Normal case: NewConnection for B arrives between DEACTIVATING and DISCONNECTED.

    Signal order:
    1. User: _set_connecting("B"), forget_connection("A") removes A from _connections
    2. DEACTIVATING(CONNECTION_REMOVED) — B NOT in _connections yet
    3. NewConnection for B → added to _connections
    4. DISCONNECTED(CONNECTION_REMOVED) — B IS in _connections, guard protects
    5. PREPARE, CONFIG, ACTIVATED
    """
    wm._connections = {"A": "/path/A"}
    wm._wifi_state = WifiState("A", ConnectStatus.CONNECTED)

    wm._set_connecting("B")
    del wm._connections["A"]

    # DEACTIVATING — B not in _connections yet, but DEACTIVATING is a no-op
    _fire(wm, NMDeviceState.DEACTIVATING, NMDeviceStateReason.CONNECTION_REMOVED)
    assert wm._wifi_state == WifiState("B", ConnectStatus.CONNECTING), \
      "DEACTIVATING must be no-op — this was the original bug when it wasn't"

    # NewConnection for B arrives (AddAndActivateConnection2 completed)
    wm._connections["B"] = "/path/B"

    # DISCONNECTED — B is now in _connections, guard protects
    _fire(wm, NMDeviceState.DISCONNECTED, NMDeviceStateReason.CONNECTION_REMOVED)
    assert wm._wifi_state == WifiState("B", ConnectStatus.CONNECTING)

    # PREPARE — ssid already set, no DBus lookup
    _fire(wm, NMDeviceState.PREPARE)
    assert wm._wifi_state == WifiState("B", ConnectStatus.CONNECTING)
    wm._get_active_wifi_connection.assert_not_called()

    _fire(wm, NMDeviceState.CONFIG)
    assert wm._wifi_state == WifiState("B", ConnectStatus.CONNECTING)

    wm._get_active_wifi_connection = MagicMock(return_value=("/path/B", {}))
    router = MagicMock()
    save_reply = MagicMock()
    save_reply.header.message_type = 1
    router.send_and_get_reply = MagicMock(return_value=save_reply)
    wm._handle_state_change(NMDeviceState.ACTIVATED, NMDeviceState.CONFIG, NMDeviceStateReason.NONE, router)
    assert wm._wifi_state == WifiState("B", ConnectStatus.CONNECTED)

  def test_forget_A_connect_B__new_connection_AFTER_disconnected(self, wm):
    """Dangerous case: NewConnection for B arrives AFTER DISCONNECTED.

    This is the exact race condition that caused the flickering bug.
    Without DEACTIVATING being a no-op, state would be cleared at DEACTIVATING.
    Even with DEACTIVATING as no-op, if NewConnection arrives after DISCONNECTED,
    the DISCONNECTED guard sees B NOT in _connections and clears state.

    Signal order:
    1. User: _set_connecting("B"), forget_connection("A")
    2. DEACTIVATING(CONNECTION_REMOVED) — B NOT in _connections
    3. DISCONNECTED(CONNECTION_REMOVED) — B STILL NOT in _connections
    4. NewConnection for B → added to _connections
    5. PREPARE, CONFIG, ACTIVATED

    This tests that even if B isn't in _connections at DISCONNECTED time,
    the PREPARE handler recovers by looking up ssid from DBus.
    """
    wm._connections = {"A": "/path/A"}
    wm._wifi_state = WifiState("A", ConnectStatus.CONNECTED)

    wm._set_connecting("B")
    del wm._connections["A"]

    # DEACTIVATING — no-op
    _fire(wm, NMDeviceState.DEACTIVATING, NMDeviceStateReason.CONNECTION_REMOVED)
    assert wm._wifi_state == WifiState("B", ConnectStatus.CONNECTING)

    # DISCONNECTED — B NOT in _connections, guard clears state
    _fire(wm, NMDeviceState.DISCONNECTED, NMDeviceStateReason.CONNECTION_REMOVED)
    assert wm._wifi_state == WifiState(None, ConnectStatus.DISCONNECTED), \
      "B not in _connections, so state must clear (this is the remaining edge case)"

    # NewConnection arrives late
    wm._connections["B"] = "/path/B"

    # PREPARE — ssid is None, so it must look up from DBus to recover
    wm._get_active_wifi_connection = MagicMock(return_value=("/path/B", {}))
    _fire(wm, NMDeviceState.PREPARE)
    assert wm._wifi_state == WifiState("B", ConnectStatus.CONNECTING), \
      "PREPARE must recover by looking up ssid from DBus when ssid is None"

    _fire(wm, NMDeviceState.CONFIG)
    assert wm._wifi_state == WifiState("B", ConnectStatus.CONNECTING)

    router = MagicMock()
    save_reply = MagicMock()
    save_reply.header.message_type = 1
    router.send_and_get_reply = MagicMock(return_value=save_reply)
    wm._handle_state_change(NMDeviceState.ACTIVATED, NMDeviceState.CONFIG, NMDeviceStateReason.NONE, router)
    assert wm._wifi_state == WifiState("B", ConnectStatus.CONNECTED)

  def test_swap_known_networks__A_to_B(self, wm):
    """Simulates switching between two saved networks A and B.

    Signal order:
    1. User calls connect_to_network(B) -> _set_connecting("B")
    2. NM fires DEACTIVATING(NEW_ACTIVATION) for A
    3. NM fires DISCONNECTED(NEW_ACTIVATION) for A — no-op (NEW_ACTIVATION)
    4. NM fires PREPARE(0) for B
    5. NM fires CONFIG(0) for B
    6. NM fires ACTIVATED(0) for B
    """
    wm._connections = {"A": "/path/A", "B": "/path/B"}
    wm._wifi_state = WifiState("A", ConnectStatus.CONNECTED)

    wm._set_connecting("B")

    _fire(wm, NMDeviceState.DEACTIVATING, NMDeviceStateReason.NEW_ACTIVATION)
    assert wm._wifi_state == WifiState("B", ConnectStatus.CONNECTING)

    _fire(wm, NMDeviceState.DISCONNECTED, NMDeviceStateReason.NEW_ACTIVATION)
    assert wm._wifi_state == WifiState("B", ConnectStatus.CONNECTING)

    _fire(wm, NMDeviceState.PREPARE)
    assert wm._wifi_state == WifiState("B", ConnectStatus.CONNECTING)

    _fire(wm, NMDeviceState.CONFIG)
    assert wm._wifi_state == WifiState("B", ConnectStatus.CONNECTING)

    wm._get_active_wifi_connection = MagicMock(return_value=("/path/B", {}))
    router = MagicMock()
    save_reply = MagicMock()
    save_reply.header.message_type = 1
    router.send_and_get_reply = MagicMock(return_value=save_reply)
    wm._handle_state_change(NMDeviceState.ACTIVATED, NMDeviceState.CONFIG, NMDeviceStateReason.NONE, router)
    assert wm._wifi_state == WifiState("B", ConnectStatus.CONNECTED)

  def test_auto_connect__no_user_action(self, wm):
    """NM auto-connects to a network (no user action, ssid starts as None).

    The PREPARE handler must look up the ssid from DBus.
    """
    wm._connections = {"AutoNet": "/path/auto"}
    wm._wifi_state = WifiState(None, ConnectStatus.DISCONNECTED)
    wm._get_active_wifi_connection = MagicMock(return_value=("/path/auto", {}))

    _fire(wm, NMDeviceState.PREPARE)
    assert wm._wifi_state == WifiState("AutoNet", ConnectStatus.CONNECTING)

    _fire(wm, NMDeviceState.CONFIG)
    assert wm._wifi_state == WifiState("AutoNet", ConnectStatus.CONNECTING)

    router = MagicMock()
    save_reply = MagicMock()
    save_reply.header.message_type = 1
    router.send_and_get_reply = MagicMock(return_value=save_reply)
    wm._handle_state_change(NMDeviceState.ACTIVATED, NMDeviceState.CONFIG, NMDeviceStateReason.NONE, router)
    assert wm._wifi_state == WifiState("AutoNet", ConnectStatus.CONNECTED)

  def test_wrong_password_then_retry(self, wm):
    """User enters wrong password, gets NEED_AUTH, then retries with correct password."""
    cb = MagicMock()
    wm._need_auth = [cb]
    wm._connections = {"SecNet": "/path/sec"}
    wm._wifi_state = WifiState("SecNet", ConnectStatus.CONNECTING)

    # Wrong password -> NEED_AUTH+SUPPLICANT_DISCONNECT
    _fire(wm, NMDeviceState.NEED_AUTH, NMDeviceStateReason.SUPPLICANT_DISCONNECT)
    assert wm._wifi_state == WifiState(None, ConnectStatus.DISCONNECTED)
    assert len(wm._callback_queue) == 1

    # User retries
    wm._callback_queue.clear()
    wm._set_connecting("SecNet")
    assert wm._wifi_state == WifiState("SecNet", ConnectStatus.CONNECTING)

    _fire(wm, NMDeviceState.PREPARE)
    assert wm._wifi_state == WifiState("SecNet", ConnectStatus.CONNECTING)

    wm._get_active_wifi_connection = MagicMock(return_value=("/path/sec", {}))
    router = MagicMock()
    save_reply = MagicMock()
    save_reply.header.message_type = 1
    router.send_and_get_reply = MagicMock(return_value=save_reply)
    wm._handle_state_change(NMDeviceState.ACTIVATED, NMDeviceState.CONFIG, NMDeviceStateReason.NONE, router)
    assert wm._wifi_state == WifiState("SecNet", ConnectStatus.CONNECTED)
