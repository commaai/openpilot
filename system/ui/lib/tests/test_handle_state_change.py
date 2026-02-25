"""Tests for WifiManager._handle_state_change.

Tests the state machine in isolation by constructing a WifiManager with mocked
DBus, then calling _handle_state_change directly with NM state transitions.

Many tests assert *desired* behavior that the current code doesn't implement yet.
These are marked with pytest.mark.xfail and document the intended fix.
"""
import threading
from unittest.mock import MagicMock, patch

import pytest

from openpilot.system.ui.lib.networkmanager import NMDeviceState, NMDeviceStateReason
from openpilot.system.ui.lib.wifi_manager import WifiManager, WifiState, ConnectStatus, MeteredType, DEFAULT_TETHERING_PASSWORD


def _make_wm(connections=None):
  """Create a WifiManager with mocked DBus."""
  with patch.object(WifiManager, '_initialize'):
    wm = WifiManager.__new__(WifiManager)
    wm._networks = []
    wm._active = True
    wm._exit = True
    wm._router_main = MagicMock()
    wm._conn_monitor = MagicMock()
    wm._wifi_device = '/mock/wifi0'
    wm._connections = dict(connections or {})
    wm._wifi_state = WifiState()
    wm._ipv4_address = ""
    wm._current_network_metered = MeteredType.UNKNOWN
    wm._tethering_password = DEFAULT_TETHERING_PASSWORD
    wm._ipv4_forward = False
    wm._last_network_scan = 0.0
    wm._callback_queue = []
    wm._tethering_ssid = "weedle"
    wm._need_auth = []
    wm._activated = []
    wm._forgotten = []
    wm._networks_updated = []
    wm._disconnected = []
    wm._lock = threading.Lock()
    wm._scan_thread = MagicMock()
    wm._state_thread = MagicMock()
    wm._update_networks = MagicMock()
    wm._update_active_connection_info = MagicMock()
    wm._get_active_wifi_connection = MagicMock(return_value=(None, None))
  return wm


def _fire(wm, new_state, reason=NMDeviceStateReason.NONE, prev_state=NMDeviceState.UNKNOWN):
  """Feed a state change into the handler."""
  wm._handle_state_change(new_state, prev_state, reason)


# ---------------------------------------------------------------------------
# Basic transitions
# ---------------------------------------------------------------------------

class TestDisconnected:
  def test_generic_disconnect_clears_state(self):
    wm = _make_wm()
    wm._wifi_state = WifiState(ssid="Net", status=ConnectStatus.CONNECTED)

    _fire(wm, NMDeviceState.DISCONNECTED, NMDeviceStateReason.UNKNOWN)

    assert wm._wifi_state.ssid is None
    assert wm._wifi_state.status == ConnectStatus.DISCONNECTED

  def test_new_activation_is_noop(self):
    """NEW_ACTIVATION means NM is about to connect to another network — don't clear."""
    wm = _make_wm()
    wm._wifi_state = WifiState(ssid="OldNet", status=ConnectStatus.CONNECTED)

    _fire(wm, NMDeviceState.DISCONNECTED, NMDeviceStateReason.NEW_ACTIVATION)

    assert wm._wifi_state.ssid == "OldNet"
    assert wm._wifi_state.status == ConnectStatus.CONNECTED

  @pytest.mark.xfail(reason="TODO: CONNECTION_REMOVED should only clear if ssid not in _connections")
  def test_connection_removed_keeps_other_connecting(self):
    """Forget A while connecting to B: CONNECTION_REMOVED for A must not clear B."""
    wm = _make_wm(connections={"B": "/path/B"})
    wm._wifi_state = WifiState(ssid="B", status=ConnectStatus.CONNECTING)

    _fire(wm, NMDeviceState.DISCONNECTED, NMDeviceStateReason.CONNECTION_REMOVED)

    assert wm._wifi_state.ssid == "B"
    assert wm._wifi_state.status == ConnectStatus.CONNECTING

  def test_connection_removed_clears_when_forgotten(self):
    """Forget A: A is no longer in _connections, so state should clear."""
    wm = _make_wm(connections={})
    wm._wifi_state = WifiState(ssid="A", status=ConnectStatus.CONNECTING)

    _fire(wm, NMDeviceState.DISCONNECTED, NMDeviceStateReason.CONNECTION_REMOVED)

    assert wm._wifi_state.ssid is None
    assert wm._wifi_state.status == ConnectStatus.DISCONNECTED


class TestDeactivating:
  @pytest.mark.xfail(reason="TODO: DEACTIVATING should be a no-op")
  def test_deactivating_is_noop(self):
    """DEACTIVATING should be a no-op — DISCONNECTED follows with correct state."""
    wm = _make_wm()
    wm._wifi_state = WifiState(ssid="Net", status=ConnectStatus.CONNECTED)

    _fire(wm, NMDeviceState.DEACTIVATING, NMDeviceStateReason.CONNECTION_REMOVED)

    assert wm._wifi_state.ssid == "Net"
    assert wm._wifi_state.status == ConnectStatus.CONNECTED


class TestPrepareConfig:
  @pytest.mark.xfail(reason="TODO: should skip DBus lookup when ssid already set")
  def test_user_initiated_skips_dbus_lookup(self):
    """User called _set_connecting('B') — PREPARE must not overwrite via DBus."""
    wm = _make_wm(connections={"A": "/path/A", "B": "/path/B"})
    wm._wifi_state = WifiState(ssid="B", status=ConnectStatus.CONNECTING)
    wm._get_active_wifi_connection = MagicMock(return_value=("/path/A", {}))

    _fire(wm, NMDeviceState.PREPARE)

    assert wm._wifi_state.ssid == "B"
    assert wm._wifi_state.status == ConnectStatus.CONNECTING
    wm._get_active_wifi_connection.assert_not_called()

  def test_auto_connect_looks_up_ssid(self):
    """Auto-connection (ssid=None): must look up ssid from NM."""
    wm = _make_wm(connections={"AutoNet": "/path/auto"})
    wm._wifi_state = WifiState()
    wm._get_active_wifi_connection = MagicMock(return_value=("/path/auto", {}))

    _fire(wm, NMDeviceState.PREPARE)

    assert wm._wifi_state.ssid == "AutoNet"
    assert wm._wifi_state.status == ConnectStatus.CONNECTING

  def test_auto_connect_dbus_fails(self):
    """Auto-connection but DBus returns None: ssid stays None, status CONNECTING."""
    wm = _make_wm()
    wm._wifi_state = WifiState()
    wm._get_active_wifi_connection = MagicMock(return_value=(None, None))

    _fire(wm, NMDeviceState.PREPARE)

    assert wm._wifi_state.ssid is None
    assert wm._wifi_state.status == ConnectStatus.CONNECTING


class TestNeedAuth:
  def test_wrong_password_fires_callback(self):
    """NEED_AUTH+SUPPLICANT_DISCONNECT from CONFIG = real wrong password."""
    wm = _make_wm()
    cb = MagicMock()
    wm._need_auth = [cb]
    wm._wifi_state = WifiState(ssid="SecNet", status=ConnectStatus.CONNECTING)

    _fire(wm, NMDeviceState.NEED_AUTH, NMDeviceStateReason.SUPPLICANT_DISCONNECT,
          prev_state=NMDeviceState.CONFIG)

    assert wm._wifi_state.status == ConnectStatus.DISCONNECTED
    assert len(wm._callback_queue) == 1
    wm._callback_queue[0]()
    cb.assert_called_once_with("SecNet")

  def test_failed_no_secrets_fires_callback(self):
    """FAILED+NO_SECRETS = wrong password (weak/gone network)."""
    wm = _make_wm()
    cb = MagicMock()
    wm._need_auth = [cb]
    wm._wifi_state = WifiState(ssid="WeakNet", status=ConnectStatus.CONNECTING)

    _fire(wm, NMDeviceState.FAILED, NMDeviceStateReason.NO_SECRETS)

    assert wm._wifi_state.status == ConnectStatus.DISCONNECTED
    assert len(wm._callback_queue) == 1
    wm._callback_queue[0]()
    cb.assert_called_once_with("WeakNet")

  def test_no_ssid_no_callback(self):
    """If ssid is None when NEED_AUTH fires, no callback enqueued."""
    wm = _make_wm()
    cb = MagicMock()
    wm._need_auth = [cb]
    wm._wifi_state = WifiState()

    _fire(wm, NMDeviceState.NEED_AUTH, NMDeviceStateReason.SUPPLICANT_DISCONNECT)

    assert len(wm._callback_queue) == 0

  @pytest.mark.xfail(reason="TODO: interrupted auth (prev=DISCONNECTED) should be ignored")
  def test_interrupted_auth_ignored(self):
    """Switching A->B: NEED_AUTH from A (prev=DISCONNECTED) must not fire callback."""
    wm = _make_wm()
    cb = MagicMock()
    wm._need_auth = [cb]
    wm._set_connecting("A")
    wm._set_connecting("B")

    _fire(wm, NMDeviceState.NEED_AUTH, NMDeviceStateReason.SUPPLICANT_DISCONNECT,
          prev_state=NMDeviceState.DISCONNECTED)

    assert wm._wifi_state.ssid == "B"
    assert wm._wifi_state.status == ConnectStatus.CONNECTING
    assert len(wm._callback_queue) == 0


class TestPassthroughStates:
  """NEED_AUTH (generic), IP_CONFIG, IP_CHECK, SECONDARIES, FAILED (generic) are no-ops."""

  @pytest.mark.parametrize("state", [
    NMDeviceState.NEED_AUTH,
    NMDeviceState.IP_CONFIG,
    NMDeviceState.IP_CHECK,
    NMDeviceState.SECONDARIES,
    NMDeviceState.FAILED,
  ])
  def test_passthrough_is_noop(self, state):
    wm = _make_wm()
    wm._wifi_state = WifiState(ssid="Net", status=ConnectStatus.CONNECTING)

    _fire(wm, state, NMDeviceStateReason.NONE)

    assert wm._wifi_state.ssid == "Net"
    assert wm._wifi_state.status == ConnectStatus.CONNECTING


class TestActivated:
  def test_sets_connected(self):
    """ACTIVATED sets status to CONNECTED and fires callback."""
    wm = _make_wm(connections={"MyNet": "/path/mynet"})
    cb = MagicMock()
    wm._activated = [cb]
    wm._wifi_state = WifiState(ssid="MyNet", status=ConnectStatus.CONNECTING)
    wm._get_active_wifi_connection = MagicMock(return_value=("/path/mynet", {}))

    _fire(wm, NMDeviceState.ACTIVATED)

    assert wm._wifi_state.status == ConnectStatus.CONNECTED
    assert wm._wifi_state.ssid == "MyNet"
    assert len(wm._callback_queue) == 1

  def test_conn_path_none_still_connected(self):
    """ACTIVATED but DBus returns None: status CONNECTED, ssid unchanged."""
    wm = _make_wm()
    wm._wifi_state = WifiState(ssid="MyNet", status=ConnectStatus.CONNECTING)
    wm._get_active_wifi_connection = MagicMock(return_value=(None, None))

    _fire(wm, NMDeviceState.ACTIVATED)

    assert wm._wifi_state.status == ConnectStatus.CONNECTED
    assert wm._wifi_state.ssid == "MyNet"


# ---------------------------------------------------------------------------
# Full sequences (NM signal order from real devices)
# ---------------------------------------------------------------------------

class TestFullSequences:
  def test_normal_connect(self):
    """User connects to saved network: full happy path."""
    wm = _make_wm(connections={"Home": "/path/home"})
    wm._get_active_wifi_connection = MagicMock(return_value=("/path/home", {}))

    wm._set_connecting("Home")
    _fire(wm, NMDeviceState.PREPARE)
    assert wm._wifi_state.status == ConnectStatus.CONNECTING

    _fire(wm, NMDeviceState.CONFIG)
    assert wm._wifi_state.status == ConnectStatus.CONNECTING

    _fire(wm, NMDeviceState.IP_CONFIG)
    _fire(wm, NMDeviceState.IP_CHECK)
    _fire(wm, NMDeviceState.SECONDARIES)
    _fire(wm, NMDeviceState.ACTIVATED)

    assert wm._wifi_state.status == ConnectStatus.CONNECTED
    assert wm._wifi_state.ssid == "Home"

  def test_wrong_password_then_retry(self):
    """Wrong password → NEED_AUTH → user retries → success."""
    wm = _make_wm(connections={"Sec": "/path/sec"})
    cb = MagicMock()
    wm._need_auth = [cb]

    wm._set_connecting("Sec")
    _fire(wm, NMDeviceState.PREPARE)
    _fire(wm, NMDeviceState.CONFIG)

    _fire(wm, NMDeviceState.NEED_AUTH, NMDeviceStateReason.SUPPLICANT_DISCONNECT,
          prev_state=NMDeviceState.CONFIG)
    assert wm._wifi_state.status == ConnectStatus.DISCONNECTED
    assert len(wm._callback_queue) == 1

    wm._callback_queue.clear()
    wm._set_connecting("Sec")
    wm._get_active_wifi_connection = MagicMock(return_value=("/path/sec", {}))
    _fire(wm, NMDeviceState.PREPARE)
    _fire(wm, NMDeviceState.CONFIG)
    _fire(wm, NMDeviceState.ACTIVATED)
    assert wm._wifi_state.status == ConnectStatus.CONNECTED

  def test_switch_saved_networks(self):
    """Switch from A to B (both saved): NM signal sequence from real device."""
    wm = _make_wm(connections={"A": "/path/A", "B": "/path/B"})
    wm._wifi_state = WifiState(ssid="A", status=ConnectStatus.CONNECTED)
    wm._get_active_wifi_connection = MagicMock(return_value=("/path/B", {}))

    wm._set_connecting("B")

    _fire(wm, NMDeviceState.DEACTIVATING, NMDeviceStateReason.NEW_ACTIVATION)
    _fire(wm, NMDeviceState.DISCONNECTED, NMDeviceStateReason.NEW_ACTIVATION)
    assert wm._wifi_state.ssid == "B"
    assert wm._wifi_state.status == ConnectStatus.CONNECTING

    _fire(wm, NMDeviceState.PREPARE)
    _fire(wm, NMDeviceState.CONFIG)
    _fire(wm, NMDeviceState.ACTIVATED)
    assert wm._wifi_state.status == ConnectStatus.CONNECTED
    assert wm._wifi_state.ssid == "B"

  @pytest.mark.xfail(reason="TODO: interrupted auth from switching should not fire need_auth")
  def test_rapid_switch_no_false_wrong_password(self):
    """Switch A→B quickly: A's interrupted NEED_AUTH must NOT show wrong password.

    Real NM signal sequence observed on device:
      DEACTIVATING (NEW_ACTIVATION)
      DISCONNECTED (NEW_ACTIVATION)
      NEED_AUTH (SUPPLICANT_DISCONNECT)  ← A's interrupted auth
      PREPARE → CONFIG → ... → ACTIVATED  ← B connects
    """
    wm = _make_wm(connections={"A": "/path/A", "B": "/path/B"})
    cb = MagicMock()
    wm._need_auth = [cb]
    wm._wifi_state = WifiState(ssid="A", status=ConnectStatus.CONNECTED)
    wm._get_active_wifi_connection = MagicMock(return_value=("/path/B", {}))

    wm._set_connecting("B")

    _fire(wm, NMDeviceState.DEACTIVATING, NMDeviceStateReason.NEW_ACTIVATION)
    _fire(wm, NMDeviceState.DISCONNECTED, NMDeviceStateReason.NEW_ACTIVATION)
    _fire(wm, NMDeviceState.NEED_AUTH, NMDeviceStateReason.SUPPLICANT_DISCONNECT,
          prev_state=NMDeviceState.DISCONNECTED)

    # A's interrupted auth must not fire callback or clear B's connecting state
    assert wm._wifi_state.ssid == "B"
    assert wm._wifi_state.status == ConnectStatus.CONNECTING
    assert len(wm._callback_queue) == 0

    _fire(wm, NMDeviceState.PREPARE)
    _fire(wm, NMDeviceState.CONFIG)
    _fire(wm, NMDeviceState.ACTIVATED)
    assert wm._wifi_state.status == ConnectStatus.CONNECTED

  @pytest.mark.xfail(reason="TODO: forget A while connecting to B should not clear B")
  def test_forget_A_connect_B(self):
    """Forget A while connecting to B: full signal sequence.

    Signal order:
      1. User: _set_connecting("B"), forget("A") removes A from _connections
      2. NewConnection for B arrives → _connections["B"] = ...
      3. DEACTIVATING(CONNECTION_REMOVED) — should be no-op
      4. DISCONNECTED(CONNECTION_REMOVED) — B is in _connections, must not clear
      5. PREPARE → CONFIG → ACTIVATED
    """
    wm = _make_wm(connections={"A": "/path/A"})
    wm._wifi_state = WifiState(ssid="A", status=ConnectStatus.CONNECTED)

    wm._set_connecting("B")
    del wm._connections["A"]
    wm._connections["B"] = "/path/B"

    _fire(wm, NMDeviceState.DEACTIVATING, NMDeviceStateReason.CONNECTION_REMOVED)
    assert wm._wifi_state.ssid == "B"
    assert wm._wifi_state.status == ConnectStatus.CONNECTING

    _fire(wm, NMDeviceState.DISCONNECTED, NMDeviceStateReason.CONNECTION_REMOVED)
    assert wm._wifi_state.ssid == "B"
    assert wm._wifi_state.status == ConnectStatus.CONNECTING

    wm._get_active_wifi_connection = MagicMock(return_value=("/path/B", {}))
    _fire(wm, NMDeviceState.PREPARE)
    _fire(wm, NMDeviceState.CONFIG)
    _fire(wm, NMDeviceState.ACTIVATED)
    assert wm._wifi_state.status == ConnectStatus.CONNECTED
    assert wm._wifi_state.ssid == "B"

  def test_auto_connect(self):
    """NM auto-connects (no user action, ssid starts None)."""
    wm = _make_wm(connections={"AutoNet": "/path/auto"})
    wm._get_active_wifi_connection = MagicMock(return_value=("/path/auto", {}))

    _fire(wm, NMDeviceState.PREPARE)
    assert wm._wifi_state.ssid == "AutoNet"
    assert wm._wifi_state.status == ConnectStatus.CONNECTING

    _fire(wm, NMDeviceState.CONFIG)
    _fire(wm, NMDeviceState.ACTIVATED)
    assert wm._wifi_state.status == ConnectStatus.CONNECTED
    assert wm._wifi_state.ssid == "AutoNet"
