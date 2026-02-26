"""Tests for WifiManager._handle_state_change.

Tests the state machine in isolation by constructing a WifiManager with mocked
DBus, then calling _handle_state_change directly with NM state transitions.
"""
import pytest
from jeepney.low_level import MessageType
from pytest_mock import MockerFixture

from openpilot.system.ui.lib.networkmanager import NMDeviceState, NMDeviceStateReason
from openpilot.system.ui.lib.wifi_manager import WifiManager, WifiState, ConnectStatus


def _make_wm(mocker: MockerFixture, connections=None):
  """Create a WifiManager with only the fields _handle_state_change touches."""
  mocker.patch.object(WifiManager, '_initialize')
  wm = WifiManager.__new__(WifiManager)
  wm._exit = True  # prevent stop() from doing anything in __del__
  wm._conn_monitor = mocker.MagicMock()
  wm._connections = dict(connections or {})
  wm._wifi_state = WifiState()
  wm._user_epoch = 0
  wm._callback_queue = []
  wm._need_auth = []
  wm._activated = []
  wm._update_networks = mocker.MagicMock()
  wm._update_active_connection_info = mocker.MagicMock()
  wm._get_active_wifi_connection = mocker.MagicMock(return_value=(None, None))
  return wm


def fire(wm: WifiManager, new_state: int, prev_state: int = NMDeviceState.UNKNOWN,
         reason: int = NMDeviceStateReason.NONE) -> None:
  """Feed a state change into the handler."""
  wm._handle_state_change(new_state, prev_state, reason)


def fire_wpa_connect(wm: WifiManager) -> None:
  """WPA handshake then IP negotiation through ACTIVATED, as seen on device."""
  fire(wm, NMDeviceState.NEED_AUTH)
  fire(wm, NMDeviceState.PREPARE, prev_state=NMDeviceState.NEED_AUTH)
  fire(wm, NMDeviceState.CONFIG)
  fire(wm, NMDeviceState.IP_CONFIG)
  fire(wm, NMDeviceState.IP_CHECK)
  fire(wm, NMDeviceState.SECONDARIES)
  fire(wm, NMDeviceState.ACTIVATED)


# ---------------------------------------------------------------------------
# Basic transitions
# ---------------------------------------------------------------------------

class TestDisconnected:
  def test_generic_disconnect_clears_state(self, mocker):
    wm = _make_wm(mocker)
    wm._wifi_state = WifiState(ssid="Net", status=ConnectStatus.CONNECTED)

    fire(wm, NMDeviceState.DISCONNECTED, reason=NMDeviceStateReason.UNKNOWN)

    assert wm._wifi_state.ssid is None
    assert wm._wifi_state.status == ConnectStatus.DISCONNECTED
    wm._update_networks.assert_not_called()

  def test_new_activation_is_noop(self, mocker):
    """NEW_ACTIVATION means NM is about to connect to another network — don't clear."""
    wm = _make_wm(mocker)
    wm._wifi_state = WifiState(ssid="OldNet", status=ConnectStatus.CONNECTED)

    fire(wm, NMDeviceState.DISCONNECTED, reason=NMDeviceStateReason.NEW_ACTIVATION)

    assert wm._wifi_state.ssid == "OldNet"
    assert wm._wifi_state.status == ConnectStatus.CONNECTED

  def test_connection_removed_keeps_other_connecting(self, mocker):
    """Forget A while connecting to B: CONNECTION_REMOVED for A must not clear B."""
    wm = _make_wm(mocker, connections={"B": "/path/B"})
    wm._set_connecting("B")

    fire(wm, NMDeviceState.DISCONNECTED, reason=NMDeviceStateReason.CONNECTION_REMOVED)

    assert wm._wifi_state.ssid == "B"
    assert wm._wifi_state.status == ConnectStatus.CONNECTING

  def test_connection_removed_clears_when_forgotten(self, mocker):
    """Forget A: A is no longer in _connections, so state should clear."""
    wm = _make_wm(mocker, connections={})
    wm._wifi_state = WifiState(ssid="A", status=ConnectStatus.CONNECTED)

    fire(wm, NMDeviceState.DISCONNECTED, reason=NMDeviceStateReason.CONNECTION_REMOVED)

    assert wm._wifi_state.ssid is None
    assert wm._wifi_state.status == ConnectStatus.DISCONNECTED


class TestDeactivating:
  def test_deactivating_noop_for_non_connection_removed(self, mocker):
    """DEACTIVATING with non-CONNECTION_REMOVED reason is a no-op."""
    wm = _make_wm(mocker)
    wm._wifi_state = WifiState(ssid="Net", status=ConnectStatus.CONNECTED)

    fire(wm, NMDeviceState.DEACTIVATING, reason=NMDeviceStateReason.USER_REQUESTED)

    assert wm._wifi_state.ssid == "Net"
    assert wm._wifi_state.status == ConnectStatus.CONNECTED

  @pytest.mark.parametrize("status, expected_clears", [
    (ConnectStatus.CONNECTED, True),
    (ConnectStatus.CONNECTING, False),
  ])
  def test_deactivating_connection_removed(self, mocker, status, expected_clears):
    """DEACTIVATING(CONNECTION_REMOVED) clears CONNECTED but preserves CONNECTING.

    CONNECTED: forgetting the current network. The forgotten callback fires between
    DEACTIVATING and DISCONNECTED — must clear here so the UI doesn't flash "connected"
    after the eager _network_forgetting flag resets.

    CONNECTING: forget A while connecting to B. DEACTIVATING fires for A's removal,
    but B's CONNECTING state must be preserved.
    """
    wm = _make_wm(mocker, connections={"B": "/path/B"})
    wm._wifi_state = WifiState(ssid="B" if status == ConnectStatus.CONNECTING else "A", status=status)

    fire(wm, NMDeviceState.DEACTIVATING, reason=NMDeviceStateReason.CONNECTION_REMOVED)

    if expected_clears:
      assert wm._wifi_state.ssid is None
      assert wm._wifi_state.status == ConnectStatus.DISCONNECTED
    else:
      assert wm._wifi_state.ssid == "B"
      assert wm._wifi_state.status == ConnectStatus.CONNECTING


class TestPrepareConfig:
  def test_user_initiated_skips_dbus_lookup(self, mocker):
    """User called _set_connecting('B') — PREPARE must not overwrite via DBus.

    Reproduced on device: rapidly tap A then B. PREPARE's DBus lookup returns A's
    stale conn_path, overwriting ssid to A for 1-2 frames. UI shows the "connecting"
    indicator briefly jump to the wrong network row then back.
    """
    wm = _make_wm(mocker, connections={"A": "/path/A", "B": "/path/B"})
    wm._set_connecting("B")
    wm._get_active_wifi_connection.return_value = ("/path/A", {})

    fire(wm, NMDeviceState.PREPARE)

    assert wm._wifi_state.ssid == "B"
    assert wm._wifi_state.status == ConnectStatus.CONNECTING
    wm._get_active_wifi_connection.assert_not_called()

  @pytest.mark.parametrize("state", [NMDeviceState.PREPARE, NMDeviceState.CONFIG])
  def test_auto_connect_looks_up_ssid(self, mocker, state):
    """Auto-connection (ssid=None): PREPARE and CONFIG must look up ssid from NM."""
    wm = _make_wm(mocker, connections={"AutoNet": "/path/auto"})
    wm._get_active_wifi_connection.return_value = ("/path/auto", {})

    fire(wm, state)

    assert wm._wifi_state.ssid == "AutoNet"
    assert wm._wifi_state.status == ConnectStatus.CONNECTING

  def test_auto_connect_dbus_fails(self, mocker):
    """Auto-connection but DBus returns None: ssid stays None, status CONNECTING."""
    wm = _make_wm(mocker)

    fire(wm, NMDeviceState.PREPARE)

    assert wm._wifi_state.ssid is None
    assert wm._wifi_state.status == ConnectStatus.CONNECTING

  def test_auto_connect_conn_path_not_in_connections(self, mocker):
    """DBus returns a conn_path that doesn't match any known connection."""
    wm = _make_wm(mocker, connections={"Other": "/path/other"})
    wm._get_active_wifi_connection.return_value = ("/path/unknown", {})

    fire(wm, NMDeviceState.PREPARE)

    assert wm._wifi_state.ssid is None
    assert wm._wifi_state.status == ConnectStatus.CONNECTING


class TestNeedAuth:
  def test_wrong_password_fires_callback(self, mocker):
    """NEED_AUTH+SUPPLICANT_DISCONNECT from CONFIG = real wrong password."""
    wm = _make_wm(mocker)
    cb = mocker.MagicMock()
    wm.add_callbacks(need_auth=cb)
    wm._set_connecting("SecNet")

    fire(wm, NMDeviceState.NEED_AUTH, prev_state=NMDeviceState.CONFIG,
         reason=NMDeviceStateReason.SUPPLICANT_DISCONNECT)

    assert wm._wifi_state.status == ConnectStatus.DISCONNECTED
    assert len(wm._callback_queue) == 1
    wm.process_callbacks()
    cb.assert_called_once_with("SecNet")

  def test_failed_no_secrets_fires_callback(self, mocker):
    """FAILED+NO_SECRETS = wrong password (weak/gone network).

    Confirmed on device: also fires when a hotspot turns off during connection.
    NM can't complete the WPA handshake (AP vanished) and reports NO_SECRETS
    rather than SSID_NOT_FOUND. The need_auth callback fires, so the UI shows
    "wrong password" — a false positive, but same signal path.

    Real device sequence (new connection, hotspot turned off immediately):
      PREPARE → CONFIG → NEED_AUTH(CONFIG, NONE) → PREPARE(NEED_AUTH) → CONFIG
      → NEED_AUTH(CONFIG, NONE) → FAILED(NEED_AUTH, NO_SECRETS) → DISCONNECTED(FAILED, NONE)
    """
    wm = _make_wm(mocker)
    cb = mocker.MagicMock()
    wm.add_callbacks(need_auth=cb)
    wm._set_connecting("WeakNet")

    fire(wm, NMDeviceState.FAILED, reason=NMDeviceStateReason.NO_SECRETS)

    assert wm._wifi_state.status == ConnectStatus.DISCONNECTED
    assert len(wm._callback_queue) == 1
    wm.process_callbacks()
    cb.assert_called_once_with("WeakNet")

  def test_need_auth_then_failed_no_double_fire(self, mocker):
    """Real device sends NEED_AUTH(SUPPLICANT_DISCONNECT) then FAILED(NO_SECRETS) back-to-back.

    The first clears ssid, so the second must not fire a duplicate callback.
    Real device sequence: NEED_AUTH(CONFIG, SUPPLICANT_DISCONNECT) → FAILED(NEED_AUTH, NO_SECRETS)
    """
    wm = _make_wm(mocker)
    cb = mocker.MagicMock()
    wm.add_callbacks(need_auth=cb)
    wm._set_connecting("BadPass")

    fire(wm, NMDeviceState.NEED_AUTH, prev_state=NMDeviceState.CONFIG,
         reason=NMDeviceStateReason.SUPPLICANT_DISCONNECT)
    assert len(wm._callback_queue) == 1

    fire(wm, NMDeviceState.FAILED, prev_state=NMDeviceState.NEED_AUTH,
         reason=NMDeviceStateReason.NO_SECRETS)
    assert len(wm._callback_queue) == 1  # no duplicate

    wm.process_callbacks()
    cb.assert_called_once_with("BadPass")

  def test_no_ssid_no_callback(self, mocker):
    """If ssid is None when NEED_AUTH fires, no callback enqueued."""
    wm = _make_wm(mocker)
    cb = mocker.MagicMock()
    wm.add_callbacks(need_auth=cb)

    fire(wm, NMDeviceState.NEED_AUTH, reason=NMDeviceStateReason.SUPPLICANT_DISCONNECT)

    assert len(wm._callback_queue) == 0

  def test_interrupted_auth_ignored(self, mocker):
    """Switching A->B: NEED_AUTH from A (prev=DISCONNECTED) must not fire callback.

    Reproduced on device: rapidly switching between two saved networks can trigger a
    rare false "wrong password" dialog for the previous network, even though both have
    correct passwords. The stale NEED_AUTH has prev_state=DISCONNECTED (not CONFIG).
    """
    wm = _make_wm(mocker)
    cb = mocker.MagicMock()
    wm.add_callbacks(need_auth=cb)
    wm._set_connecting("A")
    wm._set_connecting("B")

    fire(wm, NMDeviceState.NEED_AUTH, prev_state=NMDeviceState.DISCONNECTED,
         reason=NMDeviceStateReason.SUPPLICANT_DISCONNECT)

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
  def test_passthrough_is_noop(self, mocker, state):
    wm = _make_wm(mocker)
    wm._set_connecting("Net")

    fire(wm, state, reason=NMDeviceStateReason.NONE)

    assert wm._wifi_state.ssid == "Net"
    assert wm._wifi_state.status == ConnectStatus.CONNECTING
    assert len(wm._callback_queue) == 0


class TestActivated:
  def test_sets_connected(self, mocker):
    """ACTIVATED sets status to CONNECTED and fires callback."""
    wm = _make_wm(mocker, connections={"MyNet": "/path/mynet"})
    cb = mocker.MagicMock()
    wm.add_callbacks(activated=cb)
    wm._set_connecting("MyNet")
    wm._get_active_wifi_connection.return_value = ("/path/mynet", {})

    fire(wm, NMDeviceState.ACTIVATED)

    assert wm._wifi_state.status == ConnectStatus.CONNECTED
    assert wm._wifi_state.ssid == "MyNet"
    assert len(wm._callback_queue) == 1
    wm.process_callbacks()
    cb.assert_called_once()

  def test_conn_path_none_still_connected(self, mocker):
    """ACTIVATED but DBus returns None: status CONNECTED, ssid unchanged."""
    wm = _make_wm(mocker)
    wm._set_connecting("MyNet")

    fire(wm, NMDeviceState.ACTIVATED)

    assert wm._wifi_state.status == ConnectStatus.CONNECTED
    assert wm._wifi_state.ssid == "MyNet"

  def test_activated_side_effects(self, mocker):
    """ACTIVATED persists the volatile connection to disk and updates active connection info."""
    wm = _make_wm(mocker, connections={"Net": "/path/net"})
    wm._set_connecting("Net")
    wm._get_active_wifi_connection.return_value = ("/path/net", {})

    fire(wm, NMDeviceState.ACTIVATED)

    wm._conn_monitor.send_and_get_reply.assert_called_once()
    wm._update_active_connection_info.assert_called_once()
    wm._update_networks.assert_not_called()


# ---------------------------------------------------------------------------
# Thread races: _set_connecting on main thread vs _handle_state_change on monitor thread.
# Uses side_effect on the DBus mock to simulate _set_connecting running mid-handler.
# The epoch counter detects that a user action occurred during the slow DBus call
# and discards the stale update.
# ---------------------------------------------------------------------------
# The deterministic fixes (skip DBus lookup when ssid already set, prev_state guard
# on NEED_AUTH, DEACTIVATING clears CONNECTED on CONNECTION_REMOVED, CONNECTION_REMOVED
# guard) shrink these race windows significantly. The epoch counter closes the
# remaining gaps.

class TestThreadRaces:
  def test_prepare_race_user_tap_during_dbus(self, mocker):
    """User taps B while PREPARE's DBus call is in flight for auto-connect.

    Monitor thread reads wifi_state (ssid=None), starts DBus call.
    Main thread: _set_connecting("B"). Monitor thread writes back stale ssid from DBus.
    """
    wm = _make_wm(mocker, connections={"A": "/path/A", "B": "/path/B"})

    def user_taps_b_during_dbus(*args, **kwargs):
      wm._set_connecting("B")
      return ("/path/A", {})

    wm._get_active_wifi_connection.side_effect = user_taps_b_during_dbus

    fire(wm, NMDeviceState.PREPARE)

    assert wm._wifi_state.ssid == "B"
    assert wm._wifi_state.status == ConnectStatus.CONNECTING

  def test_activated_race_user_tap_during_dbus(self, mocker):
    """User taps B right as A finishes connecting (ACTIVATED handler running).

    Monitor thread reads wifi_state (A, CONNECTING), starts DBus call.
    Main thread: _set_connecting("B"). Monitor thread writes (A, CONNECTED), losing B.
    """
    wm = _make_wm(mocker, connections={"A": "/path/A", "B": "/path/B"})
    wm._set_connecting("A")

    def user_taps_b_during_dbus(*args, **kwargs):
      wm._set_connecting("B")
      return ("/path/A", {})

    wm._get_active_wifi_connection.side_effect = user_taps_b_during_dbus

    fire(wm, NMDeviceState.ACTIVATED)

    assert wm._wifi_state.ssid == "B"
    assert wm._wifi_state.status == ConnectStatus.CONNECTING

  def test_init_wifi_state_race_user_tap_during_dbus(self, mocker):
    """User taps B while _init_wifi_state's DBus calls are in flight.

    _init_wifi_state runs from set_active(True) or worker error paths. It does
    2 DBus calls (device State property + _get_active_wifi_connection) then
    unconditionally writes _wifi_state. If the user taps a network during those
    calls, _set_connecting("B") is overwritten with stale NM ground truth.
    """
    wm = _make_wm(mocker, connections={"A": "/path/A", "B": "/path/B"})
    wm._wifi_device = "/dev/wifi0"
    wm._router_main = mocker.MagicMock()

    state_reply = mocker.MagicMock()
    state_reply.body = [('u', NMDeviceState.ACTIVATED)]
    wm._router_main.send_and_get_reply.return_value = state_reply

    def user_taps_b_during_dbus(*args, **kwargs):
      wm._set_connecting("B")
      return ("/path/A", {})

    wm._get_active_wifi_connection.side_effect = user_taps_b_during_dbus

    wm._init_wifi_state()

    assert wm._wifi_state.ssid == "B"
    assert wm._wifi_state.status == ConnectStatus.CONNECTING


# ---------------------------------------------------------------------------
# Full sequences (NM signal order from real devices)
# ---------------------------------------------------------------------------

class TestFullSequences:
  def test_normal_connect(self, mocker):
    """User connects to saved network: full happy path.

    Real device sequence (switching from another connected network):
      DEACTIVATING(ACTIVATED, NEW_ACTIVATION) → DISCONNECTED(DEACTIVATING, NEW_ACTIVATION)
      PREPARE → CONFIG → NEED_AUTH(CONFIG, NONE) → PREPARE(NEED_AUTH, NONE) → CONFIG
      → IP_CONFIG → IP_CHECK → SECONDARIES → ACTIVATED
    """
    wm = _make_wm(mocker, connections={"Home": "/path/home"})
    wm._get_active_wifi_connection.return_value = ("/path/home", {})

    wm._set_connecting("Home")
    fire(wm, NMDeviceState.PREPARE)
    fire(wm, NMDeviceState.CONFIG)
    fire(wm, NMDeviceState.NEED_AUTH)  # WPA handshake (reason=NONE)
    fire(wm, NMDeviceState.PREPARE, prev_state=NMDeviceState.NEED_AUTH)
    fire(wm, NMDeviceState.CONFIG)
    assert wm._wifi_state.status == ConnectStatus.CONNECTING

    fire(wm, NMDeviceState.IP_CONFIG)
    fire(wm, NMDeviceState.IP_CHECK)
    fire(wm, NMDeviceState.SECONDARIES)
    fire(wm, NMDeviceState.ACTIVATED)

    assert wm._wifi_state.status == ConnectStatus.CONNECTED
    assert wm._wifi_state.ssid == "Home"

  def test_wrong_password_then_retry(self, mocker):
    """Wrong password → NEED_AUTH → FAILED → NM auto-reconnects to saved network.

    Confirmed on device: wrong password for Shane's iPhone, NM auto-connected to unifi.

    Real device sequence (switching from a connected network):
      DEACTIVATING(ACTIVATED, NEW_ACTIVATION) → DISCONNECTED(DEACTIVATING, NEW_ACTIVATION)
      → PREPARE → CONFIG → NEED_AUTH(CONFIG, NONE)              ← WPA handshake
      → PREPARE(NEED_AUTH, NONE) → CONFIG
      → NEED_AUTH(CONFIG, SUPPLICANT_DISCONNECT)                 ← wrong password
      → FAILED(NEED_AUTH, NO_SECRETS)                            ← NM gives up
      → DISCONNECTED(FAILED, NONE)
      → PREPARE → CONFIG → NEED_AUTH(CONFIG, NONE) → PREPARE(NEED_AUTH) → CONFIG
      → IP_CONFIG → IP_CHECK → SECONDARIES → ACTIVATED          ← auto-reconnect to other saved network
    """
    wm = _make_wm(mocker, connections={"Sec": "/path/sec"})
    cb = mocker.MagicMock()
    wm.add_callbacks(need_auth=cb)

    wm._set_connecting("Sec")
    fire(wm, NMDeviceState.PREPARE)
    fire(wm, NMDeviceState.CONFIG)
    fire(wm, NMDeviceState.NEED_AUTH)  # WPA handshake (reason=NONE)
    fire(wm, NMDeviceState.PREPARE, prev_state=NMDeviceState.NEED_AUTH)
    fire(wm, NMDeviceState.CONFIG)

    fire(wm, NMDeviceState.NEED_AUTH, prev_state=NMDeviceState.CONFIG,
         reason=NMDeviceStateReason.SUPPLICANT_DISCONNECT)
    assert wm._wifi_state.status == ConnectStatus.DISCONNECTED
    assert len(wm._callback_queue) == 1

    # FAILED(NO_SECRETS) follows but ssid is already cleared — no double-fire
    fire(wm, NMDeviceState.FAILED, reason=NMDeviceStateReason.NO_SECRETS)
    assert len(wm._callback_queue) == 1

    fire(wm, NMDeviceState.DISCONNECTED, prev_state=NMDeviceState.FAILED)

    # Retry
    wm._callback_queue.clear()
    wm._set_connecting("Sec")
    wm._get_active_wifi_connection.return_value = ("/path/sec", {})
    fire(wm, NMDeviceState.PREPARE)
    fire(wm, NMDeviceState.CONFIG)
    fire_wpa_connect(wm)
    assert wm._wifi_state.status == ConnectStatus.CONNECTED

  def test_switch_saved_networks(self, mocker):
    """Switch from A to B (both saved): NM signal sequence from real device.

    Real device sequence:
      DEACTIVATING(ACTIVATED, NEW_ACTIVATION) → DISCONNECTED(DEACTIVATING, NEW_ACTIVATION)
      → PREPARE → CONFIG → NEED_AUTH(CONFIG, NONE) → PREPARE(NEED_AUTH, NONE) → CONFIG
      → IP_CONFIG → IP_CHECK → SECONDARIES → ACTIVATED
    """
    wm = _make_wm(mocker, connections={"A": "/path/A", "B": "/path/B"})
    wm._wifi_state = WifiState(ssid="A", status=ConnectStatus.CONNECTED)
    wm._get_active_wifi_connection.return_value = ("/path/B", {})

    wm._set_connecting("B")

    fire(wm, NMDeviceState.DEACTIVATING, prev_state=NMDeviceState.ACTIVATED,
         reason=NMDeviceStateReason.NEW_ACTIVATION)
    fire(wm, NMDeviceState.DISCONNECTED, prev_state=NMDeviceState.DEACTIVATING,
         reason=NMDeviceStateReason.NEW_ACTIVATION)
    assert wm._wifi_state.ssid == "B"
    assert wm._wifi_state.status == ConnectStatus.CONNECTING

    fire(wm, NMDeviceState.PREPARE)
    fire(wm, NMDeviceState.CONFIG)
    fire_wpa_connect(wm)
    assert wm._wifi_state.status == ConnectStatus.CONNECTED
    assert wm._wifi_state.ssid == "B"

  def test_rapid_switch_no_false_wrong_password(self, mocker):
    """Switch A→B quickly: A's interrupted NEED_AUTH must NOT show wrong password.

    NOTE: The late NEED_AUTH(DISCONNECTED, SUPPLICANT_DISCONNECT) is common when rapidly
    switching between networks with wrong/new passwords. Less common when switching between
    saved networks with correct passwords. Not guaranteed — some switches skip it and go
    straight from DISCONNECTED to PREPARE. The prev_state is consistently DISCONNECTED
    for stale signals, so the prev_state guard reliably distinguishes them.

    Worst-case signal sequence this protects against:
      DEACTIVATING(NEW_ACTIVATION) → DISCONNECTED(NEW_ACTIVATION)
      → NEED_AUTH(DISCONNECTED, SUPPLICANT_DISCONNECT)  ← A's stale auth failure
      → PREPARE → CONFIG → ... → ACTIVATED  ← B connects
    """
    wm = _make_wm(mocker, connections={"A": "/path/A", "B": "/path/B"})
    cb = mocker.MagicMock()
    wm.add_callbacks(need_auth=cb)
    wm._wifi_state = WifiState(ssid="A", status=ConnectStatus.CONNECTED)
    wm._get_active_wifi_connection.return_value = ("/path/B", {})

    wm._set_connecting("B")

    fire(wm, NMDeviceState.DEACTIVATING, prev_state=NMDeviceState.ACTIVATED,
         reason=NMDeviceStateReason.NEW_ACTIVATION)
    fire(wm, NMDeviceState.DISCONNECTED, prev_state=NMDeviceState.DEACTIVATING,
         reason=NMDeviceStateReason.NEW_ACTIVATION)
    fire(wm, NMDeviceState.NEED_AUTH, prev_state=NMDeviceState.DISCONNECTED,
         reason=NMDeviceStateReason.SUPPLICANT_DISCONNECT)

    assert wm._wifi_state.ssid == "B"
    assert wm._wifi_state.status == ConnectStatus.CONNECTING
    assert len(wm._callback_queue) == 0

    fire(wm, NMDeviceState.PREPARE)
    fire(wm, NMDeviceState.CONFIG)
    fire_wpa_connect(wm)
    assert wm._wifi_state.status == ConnectStatus.CONNECTED

  def test_forget_while_connecting(self, mocker):
    """Forget the network we're currently connecting to (not yet ACTIVATED).

    Confirmed on device: connected to unifi, tapped Shane's iPhone, then forgot
    Shane's iPhone while at CONFIG. NM auto-connected to unifi afterward.

    Real device sequence (switching then forgetting mid-connection):
      DEACTIVATING(ACTIVATED, NEW_ACTIVATION) → DISCONNECTED(DEACTIVATING, NEW_ACTIVATION)
      → PREPARE → CONFIG → NEED_AUTH(CONFIG, NONE) → PREPARE(NEED_AUTH) → CONFIG
      → DEACTIVATING(CONFIG, CONNECTION_REMOVED)                ← forget at CONFIG
      → DISCONNECTED(DEACTIVATING, CONNECTION_REMOVED)
      → PREPARE → CONFIG → ... → ACTIVATED                     ← NM auto-connects to other saved network

    Note: DEACTIVATING fires from CONFIG (not ACTIVATED). wifi_state.status is
    CONNECTING, so the DEACTIVATING handler is a no-op. DISCONNECTED clears state
    (ssid removed from _connections by ConnectionRemoved), then PREPARE recovers
    via DBus lookup for the auto-connect.
    """
    wm = _make_wm(mocker, connections={"A": "/path/A", "Other": "/path/other"})
    wm._get_active_wifi_connection.return_value = ("/path/other", {})

    wm._set_connecting("A")

    fire(wm, NMDeviceState.PREPARE)
    fire(wm, NMDeviceState.CONFIG)
    assert wm._wifi_state.ssid == "A"
    assert wm._wifi_state.status == ConnectStatus.CONNECTING

    # User forgets A: ConnectionRemoved processed first, then state changes
    del wm._connections["A"]

    fire(wm, NMDeviceState.DEACTIVATING, prev_state=NMDeviceState.CONFIG,
         reason=NMDeviceStateReason.CONNECTION_REMOVED)
    assert wm._wifi_state.ssid == "A"
    assert wm._wifi_state.status == ConnectStatus.CONNECTING  # DEACTIVATING preserves CONNECTING

    fire(wm, NMDeviceState.DISCONNECTED, prev_state=NMDeviceState.DEACTIVATING,
         reason=NMDeviceStateReason.CONNECTION_REMOVED)
    assert wm._wifi_state.ssid is None
    assert wm._wifi_state.status == ConnectStatus.DISCONNECTED

    # NM auto-connects to another saved network
    fire(wm, NMDeviceState.PREPARE)
    assert wm._wifi_state.ssid == "Other"
    assert wm._wifi_state.status == ConnectStatus.CONNECTING

    fire(wm, NMDeviceState.CONFIG)
    fire_wpa_connect(wm)
    assert wm._wifi_state.status == ConnectStatus.CONNECTED
    assert wm._wifi_state.ssid == "Other"

  def test_forget_connected_network(self, mocker):
    """Forget the currently connected network (not switching to another).

    Real device sequence:
      DEACTIVATING(ACTIVATED, CONNECTION_REMOVED) → DISCONNECTED(DEACTIVATING, CONNECTION_REMOVED)

    ConnectionRemoved signal may or may not have been processed before state changes.
    Either way, state must clear — we're forgetting what we're connected to, not switching.
    """
    wm = _make_wm(mocker, connections={"A": "/path/A"})
    wm._wifi_state = WifiState(ssid="A", status=ConnectStatus.CONNECTED)

    fire(wm, NMDeviceState.DEACTIVATING, prev_state=NMDeviceState.ACTIVATED,
         reason=NMDeviceStateReason.CONNECTION_REMOVED)
    assert wm._wifi_state.ssid is None
    assert wm._wifi_state.status == ConnectStatus.DISCONNECTED

    # DISCONNECTED follows — harmless since state is already cleared
    fire(wm, NMDeviceState.DISCONNECTED, prev_state=NMDeviceState.DEACTIVATING,
         reason=NMDeviceStateReason.CONNECTION_REMOVED)
    assert wm._wifi_state.ssid is None
    assert wm._wifi_state.status == ConnectStatus.DISCONNECTED

  def test_forget_A_connect_B(self, mocker):
    """Forget A while connecting to B: full signal sequence.

    Real device sequence:
      DEACTIVATING(ACTIVATED, CONNECTION_REMOVED) → DISCONNECTED(DEACTIVATING, CONNECTION_REMOVED)
      → PREPARE → CONFIG → NEED_AUTH(CONFIG, NONE) → PREPARE(NEED_AUTH, NONE) → CONFIG
      → IP_CONFIG → IP_CHECK → SECONDARIES → ACTIVATED

    Signal order:
      1. User: _set_connecting("B"), forget("A") removes A from _connections
      2. NewConnection for B arrives → _connections["B"] = ...
      3. DEACTIVATING(CONNECTION_REMOVED) — no-op
      4. DISCONNECTED(CONNECTION_REMOVED) — B is in _connections, must not clear
      5. PREPARE → CONFIG → NEED_AUTH → PREPARE → CONFIG → ... → ACTIVATED
    """
    wm = _make_wm(mocker, connections={"A": "/path/A"})
    wm._wifi_state = WifiState(ssid="A", status=ConnectStatus.CONNECTED)

    wm._set_connecting("B")
    del wm._connections["A"]
    wm._connections["B"] = "/path/B"

    fire(wm, NMDeviceState.DEACTIVATING, prev_state=NMDeviceState.ACTIVATED,
         reason=NMDeviceStateReason.CONNECTION_REMOVED)
    assert wm._wifi_state.ssid == "B"
    assert wm._wifi_state.status == ConnectStatus.CONNECTING

    fire(wm, NMDeviceState.DISCONNECTED, prev_state=NMDeviceState.DEACTIVATING,
         reason=NMDeviceStateReason.CONNECTION_REMOVED)
    assert wm._wifi_state.ssid == "B"
    assert wm._wifi_state.status == ConnectStatus.CONNECTING

    wm._get_active_wifi_connection.return_value = ("/path/B", {})
    fire(wm, NMDeviceState.PREPARE)
    fire(wm, NMDeviceState.CONFIG)
    fire_wpa_connect(wm)
    assert wm._wifi_state.status == ConnectStatus.CONNECTED
    assert wm._wifi_state.ssid == "B"

  def test_forget_A_connect_B_late_new_connection(self, mocker):
    """Forget A, connect B: NewConnection for B arrives AFTER DISCONNECTED.

    This is the worst-case race: B isn't in _connections when DISCONNECTED fires,
    so the guard can't protect it and state clears. PREPARE must recover by doing
    the DBus lookup (ssid is None at that point).

    Signal order:
      1. User: _set_connecting("B"), forget("A") removes A from _connections
      2. DEACTIVATING(CONNECTION_REMOVED) — B NOT in _connections, should be no-op
      3. DISCONNECTED(CONNECTION_REMOVED) — B STILL NOT in _connections, clears state
      4. NewConnection for B arrives late → _connections["B"] = ...
      5. PREPARE (ssid=None, so DBus lookup recovers) → CONFIG → ACTIVATED
    """
    wm = _make_wm(mocker, connections={"A": "/path/A"})
    wm._wifi_state = WifiState(ssid="A", status=ConnectStatus.CONNECTED)

    wm._set_connecting("B")
    del wm._connections["A"]

    fire(wm, NMDeviceState.DEACTIVATING, prev_state=NMDeviceState.ACTIVATED,
         reason=NMDeviceStateReason.CONNECTION_REMOVED)
    assert wm._wifi_state.ssid == "B"
    assert wm._wifi_state.status == ConnectStatus.CONNECTING

    fire(wm, NMDeviceState.DISCONNECTED, prev_state=NMDeviceState.DEACTIVATING,
         reason=NMDeviceStateReason.CONNECTION_REMOVED)
    # B not in _connections yet, so state clears — this is the known edge case
    assert wm._wifi_state.ssid is None
    assert wm._wifi_state.status == ConnectStatus.DISCONNECTED

    # NewConnection arrives late
    wm._connections["B"] = "/path/B"
    wm._get_active_wifi_connection.return_value = ("/path/B", {})

    # PREPARE recovers: ssid is None so it looks up from DBus
    fire(wm, NMDeviceState.PREPARE)
    assert wm._wifi_state.ssid == "B"
    assert wm._wifi_state.status == ConnectStatus.CONNECTING

    fire(wm, NMDeviceState.CONFIG)
    fire_wpa_connect(wm)
    assert wm._wifi_state.status == ConnectStatus.CONNECTED
    assert wm._wifi_state.ssid == "B"

  def test_auto_connect(self, mocker):
    """NM auto-connects (no user action, ssid starts None)."""
    wm = _make_wm(mocker, connections={"AutoNet": "/path/auto"})
    wm._get_active_wifi_connection.return_value = ("/path/auto", {})

    fire(wm, NMDeviceState.PREPARE)
    assert wm._wifi_state.ssid == "AutoNet"
    assert wm._wifi_state.status == ConnectStatus.CONNECTING

    fire(wm, NMDeviceState.CONFIG)
    fire_wpa_connect(wm)
    assert wm._wifi_state.status == ConnectStatus.CONNECTED
    assert wm._wifi_state.ssid == "AutoNet"

  def test_network_lost_during_connection(self, mocker):
    """Hotspot turned off while connecting (before ACTIVATED).

    Confirmed on device: started new connection to Shane's iPhone, immediately
    turned off the hotspot. NM can't complete WPA handshake and reports
    FAILED(NO_SECRETS) — same signal as wrong password (false positive).

    Real device sequence:
      PREPARE → CONFIG → NEED_AUTH(CONFIG, NONE) → PREPARE(NEED_AUTH) → CONFIG
      → NEED_AUTH(CONFIG, NONE) → FAILED(NEED_AUTH, NO_SECRETS) → DISCONNECTED(FAILED, NONE)

    Note: no DEACTIVATING, no SUPPLICANT_DISCONNECT. The NEED_AUTH(CONFIG, NONE) is the
    normal WPA handshake (not an error). NM gives up with NO_SECRETS because the AP
    vanished mid-handshake.
    """
    wm = _make_wm(mocker, connections={"Hotspot": "/path/hs"})
    cb = mocker.MagicMock()
    wm.add_callbacks(need_auth=cb)

    wm._set_connecting("Hotspot")
    fire(wm, NMDeviceState.PREPARE)
    fire(wm, NMDeviceState.CONFIG)
    fire(wm, NMDeviceState.NEED_AUTH)  # WPA handshake (reason=NONE)
    fire(wm, NMDeviceState.PREPARE, prev_state=NMDeviceState.NEED_AUTH)
    fire(wm, NMDeviceState.CONFIG)
    assert wm._wifi_state.status == ConnectStatus.CONNECTING

    # Second NEED_AUTH(CONFIG, NONE) — NM retries handshake, AP vanishing
    fire(wm, NMDeviceState.NEED_AUTH)
    assert wm._wifi_state.status == ConnectStatus.CONNECTING

    # NM gives up — reports NO_SECRETS (same as wrong password)
    fire(wm, NMDeviceState.FAILED, prev_state=NMDeviceState.NEED_AUTH,
         reason=NMDeviceStateReason.NO_SECRETS)
    assert wm._wifi_state.status == ConnectStatus.DISCONNECTED
    assert len(wm._callback_queue) == 1

    fire(wm, NMDeviceState.DISCONNECTED, prev_state=NMDeviceState.FAILED)
    assert wm._wifi_state.ssid is None
    assert wm._wifi_state.status == ConnectStatus.DISCONNECTED

    wm.process_callbacks()
    cb.assert_called_once_with("Hotspot")

  @pytest.mark.xfail(reason="TODO: FAILED(SSID_NOT_FOUND) should emit error for UI")
  def test_ssid_not_found(self, mocker):
    """Network drops off while connected — hotspot turned off.

    NM docs: SSID_NOT_FOUND (53) = "The WiFi network could not be found"

    Confirmed on device: connected to Shane's iPhone, then turned off the hotspot.
    No DEACTIVATING fires — NM goes straight from ACTIVATED to FAILED(SSID_NOT_FOUND).
    NM retries connecting (PREPARE → CONFIG → ... → FAILED(CONFIG, SSID_NOT_FOUND))
    before finally giving up with DISCONNECTED.

    NOTE: turning off a hotspot during initial connection (before ACTIVATED) typically
    produces FAILED(NO_SECRETS) instead of SSID_NOT_FOUND (see test_failed_no_secrets).

    Real device sequence (hotspot turned off while connected):
      FAILED(ACTIVATED, SSID_NOT_FOUND) → DISCONNECTED(FAILED, NONE)
      → PREPARE → CONFIG → NEED_AUTH(CONFIG, NONE) → PREPARE(NEED_AUTH) → CONFIG
      → NEED_AUTH(CONFIG, NONE) → PREPARE(NEED_AUTH) → CONFIG
      → FAILED(CONFIG, SSID_NOT_FOUND) → DISCONNECTED(FAILED, NONE)

    The UI error callback mechanism is intentionally deferred — for now just clear state.
    """
    wm = _make_wm(mocker, connections={"GoneNet": "/path/gone"})
    cb = mocker.MagicMock()
    wm.add_callbacks(need_auth=cb)

    wm._set_connecting("GoneNet")
    fire(wm, NMDeviceState.PREPARE)
    fire(wm, NMDeviceState.CONFIG)
    fire(wm, NMDeviceState.FAILED, reason=NMDeviceStateReason.SSID_NOT_FOUND)

    assert wm._wifi_state.status == ConnectStatus.DISCONNECTED
    assert wm._wifi_state.ssid is None

  def test_failed_then_disconnected_clears_state(self, mocker):
    """After FAILED, NM always transitions to DISCONNECTED to clean up.

    NM docs: FAILED (120) = "failed to connect, cleaning up the connection request"
    Full sequence: ... → FAILED(reason) → DISCONNECTED(NONE)
    """
    wm = _make_wm(mocker)
    wm._set_connecting("Net")

    fire(wm, NMDeviceState.FAILED, reason=NMDeviceStateReason.NONE)
    assert wm._wifi_state.status == ConnectStatus.CONNECTING  # FAILED(NONE) is a no-op

    fire(wm, NMDeviceState.DISCONNECTED, reason=NMDeviceStateReason.NONE)
    assert wm._wifi_state.ssid is None
    assert wm._wifi_state.status == ConnectStatus.DISCONNECTED

  def test_user_requested_disconnect(self, mocker):
    """User explicitly disconnects from the network.

    NM docs: USER_REQUESTED (39) = "Device disconnected by user or client"
    Expected sequence: DEACTIVATING(USER_REQUESTED) → DISCONNECTED(USER_REQUESTED)
    """
    wm = _make_wm(mocker)
    wm._wifi_state = WifiState(ssid="MyNet", status=ConnectStatus.CONNECTED)

    fire(wm, NMDeviceState.DEACTIVATING, reason=NMDeviceStateReason.USER_REQUESTED)
    fire(wm, NMDeviceState.DISCONNECTED, reason=NMDeviceStateReason.USER_REQUESTED)

    assert wm._wifi_state.ssid is None
    assert wm._wifi_state.status == ConnectStatus.DISCONNECTED


# ---------------------------------------------------------------------------
# Worker error recovery: DBus errors in activate/connect re-sync with NM
# ---------------------------------------------------------------------------
# Verified on device: when ActivateConnection returns UnknownConnection error,
# NM emits no state signals. The worker error path is the only recovery point.

class TestWorkerErrorRecovery:
  """Worker threads re-sync with NM via _init_wifi_state on DBus errors,
  preserving actual NM state instead of blindly clearing to DISCONNECTED."""

  def _mock_init_restores(self, wm, mocker, ssid, status):
    """Replace _init_wifi_state with a mock that simulates NM reporting the given state."""
    mock = mocker.MagicMock(
      side_effect=lambda: setattr(wm, '_wifi_state', WifiState(ssid=ssid, status=status))
    )
    wm._init_wifi_state = mock
    return mock

  def test_activate_dbus_error_resyncs(self, mocker):
    """ActivateConnection returns DBus error while A is connected.
    NM rejects the request — no state signals emitted. Worker must re-read NM
    state to discover A is still connected, not clear to DISCONNECTED.
    """
    wm = _make_wm(mocker, connections={"A": "/path/A", "B": "/path/B"})
    wm._wifi_device = "/dev/wifi0"
    wm._nm = mocker.MagicMock()
    wm._wifi_state = WifiState(ssid="A", status=ConnectStatus.CONNECTED)
    wm._router_main = mocker.MagicMock()

    error_reply = mocker.MagicMock()
    error_reply.header.message_type = MessageType.error
    wm._router_main.send_and_get_reply.return_value = error_reply

    mock_init = self._mock_init_restores(wm, mocker, "A", ConnectStatus.CONNECTED)

    wm.activate_connection("B", block=True)

    mock_init.assert_called_once()
    assert wm._wifi_state.ssid == "A"
    assert wm._wifi_state.status == ConnectStatus.CONNECTED

  def test_connect_to_network_dbus_error_resyncs(self, mocker):
    """AddAndActivateConnection2 returns DBus error while A is connected."""
    wm = _make_wm(mocker, connections={"A": "/path/A"})
    wm._wifi_device = "/dev/wifi0"
    wm._nm = mocker.MagicMock()
    wm._wifi_state = WifiState(ssid="A", status=ConnectStatus.CONNECTED)
    wm._router_main = mocker.MagicMock()
    wm._forgotten = []

    error_reply = mocker.MagicMock()
    error_reply.header.message_type = MessageType.error
    wm._router_main.send_and_get_reply.return_value = error_reply

    mock_init = self._mock_init_restores(wm, mocker, "A", ConnectStatus.CONNECTED)

    # Run worker thread synchronously
    workers = []
    mocker.patch('openpilot.system.ui.lib.wifi_manager.threading.Thread',
                 side_effect=lambda target, **kw: type('T', (), {'start': lambda self: workers.append(target)})())

    wm.connect_to_network("B", "password123")
    workers[-1]()

    mock_init.assert_called_once()
    assert wm._wifi_state.ssid == "A"
    assert wm._wifi_state.status == ConnectStatus.CONNECTED
