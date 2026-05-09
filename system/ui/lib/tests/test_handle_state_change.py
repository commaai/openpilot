"""Tests for WifiManager wpa_supplicant event-based state machine.

Tests the state machine in isolation by constructing a WifiManager with mocked
wpa_supplicant, then calling _handle_event directly with wpa_supplicant events.
"""
import pytest

from openpilot.system.ui.lib import wifi_manager as wifi_manager_module
from openpilot.system.ui.lib.wifi_manager import (
  ConnectStatus,
  MeteredType,
  Network,
  PendingConnection,
  SecurityType,
  WifiManager,
  WifiState,
)


def fire(wm: WifiManager, event: str) -> None:
  """Feed a wpa_supplicant event into the handler."""
  wm._handle_event(event)


# ---------------------------------------------------------------------------
# Basic transitions
# ---------------------------------------------------------------------------

class TestConnected:
  def test_connected_sets_state(self, wm):
    wm._set_connecting("MyNet")
    wm._ctrl.request.return_value = "wpa_state=COMPLETED\nssid=MyNet\n"

    fire(wm, "CTRL-EVENT-CONNECTED - Connection to aa:bb:cc:dd:ee:ff completed [id=0 id_str=]")

    assert wm._wifi_state.status == ConnectStatus.CONNECTED
    assert wm._wifi_state.ssid == "MyNet"
    wm._dhcp.start.assert_called_once()

  def test_connected_fires_activated_callback(self, wm, mocker):
    cb = mocker.MagicMock()
    wm.add_callbacks(activated=cb)
    wm._set_connecting("Net")
    wm._ctrl.request.return_value = "wpa_state=COMPLETED\nssid=Net\n"

    fire(wm, "CTRL-EVENT-CONNECTED - Connection to aa:bb:cc:dd:ee:ff completed [id=0]")

    wm.process_callbacks()
    cb.assert_called_once()

  def test_connected_persists_pending_connection(self, wm, mocker):
    wm._set_connecting("MyNet")
    wm._set_pending_connection("MyNet", "pass1234", False)
    wm._ctrl.request.return_value = "wpa_state=COMPLETED\nssid=MyNet\n"
    mocker.patch.object(wifi_manager_module, "_generate_wpa_conf")

    fire(wm, "CTRL-EVENT-CONNECTED - Connection to aa:bb:cc:dd:ee:ff completed [id=0 id_str=]")

    wm._store.save_network.assert_called_once_with("MyNet", psk="pass1234", hidden=False)
    assert wm._pending_connection is None

  def test_handle_connected_is_idempotent(self, wm, mocker):
    """The scanner's reconcile loop and the monitor thread can both call
    _handle_connected for the same transition. The second call must not
    restart DHCP or fire another activated callback."""
    cb = mocker.MagicMock()
    wm.add_callbacks(activated=cb)

    wm._handle_connected("MyNet")
    # Simulate the second caller arriving after state is already CONNECTED.
    wm._handle_connected("MyNet")

    wm.process_callbacks()
    wm._dhcp.start.assert_called_once()
    cb.assert_called_once()

  def test_handle_connected_re_fires_on_ssid_change(self, wm, mocker):
    """Switching networks — second _handle_connected with a different ssid
    must still transition (not treated as a dup)."""
    cb = mocker.MagicMock()
    wm.add_callbacks(activated=cb)

    wm._handle_connected("First")
    wm._handle_connected("Second")

    wm.process_callbacks()
    assert wm._wifi_state.ssid == "Second"
    assert wm._dhcp.start.call_count == 2
    assert cb.call_count == 2

  def test_handle_connected_enables_all_networks_for_auto_roam(self, wm):
    """SELECT_NETWORK disables every other network as a side effect, so the
    runtime daemon would only have one enabled network after a UI-driven
    connect. _handle_connected must re-enable all saved networks so
    wpa_supplicant can auto-roam when the current AP disappears."""
    wm._handle_connected("MyNet")

    requests = [call.args[0] for call in wm._ctrl.request.call_args_list]
    assert "ENABLE_NETWORK all" in requests

  def test_handle_connected_swallows_enable_network_failure(self, wm):
    """A transient ctrl error on ENABLE_NETWORK must not tear down the
    CONNECTED transition itself."""
    wm._ctrl.request.side_effect = OSError("ctrl busy")

    wm._handle_connected("MyNet")

    # State still transitioned, DHCP still started.
    assert wm._wifi_state.status == ConnectStatus.CONNECTED
    assert wm._wifi_state.ssid == "MyNet"
    wm._dhcp.start.assert_called_once()


class TestDisconnected:
  def test_disconnected_clears_state(self, wm):
    wm._wifi_state = WifiState(ssid="Net", status=ConnectStatus.CONNECTED)

    fire(wm, "CTRL-EVENT-DISCONNECTED bssid=aa:bb:cc:dd:ee:ff reason=3")

    assert wm._wifi_state.ssid is None
    assert wm._wifi_state.status == ConnectStatus.DISCONNECTED
    wm._dhcp.stop.assert_called_once()

  def test_disconnected_preserves_connecting(self, wm):
    """If user just initiated a connect, don't clear the connecting state."""
    wm._set_connecting("NewNet")

    fire(wm, "CTRL-EVENT-DISCONNECTED bssid=aa:bb:cc:dd:ee:ff reason=3")

    assert wm._wifi_state.ssid == "NewNet"
    assert wm._wifi_state.status == ConnectStatus.CONNECTING

  def test_disconnected_during_tethering_ignored(self, wm):
    wm._wifi_state = WifiState(ssid="tether", status=ConnectStatus.CONNECTED)
    wm._tethering_active = True

    fire(wm, "CTRL-EVENT-DISCONNECTED bssid=aa:bb:cc:dd:ee:ff reason=3")

    assert wm._wifi_state.ssid == "tether"
    assert wm._wifi_state.status == ConnectStatus.CONNECTED

  def test_disconnected_fires_callback(self, wm, mocker):
    cb = mocker.MagicMock()
    wm.add_callbacks(disconnected=cb)
    wm._wifi_state = WifiState(ssid="Net", status=ConnectStatus.CONNECTED)

    fire(wm, "CTRL-EVENT-DISCONNECTED bssid=aa:bb:cc:dd:ee:ff reason=3")

    wm.process_callbacks()
    cb.assert_called_once()


class TestWrongPassword:
  def test_wrong_key_fires_need_auth(self, wm, mocker):
    cb = mocker.MagicMock()
    wm.add_callbacks(need_auth=cb)
    wm._set_connecting("SecNet")

    fire(wm, "CTRL-EVENT-SSID-TEMP-DISABLED id=0 ssid=\"SecNet\" auth_failures=1 duration=10 reason=WRONG_KEY")

    assert wm._wifi_state.status == ConnectStatus.DISCONNECTED
    wm.process_callbacks()
    cb.assert_called_once_with("SecNet")

  def test_wrong_key_removes_failed_runtime_network(self, wm, mocker):
    """The runtime network from _add_and_select_network was never persisted; if it
    survives the WRONG_KEY, ENABLE_NETWORK all just re-arms the bad PSK for retry."""
    wm._set_connecting("SecNet")
    wm._remove_wpa_network = mocker.MagicMock()

    fire(wm, "CTRL-EVENT-SSID-TEMP-DISABLED id=0 ssid=\"SecNet\" auth_failures=1 duration=10 reason=WRONG_KEY")

    wm._remove_wpa_network.assert_called_once_with("SecNet")
    requests = [c.args[0] for c in wm._ctrl.request.call_args_list]
    assert "ENABLE_NETWORK all" in requests

  def test_wrong_key_no_ssid_no_callback(self, wm, mocker):
    cb = mocker.MagicMock()
    wm.add_callbacks(need_auth=cb)

    fire(wm, "CTRL-EVENT-SSID-TEMP-DISABLED id=0 ssid=\"Net\" auth_failures=1 duration=10 reason=WRONG_KEY")

    assert len(wm._callback_queue) == 0

  def test_wrong_key_tears_down_dhcp_state(self, wm, mocker):
    """CTRL-EVENT-DISCONNECTED is dropped while CONNECTING; WRONG_KEY must
    perform the same teardown itself so DHCP/IP/metered don't go stale."""
    disconnected = mocker.MagicMock()
    wm.add_callbacks(disconnected=disconnected)
    wm._set_connecting("SecNet")
    wm._ipv4_address = "10.0.0.5"
    wm._current_network_metered = MeteredType.NO

    fire(wm, "CTRL-EVENT-SSID-TEMP-DISABLED id=0 ssid=\"SecNet\" auth_failures=1 duration=10 reason=WRONG_KEY")

    wm._dhcp.stop.assert_called_once()
    assert wm._ipv4_address == ""
    assert wm._current_network_metered == MeteredType.UNKNOWN
    wm.process_callbacks()
    disconnected.assert_called_once()

  def test_wrong_key_clears_pending_without_saving(self, wm):
    wm._set_connecting("SecNet")
    wm._set_pending_connection("SecNet", "wrongpass", False)

    fire(wm, "CTRL-EVENT-SSID-TEMP-DISABLED id=0 ssid=\"SecNet\" auth_failures=1 duration=10 reason=WRONG_KEY")

    wm._store.save_network.assert_not_called()
    assert wm._pending_connection is None

  def test_wrong_key_ignores_stale_event_for_previous_ssid(self, wm, mocker):
    """A delayed TEMP-DISABLED for a previously-attempted SSID must not
    tear down the user's current connection attempt."""
    cb = mocker.MagicMock()
    wm.add_callbacks(need_auth=cb)
    wm._set_connecting("CurrentNet")

    fire(wm, "CTRL-EVENT-SSID-TEMP-DISABLED id=0 ssid=\"OldNet\" auth_failures=1 duration=10 reason=WRONG_KEY")

    assert wm._wifi_state.ssid == "CurrentNet"
    assert wm._wifi_state.status == ConnectStatus.CONNECTING
    wm.process_callbacks()
    cb.assert_not_called()

  def test_wrong_key_debounced_after_recent_dispatch(self, wm, mocker):
    """Regression: if the user retries with fresh credentials right after
    a WRONG_KEY fired, a delayed second WRONG_KEY from the prior attempt
    must not clobber the new pending password. The debounce window
    allows the fresh attempt's real outcome to surface as the next event."""
    import time
    cb = mocker.MagicMock()
    wm.add_callbacks(need_auth=cb)
    # Simulate: a WRONG_KEY was dispatched a moment ago for this SSID.
    wm._last_wrong_key_dispatch["SecNet"] = time.monotonic()
    # User retried, new pending credentials are in flight.
    wm._set_connecting("SecNet")
    wm._set_pending_connection("SecNet", "retry-password", False)

    # Stale WRONG_KEY event from the previous attempt arrives.
    fire(wm, "CTRL-EVENT-SSID-TEMP-DISABLED id=0 ssid=\"SecNet\" auth_failures=1 duration=10 reason=WRONG_KEY")

    # Fresh pending credentials must survive.
    assert wm._pending_connection is not None
    assert wm._pending_connection.password == "retry-password"
    # And we stay in CONNECTING waiting for the real outcome.
    assert wm._wifi_state.status == ConnectStatus.CONNECTING
    wm.process_callbacks()
    cb.assert_not_called()

  def test_wrong_key_connecting_with_unknown_ssid_accepts_event(self, wm, mocker):
    """Auto-connect path sets CONNECTING with ssid=None when STATUS was
    briefly unavailable. A subsequent WRONG_KEY event is still the
    authoritative target identifier and must fire need_auth — otherwise
    the user sees a silent auth failure with no password prompt."""
    cb = mocker.MagicMock()
    wm.add_callbacks(need_auth=cb)
    wm._wifi_state = WifiState(ssid=None, status=ConnectStatus.CONNECTING)

    fire(wm, "CTRL-EVENT-SSID-TEMP-DISABLED id=0 ssid=\"AutoNet\" auth_failures=1 duration=10 reason=WRONG_KEY")

    assert wm._wifi_state.status == ConnectStatus.DISCONNECTED
    wm.process_callbacks()
    cb.assert_called_once_with("AutoNet")


class TestAutoConnect:
  def test_trying_to_associate_sets_connecting(self, wm):
    """Auto-connect: wpa_supplicant connects on its own."""
    wm._ctrl.request.return_value = "wpa_state=ASSOCIATING\nssid=AutoNet\n"

    fire(wm, "Trying to associate with aa:bb:cc:dd:ee:ff (SSID='AutoNet' freq=2437 MHz)")

    assert wm._wifi_state.ssid == "AutoNet"
    assert wm._wifi_state.status == ConnectStatus.CONNECTING

  def test_auto_connect_doesnt_overwrite_user_connecting(self, wm):
    """If user initiated connect, auto-connect event is ignored."""
    wm._set_connecting("UserNet")

    fire(wm, "Trying to associate with aa:bb:cc:dd:ee:ff (SSID='OtherNet' freq=2437 MHz)")

    assert wm._wifi_state.ssid == "UserNet"
    assert wm._wifi_state.status == ConnectStatus.CONNECTING


class TestScanResults:
  def test_scan_results_triggers_update(self, wm, mocker):
    wm._active = True
    wm._scan_lock = mocker.MagicMock()
    wm._tethering_ssid = "weedle"
    # Mock scan results
    wm._ctrl.request.return_value = "bssid / frequency / signal level / flags / ssid\naa:bb:cc:dd:ee:ff\t2437\t-50\t[WPA2-PSK-CCMP][ESS]\tTestNet\n"
    wm._update_networks = mocker.MagicMock()

    fire(wm, "CTRL-EVENT-SCAN-RESULTS")

    wm._update_networks.assert_called_once()


# ---------------------------------------------------------------------------
# Thread races: _set_connecting vs _handle_event
# ---------------------------------------------------------------------------

class TestThreadRaces:
  def test_connected_race_user_tap_during_status(self, wm):
    """User taps B right as A finishes connecting (STATUS call in flight)."""
    wm._set_connecting("A")

    def user_taps_b_during_status(cmd):
      if cmd == "STATUS":
        wm._set_connecting("B")
        return "wpa_state=COMPLETED\nssid=A\n"
      return ""

    wm._ctrl.request.side_effect = user_taps_b_during_status

    fire(wm, "CTRL-EVENT-CONNECTED - Connection to aa:bb:cc:dd:ee:ff completed [id=0]")

    assert wm._wifi_state.ssid == "B"
    assert wm._wifi_state.status == ConnectStatus.CONNECTING

  def test_auto_connect_race_user_tap_during_status(self, wm):
    """User taps B while auto-connect STATUS lookup is in flight."""
    def user_taps_b_during_status(cmd):
      if cmd == "STATUS":
        wm._set_connecting("B")
        return "wpa_state=ASSOCIATING\nssid=A\n"
      return ""

    wm._ctrl.request.side_effect = user_taps_b_during_status

    fire(wm, "Trying to associate with aa:bb:cc:dd:ee:ff (SSID='A' freq=2437 MHz)")

    assert wm._wifi_state.ssid == "B"
    assert wm._wifi_state.status == ConnectStatus.CONNECTING

  def test_disconnected_does_not_stomp_connecting(self, wm):
    """_set_connecting() between CONNECTING check and state write is preserved."""
    wm._wifi_state = WifiState(ssid="A", status=ConnectStatus.CONNECTED)

    original_handle = wm._handle_event.__func__

    def intercept(event):
      # Simulate: just after the CONNECTING check passes, user taps connect
      if "CTRL-EVENT-DISCONNECTED" in event:
        wm._set_connecting("B")
      original_handle(wm, event)

    wm._handle_event = intercept
    fire(wm, "CTRL-EVENT-DISCONNECTED bssid=aa:bb:cc:dd:ee:ff reason=3")

    assert wm._wifi_state.ssid == "B"
    assert wm._wifi_state.status == ConnectStatus.CONNECTING

  def test_connected_with_none_ssid_is_ignored(self, wm):
    """CONNECTED event with no SSID (STATUS parse fails) should not transition."""
    wm._wifi_state = WifiState()  # DISCONNECTED, ssid=None
    wm._ctrl.request.side_effect = Exception("wpa_supplicant gone")

    fire(wm, "CTRL-EVENT-CONNECTED - Connection to aa:bb:cc:dd:ee:ff completed [id=0]")

    assert wm._wifi_state.status == ConnectStatus.DISCONNECTED
    wm._dhcp.start.assert_not_called()


# ---------------------------------------------------------------------------
# Full sequences
# ---------------------------------------------------------------------------

class TestFullSequences:
  def test_normal_connect(self, wm):
    """User connects → CONNECTED event → gets IP."""
    wm._set_connecting("Home")
    wm._ctrl.request.return_value = "wpa_state=COMPLETED\nssid=Home\n"

    fire(wm, "CTRL-EVENT-CONNECTED - Connection to aa:bb:cc:dd:ee:ff completed [id=0]")

    assert wm._wifi_state.status == ConnectStatus.CONNECTED
    assert wm._wifi_state.ssid == "Home"
    wm._dhcp.start.assert_called_once()

  def test_wrong_password_then_retry(self, wm, mocker):
    """Wrong password → need_auth callback → user retries."""
    cb = mocker.MagicMock()
    wm.add_callbacks(need_auth=cb)

    wm._set_connecting("Sec")
    fire(wm, "CTRL-EVENT-SSID-TEMP-DISABLED id=0 ssid=\"Sec\" auth_failures=1 duration=10 reason=WRONG_KEY")

    assert wm._wifi_state.status == ConnectStatus.DISCONNECTED
    wm.process_callbacks()
    cb.assert_called_once_with("Sec")

    # Retry
    wm._set_connecting("Sec")
    wm._ctrl.request.return_value = "wpa_state=COMPLETED\nssid=Sec\n"
    fire(wm, "CTRL-EVENT-CONNECTED - Connection to aa:bb:cc:dd:ee:ff completed [id=0]")

    assert wm._wifi_state.status == ConnectStatus.CONNECTED
    assert wm._wifi_state.ssid == "Sec"

  def test_connect_then_disconnect(self, wm):
    """Connect, then network drops."""
    wm._set_connecting("Net")
    wm._ctrl.request.return_value = "wpa_state=COMPLETED\nssid=Net\n"

    fire(wm, "CTRL-EVENT-CONNECTED - Connection to aa:bb:cc:dd:ee:ff completed [id=0]")
    assert wm._wifi_state.status == ConnectStatus.CONNECTED

    fire(wm, "CTRL-EVENT-DISCONNECTED bssid=aa:bb:cc:dd:ee:ff reason=3")
    assert wm._wifi_state.status == ConnectStatus.DISCONNECTED
    assert wm._wifi_state.ssid is None

  def test_auto_connect_full_sequence(self, wm):
    """wpa_supplicant auto-connects to saved network."""
    wm._ctrl.request.return_value = "wpa_state=ASSOCIATING\nssid=AutoNet\n"

    fire(wm, "Trying to associate with aa:bb:cc:dd:ee:ff (SSID='AutoNet' freq=2437 MHz)")
    assert wm._wifi_state.ssid == "AutoNet"
    assert wm._wifi_state.status == ConnectStatus.CONNECTING

    wm._ctrl.request.return_value = "wpa_state=COMPLETED\nssid=AutoNet\n"
    fire(wm, "CTRL-EVENT-CONNECTED - Connection to aa:bb:cc:dd:ee:ff completed [id=0]")
    assert wm._wifi_state.status == ConnectStatus.CONNECTED
    assert wm._wifi_state.ssid == "AutoNet"

  def test_switch_networks(self, wm):
    """User switches from A to B."""
    wm._ctrl.request.return_value = "wpa_state=COMPLETED\nssid=A\n"
    wm._set_connecting("A")
    fire(wm, "CTRL-EVENT-CONNECTED - Connection to 11:22:33:44:55:66 completed [id=0]")
    assert wm._wifi_state.status == ConnectStatus.CONNECTED
    assert wm._wifi_state.ssid == "A"

    # User taps B
    wm._set_connecting("B")

    # Disconnect from A (preserved because CONNECTING)
    fire(wm, "CTRL-EVENT-DISCONNECTED bssid=11:22:33:44:55:66 reason=3")
    assert wm._wifi_state.ssid == "B"
    assert wm._wifi_state.status == ConnectStatus.CONNECTING

    # Connect to B
    wm._ctrl.request.return_value = "wpa_state=COMPLETED\nssid=B\n"
    fire(wm, "CTRL-EVENT-CONNECTED - Connection to aa:bb:cc:dd:ee:ff completed [id=1]")
    assert wm._wifi_state.status == ConnectStatus.CONNECTED
    assert wm._wifi_state.ssid == "B"


class TestAddAndSelectNetworkResponseChecks:
  """Verify that wpa_supplicant FAIL responses are caught and orphaned
  network entries are cleaned up (see _add_and_select_network)."""

  def _stub_ctrl(self, wm, responses):
    """Back wm._ctrl with a canned response sequence."""
    calls = []

    def fake_request(cmd):
      calls.append(cmd)
      if not responses:
        return "OK"
      return responses.pop(0)

    wm._ctrl.request.side_effect = fake_request
    return calls

  def test_happy_path_all_ok(self, wm):
    calls = self._stub_ctrl(wm, ["7", "OK", "OK", "OK"])

    wm._add_and_select_network("Net", "pass1234", hidden=False)

    assert calls == [
      "ADD_NETWORK",
      'SET_NETWORK 7 ssid "Net"',
      'SET_NETWORK 7 psk "pass1234"',
      "SELECT_NETWORK 7",
    ]

  def test_set_network_psk_fail_removes_orphan(self, wm):
    """FAIL on PSK SET_NETWORK (e.g. too short) must raise and clean up."""
    calls = self._stub_ctrl(wm, ["3", "OK", "FAIL", "OK"])

    with pytest.raises(RuntimeError, match="SET_NETWORK 3 psk failed"):
      wm._add_and_select_network("Net", "bad", hidden=False)

    assert "REMOVE_NETWORK 3" in calls
    # SELECT_NETWORK must NOT have been called with the orphan id.
    assert "SELECT_NETWORK 3" not in calls

  def test_set_network_ssid_fail_removes_orphan(self, wm):
    calls = self._stub_ctrl(wm, ["4", "FAIL"])

    with pytest.raises(RuntimeError, match="SET_NETWORK 4 ssid failed"):
      wm._add_and_select_network("BadSsid", "pw12345678", hidden=False)

    assert "REMOVE_NETWORK 4" in calls

  def test_select_network_fail_removes_orphan(self, wm):
    calls = self._stub_ctrl(wm, ["2", "OK", "OK", "FAIL"])

    with pytest.raises(RuntimeError, match="SELECT_NETWORK 2 failed"):
      wm._add_and_select_network("Net", "pw12345678", hidden=False)

    assert "REMOVE_NETWORK 2" in calls

  def test_add_network_fail_raises_without_orphan(self, wm):
    self._stub_ctrl(wm, ["FAIL"])

    with pytest.raises(RuntimeError, match="ADD_NETWORK failed"):
      wm._add_and_select_network("Net", "pw12345678", hidden=False)

    # Nothing to remove — ADD_NETWORK itself failed.
    assert wm._ctrl.request.call_count == 1

  def test_hidden_network_sets_scan_ssid(self, wm):
    calls = self._stub_ctrl(wm, ["1", "OK", "OK", "OK", "OK"])

    wm._add_and_select_network("Hidden", "pw12345678", hidden=True)

    assert "SET_NETWORK 1 scan_ssid 1" in calls
    assert "SELECT_NETWORK 1" in calls

  def test_open_network_sets_key_mgmt_none(self, wm):
    calls = self._stub_ctrl(wm, ["5", "OK", "OK", "OK"])

    wm._add_and_select_network("Open", "", hidden=False)

    assert "SET_NETWORK 5 key_mgmt NONE" in calls

  def test_raw_hex_psk_sent_unquoted(self, wm):
    """A 64-hex PSK must be passed unquoted; wpa_supplicant rejects quoted
    values of length 64 as too-long passphrases (hostap config.c:650)."""
    raw_psk = "0123456789abcdef" * 4
    assert len(raw_psk) == 64
    calls = self._stub_ctrl(wm, ["8", "OK", "OK", "OK"])

    wm._add_and_select_network("Net", raw_psk, hidden=False)

    assert f"SET_NETWORK 8 psk {raw_psk}" in calls
    assert f'SET_NETWORK 8 psk "{raw_psk}"' not in calls

  def test_passphrase_psk_sent_quoted(self, wm):
    """Anything not exactly 64 hex chars goes through as a quoted passphrase."""
    calls = self._stub_ctrl(wm, ["9", "OK", "OK", "OK"])

    wm._add_and_select_network("Net", "s3cretpass", hidden=False)

    assert 'SET_NETWORK 9 psk "s3cretpass"' in calls

  def test_63_hex_chars_is_passphrase(self, wm):
    """A 63-char hex-looking string is a passphrase, not a raw PSK."""
    psk63 = "0123456789abcdef" * 3 + "0123456789abcde"
    assert len(psk63) == 63
    calls = self._stub_ctrl(wm, ["10", "OK", "OK", "OK"])

    wm._add_and_select_network("Net", psk63, hidden=False)

    assert f'SET_NETWORK 10 psk "{psk63}"' in calls


class TestListNetworkIdsDecoding:
  def test_list_network_ids_decodes_printf_encoded_ssids(self, wm):
    """LIST_NETWORKS emits SSIDs in printf_encode form. Without decoding,
    non-ASCII SSIDs never match the caller's already-decoded string, so
    forget_connection / activate_connection silently leak runtime
    entries for any UTF-8 network name."""
    # "café" → 5 UTF-8 bytes, printf_encoded as caf\\xc3\\xa9
    wm._ctrl.request.return_value = "\n".join([
      "network id / ssid / bssid / flags",
      "0\tcaf\\xc3\\xa9\tany\t[CURRENT]",
      "1\tOtherNet\tany\t",
    ]) + "\n"

    ids = wm._list_network_ids("café")

    assert ids == ["0"]

  def test_list_network_ids_matches_plain_ascii(self, wm):
    """Regression guard: decoding must be a no-op for pure ASCII SSIDs."""
    wm._ctrl.request.return_value = "\n".join([
      "network id / ssid / bssid / flags",
      "0\tHomeNet\tany\t[CURRENT]",
      "1\tOtherNet\tany\t",
      "2\tHomeNet\tany\t",
    ]) + "\n"

    assert wm._list_network_ids("HomeNet") == ["0", "2"]


class TestConnectBlockedDuringTethering:
  def test_connect_to_network_noop_while_tethering(self, wm, mocker):
    """Backend guard: connect_to_network must refuse while tethering is
    active. The AP daemon can't service ADD_NETWORK/SELECT_NETWORK for
    STA networks, so dispatching would churn UI state and could disrupt
    the running hotspot."""
    wm._tethering_active = True
    mocker.patch.object(wifi_manager_module.threading, "Thread")

    wm.connect_to_network("HomeNet", "password", hidden=False)

    # No state change: wifi_state untouched, no pending connection.
    assert wm._wifi_state.status == ConnectStatus.DISCONNECTED
    assert wm._pending_connection is None
    wifi_manager_module.threading.Thread.assert_not_called()

  def test_activate_connection_noop_while_tethering(self, wm, mocker):
    wm._tethering_active = True
    mocker.patch.object(wifi_manager_module.threading, "Thread")

    wm.activate_connection("HomeNet")

    assert wm._wifi_state.status == ConnectStatus.DISCONNECTED
    wifi_manager_module.threading.Thread.assert_not_called()


class TestConnectWithoutCtrl:
  """When _ctrl is None (pre-attach init, post-daemon-loss), the worker must
  reset the CONNECTING state inline. _init_wifi_state no-ops in that condition,
  so relying on it leaves the UI wedged at CONNECTING forever."""

  class ImmediateThread:
    def __init__(self, target=None, daemon=None):
      self._target = target

    def start(self):
      if self._target is not None:
        self._target()

  def test_connect_to_network_without_ctrl_resets_state(self, wm, mocker):
    wm._ctrl = None
    disconnected_cb = mocker.MagicMock()
    wm.add_callbacks(disconnected=disconnected_cb)
    mocker.patch.object(wifi_manager_module.threading, "Thread", self.ImmediateThread)

    wm.connect_to_network("HomeNet", "password", hidden=False)

    assert wm._wifi_state.status == ConnectStatus.DISCONNECTED
    assert wm._wifi_state.ssid is None
    assert wm._pending_connection is None
    wm.process_callbacks()
    disconnected_cb.assert_called_once()

  def test_connect_to_network_aborts_when_user_taps_another_network(self, wm, mocker):
    """If the user taps SSID A, then quickly taps SSID B before the worker runs,
    the stale worker for A must not SELECT_NETWORK A and force the wrong network."""
    wm._remove_wpa_network = mocker.MagicMock()
    wm._add_and_select_network = mocker.MagicMock()

    captured = []

    def deferred_thread(target=None, daemon=None):
      class T:
        def start(_):
          captured.append(target)
      return T()

    mocker.patch.object(wifi_manager_module.threading, "Thread", deferred_thread)

    wm.connect_to_network("NetA", "passA", hidden=False)
    # User taps the next network before the worker runs.
    wm._set_connecting("NetB")
    captured[0]()  # run the deferred A worker

    wm._remove_wpa_network.assert_not_called()
    wm._add_and_select_network.assert_not_called()

  def test_connect_to_network_setup_failure_resets_state(self, wm, mocker):
    """If _add_and_select_network raises (e.g. wpa_supplicant rejects a bad PSK),
    the worker must reset CONNECTING and fire disconnected. Otherwise the UI
    stays wedged at CONNECTING until an unrelated event clears it."""
    wm._remove_wpa_network = mocker.MagicMock()
    wm._add_and_select_network = mocker.MagicMock(side_effect=OSError("FAIL"))
    disconnected_cb = mocker.MagicMock()
    wm.add_callbacks(disconnected=disconnected_cb)
    mocker.patch.object(wifi_manager_module.threading, "Thread", self.ImmediateThread)

    wm.connect_to_network("HomeNet", "password", hidden=False)

    assert wm._wifi_state.status == ConnectStatus.DISCONNECTED
    assert wm._wifi_state.ssid is None
    assert wm._pending_connection is None
    wm.process_callbacks()
    disconnected_cb.assert_called_once()

  def test_activate_connection_setup_failure_fires_disconnected(self, wm, mocker):
    """If _add_and_select_network raises (e.g. migrated keyfile has bad PSK),
    activate_connection must reset CONNECTING and fire disconnected. Otherwise
    the UI sticks at CONNECTING until an unrelated event clears it."""
    wm._list_network_ids = mocker.MagicMock(return_value=[])
    wm._store.get = mocker.MagicMock(return_value={"psk": "x", "hidden": False})
    wm._add_and_select_network = mocker.MagicMock(side_effect=OSError("FAIL"))
    disconnected_cb = mocker.MagicMock()
    wm.add_callbacks(disconnected=disconnected_cb)

    wm.activate_connection("HomeNet", block=True)

    assert wm._wifi_state.status == ConnectStatus.DISCONNECTED
    assert wm._wifi_state.ssid is None
    wm.process_callbacks()
    disconnected_cb.assert_called_once()

  def test_activate_connection_missing_network_fires_disconnected(self, wm, mocker):
    """Activating a non-existent saved network must reset CONNECTING and notify
    the UI rather than silently leaving _init_wifi_state to handle it."""
    wm._list_network_ids = mocker.MagicMock(return_value=[])
    wm._store.get = mocker.MagicMock(return_value=None)
    disconnected_cb = mocker.MagicMock()
    wm.add_callbacks(disconnected=disconnected_cb)

    wm.activate_connection("Vanished", block=True)

    assert wm._wifi_state.status == ConnectStatus.DISCONNECTED
    wm.process_callbacks()
    disconnected_cb.assert_called_once()

  def test_activate_connection_without_ctrl_resets_state(self, wm, mocker):
    wm._ctrl = None
    disconnected_cb = mocker.MagicMock()
    wm.add_callbacks(disconnected=disconnected_cb)
    mocker.patch.object(wifi_manager_module.threading, "Thread", self.ImmediateThread)

    wm.activate_connection("HomeNet")

    assert wm._wifi_state.status == ConnectStatus.DISCONNECTED
    assert wm._wifi_state.ssid is None
    wm.process_callbacks()
    disconnected_cb.assert_called_once()

  def test_forget_inactive_skips_reassociate(self, wm, mocker):
    """Forgetting a saved network that isn't the current connection must not
    REASSOCIATE — the active link is unrelated and a reassociate would briefly
    drop or renegotiate it for no reason."""
    wm._remove_wpa_network = mocker.MagicMock()
    wm._wifi_state = WifiState(ssid="ActiveNet", status=ConnectStatus.CONNECTED)
    mocker.patch.object(wifi_manager_module, "_generate_wpa_conf")

    wm.forget_connection("OtherSavedNet", block=True)

    requests = [c.args[0] for c in wm._ctrl.request.call_args_list]
    assert "REASSOCIATE" not in requests
    assert "DISCONNECT" not in requests
    assert "ENABLE_NETWORK all" in requests

  def test_forget_active_reassociates(self, wm, mocker):
    """Forgetting the active connection must DISCONNECT and REASSOCIATE so the
    device falls back to the next saved network."""
    wm._remove_wpa_network = mocker.MagicMock()
    wm._wifi_state = WifiState(ssid="ActiveNet", status=ConnectStatus.CONNECTED)
    mocker.patch.object(wifi_manager_module, "_generate_wpa_conf")

    wm.forget_connection("ActiveNet", block=True)

    requests = [c.args[0] for c in wm._ctrl.request.call_args_list]
    assert "DISCONNECT" in requests
    assert "REASSOCIATE" in requests


class TestConnectPersistence:
  def test_connect_to_network_does_not_save_before_auth(self, wm, mocker):
    wm._remove_wpa_network = mocker.MagicMock()
    wm._add_and_select_network = mocker.MagicMock()

    class ImmediateThread:
      def __init__(self, target=None, daemon=None):
        self._target = target

      def start(self):
        if self._target is not None:
          self._target()

    mocker.patch.object(wifi_manager_module.threading, "Thread", ImmediateThread)
    mocker.patch.object(wifi_manager_module, "_generate_wpa_conf")
    wm.connect_to_network("SecNet", "secretpass", hidden=True)

    wm._store.save_network.assert_not_called()
    assert wm._pending_connection == PendingConnection(ssid="SecNet", password="secretpass", hidden=True, epoch=1)
    wm._remove_wpa_network.assert_called_once_with("SecNet")
    wm._add_and_select_network.assert_called_once_with("SecNet", "secretpass", True)


class TestNetworksUpdatedCoalescing:
  def test_mark_networks_updated_is_idempotent(self, wm):
    wm._mark_networks_updated()
    wm._mark_networks_updated()
    wm._mark_networks_updated()
    # Only the single dirty flag is buffered — no queue growth.
    assert wm._networks_updated_pending is True
    assert wm._callback_queue == []

  def test_many_scan_ticks_while_panel_hidden_collapse_to_one_call(self, wm, mocker):
    cb = mocker.MagicMock()
    wm.add_callbacks(networks_updated=cb)

    for _ in range(50):
      wm._mark_networks_updated()

    assert wm._callback_queue == []
    wm.process_callbacks()

    cb.assert_called_once_with(wm.networks)
    assert wm._networks_updated_pending is False

  def test_process_callbacks_uses_latest_networks_snapshot(self, wm, mocker):
    seen = []
    wm.add_callbacks(networks_updated=lambda nets: seen.append(list(nets)))
    wm._store.saved_ssids.return_value = set()

    stale = Network(ssid="Stale", strength=50, security_type=SecurityType.OPEN, is_tethering=False)
    fresh = Network(ssid="Fresh", strength=80, security_type=SecurityType.OPEN, is_tethering=False)

    wm._networks = [stale]
    wm._mark_networks_updated()

    # Simulate newer scan landing before the drain.
    wm._networks = [fresh]

    wm.process_callbacks()

    assert len(seen) == 1
    assert [n.ssid for n in seen[0]] == ["Fresh"]

  def test_process_callbacks_without_flag_does_not_fire(self, wm, mocker):
    cb = mocker.MagicMock()
    wm.add_callbacks(networks_updated=cb)

    wm.process_callbacks()

    cb.assert_not_called()


class TestStop:
  def test_stop_calls_stop_tethering_when_active(self, wm, mocker):
    wm._tethering_active = True
    wm._scan_thread = mocker.MagicMock(is_alive=mocker.MagicMock(return_value=False))
    wm._state_thread = mocker.MagicMock(is_alive=mocker.MagicMock(return_value=False))
    wm._stop_tethering = mocker.MagicMock()
    wm._exit = False

    wm.stop()

    wm._stop_tethering.assert_called_once()

  def test_stop_skips_tethering_when_not_active(self, wm, mocker):
    wm._tethering_active = False
    wm._scan_thread = mocker.MagicMock(is_alive=mocker.MagicMock(return_value=False))
    wm._state_thread = mocker.MagicMock(is_alive=mocker.MagicMock(return_value=False))
    wm._stop_tethering = mocker.MagicMock()
    wm._exit = False

    wm.stop()

    wm._stop_tethering.assert_not_called()


class TestInitWifiState:
  """_init_wifi_state must distinguish a running AP (tethering) from a
  station-mode association. STATUS reports wpa_state=COMPLETED in both
  cases; routing an active hotspot through the station path would call
  _dhcp.start() → ip addr flush wlan0 → drop TETHERING_IP_ADDRESS."""

  def test_ap_mode_adopts_without_starting_dhcp(self, wm):
    """Regression: process restart while tethering active must not flush wlan0."""
    from openpilot.system.ui.lib.wifi_manager import TETHERING_IP_ADDRESS
    wm._tethering_ssid = "weedle-test"
    wm._ctrl.request.return_value = "wpa_state=COMPLETED\nmode=AP\nssid=weedle-test\n"

    wm._init_wifi_state()

    assert wm._tethering_active is True
    assert wm._wifi_state.status == ConnectStatus.CONNECTED
    assert wm._wifi_state.ssid == "weedle-test"
    assert wm._ipv4_address == TETHERING_IP_ADDRESS
    wm._dhcp.start.assert_not_called()
    wm._dhcp.stop.assert_not_called()

  def test_ap_mode_falls_back_to_configured_ssid_if_status_missing(self, wm):
    """If STATUS doesn't echo ssid (should never happen, but be defensive),
    use the configured tethering SSID so we still report CONNECTED."""
    wm._tethering_ssid = "weedle-fallback"
    wm._ctrl.request.return_value = "wpa_state=COMPLETED\nmode=AP\n"

    wm._init_wifi_state()

    assert wm._tethering_active is True
    assert wm._wifi_state.ssid == "weedle-fallback"
    wm._dhcp.start.assert_not_called()

  def test_station_completed_still_starts_dhcp(self, wm):
    """Happy path: attaching to a connected STA daemon must re-launch udhcpc
    (the prior UI's udhcpc died with its parent)."""
    wm._ctrl.request.return_value = "wpa_state=COMPLETED\nmode=station\nssid=HomeNet\n"

    wm._init_wifi_state()

    assert wm._tethering_active is False
    assert wm._wifi_state.status == ConnectStatus.CONNECTED
    assert wm._wifi_state.ssid == "HomeNet"
    wm._dhcp.start.assert_called_once()

  def test_station_disconnected_does_not_touch_dhcp(self, wm):
    wm._ctrl.request.return_value = "wpa_state=DISCONNECTED\n"

    wm._init_wifi_state()

    assert wm._wifi_state.status == ConnectStatus.DISCONNECTED
    wm._dhcp.start.assert_not_called()

  @pytest.mark.parametrize("wpa_state", [
    "SCANNING", "AUTHENTICATING", "ASSOCIATING", "ASSOCIATED",
    "4WAY_HANDSHAKE", "GROUP_HANDSHAKE",
  ])
  def test_mid_connect_states_adopt_connecting(self, wm, wpa_state):
    """On UI/daemon restart during any transient wpa_supplicant state, the
    manager must start in CONNECTING so recovery paths (stale reconcile,
    WRONG_KEY dispatch) keyed on CONNECTING still run for that attempt."""
    wm._ctrl.request.return_value = f"wpa_state={wpa_state}\nmode=station\nssid=HomeNet\n"

    wm._init_wifi_state()

    assert wm._wifi_state.status == ConnectStatus.CONNECTING, f"{wpa_state} must map to CONNECTING"
    assert wm._wifi_state.ssid == "HomeNet"
