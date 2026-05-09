import time

from openpilot.system.ui.lib import wifi_manager as wifi_manager_module
from openpilot.system.ui.lib.wifi_manager import (
  CONNECTING_STALE_TIMEOUT_SECONDS,
  ConnectStatus,
  Network,
  PendingConnection,
  SecurityType,
  WifiState,
)


def test_reconcile_stale_connecting_to_disconnected(wm, mocker):
  disconnected = mocker.MagicMock()
  wm._disconnected.append(disconnected)
  wm._wifi_state = WifiState(ssid="systeam", status=ConnectStatus.CONNECTING)
  wm._ctrl.request.return_value = "wpa_state=DISCONNECTED\n"

  wm._reconcile_connecting_state()
  wm.process_callbacks()

  assert wm._wifi_state.status == ConnectStatus.DISCONNECTED
  assert wm._wifi_state.ssid is None
  disconnected.assert_called_once()


def test_reconcile_stale_connecting_to_connected(wm, mocker):
  activated = mocker.MagicMock()
  wm._activated.append(activated)
  wm._wifi_state = WifiState(ssid="systeam", status=ConnectStatus.CONNECTING)
  wm._ctrl.request.return_value = "wpa_state=COMPLETED\nssid=systeam\n"

  wm._reconcile_connecting_state()
  wm.process_callbacks()

  assert wm._wifi_state.status == ConnectStatus.CONNECTED
  assert wm._wifi_state.ssid == "systeam"
  wm._dhcp.start.assert_called_once()
  activated.assert_called_once()


def test_reconcile_stale_connecting_adopts_actual_connected_ssid(wm, mocker):
  activated = mocker.MagicMock()
  wm._activated.append(activated)
  wm._wifi_state = WifiState(ssid="systeam", status=ConnectStatus.CONNECTING)
  wm._ctrl.request.return_value = "wpa_state=COMPLETED\nssid=systeam5\n"

  wm._reconcile_connecting_state()
  wm.process_callbacks()

  assert wm._wifi_state.status == ConnectStatus.CONNECTED
  assert wm._wifi_state.ssid == "systeam5"
  wm._dhcp.start.assert_called_once()
  activated.assert_called_once()


def test_reconcile_stale_connecting_to_disconnected_reenables_networks(wm, mocker):
  wm._wifi_state = WifiState(ssid="systeam", status=ConnectStatus.CONNECTING)
  wm._ctrl.request.return_value = "wpa_state=DISCONNECTED\n"

  wm._reconcile_connecting_state()

  requests = [call.args[0] for call in wm._ctrl.request.call_args_list]
  assert "ENABLE_NETWORK all" in requests


def test_reconcile_stale_connecting_drops_result_when_user_starts_new_attempt(wm, mocker):
  """If the user starts connecting to a different network while we're blocked in
  STATUS, a stale DISCONNECTED for the original SSID must not call _set_connecting(None)
  and clobber the fresh attempt."""
  disconnected = mocker.MagicMock()
  wm._disconnected.append(disconnected)
  wm._wifi_state = WifiState(ssid="oldnet", status=ConnectStatus.CONNECTING)

  def status_then_user_taps(_cmd):
    # Mid-STATUS the user taps a new network — _set_connecting bumps _user_epoch.
    wm._set_connecting("newnet")
    return "wpa_state=DISCONNECTED\n"
  wm._ctrl.request.side_effect = status_then_user_taps

  wm._reconcile_connecting_state()
  wm.process_callbacks()

  # Fresh attempt for "newnet" must survive untouched.
  assert wm._wifi_state.status == ConnectStatus.CONNECTING
  assert wm._wifi_state.ssid == "newnet"
  disconnected.assert_not_called()


def test_reconcile_stale_secure_network_prompts_auth(wm, mocker):
  need_auth = mocker.MagicMock()
  wm._need_auth.append(need_auth)
  wm._wifi_state = WifiState(ssid="systeam", status=ConnectStatus.CONNECTING)
  wm._networks = [Network(ssid="systeam", strength=90, security_type=SecurityType.WPA, is_tethering=False)]
  wm._ctrl.request.return_value = "wpa_state=DISCONNECTED\n"

  wm._reconcile_connecting_state()
  wm.process_callbacks()

  need_auth.assert_called_once_with("systeam")


def test_reconcile_disconnected_detects_missed_connected(wm, mocker):
  """After tethering stops, monitor may miss CONNECTED event."""
  activated = mocker.MagicMock()
  wm._activated.append(activated)
  wm._wifi_state = WifiState(ssid=None, status=ConnectStatus.DISCONNECTED)
  wm._ctrl.request.return_value = "wpa_state=COMPLETED\nssid=systeam5\n"

  wm._reconcile_connecting_state()
  wm.process_callbacks()

  assert wm._wifi_state.status == ConnectStatus.CONNECTED
  assert wm._wifi_state.ssid == "systeam5"
  wm._dhcp.start.assert_called_once()
  activated.assert_called_once()


def test_reconcile_disconnected_stays_disconnected(wm, mocker):
  """Don't falsely connect when wpa_supplicant is also disconnected."""
  activated = mocker.MagicMock()
  wm._activated.append(activated)
  wm._wifi_state = WifiState(ssid=None, status=ConnectStatus.DISCONNECTED)
  wm._ctrl.request.return_value = "wpa_state=DISCONNECTED\n"

  wm._reconcile_connecting_state()
  wm.process_callbacks()

  assert wm._wifi_state.status == ConnectStatus.DISCONNECTED
  wm._dhcp.start.assert_not_called()
  activated.assert_not_called()


def test_reconcile_disconnected_adopts_ap_mode_status(wm, mocker):
  """If startup adoption missed (e.g. transient STATUS failure), reconcile must re-adopt
  AP state so we don't stay DISCONNECTED while attached to the AP daemon."""
  mocker.patch.object(wifi_manager_module, "_our_dnsmasq_running", return_value=True)
  activated = mocker.MagicMock()
  wm._activated.append(activated)
  wm._tethering_ssid = "weedle"
  wm._wifi_state = WifiState(ssid=None, status=ConnectStatus.DISCONNECTED)
  wm._ctrl.request.return_value = "wpa_state=COMPLETED\nmode=AP\nssid=tether\n"

  wm._reconcile_connecting_state()
  wm.process_callbacks()

  assert wm._tethering_active
  assert wm._wifi_state.status == ConnectStatus.CONNECTED
  assert wm._wifi_state.ssid == "tether"
  wm._dhcp.start.assert_not_called()
  activated.assert_called_once()


def test_reconcile_disconnected_refuses_ap_adoption_without_dnsmasq(wm, mocker):
  """AP daemon up but dnsmasq dead = half-broken hotspot. Don't adopt; stay DISCONNECTED
  so the user toggling tethering can recover instead of advertising a healthy AP."""
  mocker.patch.object(wifi_manager_module, "_our_dnsmasq_running", return_value=False)
  activated = mocker.MagicMock()
  wm._activated.append(activated)
  wm._wifi_state = WifiState(ssid=None, status=ConnectStatus.DISCONNECTED)
  wm._ctrl.request.return_value = "wpa_state=COMPLETED\nmode=AP\nssid=tether\n"

  wm._reconcile_connecting_state()
  wm.process_callbacks()

  assert wm._tethering_active is False
  assert wm._wifi_state.status == ConnectStatus.DISCONNECTED
  activated.assert_not_called()


def test_reconcile_disconnected_skipped_during_tethering(wm):
  """Don't reconcile while tethering is active."""
  wm._tethering_active = True
  wm._wifi_state = WifiState(ssid=None, status=ConnectStatus.DISCONNECTED)
  wm._ctrl.request.return_value = "wpa_state=COMPLETED\nssid=systeam\n"

  wm._reconcile_connecting_state()

  assert wm._wifi_state.status == ConnectStatus.DISCONNECTED
  wm._ctrl.request.assert_not_called()


def test_reconcile_scanning_keeps_connecting(wm, mocker):
  """wpa_supplicant can remain in SCANNING past the stale window for
  legitimate reasons (hidden SSIDs, slow directed-probe responses). We
  must not synthesize a wrong-password failure — that would clear the
  pending credentials and drop back to disconnected, preventing the
  eventual successful connect from being persisted."""
  need_auth = mocker.MagicMock()
  disconnected = mocker.MagicMock()
  wm._need_auth.append(need_auth)
  wm._disconnected.append(disconnected)
  wm._wifi_state = WifiState(ssid="HiddenAP", status=ConnectStatus.CONNECTING)
  wm._networks = [Network(ssid="HiddenAP", strength=90, security_type=SecurityType.WPA, is_tethering=False)]
  wm._pending_connection = PendingConnection(ssid="HiddenAP", password="secret", hidden=True, epoch=1)
  wm._ctrl.request.return_value = "wpa_state=SCANNING\n"

  before = time.monotonic()
  wm._reconcile_connecting_state()
  wm.process_callbacks()

  assert wm._wifi_state.status == ConnectStatus.CONNECTING
  assert wm._wifi_state.ssid == "HiddenAP"
  assert wm._pending_connection is not None, "pending credentials must survive SCANNING"
  need_auth.assert_not_called()
  disconnected.assert_not_called()
  # Timestamp refreshed so we wait another full window before re-checking.
  assert wm._last_connecting_at >= before


def test_reconcile_connected_detects_stale_wifi_state(wm, mocker):
  """Monitor socket drop/reconnect can drop a CTRL-EVENT-DISCONNECTED,
  leaving self._wifi_state stuck at CONNECTED while wpa_supplicant has
  actually moved on. The scanner-driven reconcile must detect this and
  synthesize a disconnect so DHCP stops, IP/metered clear, and the UI
  fires the disconnected callback."""
  disconnected = mocker.MagicMock()
  wm._disconnected.append(disconnected)
  wm._wifi_state = WifiState(ssid="systeam", status=ConnectStatus.CONNECTED)
  wm._ipv4_address = "192.168.1.105"
  wm._last_connected_recheck = 0.0
  wm._ctrl.request.return_value = "wpa_state=DISCONNECTED\n"

  wm._reconcile_connecting_state()
  wm.process_callbacks()

  assert wm._wifi_state.status == ConnectStatus.DISCONNECTED
  assert wm._wifi_state.ssid is None
  assert wm._ipv4_address == ""
  wm._dhcp.stop.assert_called_once()
  disconnected.assert_called_once()


def test_reconcile_connected_stable_when_still_completed(wm, mocker):
  """If STATUS still reports COMPLETED on the same SSID, leave state
  untouched — no spurious disconnect callback fan-out."""
  disconnected = mocker.MagicMock()
  wm._disconnected.append(disconnected)
  wm._wifi_state = WifiState(ssid="systeam", status=ConnectStatus.CONNECTED)
  wm._ipv4_address = "192.168.1.105"
  wm._last_connected_recheck = 0.0
  wm._ctrl.request.return_value = "wpa_state=COMPLETED\nssid=systeam\n"

  wm._reconcile_connecting_state()
  wm.process_callbacks()

  assert wm._wifi_state.status == ConnectStatus.CONNECTED
  assert wm._wifi_state.ssid == "systeam"
  assert wm._ipv4_address == "192.168.1.105"
  wm._dhcp.stop.assert_not_called()
  disconnected.assert_not_called()


def test_reconcile_connected_defers_transient_states(wm, mocker):
  """Mid-roam/rekey wpa_state can be ASSOCIATING/ASSOCIATED/handshake briefly.
  Treating that as disconnect would flush a live udhcpc lease for nothing."""
  disconnected = mocker.MagicMock()
  wm._disconnected.append(disconnected)
  wm._wifi_state = WifiState(ssid="HomeNet", status=ConnectStatus.CONNECTED)
  wm._ipv4_address = "10.0.0.5"
  wm._last_connected_recheck = time.monotonic() - 100

  for transient in ("ASSOCIATING", "ASSOCIATED", "AUTHENTICATING", "4WAY_HANDSHAKE", "GROUP_HANDSHAKE", "SCANNING"):
    wm._ctrl.request.return_value = f"wpa_state={transient}\nssid=HomeNet\n"
    wm._last_connected_recheck = time.monotonic() - 100
    wm._reconcile_connecting_state()
    assert wm._wifi_state.status == ConnectStatus.CONNECTED, f"{transient} must defer, not disconnect"
    assert wm._ipv4_address == "10.0.0.5"
  wm._dhcp.stop.assert_not_called()
  disconnected.assert_not_called()


def test_reconcile_connected_adopts_roamed_ssid(wm, mocker):
  """If STATUS reports COMPLETED on a different SSID than we think we're
  on (monitor socket down during a roam A→B), adopt the new network via
  _handle_connected. Synthesizing DISCONNECTED would flush wlan0's IP and
  fire a spurious disconnected callback before the next loop recovers."""
  activated = mocker.MagicMock()
  disconnected = mocker.MagicMock()
  wm._activated.append(activated)
  wm._disconnected.append(disconnected)
  wm._wifi_state = WifiState(ssid="HomeA", status=ConnectStatus.CONNECTED)
  wm._ipv4_address = "192.168.1.50"
  wm._last_connected_recheck = 0.0
  wm._ctrl.request.return_value = "wpa_state=COMPLETED\nssid=HomeB\n"

  wm._reconcile_connecting_state()
  wm.process_callbacks()

  assert wm._wifi_state.status == ConnectStatus.CONNECTED
  assert wm._wifi_state.ssid == "HomeB"
  wm._dhcp.stop.assert_not_called()
  disconnected.assert_not_called()
  activated.assert_called_once()


def test_reconcile_connected_gated_by_scan_period(wm, mocker):
  """STATUS must not be polled on every scanner tick — only once per
  SCAN_PERIOD_SECONDS — otherwise we spam wpa_supplicant at 2Hz."""
  wm._wifi_state = WifiState(ssid="systeam", status=ConnectStatus.CONNECTED)
  wm._last_connected_recheck = time.monotonic()  # just rechecked

  wm._reconcile_connecting_state()

  wm._ctrl.request.assert_not_called()


def test_persist_pending_connection_keeps_creds_on_write_failure(wm, mocker):
  """If save_network fails (e.g. /data write error), credentials must
  survive so a later retry can persist them. The exception must not
  propagate — _handle_connected needs to continue to DHCP start."""
  wm._pending_connection = PendingConnection(ssid="systeam", password="secret", hidden=False, epoch=wm._user_epoch)
  wm._store.save_network.side_effect = OSError("disk full")

  # Should not raise
  wm._persist_pending_connection("systeam")

  assert wm._pending_connection is not None
  assert wm._pending_connection.password == "secret"


def test_persist_pending_connection_clears_on_success(wm, mocker):
  """Happy path: persistence succeeds, pending credentials are cleared."""
  mocker.patch("openpilot.system.ui.lib.wifi_manager._generate_wpa_conf")
  wm._pending_connection = PendingConnection(ssid="systeam", password="secret", hidden=False, epoch=wm._user_epoch)

  wm._persist_pending_connection("systeam")

  assert wm._pending_connection is None
  wm._store.save_network.assert_called_once_with("systeam", psk="secret", hidden=False)


def test_handle_connected_idempotent_retries_pending_persist(wm, mocker):
  """If _persist_pending_connection failed (e.g. transient FS error), the pending
  entry stays populated. A subsequent CONNECTED event for the same SSID must retry
  the save — otherwise the network is forgotten after restart."""
  mocker.patch("openpilot.system.ui.lib.wifi_manager._generate_wpa_conf")
  wm._wifi_state = WifiState(ssid="systeam", status=ConnectStatus.CONNECTED)
  wm._pending_connection = PendingConnection(ssid="systeam", password="secret", hidden=False, epoch=wm._user_epoch)

  wm._handle_connected("systeam")  # idempotent early-return path

  wm._store.save_network.assert_called_once_with("systeam", psk="secret", hidden=False)
  assert wm._pending_connection is None


def test_reconcile_scanning_then_disconnected_fires_need_auth(wm, mocker):
  """Once SCANNING transitions to DISCONNECTED/INACTIVE, the terminal
  failure path still runs — we just don't run it prematurely on SCANNING."""
  need_auth = mocker.MagicMock()
  wm._need_auth.append(need_auth)
  wm._wifi_state = WifiState(ssid="HiddenAP", status=ConnectStatus.CONNECTING)
  wm._networks = [Network(ssid="HiddenAP", strength=90, security_type=SecurityType.WPA, is_tethering=False)]
  wm._pending_connection = PendingConnection(ssid="HiddenAP", password="secret", hidden=True, epoch=1)

  # First pass: SCANNING → keep waiting, refresh window.
  wm._ctrl.request.return_value = "wpa_state=SCANNING\n"
  wm._reconcile_connecting_state()
  wm.process_callbacks()
  need_auth.assert_not_called()
  assert wm._wifi_state.status == ConnectStatus.CONNECTING

  # Expire the fresh window and let DISCONNECTED resolve it.
  wm._last_connecting_at = time.monotonic() - CONNECTING_STALE_TIMEOUT_SECONDS - 1
  wm._ctrl.request.return_value = "wpa_state=DISCONNECTED\n"
  wm._reconcile_connecting_state()
  wm.process_callbacks()

  assert wm._wifi_state.status == ConnectStatus.DISCONNECTED
  need_auth.assert_called_once_with("HiddenAP")
