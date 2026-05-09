import atexit
import os
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum

from openpilot.common.swaglog import cloudlog
from openpilot.common.utils import atomic_write
from openpilot.system.ui.lib.dhcp_client import DhcpClient
from openpilot.system.ui.lib.wifi_network_store import MeteredType, NetworkStore, NM_CONNECTIONS_DIR
from openpilot.system.ui.lib.wpa_ctrl import (WpaCtrl, WpaCtrlMonitor, SecurityType,
                                               WPA_SUPPLICANT_CONF, WPA_AP_CONF,
                                               _pkill_wpa_supplicant, _wpa_supplicant_running,
                                               _sanitize_for_conf, _format_psk_value, _is_raw_psk,
                                               _generate_wpa_conf, parse_event_ssid,
                                               parse_scan_results, flags_to_security_type,
                                               parse_status, dbm_to_percent, decode_ssid,
                                               ensure_wpa_supplicant, try_attach_ctrl)

try:
  from openpilot.common.params import Params
except Exception:
  Params = None

TETHERING_IP_ADDRESS = "192.168.43.1"
TETHERING_SUBNET = "192.168.43.0/24"
TETHERING_NAT_COMMENT = "openpilot-tethering"
DEFAULT_TETHERING_PASSWORD = "swagswagcomma"
TETHERING_PASSWORD_FILE = "/data/tethering_password"
SCAN_PERIOD_SECONDS = 5
CONNECTING_STALE_TIMEOUT_SECONDS = 5
# Suppress WRONG_KEY events from prior attempts that can clobber fresh credentials on a fast retry.
WRONG_KEY_DEBOUNCE_SECONDS = 2.0


@dataclass(frozen=True)
class Network:
  ssid: str
  strength: int
  security_type: SecurityType
  is_tethering: bool


def sort_networks(networks: list[Network], current_ssid: str | None, saved_ssids: set[str]) -> list[Network]:
  """Sort networks: connected first, then saved, then by signal strength."""
  return sorted(networks, key=lambda n: (n.ssid != current_ssid, n.ssid not in saved_ssids, -n.strength, n.ssid.lower()))


class ConnectStatus(IntEnum):
  DISCONNECTED = 0
  CONNECTING = 1
  CONNECTED = 2


@dataclass(frozen=True)
class WifiState:
  ssid: str | None = None
  status: ConnectStatus = ConnectStatus.DISCONNECTED


@dataclass(frozen=True)
class PendingConnection:
  ssid: str
  password: str
  hidden: bool
  epoch: int


# Match our tethering dnsmasq specifically (system dnsmasq, if any, has a different range).
_DNSMASQ_PGREP = r"dnsmasq.*--dhcp-range=192\.168\.43\.2"


def _our_dnsmasq_running() -> bool:
  return subprocess.run(["pgrep", "-f", _DNSMASQ_PGREP], capture_output=True).returncode == 0


def _tethering_nat_rule(op: str) -> list[str]:
  # Source-subnet MASQUERADE (no `-o <iface>`) so the session survives uplink changes.
  # Mirrors NM's nm-firewall-utils.c:_share_iptables_set_masquerade_sync.
  return ["sudo", "iptables",
          "-t", "nat",
          op, "POSTROUTING",
          "-s", TETHERING_SUBNET, "!", "-d", TETHERING_SUBNET,
          "-j", "MASQUERADE",
          "-m", "comment", "--comment", TETHERING_NAT_COMMENT]


class WifiManager:
  def __init__(self):
    self._networks: list[Network] = []
    self._exit = False

    self._store = NetworkStore()
    self._ctrl: WpaCtrl | None = None
    self._dhcp = DhcpClient()

    # State
    self._wifi_state: WifiState = WifiState()
    self._user_epoch: int = 0
    self._ipv4_address: str = ""
    self._current_network_metered: MeteredType = MeteredType.UNKNOWN
    self._ipv4_forward = False
    self._tethering_active = False
    self._dnsmasq_proc: subprocess.Popen | None = None
    self._pending_connection: PendingConnection | None = None

    self._last_network_scan: float = 0.0
    self._last_connecting_at: float = 0.0
    self._last_connected_recheck: float = 0.0
    self._last_wrong_key_dispatch: dict[str, float] = {}
    self._callback_queue: list[Callable] = []
    self._callback_lock = threading.Lock()
    # Serializes connect_to_network / activate_connection workers so a stale worker
    # can't remove a runtime network just added by a fresher worker.
    self._connect_lock = threading.Lock()
    # Coalesced so an undrained queue (user on another tab) can't grow unboundedly.
    self._networks_updated_pending = False

    self._tethering_ssid = "weedle"
    if Params is not None:
      dongle_id = Params().get("DongleId")
      if dongle_id:
        self._tethering_ssid += "-" + dongle_id[:4]

    # Callbacks
    self._need_auth: list[Callable[[str], None]] = []
    self._activated: list[Callable[[], None]] = []
    self._forgotten: list[Callable[[str | None], None]] = []
    self._networks_updated: list[Callable[[list[Network]], None]] = []
    self._disconnected: list[Callable[[], None]] = []

    self._scan_lock = threading.Lock()
    self._monitor_epoch = 0
    self._scan_thread = threading.Thread(target=self._network_scanner, daemon=True)
    self._state_thread = threading.Thread(target=self._monitor_state, daemon=True)
    self._initialize()
    atexit.register(self.stop)

  def _initialize(self):
    # Load tethering password from file. WPA passphrases may legally include leading
    # or trailing spaces, so only trim the file terminator — never .strip() arbitrary
    # whitespace, or those passwords silently change on next restart.
    try:
      with open(TETHERING_PASSWORD_FILE) as f:
        raw = f.read()
      self._tethering_psk = raw[:-1] if raw.endswith("\n") else raw
    except FileNotFoundError:
      self._tethering_psk = DEFAULT_TETHERING_PASSWORD

    def worker():
      try:
        _generate_wpa_conf(self._store)
        self._ensure_wpa_supplicant()

        # Populate networks before wifi state so the connected SSID's strength is
        # known on first render; otherwise it flashes the disconnected icon.
        self._update_networks(block=True)

        self._init_wifi_state()

        self._scan_thread.start()
        self._state_thread.start()

        cloudlog.debug("WifiManager initialized")
      except Exception:
        cloudlog.exception("WifiManager initialization failed")

    threading.Thread(target=worker, daemon=True).start()

  def _ensure_wpa_supplicant(self):
    ctrl = ensure_wpa_supplicant(lambda: self._exit, NM_CONNECTIONS_DIR)
    if ctrl is not None:
      self._ctrl = ctrl

  def _request(self, cmd: str) -> str:
    ctrl = self._ctrl
    if ctrl is None:
      raise OSError("wpa_supplicant ctrl not attached")
    try:
      return ctrl.request(cmd)
    except OSError:
      # Monitor recv doesn't raise on daemon SIGKILL; the epoch bump kicks it to respawn.
      try:
        ctrl.close()
      except Exception:
        pass
      self._ctrl = None
      self._monitor_epoch += 1
      raise

  def _init_wifi_state(self, block: bool = True):
    def worker():
      if self._ctrl is None:
        return

      epoch = self._user_epoch

      try:
        status = parse_status(self._request("STATUS"))
      except Exception:
        cloudlog.exception("Failed to get wpa_supplicant status")
        return

      wpa_state = status.get("wpa_state", "")
      ssid = status.get("ssid")

      if status.get("mode") == "AP":
        # Hotspot adoption after UI restart. STATUS reports COMPLETED in AP mode too,
        # so the STA path below would flush wlan0 and kill the live hotspot.
        if self._user_epoch != epoch:
          return
        if self._adopt_ap_state(ssid):
          return
        # dnsmasq is gone — the surviving AP daemon is half-broken. Stay
        # DISCONNECTED rather than letting the COMPLETED branch below treat this
        # as a station connect (which would start STA DHCP on wlan0 and clobber
        # the hotspot's address).
        self._wifi_state = WifiState(ssid=None, status=ConnectStatus.DISCONNECTED)
        return

      if wpa_state == "COMPLETED":
        new_status = ConnectStatus.CONNECTED
      elif wpa_state in ("SCANNING", "AUTHENTICATING", "ASSOCIATING", "ASSOCIATED", "4WAY_HANDSHAKE", "GROUP_HANDSHAKE"):
        # Adopt mid-connect state; otherwise a WRONG_KEY event would bypass its current_ssid check.
        new_status = ConnectStatus.CONNECTING
      else:
        new_status = ConnectStatus.DISCONNECTED
        ssid = None

      if self._user_epoch != epoch:
        return

      if new_status == ConnectStatus.CONNECTED and ssid is not None:
        # We own DHCP — must (re)start udhcpc since the previous UI's died with its parent.
        self._handle_connected(ssid)
      else:
        self._wifi_state = WifiState(ssid=ssid, status=new_status)

    if block:
      worker()
    else:
      threading.Thread(target=worker, daemon=True).start()

  def add_callbacks(self, need_auth: Callable[[str], None] | None = None,
                    activated: Callable[[], None] | None = None,
                    forgotten: Callable[[str], None] | None = None,
                    networks_updated: Callable[[list[Network]], None] | None = None,
                    disconnected: Callable[[], None] | None = None):
    if need_auth is not None:
      self._need_auth.append(need_auth)
    if activated is not None:
      self._activated.append(activated)
    if forgotten is not None:
      self._forgotten.append(forgotten)
    if networks_updated is not None:
      self._networks_updated.append(networks_updated)
    if disconnected is not None:
      self._disconnected.append(disconnected)

  @property
  def networks(self) -> list[Network]:
    return sort_networks(self._networks, self._wifi_state.ssid, self._store.saved_ssids())

  @property
  def wifi_state(self) -> WifiState:
    return self._wifi_state

  @property
  def ipv4_address(self) -> str:
    return self._ipv4_address

  @property
  def current_network_metered(self) -> MeteredType:
    return self._current_network_metered

  @property
  def connecting_to_ssid(self) -> str | None:
    wifi_state = self._wifi_state
    return wifi_state.ssid if wifi_state.status == ConnectStatus.CONNECTING else None

  @property
  def connected_ssid(self) -> str | None:
    wifi_state = self._wifi_state
    return wifi_state.ssid if wifi_state.status == ConnectStatus.CONNECTED else None

  @property
  def tethering_password(self) -> str:
    return self._tethering_psk

  def _set_connecting(self, ssid: str | None):
    self._user_epoch += 1
    self._last_connecting_at = time.monotonic() if ssid is not None else 0.0
    self._wifi_state = WifiState(ssid=ssid, status=ConnectStatus.DISCONNECTED if ssid is None else ConnectStatus.CONNECTING)

  def _set_pending_connection(self, ssid: str, password: str, hidden: bool):
    self._pending_connection = PendingConnection(ssid=ssid, password=password, hidden=hidden, epoch=self._user_epoch)

  def _clear_pending_connection(self, ssid: str | None = None):
    if self._pending_connection is None:
      return
    if ssid is None or self._pending_connection.ssid == ssid:
      self._pending_connection = None

  def _persist_pending_connection(self, ssid: str | None):
    pending = self._pending_connection
    if pending is None:
      return

    if ssid != pending.ssid or pending.epoch != self._user_epoch:
      return

    # On filesystem error, keep credentials for later retry and swallow so
    # _handle_connected can still fire DHCP/activated callbacks.
    try:
      self._store.save_network(ssid, psk=pending.password, hidden=pending.hidden)
      _generate_wpa_conf(self._store)
    except Exception:
      cloudlog.exception("Failed to persist pending connection for %s", ssid)
      return
    self._pending_connection = None

  def _enqueue_callbacks(self, cbs: list[Callable], *args):
    with self._callback_lock:
      for cb in cbs:
        self._callback_queue.append(lambda _cb=cb: _cb(*args))

  def _mark_networks_updated(self):
    # Coalesces across scans so the queue stays O(1) when the UI isn't draining.
    with self._callback_lock:
      self._networks_updated_pending = True

  def process_callbacks(self):
    with self._callback_lock:
      to_run, self._callback_queue = self._callback_queue, []
      if self._networks_updated_pending:
        self._networks_updated_pending = False
        networks_cbs = list(self._networks_updated)
      else:
        networks_cbs = None
    for cb in to_run:
      cb()
    if networks_cbs:
      # Fire with the latest snapshot, not one captured when we were flagged.
      snapshot = self.networks
      for cb in networks_cbs:
        cb(snapshot)

  def _monitor_state(self):
    # If pgrep keeps finding our daemon but try_attach_ctrl keeps returning None,
    # the process is alive with a dead/missing ctrl socket. After this many
    # consecutive attach failures, force a full respawn instead of looping forever.
    ATTACH_FAILURES_BEFORE_RESPAWN = 3
    attach_failures = 0
    while not self._exit:
      if self._ctrl is None:
        # _start_tethering closes _ctrl and pkills the STA daemon before the AP daemon
        # is up. Spawning STA in that gap races AP bringup and can keep the hotspot
        # off wlan0. Wait for tethering to finish; _start_tethering will rebind _ctrl.
        if self._tethering_active:
          time.sleep(1)
          continue
        # No owned daemon? Spawn one so wifi doesn't stay dead after a failed
        # initial bringup or a crash. Otherwise just attach.
        daemon_alive = _wpa_supplicant_running(WPA_SUPPLICANT_CONF) or _wpa_supplicant_running(WPA_AP_CONF)
        stale_daemon = daemon_alive and attach_failures >= ATTACH_FAILURES_BEFORE_RESPAWN
        if daemon_alive and not stale_daemon:
          ctrl = try_attach_ctrl()
          if ctrl is None:
            attach_failures += 1
            time.sleep(2)
            continue
          self._ctrl = ctrl
          attach_failures = 0
        else:
          self._ensure_wpa_supplicant()
          attach_failures = 0
          if self._ctrl is None:
            time.sleep(2)
            continue
      monitor = None
      try:
        epoch = self._monitor_epoch
        monitor = WpaCtrlMonitor()
        monitor.open()
        while not self._exit and self._monitor_epoch == epoch:
          event = monitor.recv(timeout=1.0)
          if event is None:
            continue
          self._handle_event(event)
      except Exception:
        cloudlog.exception("wpa_supplicant monitor error, reconnecting...")
        # Drop the ctrl handle so the next iteration re-attaches (or respawns
        # if the daemon actually died); otherwise we'd wedge on a dead socket.
        if self._ctrl is not None:
          try:
            self._ctrl.close()
          except Exception:
            pass
          self._ctrl = None
      finally:
        if monitor is not None:
          try:
            monitor.close()
          except Exception:
            pass
        time.sleep(2)

  def _adopt_ap_state(self, ssid: str | None) -> bool:
    """Mark a live hotspot as active without touching dnsmasq/iptables — those daemons
    survive UI restart via start_new_session, so adoption only updates manager state.
    On refusal (dnsmasq dead) closes the AP ctrl handle and pkills the orphan AP
    daemon so the monitor's STA recovery picks up; otherwise station connect
    requests would still flow through the AP ctrl socket until the user toggled
    tethering manually."""
    if not _our_dnsmasq_running():
      cloudlog.warning("AP daemon present but our dnsmasq isn't; refusing adoption and tearing down orphan AP")
      if self._ctrl is not None:
        try:
          self._ctrl.close()
        except Exception:
          cloudlog.exception("Failed to close AP ctrl on adoption refusal")
        self._ctrl = None
      _pkill_wpa_supplicant(WPA_AP_CONF)
      return False
    self._tethering_active = True
    self._wifi_state = WifiState(ssid=ssid or self._tethering_ssid, status=ConnectStatus.CONNECTED)
    self._ipv4_address = TETHERING_IP_ADDRESS
    self._enqueue_callbacks(self._activated)
    return True

  def _handle_connected(self, ssid: str):
    """Transition to CONNECTED. Idempotent on (ssid, CONNECTED) so the monitor and
    reconcile paths can both call in without each one killing the previous udhcpc."""
    if (self._wifi_state.status == ConnectStatus.CONNECTED
        and self._wifi_state.ssid == ssid):
      # If a prior persist hit a transient FS error, _pending_connection is still
      # populated for this SSID. Without the retry here, repeat CONNECTED events
      # short-circuit and the network is forgotten after restart.
      pending = self._pending_connection
      if pending is not None and pending.ssid == ssid:
        self._persist_pending_connection(ssid)
      return
    self._last_connecting_at = 0.0
    self._wifi_state = WifiState(ssid=ssid, status=ConnectStatus.CONNECTED)
    self._persist_pending_connection(ssid)
    # Re-enable saved networks so wpa_supplicant can auto-roam: SELECT_NETWORK disables
    # every other network as a side effect.
    if self._ctrl is not None:
      try:
        self._request("ENABLE_NETWORK all")
      except Exception:
        cloudlog.exception("Failed to re-enable saved networks for auto-roam")
    self._dhcp.start()
    self._enqueue_callbacks(self._activated)
    self._poll_for_ip()

  def _handle_event(self, event: str):
    """Dispatch wpa_supplicant event to state machine."""
    if "CTRL-EVENT-SCAN-RESULTS" in event:
      self._update_networks(block=False)

    elif "CTRL-EVENT-CONNECTED" in event:
      # Extract SSID from "Connection to xx:xx:xx:xx:xx:xx completed [id=N id_str=]"
      epoch = self._user_epoch
      ssid = self._wifi_state.ssid

      # Get actual SSID from STATUS
      if self._ctrl:
        try:
          status = parse_status(self._request("STATUS"))
          ssid = status.get("ssid", ssid)
        except Exception:
          pass

      if self._user_epoch != epoch:
        return

      if ssid:
        self._handle_connected(ssid)

    elif "CTRL-EVENT-DISCONNECTED" in event:
      if self._tethering_active:
        return  # Ignore disconnects during tethering transitions

      epoch = self._user_epoch

      # Don't clear state if we're connecting to something (user action in progress)
      if self._wifi_state.status == ConnectStatus.CONNECTING:
        return

      if self._user_epoch != epoch:
        return

      self._wifi_state = WifiState(ssid=None, status=ConnectStatus.DISCONNECTED)
      self._dhcp.stop()
      self._ipv4_address = ""
      self._current_network_metered = MeteredType.UNKNOWN
      self._enqueue_callbacks(self._disconnected)

    elif "TEMP-DISABLED" in event and "reason=WRONG_KEY" in event:
      event_ssid = parse_event_ssid(event)
      if event_ssid is not None:
        current_ssid = self._wifi_state.ssid
        # Auto-connect may leave us in CONNECTING with ssid=None; the event's SSID is authoritative.
        connecting_unknown = (
          self._wifi_state.status == ConnectStatus.CONNECTING
          and current_ssid is None
        )
        if connecting_unknown or (current_ssid and event_ssid == current_ssid):
          # Per-SSID debounce: suppress stale repeats for this SSID without
          # masking a legitimate WRONG_KEY on a different network.
          now = time.monotonic()
          if now - self._last_wrong_key_dispatch.get(event_ssid, 0.0) < WRONG_KEY_DEBOUNCE_SECONDS:
            return
          self._last_wrong_key_dispatch[event_ssid] = now
          self._clear_pending_connection(event_ssid)
          # The runtime network from _add_and_select_network was never persisted; if
          # we leave it in wpa_supplicant, ENABLE_NETWORK all just re-arms the bad PSK
          # for retry. Drop it before re-enabling the rest.
          if self._ctrl is not None:
            try:
              self._remove_wpa_network(event_ssid)
              self._request("ENABLE_NETWORK all")
            except Exception:
              cloudlog.exception("Failed to re-enable saved networks after WRONG_KEY")
          self._enqueue_callbacks(self._need_auth, event_ssid)
          self._set_connecting(None)
          # CTRL-EVENT-DISCONNECTED is ignored while CONNECTING, so tear down
          # DHCP/IP/metered here ourselves in case it arrived before WRONG_KEY.
          self._dhcp.stop()
          self._ipv4_address = ""
          self._current_network_metered = MeteredType.UNKNOWN
          self._enqueue_callbacks(self._disconnected)

    elif "Trying to associate with" in event or "Associated with" in event:
      # Auto-connect case: wpa_supplicant is connecting on its own
      if self._wifi_state.status == ConnectStatus.DISCONNECTED:
        epoch = self._user_epoch
        ssid = None
        if self._ctrl:
          try:
            status = parse_status(self._request("STATUS"))
            ssid = status.get("ssid")
          except Exception:
            pass
        if self._user_epoch != epoch:
          return
        self._last_connecting_at = time.monotonic()
        self._wifi_state = WifiState(ssid=ssid, status=ConnectStatus.CONNECTING)

  def _network_scanner(self):
    while not self._exit:
      self._reconcile_connecting_state()
      if not self._tethering_active:
        if time.monotonic() - self._last_network_scan > SCAN_PERIOD_SECONDS:
          self._request_scan()
          self._last_network_scan = time.monotonic()
      time.sleep(1 / 2.)

  def _request_scan(self):
    if self._ctrl is None:
      return
    try:
      self._request("SCAN")
    except Exception:
      cloudlog.exception("Failed to request scan")

  def _reconcile_connecting_state(self):
    current_state = self._wifi_state
    if self._ctrl is None or self._tethering_active:
      return

    # Detect missed CONNECTED event (e.g. monitor was reconnecting after tethering stop)
    if current_state.status == ConnectStatus.DISCONNECTED:
      epoch = self._user_epoch
      try:
        status = parse_status(self._request("STATUS"))
      except Exception:
        return
      # A user tap during the blocking STATUS bumped the epoch; their CONNECTING
      # state is fresh, so don't synthesize a connected from the stale STATUS.
      if self._user_epoch != epoch:
        return
      # wpa_supplicant reports COMPLETED in AP mode too; STA path would flush the hotspot. Re-adopt
      # so a missed startup adoption (e.g. transient STATUS failure) doesn't strand us in DISCONNECTED
      # while still attached to the AP daemon, which would route station actions to the AP socket.
      if status.get("mode") == "AP":
        if self._adopt_ap_state(status.get("ssid")):
          return
        # dnsmasq missing — half-broken AP. Stay DISCONNECTED so the user can recover via tethering toggle.
        return
      if status.get("wpa_state") == "COMPLETED" and status.get("ssid"):
        self._handle_connected(status["ssid"])
      return

    # Detect missed DISCONNECTED if the monitor dropped an event. Gated at
    # SCAN_PERIOD_SECONDS to avoid STATUS spam.
    if current_state.status == ConnectStatus.CONNECTED:
      now = time.monotonic()
      if now - self._last_connected_recheck < SCAN_PERIOD_SECONDS:
        return
      self._last_connected_recheck = now
      epoch = self._user_epoch
      try:
        status = parse_status(self._request("STATUS"))
      except Exception:
        return
      # User started a new connect while we were blocked in STATUS; their fresh
      # CONNECTING state must not be clobbered by stale STATUS results below.
      if self._user_epoch != epoch:
        return
      wpa_state = status.get("wpa_state", "")
      status_ssid = status.get("ssid")
      if wpa_state == "COMPLETED" and status_ssid == current_state.ssid:
        return
      if wpa_state == "COMPLETED" and status_ssid:
        # Roamed while the monitor was down; adopt the new network instead of
        # synthesizing a disconnect that would flush the live lease.
        self._handle_connected(status_ssid)
        return
      # Normal roam/rekey transits through these states briefly; treating them as
      # disconnect would flush the live udhcpc lease for nothing. Wait for the
      # next sample to see the terminal state.
      if wpa_state in ("SCANNING", "AUTHENTICATING", "ASSOCIATING", "ASSOCIATED",
                       "4WAY_HANDSHAKE", "GROUP_HANDSHAKE"):
        return
      # Actually disconnected under us.
      self._wifi_state = WifiState(ssid=None, status=ConnectStatus.DISCONNECTED)
      self._dhcp.stop()
      self._ipv4_address = ""
      self._current_network_metered = MeteredType.UNKNOWN
      self._enqueue_callbacks(self._disconnected)
      return

    # Reconcile even if ssid is None — STATUS below tells us definitively.
    if current_state.status != ConnectStatus.CONNECTING:
      return
    if time.monotonic() - self._last_connecting_at < CONNECTING_STALE_TIMEOUT_SECONDS:
      return

    # Snapshot the user epoch so a STATUS reply for a stale connect attempt can't
    # clobber a fresh user-initiated one that started while we were blocked below.
    epoch = self._user_epoch
    try:
      status = parse_status(self._request("STATUS"))
    except Exception:
      cloudlog.exception("Failed to reconcile wifi state from STATUS")
      return
    if self._user_epoch != epoch:
      return

    wpa_state = status.get("wpa_state", "")
    status_ssid = status.get("ssid")

    if wpa_state == "COMPLETED" and status_ssid:
      self._handle_connected(status_ssid)
    elif wpa_state == "SCANNING":
      # Hidden-SSID joins can legitimately stay in SCANNING past the stale window; defer, don't fail.
      self._last_connecting_at = time.monotonic()
    elif wpa_state in ("DISCONNECTED", "INACTIVE"):
      network = next((n for n in self._networks if n.ssid == current_state.ssid), None)
      if network is not None and network.security_type != SecurityType.OPEN:
        self._enqueue_callbacks(self._need_auth, current_state.ssid)
      self._clear_pending_connection(current_state.ssid)
      # Drop the unsaved runtime network so ENABLE_NETWORK all doesn't re-arm
      # the failed credential for another retry.
      try:
        if current_state.ssid:
          self._remove_wpa_network(current_state.ssid)
        self._request("ENABLE_NETWORK all")
      except Exception:
        cloudlog.exception("Failed to re-enable saved networks after stale CONNECTING")
      self._set_connecting(None)
      self._dhcp.stop()
      self._ipv4_address = ""
      self._current_network_metered = MeteredType.UNKNOWN
      self._enqueue_callbacks(self._disconnected)

  def _update_networks(self, block: bool = True):
    def worker():
      with self._scan_lock:
        if self._ctrl is None:
          return

        try:
          raw = self._request("SCAN_RESULTS")
        except Exception:
          cloudlog.exception("Failed to get scan results")
          return

        results = parse_scan_results(raw)

        # Group by SSID, keep strongest signal
        ssid_map: dict[str, list] = {}
        for r in results:
          if not r.ssid:
            continue
          if r.ssid not in ssid_map:
            ssid_map[r.ssid] = []
          ssid_map[r.ssid].append(r)

        networks = []
        for ssid, aps in ssid_map.items():
          strongest = max(aps, key=lambda a: a.signal)
          security = flags_to_security_type(strongest.flags)
          is_tethering = ssid == self._tethering_ssid
          strength = 100 if is_tethering else dbm_to_percent(strongest.signal)
          networks.append(Network(ssid=ssid, strength=strength, security_type=security, is_tethering=is_tethering))

        # Never replace with empty — stale data is better than no data
        if networks:
          self._networks = networks
        self._update_active_connection_info()
        self._mark_networks_updated()

    if block:
      worker()
    else:
      threading.Thread(target=worker, daemon=True).start()

  def _poll_for_ip(self):
    """Poll for IP address after DHCP starts, then update connection info."""
    epoch = self._user_epoch

    def worker():
      for _ in range(50):  # 10 seconds max
        if self._wifi_state.status != ConnectStatus.CONNECTED or self._user_epoch != epoch:
          return
        self._update_active_connection_info()
        if self._ipv4_address:
          return
        time.sleep(0.2)
    threading.Thread(target=worker, daemon=True).start()

  def _update_active_connection_info(self):
    ipv4_address = ""
    metered = MeteredType.UNKNOWN

    if self._wifi_state.status == ConnectStatus.CONNECTED:
      # Try wpa_cli STATUS for ip_address first (works regardless of network namespace)
      if self._ctrl:
        try:
          status = parse_status(self._request("STATUS"))
          ipv4_address = status.get("ip_address", "")
        except Exception:
          pass

      # Fallback to ip command
      if not ipv4_address:
        try:
          result = subprocess.run(["ip", "-4", "-o", "addr", "show", "wlan0"],
                                  capture_output=True, text=True, timeout=2)
          for line in result.stdout.strip().split("\n"):
            if "inet " in line:
              parts = line.split()
              inet_idx = parts.index("inet")
              ipv4_address = parts[inet_idx + 1].split("/")[0]
              break
        except Exception:
          pass

      # Metered from store
      ssid = self._wifi_state.ssid
      if ssid:
        metered = self._store.get_metered(ssid)

    self._ipv4_address = ipv4_address
    self._current_network_metered = metered

  def connect_to_network(self, ssid: str, password: str, hidden: bool = False):
    # Backend guard: non-UI entry points (hidden-network dialog, automation) can still reach here.
    if self._tethering_active:
      cloudlog.warning(f"Ignoring connect to {ssid!r} while tethering is active")
      return
    self._set_connecting(ssid)
    self._set_pending_connection(ssid, password, hidden)
    epoch = self._user_epoch

    def worker():
      if self._ctrl is None:
        cloudlog.warning("No wpa_supplicant connection")
        # If a fresher attempt landed during the supplicant-restart window, don't
        # let this stale worker emit a false disconnect for it.
        if self._user_epoch != epoch:
          return
        self._clear_pending_connection(ssid)
        # _init_wifi_state is a no-op while _ctrl is None, so reset CONNECTING inline.
        self._set_connecting(None)
        self._enqueue_callbacks(self._disconnected)
        return

      # Serialize against other workers and recheck epoch inside the lock — otherwise
      # a stale worker that passed an early epoch check could remove the runtime
      # network a fresher worker just added (rapid same-SSID retries).
      with self._connect_lock:
        if self._user_epoch != epoch:
          return
        try:
          self._remove_wpa_network(ssid)
          self._add_and_select_network(ssid, password, hidden)
        except Exception:
          cloudlog.exception(f"Failed to connect to {ssid}")
          # Setup failed before SELECT_NETWORK could land; STATUS won't tell us
          # anything useful and _init_wifi_state would silently set DISCONNECTED
          # without notifying the UI. Reset CONNECTING and fire disconnected
          # ourselves so the UI unsticks.
          self._clear_pending_connection(ssid)
          self._set_connecting(None)
          self._enqueue_callbacks(self._disconnected)

    threading.Thread(target=worker, daemon=True).start()

  def forget_connection(self, ssid: str, block: bool = False):
    def worker():
      self._clear_pending_connection(ssid)
      was_connected = self._wifi_state.ssid == ssid and self._wifi_state.status == ConnectStatus.CONNECTED

      existed = self._store.contains(ssid)
      removed = self._store.remove(ssid)
      if existed and not removed:
        # rm failed — the on-disk file survives and _load will restore the entry
        # at next start. Don't tear down the runtime/regenerate config or fire
        # `forgotten`, or the UI will lie about state until the file gets restored.
        cloudlog.warning(f"forget_connection: failed to remove {ssid} from disk; leaving runtime intact")
        return
      if not removed:
        cloudlog.warning(f"Trying to forget unknown connection: {ssid}")

      _generate_wpa_conf(self._store)

      if self._ctrl:
        try:
          if was_connected:
            self._request("DISCONNECT")
          self._remove_wpa_network(ssid)
          self._request("ENABLE_NETWORK all")
          # Reassociate only when the forgotten profile was the live link, so the
          # device falls back to the next saved network. Otherwise REASSOCIATE
          # would briefly drop an unrelated active connection.
          if was_connected:
            self._request("REASSOCIATE")
        except Exception:
          cloudlog.exception(f"Failed to reconfigure after forgetting {ssid}")

      self._enqueue_callbacks(self._forgotten, ssid)

    if block:
      worker()
    else:
      threading.Thread(target=worker, daemon=True).start()

  def activate_connection(self, ssid: str, block: bool = False):
    if self._tethering_active:
      cloudlog.warning(f"Ignoring activate {ssid!r} while tethering is active")
      return
    self._set_connecting(ssid)
    self._clear_pending_connection()
    epoch = self._user_epoch

    def worker():
      if self._ctrl is None:
        cloudlog.warning(f"No wpa_supplicant connection for activate {ssid}")
        # Skip the reset if a fresher attempt has already moved on, otherwise
        # this stale worker would emit a false disconnect for the new attempt.
        if self._user_epoch != epoch:
          return
        # _init_wifi_state is a no-op while _ctrl is None, so reset CONNECTING inline.
        self._set_connecting(None)
        self._enqueue_callbacks(self._disconnected)
        return

      def reset_to_disconnected():
        # Mirror the _ctrl is None recovery: _init_wifi_state silently sets DISCONNECTED
        # without firing the callback, which leaves the UI wedged at CONNECTING.
        self._set_connecting(None)
        self._enqueue_callbacks(self._disconnected)

      # Serialize against other connect/activate workers and recheck epoch inside
      # the lock so a stale worker can't mutate networks added by a fresher one.
      with self._connect_lock:
        if self._user_epoch != epoch:
          return
        try:
          ids = self._list_network_ids(ssid)
          if ids:
            self._request(f"SELECT_NETWORK {ids[0]}")
          else:
            # Network not in wpa_supplicant's runtime list — add from store
            entry = self._store.get(ssid)
            if entry:
              self._add_and_select_network(ssid, entry.get("psk", ""), entry.get("hidden", False))
            else:
              cloudlog.warning(f"Network {ssid} not found for activation")
              reset_to_disconnected()
        except Exception:
          cloudlog.exception(f"Failed to activate {ssid}")
          reset_to_disconnected()

    if block:
      worker()
    else:
      threading.Thread(target=worker, daemon=True).start()

  def _add_and_select_network(self, ssid: str, psk: str = "", hidden: bool = False):
    """Add a network and select it. Every SET_NETWORK is checked so a bad PSK/key_mgmt
    surfaces an immediate error instead of a delayed WRONG_KEY; orphans get REMOVE_NETWORK'd."""
    net_id = self._request("ADD_NETWORK").strip()
    if not net_id.isdigit():
      raise RuntimeError(f"ADD_NETWORK failed: {net_id}")

    try:
      safe_ssid = _sanitize_for_conf(ssid)
      self._wpa_set_network(net_id, "ssid", f'"{safe_ssid}"')
      if psk:
        self._wpa_set_network(net_id, "psk", _format_psk_value(psk))
      else:
        self._wpa_set_network(net_id, "key_mgmt", "NONE")
      if hidden:
        self._wpa_set_network(net_id, "scan_ssid", "1")
      resp = self._request(f"SELECT_NETWORK {net_id}").strip()
      if not resp.startswith("OK"):
        raise RuntimeError(f"SELECT_NETWORK {net_id} failed: {resp}")
    except Exception:
      try:
        self._request(f"REMOVE_NETWORK {net_id}")
      except Exception:
        cloudlog.exception(f"Failed to clean up orphaned network {net_id}")
      raise

  def _wpa_set_network(self, net_id: str, key: str, value: str):
    """SET_NETWORK wrapper that raises on wpa_supplicant FAIL responses."""
    resp = self._request(f"SET_NETWORK {net_id} {key} {value}").strip()
    if not resp.startswith("OK"):
      raise RuntimeError(f"SET_NETWORK {net_id} {key} failed: {resp}")

  def _list_network_ids(self, ssid: str) -> list[str]:
    """Return all wpa_supplicant network ids matching SSID. LIST_NETWORKS emits
    printf_encode'd SSIDs — decode before comparing or non-ASCII SSIDs silently miss."""
    if self._ctrl is None:
      return []
    try:
      raw = self._request("LIST_NETWORKS")
      return [parts[0] for line in raw.strip().split("\n")[1:]
              if len(parts := line.split("\t")) >= 2 and decode_ssid(parts[1]) == ssid]
    except Exception:
      cloudlog.exception("Failed to list networks")
      return []

  def _remove_wpa_network(self, ssid: str):
    """Remove all wpa_supplicant network entries matching SSID."""
    for net_id in self._list_network_ids(ssid):
      try:
        self._request(f"REMOVE_NETWORK {net_id}")
      except Exception:
        cloudlog.exception(f"Failed to remove network {ssid}")

  def is_tethering_active(self) -> bool:
    return self._tethering_active

  def is_connection_saved(self, ssid: str) -> bool:
    return self._store.contains(ssid)

  def set_tethering_password(self, password: str):
    # wpa_supplicant accepts either an 8–63 BYTE passphrase or exactly 64 hex chars
    # as a pre-hashed PSK. Counting characters lets a 32-char non-ASCII password
    # (e.g. 32 'é' = 64 UTF-8 bytes) past the check; AP bringup then fails forever
    # because the bad value is persisted to /data/tethering_password.
    pw_bytes = len(password.encode("utf-8"))
    if not (8 <= pw_bytes <= 63 or _is_raw_psk(password)):
      cloudlog.warning(f"set_tethering_password: rejecting invalid password (bytes={pw_bytes})")
      # The UI disables tethering controls before calling this and only re-enables
      # them from activated/disconnected. Notify so the controls don't stay stuck.
      self._enqueue_callbacks(self._activated if self._tethering_active else self._disconnected)
      return
    def worker():
      try:
        with atomic_write(TETHERING_PASSWORD_FILE, overwrite=True) as f:
          f.write(password)
      except Exception:
        cloudlog.exception("Failed to persist tethering password; runtime state unchanged")
        return
      self._tethering_psk = password
      if self._tethering_active:
        # Re-assert the flag during restart so is_tethering_active() stays in sync
        # while the hotspot is still running; mirror set_tethering_active's rollback.
        self._stop_tethering()
        self._tethering_active = True
        try:
          self._start_tethering()
        except Exception:
          cloudlog.exception("Failed to restart tethering after password change")
          try:
            self._stop_tethering()
          except Exception:
            cloudlog.exception("Tethering rollback also failed")
            self._tethering_active = False
            self._wifi_state = WifiState()
            self._enqueue_callbacks(self._disconnected)
    threading.Thread(target=worker, daemon=True).start()

  def set_ipv4_forward(self, enabled: bool):
    self._ipv4_forward = enabled

  def set_tethering_active(self, active: bool):
    # Enable: assert flag synchronously so UI/reconcile see "tethering in progress" immediately.
    # Disable: _stop_tethering clears the flag at the end, after _ctrl is back in STA mode.
    if active:
      self._tethering_active = True
    def worker():
      if active:
        try:
          self._start_tethering()
          if not self._ipv4_forward:
            time.sleep(5)
            cloudlog.warning("net.ipv4.ip_forward = 0")
            subprocess.run(["sudo", "sysctl", "net.ipv4.ip_forward=0"], check=False)
        except Exception:
          cloudlog.exception("Failed to start tethering, rolling back")
          try:
            # Safe on a partial bringup.
            self._stop_tethering()
          except Exception:
            cloudlog.exception("Tethering rollback also failed")
            self._tethering_active = False
            self._wifi_state = WifiState()
            self._enqueue_callbacks(self._disconnected)
      else:
        try:
          self._stop_tethering()
        except Exception:
          cloudlog.exception("Failed to stop tethering")
          # Force-clear so the UI isn't stuck reporting tethering active.
          self._tethering_active = False
    threading.Thread(target=worker, daemon=True).start()

  def set_current_network_metered(self, metered: MeteredType):
    def worker():
      if self._tethering_active:
        return
      ssid = self._wifi_state.ssid
      if ssid:
        self._store.set_metered(ssid, int(metered))
        self._current_network_metered = metered
    threading.Thread(target=worker, daemon=True).start()

  def _start_tethering(self):
    # TODO: kill-and-respawn is incompatible with a system-managed wpa_supplicant.
    # Switch to a single-daemon flip: ADD_NETWORK mode=2/freq=2437, DISABLE/ENABLE_NETWORK.
    self._set_connecting(self._tethering_ssid)

    psk = self._tethering_psk

    # Close existing control socket
    if self._ctrl:
      self._ctrl.close()
      self._ctrl = None

    # Target only our configs — never touch a system-managed daemon. Kill any
    # surviving AP daemon too so the new spawn isn't blocked by an orphan
    # holding wlan0 with stale credentials.
    self._monitor_epoch += 1
    _pkill_wpa_supplicant(WPA_SUPPLICANT_CONF)
    _pkill_wpa_supplicant(WPA_AP_CONF)
    self._dhcp.stop()
    time.sleep(0.5)

    # Write AP config
    safe_tether_ssid = _sanitize_for_conf(self._tethering_ssid)
    lines = ["ctrl_interface=/var/run/wpa_supplicant", "ap_scan=2", "",
             "network={", f'  ssid="{safe_tether_ssid}"', "  mode=2",
             "  frequency=2437", "  key_mgmt=WPA-PSK", f'  psk={_format_psk_value(psk)}', "}", ""]
    ap_conf = "\n".join(lines)
    fd = os.open(WPA_AP_CONF, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
      f.write(ap_conf)

    # Start AP wpa_supplicant
    subprocess.run(["sudo", "wpa_supplicant", "-B", "-i", "wlan0", "-c", WPA_AP_CONF, "-D", "nl80211"], check=False)
    time.sleep(1)

    # Configure AP interface. addr/link failures here mean wlan0 has no
    # 192.168.43.1 — clients could associate and pull dnsmasq leases, but the AP
    # has no gateway IP. Raise so the existing rollback path runs instead.
    subprocess.run(["sudo", "ip", "addr", "flush", "dev", "wlan0"], check=False)
    subprocess.run(["sudo", "ip", "addr", "add", f"{TETHERING_IP_ADDRESS}/24", "dev", "wlan0"], check=True)
    subprocess.run(["sudo", "ip", "link", "set", "wlan0", "up"], check=True)

    # Start dnsmasq for DHCP
    subprocess.run(["sudo", "killall", "-q", "dnsmasq"], check=False)
    self._dnsmasq_proc = subprocess.Popen([
      "sudo", "dnsmasq",
      "--interface=wlan0",
      "--bind-interfaces",
      "--dhcp-range=192.168.43.2,192.168.43.254,24h",
      "--dhcp-leasefile=/tmp/dnsmasq.leases",
      "--no-daemon", "--log-queries",
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
      start_new_session=True)
    # Clients can't obtain leases without dnsmasq; fail fast so rollback runs
    # instead of advertising a broken hotspot.
    time.sleep(0.2)
    if self._dnsmasq_proc.poll() is not None:
      rc = self._dnsmasq_proc.returncode
      self._dnsmasq_proc = None
      raise RuntimeError(f"dnsmasq exited during tethering bringup (rc={rc})")

    # Flush stale copies (idempotent), plus legacy `-o <iface>` rules from older openpilot.
    for _ in range(4):
      result = subprocess.run(_tethering_nat_rule("-D"), capture_output=True, check=False)
      if result.returncode != 0:
        break
    for iface in ("wwan0", "rmnet_data0", "eth0"):
      for _ in range(4):
        result = subprocess.run(["sudo", "iptables", "-t", "nat", "-D", "POSTROUTING", "-o", iface, "-j", "MASQUERADE"],
                                capture_output=True, check=False)
        if result.returncode != 0:
          break
    # Without MASQUERADE or ip_forward the AP comes up but clients can't reach the
    # uplink — UI reports a healthy hotspot with broken sharing. Treat both as hard
    # failures so the existing _start_tethering rollback path tears the AP back down.
    subprocess.run(_tethering_nat_rule("-A"), check=True)
    if self._ipv4_forward:
      subprocess.run(["sudo", "sysctl", "net.ipv4.ip_forward=1"], check=True)

    # Verify bringup: pgrep our config (so a surviving foreign or stale AP daemon
    # can't masquerade as our hotspot with stale credentials), then attach + require
    # STATUS mode=AP.
    if not _wpa_supplicant_running(WPA_AP_CONF):
      raise RuntimeError("AP wpa_supplicant did not start with our config; another daemon likely still owns wlan0")
    try:
      ctrl = WpaCtrl()
      ctrl.open()
    except Exception as e:
      raise RuntimeError(f"AP wpa_supplicant bringup failed: {e}") from e
    try:
      status = parse_status(ctrl.request("STATUS"))
    except Exception as e:
      ctrl.close()
      raise RuntimeError(f"AP wpa_supplicant STATUS failed: {e}") from e
    if status.get("mode") != "AP":
      actual_mode = status.get("mode")
      ctrl.close()
      raise RuntimeError(f"AP wpa_supplicant bringup did not take over wlan0 (mode={actual_mode!r}); another daemon likely owns the interface")
    self._ctrl = ctrl

    self._wifi_state = WifiState(ssid=self._tethering_ssid, status=ConnectStatus.CONNECTED)
    self._ipv4_address = TETHERING_IP_ADDRESS
    self._enqueue_callbacks(self._activated)

  def _stop_tethering(self):
    # Kill dnsmasq
    subprocess.run(["sudo", "killall", "-q", "dnsmasq"], check=False)
    if self._dnsmasq_proc is not None:
      try:
        self._dnsmasq_proc.wait(timeout=3)
      except Exception:
        pass
      self._dnsmasq_proc = None

    # Remove NAT. Loop in case a prior start left duplicates.
    for _ in range(4):
      result = subprocess.run(_tethering_nat_rule("-D"), capture_output=True, check=False)
      if result.returncode != 0:
        break

    # Close control socket
    if self._ctrl:
      self._ctrl.close()
      self._ctrl = None

    # Stop AP wpa_supplicant (only the one running our AP config).
    self._monitor_epoch += 1
    _pkill_wpa_supplicant(WPA_AP_CONF)
    time.sleep(0.5)

    # Flush AP IP
    subprocess.run(["sudo", "ip", "addr", "flush", "dev", "wlan0"], check=False)

    # Refresh config for the spawn path; _ensure_wpa_supplicant picks attach vs spawn.
    _generate_wpa_conf(self._store)
    self._ensure_wpa_supplicant()

    self._tethering_active = False
    self._wifi_state = WifiState(ssid=None, status=ConnectStatus.DISCONNECTED)
    self._ipv4_address = ""
    self._enqueue_callbacks(self._disconnected)

  def __del__(self):
    self.stop()

  def stop(self):
    if not self._exit:
      self._exit = True
      if self._scan_thread.is_alive():
        self._scan_thread.join()
      if self._state_thread.is_alive():
        self._state_thread.join()
      if self._tethering_active:
        self._stop_tethering()
      if self._ctrl is not None:
        self._ctrl.close()
      self._dhcp.stop()
