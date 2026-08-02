import atexit
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

from openpilot.common.swaglog import cloudlog
from openpilot.common.utils import atomic_write
from openpilot.system.ui.lib.dhcp_client import DhcpClient
from openpilot.system.ui.lib.wifi_network_store import MeteredType, NetworkStore
from openpilot.system.ui.lib.wpa_ctrl import (WpaCtrl, WpaCtrlMonitor, SecurityType,
                                               WPA_SUPPLICANT_CONF, WPA_AP_CONF,
                                               WPA_CTRL_INTERFACE,
                                               stop_wpa_supplicant, wpa_supplicant_running,
                                               sanitize_for_conf, format_psk_value, format_ssid_value, is_valid_psk,
                                               generate_wpa_conf, parse_event_network_id, parse_event_ssid,
                                               parse_scan_results, flags_to_security_type,
                                               parse_status, dbm_to_percent, decode_ssid,
                                               ensure_wpa_supplicant, try_attach_ctrl,
                                               stop_tethering_dnsmasq, tethering_dnsmasq_running)

if TYPE_CHECKING:
  from openpilot.common.params import Params
else:
  try:
    from openpilot.common.params import Params
  except (ImportError, OSError):
    Params = None

TETHERING_IP_ADDRESS = "192.168.43.1"
TETHERING_SUBNET = "192.168.43.0/24"
TETHERING_NAT_COMMENT = "openpilot-tethering"
DEFAULT_TETHERING_PASSWORD = "swagswagcomma"
TETHERING_PASSWORD_FILE = "/data/tethering_password"
SCAN_PERIOD_SECONDS = 5
CONNECTING_STALE_TIMEOUT_SECONDS = 5
NETWORK_NOT_FOUND_EVENTS_REQUIRED = 2
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
  network_id: str | None = None


def _iptables_executable() -> str:
  # AGNOS 18.7 exposes NAT through xtables while /usr/sbin/iptables selects the unsupported nft frontend.
  return "iptables-legacy" if shutil.which("iptables-legacy") is not None else "iptables"


def _tethering_firewall_rules(op: str) -> list[list[str]]:
  # Source-subnet MASQUERADE (no `-o <iface>`) so the session survives uplink changes.
  # Mirrors NM's nm-firewall-utils.c:_share_iptables_set_masquerade_sync.
  command = ["sudo", _iptables_executable()]
  tagged = ["-m", "comment", "--comment", TETHERING_NAT_COMMENT]
  return [
    [*command, "-t", "nat", op, "POSTROUTING",
     "-s", TETHERING_SUBNET, "!", "-d", TETHERING_SUBNET,
     *tagged, "-j", "MASQUERADE"],
    [*command, op, "INPUT", "-i", "wlan0", "-p", "udp", "--dport", "67", *tagged, "-j", "ACCEPT"],
    [*command, op, "INPUT", "-i", "wlan0", "-p", "udp", "--dport", "53", *tagged, "-j", "ACCEPT"],
    [*command, op, "INPUT", "-i", "wlan0", "-p", "tcp", "--dport", "53", *tagged, "-j", "ACCEPT"],
    [*command, op, "FORWARD", "-i", "wlan0", "-s", TETHERING_SUBNET, *tagged, "-j", "ACCEPT"],
    [*command, op, "FORWARD", "-o", "wlan0", "-d", TETHERING_SUBNET,
     "-m", "conntrack", "--ctstate", "ESTABLISHED,RELATED", *tagged, "-j", "ACCEPT"],
  ]


def _tethering_firewall_ready() -> bool:
  try:
    return all(subprocess.run(rule, capture_output=True, check=False).returncode == 0
               for rule in _tethering_firewall_rules("-C"))
  except OSError:
    cloudlog.exception("Failed to verify tethering firewall rules")
    return False


def _delete_tethering_firewall_rules():
  for rule in _tethering_firewall_rules("-D"):
    for _ in range(4):
      result = subprocess.run(rule, capture_output=True, check=False)
      if result.returncode != 0:
        break


class WifiManager:
  def __init__(self):
    self._networks: list[Network] = []
    self._exit = False
    self._exit_event = threading.Event()
    self._active = False

    self._store: NetworkStore | None = None
    self._ctrl: WpaCtrl | None = None
    self._dhcp = DhcpClient()

    self._wifi_state: WifiState = WifiState()
    self._user_epoch: int = 0
    self._ipv4_address: str = ""
    self._dhcp_adoption_ssid: str | None = None
    self._current_network_metered: MeteredType = MeteredType.UNKNOWN
    self._ipv4_forward = False
    self._tethering_active = False
    self._tethering_psk = DEFAULT_TETHERING_PASSWORD
    self._dnsmasq_proc: subprocess.Popen | None = None
    self._pending_connection: PendingConnection | None = None
    self._requested_ssid: str | None = None
    self._network_not_found_epoch: int | None = None
    self._network_not_found_events = 0

    self._last_network_scan: float = 0.0
    self._last_connecting_at: float = 0.0
    self._last_scanning_recheck: float = 0.0
    self._last_connected_recheck: float = 0.0
    self._last_wrong_key_dispatch: dict[tuple[str, str | None], float] = {}
    self._callback_queue: list[Callable] = []
    self._callback_lock = threading.Lock()
    self._state_lock = threading.RLock()
    # Serializes supplicant network mutations and connected-state transitions.
    self._connect_lock = threading.Lock()
    self._station_lock = threading.Lock()
    self._station_cleanup_pending = False
    self._tethering_lock = threading.RLock()
    self._tethering_epoch = 0
    self._tethering_transition_pending = False
    self._tethering_password_epoch = 0
    # Coalesced so an undrained queue (user on another tab) can't grow unboundedly.
    self._networks_updated_pending = False

    self._tethering_ssid = "weedle"
    if Params is not None:
      dongle_id = Params().get("DongleId")
      if dongle_id:
        self._tethering_ssid += "-" + dongle_id[:4]

    self._need_auth: list[Callable[[str], None]] = []
    self._activated: list[Callable[[], None]] = []
    self._forgotten: list[Callable[[str | None], None]] = []
    self._forget_failed: list[Callable[[str | None], None]] = []
    self._networks_updated: list[Callable[[list[Network]], None]] = []
    self._disconnected: list[Callable[[], None]] = []

    self._scan_lock = threading.Lock()
    self._monitor_epoch = 0
    self._scan_thread = threading.Thread(target=self._network_scanner, daemon=True)
    self._state_thread = threading.Thread(target=self._monitor_state, daemon=True)
    self._initialize()
    atexit.register(self.stop)

  def _initialize(self):
    def worker():
      try:
        store = NetworkStore()
        self._store = store
        # WPA passphrases may legally include leading or trailing spaces, so only
        # trim the file terminator.
        try:
          with open(TETHERING_PASSWORD_FILE) as f:
            raw = f.read()
          self._tethering_psk = raw[:-1] if raw.endswith("\n") else raw
        except FileNotFoundError:
          self._tethering_psk = store.get_tethering_password(self._tethering_ssid) or DEFAULT_TETHERING_PASSWORD
        except (OSError, UnicodeError):
          cloudlog.exception("Failed to read tethering password")
          self._tethering_psk = store.get_tethering_password(self._tethering_ssid) or DEFAULT_TETHERING_PASSWORD

        with self._tethering_lock:
          self._ensure_wpa_supplicant()

          # Populate networks before wifi state so the connected SSID's strength is
          # known on first render; otherwise it flashes the disconnected icon.
          self._update_networks(block=True)

          self._init_wifi_state()

        cloudlog.debug("WifiManager initialized")
      except Exception:
        cloudlog.exception("WifiManager initialization failed")
      finally:
        self._scan_thread.start()
        self._state_thread.start()

    threading.Thread(target=worker, daemon=True).start()

  def _require_store(self) -> NetworkStore:
    if self._store is None:
      raise RuntimeError("WifiManager is not initialized")
    return self._store

  def _ensure_wpa_supplicant(self):
    self._dhcp_adoption_ssid = None
    if not wpa_supplicant_running(WPA_AP_CONF):
      try:
        generate_wpa_conf(self._require_store())
      except Exception:
        cloudlog.exception("Failed to generate wpa_supplicant configuration")
        return

    def station_reconfigured(ssid: str):
      self._dhcp_adoption_ssid = ssid
    ctrl = ensure_wpa_supplicant(lambda: self._exit, station_reconfigured,
                                 on_abandoned_ap=self._clear_tethering_network_state)
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
        # dnsmasq is gone, so the surviving AP daemon is half-broken. Stay
        # DISCONNECTED rather than letting the COMPLETED branch below treat this
        # as a station connect (which would start STA DHCP on wlan0 and clobber
        # the hotspot's address).
        self._wifi_state = WifiState(ssid=None, status=ConnectStatus.DISCONNECTED)
        return

      if wpa_state == "COMPLETED":
        connection_status = ConnectStatus.CONNECTED
      elif wpa_state in ("SCANNING", "AUTHENTICATING", "ASSOCIATING", "ASSOCIATED", "4WAY_HANDSHAKE", "GROUP_HANDSHAKE"):
        # Adopt mid-connect state; otherwise a WRONG_KEY event would bypass its current_ssid check.
        connection_status = ConnectStatus.CONNECTING
      else:
        connection_status = ConnectStatus.DISCONNECTED
        ssid = None

      if self._user_epoch != epoch:
        return

      if connection_status == ConnectStatus.CONNECTED and ssid is not None:
        adopt_dhcp = self._dhcp_adoption_ssid == ssid
        self._dhcp_adoption_ssid = None
        self._handle_connected(ssid, adopt_dhcp=adopt_dhcp, expected_epoch=epoch)
      else:
        if connection_status == ConnectStatus.CONNECTING and self._last_connecting_at == 0.0:
          self._last_connecting_at = time.monotonic()
        self._wifi_state = WifiState(ssid=ssid, status=connection_status)

    if block:
      worker()
    else:
      threading.Thread(target=worker, daemon=True).start()

  def add_callbacks(self, need_auth: Callable[[str], None] | None = None,
                    activated: Callable[[], None] | None = None,
                    forgotten: Callable[[str | None], None] | None = None,
                    forget_failed: Callable[[str | None], None] | None = None,
                    networks_updated: Callable[[list[Network]], None] | None = None,
                    disconnected: Callable[[], None] | None = None):
    if need_auth is not None:
      self._need_auth.append(need_auth)
    if activated is not None:
      with self._callback_lock:
        self._activated.append(activated)
        if self._tethering_active:
          self._callback_queue.append(activated)
    if forgotten is not None:
      self._forgotten.append(forgotten)
    if forget_failed is not None:
      self._forget_failed.append(forget_failed)
    if networks_updated is not None:
      self._networks_updated.append(networks_updated)
    if disconnected is not None:
      self._disconnected.append(disconnected)

  @property
  def networks(self) -> list[Network]:
    saved_ssids = self._store.saved_ssids() if self._store is not None else set()
    return sort_networks(self._networks, self._wifi_state.ssid, saved_ssids)

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

  def _set_connecting(self, ssid: str | None, requested: bool = True):
    with self._state_lock:
      self._dhcp_adoption_ssid = None
      self._user_epoch += 1
      self._requested_ssid = ssid if requested else None
      self._network_not_found_epoch = None
      self._network_not_found_events = 0
      self._last_connecting_at = time.monotonic() if ssid is not None else 0.0
      self._last_scanning_recheck = 0.0
      self._wifi_state = WifiState(ssid=ssid, status=ConnectStatus.DISCONNECTED if ssid is None else ConnectStatus.CONNECTING)

  def _clear_station_state(self):
    self._dhcp.stop()
    self._dhcp.clear_ipv6_state()
    self._ipv4_address = ""
    self._current_network_metered = MeteredType.UNKNOWN

  def _prepare_connection(self, epoch: int) -> bool:
    with self._station_lock:
      with self._state_lock:
        if self._user_epoch != epoch:
          return False
        cleanup_pending = self._station_cleanup_pending
        self._station_cleanup_pending = False
      if cleanup_pending:
        self._clear_station_state()
      with self._state_lock:
        return self._user_epoch == epoch

  def _set_pending_connection(self, ssid: str, password: str, hidden: bool):
    with self._state_lock:
      self._pending_connection = PendingConnection(ssid=ssid, password=password, hidden=hidden, epoch=self._user_epoch)

  def _set_pending_network_id(self, net_id: str, epoch: int):
    with self._state_lock:
      pending = self._pending_connection
      if pending is None or pending.epoch != epoch or self._user_epoch != epoch:
        return
      self._pending_connection = PendingConnection(
        ssid=pending.ssid,
        password=pending.password,
        hidden=pending.hidden,
        epoch=pending.epoch,
        network_id=net_id,
      )

  def _clear_pending_connection(self, ssid: str | None = None):
    with self._state_lock:
      if self._pending_connection is None:
        return
      if ssid is None or self._pending_connection.ssid == ssid:
        self._pending_connection = None

  def _persist_pending_connection(self, ssid: str | None):
    with self._state_lock:
      pending = self._pending_connection
      if pending is None or ssid is None:
        return

      if ssid != pending.ssid or pending.epoch != self._user_epoch:
        return

    # On filesystem error, keep credentials for later retry and swallow so
    # _handle_connected can still fire DHCP/activated callbacks.
    try:
      store = self._require_store()
      store.save_network(ssid, psk=pending.password, hidden=pending.hidden)
      generate_wpa_conf(store)
    except Exception:
      cloudlog.exception("Failed to persist pending connection for %s", ssid)
      return
    with self._state_lock:
      if self._pending_connection is pending:
        self._pending_connection = None

  def _connected_transition_is_current(self, ssid: str, epoch: int) -> bool:
    with self._state_lock:
      return self._user_epoch == epoch and self._wifi_state == WifiState(ssid, ConnectStatus.CONNECTED)

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

  def set_active(self, active: bool):
    self._active = active
    if active:
      self._init_wifi_state(block=False)
      self._update_networks(block=False)

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
          self._exit_event.wait(1)
          continue
        # No owned daemon? Spawn one so wifi doesn't stay dead after a failed
        # initial bringup or a crash. Otherwise just attach.
        daemon_alive = wpa_supplicant_running(WPA_SUPPLICANT_CONF) or wpa_supplicant_running(WPA_AP_CONF)
        stale_daemon = daemon_alive and attach_failures >= ATTACH_FAILURES_BEFORE_RESPAWN
        if daemon_alive and not stale_daemon:
          ctrl = try_attach_ctrl()
          if ctrl is None:
            attach_failures += 1
            self._exit_event.wait(SCAN_PERIOD_SECONDS)
            continue
          self._ctrl = ctrl
          attach_failures = 0
        else:
          self._ensure_wpa_supplicant()
          attach_failures = 0
          if self._ctrl is None:
            self._exit_event.wait(SCAN_PERIOD_SECONDS)
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
        if not self._exit:
          self._exit_event.wait(SCAN_PERIOD_SECONDS)

  def _adopt_ap_state(self, ssid: str | None) -> bool:
    """Adopt a hotspot only when its DHCP and NAT services are ready. On refusal,
    tear down its network services so the monitor can recover station mode."""
    with self._tethering_lock:
      if not (tethering_dnsmasq_running() and _tethering_firewall_ready()):
        cloudlog.warning("AP services are incomplete; refusing adoption and tearing down orphan AP")
        self._stop_tethering()
        return False
      if not self._ap_config_matches_password():
        cloudlog.warning("Persisted tethering password differs from the running AP; rebuilding hotspot")
        self._tethering_active = True
        try:
          self._start_tethering()
          return True
        except Exception:
          cloudlog.exception("Failed to rebuild hotspot with persisted password")
          try:
            self._stop_tethering()
          except Exception:
            cloudlog.exception("Hotspot password reconciliation rollback also failed")
            self._tethering_active = False
            self._wifi_state = WifiState()
            self._ipv4_address = ""
            self._enqueue_callbacks(self._disconnected)
          return False
      with self._callback_lock:
        self._tethering_active = True
        self._wifi_state = WifiState(ssid=ssid or self._tethering_ssid, status=ConnectStatus.CONNECTED)
        self._ipv4_address = TETHERING_IP_ADDRESS
        self._callback_queue.extend(self._activated)
      return True

  def _ap_config_matches_password(self) -> bool:
    try:
      with open(WPA_AP_CONF) as f:
        expected = f"psk={format_psk_value(self._tethering_psk)}"
        return any(line.strip() == expected for line in f)
    except OSError:
      cloudlog.exception("Failed to read running AP configuration")
      return False

  def _handle_connected(self, ssid: str, adopt_dhcp: bool = False, expected_epoch: int | None = None):
    """Transition to CONNECTED. Idempotent on (ssid, CONNECTED) so the monitor and
    reconcile paths can both call in without each one killing the previous udhcpc."""
    with self._connect_lock:
      with self._state_lock:
        if expected_epoch is not None and self._user_epoch != expected_epoch:
          return
        if self._requested_ssid is not None and self._requested_ssid != ssid:
          return
        already_connected = self._wifi_state == WifiState(ssid, ConnectStatus.CONNECTED)
        transition_epoch = self._user_epoch
        if not already_connected:
          self._requested_ssid = None
          self._last_connecting_at = 0.0
          self._last_scanning_recheck = 0.0
          self._network_not_found_epoch = None
          self._network_not_found_events = 0
          self._wifi_state = WifiState(ssid=ssid, status=ConnectStatus.CONNECTED)

      if already_connected:
        # If a prior persist hit a transient FS error, _pending_connection is still
        # populated for this SSID. Without the retry here, repeat CONNECTED events
        # short-circuit and the network is forgotten after restart.
        with self._state_lock:
          pending = self._pending_connection
        if pending is not None and pending.ssid == ssid:
          self._persist_pending_connection(ssid)
        return
      self._persist_pending_connection(ssid)
      if not self._connected_transition_is_current(ssid, transition_epoch):
        return
      # Re-enable saved networks so wpa_supplicant can auto-roam: SELECT_NETWORK disables
      # every other network as a side effect.
      if self._ctrl is not None:
        try:
          self._request("ENABLE_NETWORK all")
        except Exception:
          cloudlog.exception("Failed to re-enable saved networks for auto-roam")
      if not self._connected_transition_is_current(ssid, transition_epoch):
        return
      if not adopt_dhcp or not self._dhcp.adopt():
        self._ipv4_address = ""
        self._dhcp.start()
      if not self._connected_transition_is_current(ssid, transition_epoch):
        return
      self._enqueue_callbacks(self._activated)
      self._poll_for_ip()

  def _handle_event(self, event: str):
    """Dispatch wpa_supplicant event to state machine."""
    if "CTRL-EVENT-SCAN-RESULTS" in event:
      self._update_networks(block=False)

    elif "CTRL-EVENT-CONNECTED" in event:
      epoch = self._user_epoch

      try:
        status = parse_status(self._request("STATUS"))
      except Exception:
        cloudlog.exception("Failed to verify wpa_supplicant connected state")
        return

      if self._user_epoch != epoch:
        return

      if status.get("wpa_state") != "COMPLETED":
        return

      ssid = status.get("ssid")
      if ssid:
        adopt_dhcp = self._dhcp_adoption_ssid == ssid
        self._dhcp_adoption_ssid = None
        self._handle_connected(ssid, adopt_dhcp=adopt_dhcp, expected_epoch=epoch)

    elif "CTRL-EVENT-DISCONNECTED" in event:
      if self._tethering_active:
        return  # Ignore disconnects during tethering transitions

      epoch = self._user_epoch

      # Don't clear state if we're connecting to something (user action in progress)
      if self._wifi_state.status == ConnectStatus.CONNECTING:
        return

      if self._user_epoch != epoch:
        return

      ssid = self._wifi_state.ssid
      if self._wifi_state.status == ConnectStatus.CONNECTED and ssid is not None:
        self._dhcp_adoption_ssid = ssid
        self._last_connecting_at = time.monotonic()
        self._last_scanning_recheck = 0.0
        self._wifi_state = WifiState(ssid=ssid, status=ConnectStatus.CONNECTING)
        return

      if self._wifi_state.status == ConnectStatus.DISCONNECTED:
        return

      self._wifi_state = WifiState(ssid=None, status=ConnectStatus.DISCONNECTED)
      self._clear_station_state()
      self._enqueue_callbacks(self._disconnected)

    elif "TEMP-DISABLED" in event and "reason=WRONG_KEY" in event:
      event_ssid = parse_event_ssid(event)
      event_network_id = parse_event_network_id(event)
      if event_ssid is not None:
        with self._connect_lock:
          current_ssid = self._wifi_state.ssid
          # Auto-connect may leave us in CONNECTING with ssid=None; the event's SSID is authoritative.
          connecting_unknown = (
            self._wifi_state.status == ConnectStatus.CONNECTING
            and current_ssid is None
          )
          if not connecting_unknown and (not current_ssid or event_ssid != current_ssid):
            return

          pending = self._pending_connection
          if pending is not None and pending.ssid == event_ssid:
            if (pending.epoch != self._user_epoch
                or pending.network_id is None
                or event_network_id != pending.network_id):
              return

          # Per-profile debounce suppresses duplicate events without masking a
          # legitimate WRONG_KEY from another profile sharing the same SSID.
          dispatch_key = (event_ssid, event_network_id)
          now = time.monotonic()
          self._last_wrong_key_dispatch = {
            key: timestamp for key, timestamp in self._last_wrong_key_dispatch.items()
            if now - timestamp < WRONG_KEY_DEBOUNCE_SECONDS
          }
          last_dispatch = self._last_wrong_key_dispatch.get(dispatch_key)
          if last_dispatch is not None and now - last_dispatch < WRONG_KEY_DEBOUNCE_SECONDS:
            return
          self._last_wrong_key_dispatch[dispatch_key] = now

          # Drop only the failed profile. If another profile for this SSID is
          # available, keep the current attempt alive and try it before asking
          # the user for replacement credentials.
          if self._ctrl is not None:
            try:
              if event_network_id is not None:
                matching_ids = self._list_network_ids(event_ssid)
                if event_network_id not in matching_ids:
                  return
                self._remove_wpa_network_id(event_network_id)
                remaining_ids = [net_id for net_id in matching_ids if net_id != event_network_id]
              else:
                self._remove_wpa_network(event_ssid)
                remaining_ids = []
              if remaining_ids:
                self._select_network_ids(remaining_ids)
                self._last_connecting_at = now
                self._last_scanning_recheck = 0.0
                self._network_not_found_epoch = None
                self._network_not_found_events = 0
                return
              self._request("ENABLE_NETWORK all")
            except Exception:
              cloudlog.exception("Failed to re-enable saved networks after WRONG_KEY")
          self._clear_pending_connection(event_ssid)
          self._enqueue_callbacks(self._need_auth, event_ssid)
          self._set_connecting(None)
          # CTRL-EVENT-DISCONNECTED is ignored while CONNECTING, so tear down
          # DHCP/IP/metered here ourselves in case it arrived before WRONG_KEY.
          self._dhcp.stop()
          self._ipv4_address = ""
          self._current_network_metered = MeteredType.UNKNOWN
          self._enqueue_callbacks(self._disconnected)

    elif "CTRL-EVENT-NETWORK-NOT-FOUND" in event:
      if self._wifi_state.status != ConnectStatus.CONNECTING:
        return
      # The event has no network ID or SSID. A delayed event from the previous
      # profile can arrive after a fresh SELECT_NETWORK, so let the existing
      # stale-connection reconciliation confirm that this attempt also failed.
      if time.monotonic() - self._last_connecting_at >= CONNECTING_STALE_TIMEOUT_SECONDS:
        self._network_not_found_events += 1
        if self._network_not_found_events >= NETWORK_NOT_FOUND_EVENTS_REQUIRED:
          self._network_not_found_epoch = self._user_epoch

    elif "Trying to associate with" in event or "Associated with" in event:
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
        self._last_scanning_recheck = 0.0
        self._wifi_state = WifiState(ssid=ssid, status=ConnectStatus.CONNECTING)

  def _network_scanner(self):
    while not self._exit:
      self._reconcile_connecting_state()
      if self._active and not self._tethering_active:
        if time.monotonic() - self._last_network_scan > SCAN_PERIOD_SECONDS:
          self._request_scan()
          self._last_network_scan = time.monotonic()
      time.sleep(1 / 2.)

  def _request_scan(self):
    if self._ctrl is None:
      return
    try:
      self._request("SCAN TYPE=ONLY" if self._wifi_state.status == ConnectStatus.CONNECTED else "SCAN")
    except Exception:
      cloudlog.exception("Failed to request scan")

  def _reconcile_tethering_state(self):
    now = time.monotonic()
    if now - self._last_connected_recheck < SCAN_PERIOD_SECONDS:
      return
    self._last_connected_recheck = now

    with self._tethering_lock:
      if self._tethering_transition_pending or not self._tethering_active:
        return

      try:
        status = parse_status(self._request("STATUS"))
        if (status.get("mode") == "AP" and status.get("wpa_state") == "COMPLETED"
            and tethering_dnsmasq_running() and _tethering_firewall_ready()):
          return
      except Exception:
        cloudlog.exception("Failed to verify tethering state")

      cloudlog.warning("Tethering services stopped unexpectedly, restoring station mode")
      try:
        self._stop_tethering()
      except Exception:
        cloudlog.exception("Failed to restore station mode after tethering failure")
        self._tethering_active = False
        self._wifi_state = WifiState()
        self._ipv4_address = ""
        self._enqueue_callbacks(self._disconnected)

  def _reconcile_connecting_state(self):
    current_state = self._wifi_state
    if self._tethering_active:
      self._reconcile_tethering_state()
      return
    if self._ctrl is None:
      return

    # Detect missed CONNECTED event (e.g. monitor was reconnecting after tethering stop)
    if current_state.status == ConnectStatus.DISCONNECTED:
      now = time.monotonic()
      if now - self._last_connected_recheck < SCAN_PERIOD_SECONDS:
        return
      self._last_connected_recheck = now
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
        # dnsmasq is missing, so the AP is incomplete. Stay DISCONNECTED so the user can recover via tethering toggle.
        return
      if status.get("wpa_state") == "COMPLETED" and status.get("ssid"):
        self._handle_connected(status["ssid"], expected_epoch=epoch)
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
      # User started another connect while we were blocked in STATUS; their current
      # CONNECTING state must not be clobbered by stale STATUS results below.
      if self._user_epoch != epoch:
        return
      wpa_state = status.get("wpa_state", "")
      status_ssid = status.get("ssid")
      if wpa_state == "COMPLETED" and status_ssid is not None and status_ssid == current_state.ssid:
        self._handle_connected(status_ssid, expected_epoch=epoch)
        return
      if wpa_state == "COMPLETED" and status_ssid:
        # Roamed while the monitor was down; adopt the current network instead of
        # synthesizing a disconnect that would flush the live lease.
        self._dhcp.clear_ipv6_state()
        self._handle_connected(status_ssid, expected_epoch=epoch)
        return
      # Normal roam/rekey transits through these states briefly; treating them as
      # disconnect would flush the live udhcpc lease for nothing. Wait for the
      # next sample to see the terminal state.
      if wpa_state in ("SCANNING", "AUTHENTICATING", "ASSOCIATING", "ASSOCIATED",
                       "4WAY_HANDSHAKE", "GROUP_HANDSHAKE"):
        return
      self._wifi_state = WifiState(ssid=None, status=ConnectStatus.DISCONNECTED)
      self._dhcp.stop()
      self._dhcp.clear_ipv6_state()
      self._ipv4_address = ""
      self._current_network_metered = MeteredType.UNKNOWN
      self._enqueue_callbacks(self._disconnected)
      return

    # Reconcile even if ssid is None. STATUS below tells us definitively.
    if current_state.status != ConnectStatus.CONNECTING:
      return
    now = time.monotonic()
    if now - self._last_connecting_at < CONNECTING_STALE_TIMEOUT_SECONDS:
      return

    # Snapshot the user epoch so a STATUS reply for a stale connect attempt can't
    # clobber a fresh user-initiated one that started while we were blocked below.
    epoch = self._user_epoch
    if self._network_not_found_epoch != epoch and now - self._last_scanning_recheck < CONNECTING_STALE_TIMEOUT_SECONDS:
      return
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
      self._handle_connected(status_ssid, expected_epoch=epoch)
    elif wpa_state == "SCANNING" and self._network_not_found_epoch != epoch:
      # Hidden-SSID joins can legitimately stay in SCANNING past the stale window; defer, don't fail.
      self._last_scanning_recheck = time.monotonic()
    elif wpa_state in ("DISCONNECTED", "INACTIVE", "SCANNING", "AUTHENTICATING", "ASSOCIATING",
                      "ASSOCIATED", "4WAY_HANDSHAKE", "GROUP_HANDSHAKE"):
      ssid = current_state.ssid
      pending = self._pending_connection
      temporary_ssid = ssid if (
        pending is not None
        and ssid is not None
        and pending.ssid == ssid
        and not self._require_store().contains(ssid)
      ) else None
      self._clear_pending_connection(ssid)
      # Drop the unsaved runtime network so ENABLE_NETWORK all doesn't re-arm
      # the failed credential for another retry.
      try:
        if temporary_ssid is not None:
          self._remove_wpa_network(temporary_ssid)
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
          security_types = {flags_to_security_type(ap.flags) for ap in aps}
          if len(security_types) == 1:
            security = security_types.pop()
          elif SecurityType.WPA in security_types and SecurityType.OPEN not in security_types:
            security = SecurityType.WPA
          else:
            security = SecurityType.UNSUPPORTED
          is_tethering = ssid == self._tethering_ssid
          strength = 100 if is_tethering else dbm_to_percent(strongest.signal)
          networks.append(Network(ssid=ssid, strength=strength, security_type=security, is_tethering=is_tethering))

        # SCAN_RESULTS command failure already early-returns above, so reaching
        # here means the scan succeeded; an empty result is a real "no APs in
        # range" signal (drove away, area with no wifi, etc.) and the UI must
        # see vanished SSIDs disappear instead of holding a stale snapshot.
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
      if self._ctrl:
        try:
          status = parse_status(self._request("STATUS"))
          ipv4_address = status.get("ip_address", "")
        except Exception:
          pass

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

      ssid = self._wifi_state.ssid
      if ssid and self._store is not None:
        metered = self._store.get_metered(ssid)

    self._ipv4_address = ipv4_address
    self._current_network_metered = metered

  def connect_to_network(self, ssid: str, password: str, hidden: bool = False):
    # Backend guard: non-UI entry points (hidden-network dialog, automation) can still reach here.
    if self._tethering_active:
      cloudlog.warning(f"Ignoring connect to {ssid!r} while tethering is active")
      return
    if password and not is_valid_psk(password):
      cloudlog.warning(f"Ignoring connect to {ssid!r} with invalid passphrase")
      self._enqueue_callbacks(self._need_auth, ssid)
      return
    with self._state_lock:
      self._station_cleanup_pending |= self._wifi_state.status == ConnectStatus.CONNECTED or self._dhcp_adoption_ssid is not None
      self._set_connecting(ssid)
      self._set_pending_connection(ssid, password, hidden)
      epoch = self._user_epoch

    def worker():
      with self._connect_lock:
        if not self._prepare_connection(epoch):
          return
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

        # Recheck inside the serialization lock so a stale worker cannot remove
        # the runtime network a fresher worker just added.
        if self._user_epoch != epoch:
          return
        try:
          existing_ids = self._list_network_ids(ssid)
          net_id = self._add_and_select_network(ssid, password, hidden)
          for existing_id in existing_ids:
            self._remove_wpa_network_id(existing_id)
          self._set_pending_network_id(net_id, epoch)
        except Exception:
          cloudlog.exception(f"Failed to connect to {ssid}")
          if self._user_epoch != epoch:
            return
          try:
            self._request("ENABLE_NETWORK all")
          except Exception:
            cloudlog.exception("Failed to re-enable saved networks after connect failure")
          # Setup failed before SELECT_NETWORK could land; STATUS won't tell us
          # anything useful and _init_wifi_state would silently set DISCONNECTED
          # without notifying the UI. Reset CONNECTING and fire disconnected
          # ourselves so the UI unsticks.
          self._clear_pending_connection(ssid)
          self._set_connecting(None)
          self._enqueue_callbacks(self._disconnected)

    threading.Thread(target=worker, daemon=True).start()

  def forget_connection(self, ssid: str, block: bool = False):
    if self._wifi_state.status == ConnectStatus.CONNECTING and self._wifi_state.ssid == ssid:
      self._set_connecting(None)

    def worker():
      self._clear_pending_connection(ssid)

      try:
        store = self._require_store()
      except RuntimeError:
        cloudlog.exception(f"forget_connection: manager not initialized for {ssid}")
        self._enqueue_callbacks(self._forget_failed, ssid)
        return

      existed = store.contains(ssid)
      removed = store.remove(ssid)
      if existed and not removed:
        # rm failed, so the on-disk file survives and _load will restore the entry
        # at next start. Don't tear down the runtime/regenerate config or fire
        # `forgotten`, or the UI will lie about state until the file gets restored.
        cloudlog.warning(f"forget_connection: failed to remove {ssid} from disk; leaving runtime intact")
        self._enqueue_callbacks(self._forget_failed, ssid)
        return
      if not removed:
        cloudlog.warning(f"Trying to forget unknown connection: {ssid}")

      with self._connect_lock:
        try:
          generate_wpa_conf(store)
          if self._ctrl:
            with self._state_lock:
              was_connected = self._wifi_state.ssid == ssid and self._wifi_state.status == ConnectStatus.CONNECTED
              preserve_selection = self._wifi_state.status == ConnectStatus.CONNECTING and self._wifi_state.ssid != ssid
            if was_connected:
              self._set_connecting(None)
              self._clear_station_state()
              self._request("DISCONNECT")
            self._remove_wpa_network(ssid)
            if not preserve_selection:
              self._request("ENABLE_NETWORK all")
            # Reassociate only when the forgotten profile was the live link, so the
            # device falls back to the next saved network. Otherwise REASSOCIATE
            # would briefly drop an unrelated active connection.
            if was_connected:
              self._request("REASSOCIATE")
        except Exception:
          cloudlog.exception(f"Failed to reconfigure after forgetting {ssid}")
          self._enqueue_callbacks(self._forget_failed, ssid)
          return

      self._enqueue_callbacks(self._forgotten, ssid)

    if block:
      worker()
    else:
      threading.Thread(target=worker, daemon=True).start()

  def activate_connection(self, ssid: str, block: bool = False):
    if self._tethering_active:
      cloudlog.warning(f"Ignoring activate {ssid!r} while tethering is active")
      return
    with self._state_lock:
      self._station_cleanup_pending |= self._wifi_state.status == ConnectStatus.CONNECTED or self._dhcp_adoption_ssid is not None
      self._set_connecting(ssid)
      self._clear_pending_connection()
      epoch = self._user_epoch

    def worker():
      with self._connect_lock:
        if not self._prepare_connection(epoch):
          return
        if self._ctrl is None:
          cloudlog.warning(f"No wpa_supplicant connection for activate {ssid}")
          # Skip the reset if a fresher attempt has already moved on, otherwise
          # this stale worker would emit a false disconnect for the current attempt.
          if self._user_epoch != epoch:
            return
          # _init_wifi_state is a no-op while _ctrl is None, so reset CONNECTING inline.
          self._set_connecting(None)
          self._enqueue_callbacks(self._disconnected)
          return

        def reset_to_disconnected():
          if self._user_epoch != epoch:
            return
          try:
            self._request("ENABLE_NETWORK all")
          except Exception:
            cloudlog.exception("Failed to re-enable saved networks after activation failure")
          # Mirror the _ctrl is None recovery: _init_wifi_state silently sets DISCONNECTED
          # without firing the callback, which leaves the UI wedged at CONNECTING.
          self._set_connecting(None)
          self._enqueue_callbacks(self._disconnected)

        # Recheck inside the serialization lock so a stale worker cannot mutate
        # networks added by a fresher one.
        if self._user_epoch != epoch:
          return
        try:
          ids = self._list_network_ids(ssid)
          if ids:
            self._select_network_ids(ids)
          else:
            # Network not in wpa_supplicant's runtime list, so add it from the store.
            entry = self._require_store().get(ssid)
            if entry:
              self._add_and_select_network(
                ssid,
                entry.get("psk", ""),
                entry.get("hidden", False),
                entry.get("priority", 0),
                bssid=entry.get("bssid") or None,
              )
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

  def _select_network_ids(self, net_ids: list[str]):
    """Make only the supplied runtime profiles eligible, then reassociate."""
    commands = ["DISABLE_NETWORK all", *(f"ENABLE_NETWORK {net_id}" for net_id in net_ids), "REASSOCIATE"]
    for command in commands:
      resp = self._request(command).strip()
      if not resp.startswith("OK"):
        raise RuntimeError(f"{command} failed: {resp}")

  def _add_and_select_network(self, ssid: str, psk: str = "", hidden: bool = False,
                              priority: int = 0, bssid: str | None = None) -> str:
    """Add a network and select it. Every SET_NETWORK is checked so a bad PSK/key_mgmt
    surfaces an immediate error instead of a delayed WRONG_KEY; orphans get REMOVE_NETWORK'd."""
    net_id = self._request("ADD_NETWORK").strip()
    if not net_id.isdigit():
      raise RuntimeError(f"ADD_NETWORK failed: {net_id}")

    try:
      self._wpa_set_network(net_id, "ssid", format_ssid_value(ssid))
      if psk:
        self._wpa_set_network(net_id, "psk", format_psk_value(psk))
      else:
        self._wpa_set_network(net_id, "key_mgmt", "NONE")
      if hidden:
        self._wpa_set_network(net_id, "scan_ssid", "1")
      if bssid:
        self._wpa_set_network(net_id, "bssid", bssid)
      self._wpa_set_network(net_id, "priority", str(priority))
      resp = self._request(f"SELECT_NETWORK {net_id}").strip()
      if not resp.startswith("OK"):
        raise RuntimeError(f"SELECT_NETWORK {net_id} failed: {resp}")
      return net_id
    except Exception:
      try:
        self._request(f"REMOVE_NETWORK {net_id}")
      except Exception:
        cloudlog.exception(f"Failed to clean up orphaned network {net_id}")
      raise

  def _wpa_set_network(self, net_id: str, key: str, value: str):
    resp = self._request(f"SET_NETWORK {net_id} {key} {value}").strip()
    if not resp.startswith("OK"):
      raise RuntimeError(f"SET_NETWORK {net_id} {key} failed: {resp}")

  def _list_network_ids(self, ssid: str) -> list[str]:
    """Return all wpa_supplicant network ids matching SSID. LIST_NETWORKS emits
    printf_encode'd SSIDs. Decode before comparing or non-ASCII SSIDs silently miss.
    Don't .strip() the whole reply: SSIDs may end with spaces, so a trailing-space
    SSID on the last line would be clipped and miss the match."""
    if self._ctrl is None:
      raise OSError("wpa_supplicant ctrl not attached")
    raw = self._request("LIST_NETWORKS")
    lines = raw.splitlines()
    if not lines or not lines[0].startswith("network id"):
      raise RuntimeError(f"LIST_NETWORKS failed: {raw.strip()}")
    return [parts[0] for line in lines[1:]
            if len(parts := line.split("\t")) >= 2 and decode_ssid(parts[1]) == ssid]

  def _remove_wpa_network(self, ssid: str):
    for net_id in self._list_network_ids(ssid):
      self._remove_wpa_network_id(net_id)

  def _remove_wpa_network_id(self, net_id: str):
    resp = self._request(f"REMOVE_NETWORK {net_id}").strip()
    if not resp.startswith("OK"):
      raise RuntimeError(f"REMOVE_NETWORK {net_id} failed: {resp}")

  def is_tethering_active(self) -> bool:
    return self._tethering_active

  def is_connection_saved(self, ssid: str) -> bool:
    return self._store.contains(ssid) if self._store is not None else False

  def set_tethering_password(self, password: str):
    # WPA PSKs are 8-63 UTF-8 bytes or exactly 64 hexadecimal characters.
    pw_bytes = len(password.encode("utf-8"))
    if not is_valid_psk(password):
      cloudlog.warning(f"set_tethering_password: rejecting invalid password (bytes={pw_bytes})")
      # Notify the UI so it re-enables the tethering controls.
      self._enqueue_callbacks(self._activated if self._tethering_active else self._disconnected)
      return
    self._tethering_password_epoch += 1
    epoch = self._tethering_password_epoch
    def transition():
      try:
        with atomic_write(TETHERING_PASSWORD_FILE, overwrite=True) as f:
          f.write(password)
      except Exception:
        cloudlog.exception("Failed to persist tethering password; runtime state unchanged")
        self._enqueue_callbacks(self._activated if self._tethering_active else self._disconnected)
        return
      try:
        if self._store is not None:
          self._store.set_tethering_password(self._tethering_ssid, password)
      except Exception:
        cloudlog.exception("Failed to update NetworkManager tethering profile")
      self._tethering_psk = password
      if self._tethering_active:
        try:
          # Keep the hotspot active while the password restart is in progress.
          self._stop_tethering()
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

    def worker():
      with self._tethering_lock:
        if self._tethering_password_epoch == epoch:
          transition()
    threading.Thread(target=worker, daemon=True).start()

  def set_ipv4_forward(self, enabled: bool):
    self._ipv4_forward = enabled

  def set_tethering_active(self, active: bool):
    # Enabling is visible immediately; disabling completes after station mode is restored.
    self._tethering_epoch += 1
    epoch = self._tethering_epoch
    self._tethering_transition_pending = True
    if active:
      self._tethering_active = True
    def transition():
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
          self._wifi_state = WifiState()
          self._ipv4_address = ""
          self._enqueue_callbacks(self._disconnected)

    def worker():
      with self._tethering_lock:
        if self._tethering_epoch == epoch:
          try:
            transition()
          finally:
            self._tethering_transition_pending = False
    threading.Thread(target=worker, daemon=True).start()

  def set_current_network_metered(self, metered: MeteredType):
    if self._tethering_active:
      return
    ssid = self.connected_ssid
    if ssid is None:
      return

    def worker():
      try:
        self._require_store().set_metered(ssid, int(metered))
      except Exception:
        cloudlog.exception(f"Failed to update metered state for {ssid}")
        return
      if self.connected_ssid == ssid:
        self._current_network_metered = metered
    threading.Thread(target=worker, daemon=True).start()

  def _start_tethering(self):
    self._tethering_active = True
    self._set_connecting(self._tethering_ssid, requested=False)

    psk = self._tethering_psk

    if self._ctrl:
      self._ctrl.close()
      self._ctrl = None

    # Target only openpilot-owned daemons, including surviving AP instances.
    self._monitor_epoch += 1
    stop_wpa_supplicant(WPA_SUPPLICANT_CONF)
    stop_wpa_supplicant(WPA_AP_CONF)
    self._dhcp.stop()
    time.sleep(0.5)

    safe_tether_ssid = sanitize_for_conf(self._tethering_ssid)
    lines = [WPA_CTRL_INTERFACE, "ap_scan=2", "",
             "network={", f'  ssid="{safe_tether_ssid}"', "  mode=2",
             "  frequency=2437", "  key_mgmt=WPA-PSK", "  proto=RSN",
             "  pairwise=CCMP", "  group=CCMP", f'  psk={format_psk_value(psk)}', "}", ""]
    ap_conf = "\n".join(lines)
    with atomic_write(WPA_AP_CONF, overwrite=True) as f:
      f.write(ap_conf)

    subprocess.run(["sudo", "wpa_supplicant", "-B", "-i", "wlan0", "-c", WPA_AP_CONF, "-D", "nl80211"], check=False)
    time.sleep(1)

    # Treat interface configuration failures as incomplete AP bringup.
    subprocess.run(["sudo", "ip", "addr", "flush", "dev", "wlan0"], check=False)
    subprocess.run(["sudo", "ip", "addr", "add", f"{TETHERING_IP_ADDRESS}/24", "dev", "wlan0"], check=True)
    subprocess.run(["sudo", "ip", "link", "set", "wlan0", "up"], check=True)

    stop_tethering_dnsmasq()
    self._dnsmasq_proc = subprocess.Popen([
      "sudo", "dnsmasq",
      "--interface=wlan0",
      "--bind-interfaces",
      "--dhcp-range=192.168.43.2,192.168.43.254,24h",
      "--dhcp-leasefile=/tmp/dnsmasq.leases",
      "--no-daemon", "--log-queries",
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
      start_new_session=True)
    # Fail bringup if clients cannot obtain leases.
    time.sleep(0.2)
    if self._dnsmasq_proc.poll() is not None:
      rc = self._dnsmasq_proc.returncode
      self._dnsmasq_proc = None
      raise RuntimeError(f"dnsmasq exited during tethering bringup (rc={rc})")

    # Flush tagged copies so repeated starts remain idempotent.
    _delete_tethering_firewall_rules()
    # Firewall and forwarding failures must roll back the hotspot.
    for rule in _tethering_firewall_rules("-A"):
      subprocess.run(rule, check=True)
    if self._ipv4_forward:
      subprocess.run(["sudo", "sysctl", "net.ipv4.ip_forward=1"], check=True)

    # Verify that the owned daemon has control of wlan0 in AP mode.
    if not wpa_supplicant_running(WPA_AP_CONF):
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

  def _clear_tethering_network_state(self):
    try:
      subprocess.run(["sudo", "sysctl", "net.ipv4.ip_forward=0"], check=True)
    except (OSError, subprocess.CalledProcessError):
      cloudlog.exception("Failed to disable IPv4 forwarding during tethering teardown")

    try:
      stop_tethering_dnsmasq()
    except OSError:
      cloudlog.exception("Failed to stop tethering dnsmasq")
    if self._dnsmasq_proc is not None:
      try:
        self._dnsmasq_proc.wait(timeout=3)
      except Exception:
        cloudlog.exception("Failed waiting for tethering dnsmasq to stop")
      self._dnsmasq_proc = None

    try:
      _delete_tethering_firewall_rules()
    except OSError:
      cloudlog.exception("Failed to remove tethering firewall rules")

  def _stop_tethering(self):
    self._clear_tethering_network_state()

    if self._ctrl:
      self._ctrl.close()
      self._ctrl = None

    # Stop AP wpa_supplicant (only the one running our AP config).
    self._monitor_epoch += 1
    stop_wpa_supplicant(WPA_AP_CONF)
    time.sleep(0.5)

    subprocess.run(["sudo", "ip", "addr", "flush", "dev", "wlan0"], check=False)

    generate_wpa_conf(self._require_store())
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
      self._exit_event.set()
      ctrl, self._ctrl = self._ctrl, None
      if ctrl is not None:
        ctrl.interrupt()
      if self._scan_thread.is_alive():
        self._scan_thread.join()
      if self._state_thread.is_alive():
        self._state_thread.join()
      if ctrl is not None:
        ctrl.close()
      # Network daemons outlive the UI and are adopted by the next controller.
