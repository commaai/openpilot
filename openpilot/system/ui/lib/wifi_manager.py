import atexit
import shutil
import subprocess
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING

from openpilot.common.swaglog import cloudlog
from openpilot.common.utils import atomic_write
from openpilot.system.ui.lib.dhcp_client import DhcpClient
from openpilot.system.ui.lib.wifi_network_store import MeteredType, NetworkStore
from openpilot.system.ui.lib.wpa_ctrl import (WpaCtrl, WpaCtrlMonitor, SecurityType,
                                               WPA_SUPPLICANT_CONF, WPA_AP_CONF,
                                               WPA_CTRL_INTERFACE, WPA_PID_FILE,
                                               stop_wpa_supplicant, wpa_supplicant_running,
                                               sanitize_for_conf, format_psk_value, format_ssid_value, is_valid_psk, is_valid_ssid,
                                               generate_wpa_conf, parse_event_network_id, parse_event_ssid,
                                               parse_scan_results, flags_to_security_type,
                                               parse_status, dbm_to_percent, decode_ssid,
                                               ensure_wpa_supplicant, prepare_wpa_runtime, try_attach_ctrl,
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
TETHERING_NAT_CHAIN = "OPENPILOT_TETHERING_NAT"
TETHERING_INPUT_CHAIN = "OPENPILOT_TETHERING_INPUT"
TETHERING_FORWARD_CHAIN = "OPENPILOT_TETHERING_FORWARD"
DEFAULT_TETHERING_PASSWORD = "swagswagcomma"
TETHERING_PASSWORD_FILE = "/data/tethering_password"
SCAN_PERIOD_SECONDS = 5
CONNECTING_STALE_TIMEOUT_SECONDS = 5
NETWORK_NOT_FOUND_EVENTS_REQUIRED = 2
# Ignore stale WRONG_KEY events after a fast retry
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


class StationOperationKind(IntEnum):
  CONNECT = 0
  ACTIVATE = 1
  ASSOCIATED = 2
  FORGET = 3
  AUTH_FAILURE = 4
  TIMEOUT = 5


@dataclass(frozen=True)
class StationOperation:
  epoch: int
  kind: StationOperationKind
  ssid: str | None
  profile_uuid: str | None = None
  runtime_network_id: str | None = None


@dataclass(frozen=True)
class PendingConnection:
  ssid: str
  password: str
  hidden: bool
  security: SecurityType
  epoch: int
  profile_uuid: str
  network_id: str | None = None


def _iptables_executable() -> str:
  # AGNOS 18.7 requires iptables-legacy for NAT
  return "iptables-legacy" if shutil.which("iptables-legacy") is not None else "iptables"


def _tethering_firewall_rules(op: str) -> list[list[str]]:
  # Match NetworkManager's source-subnet MASQUERADE so NAT survives uplink changes
  command = ["sudo", _iptables_executable()]
  tagged = ["-m", "comment", "--comment", TETHERING_NAT_COMMENT]
  return [
    [*command, "-t", "nat", op, TETHERING_NAT_CHAIN,
     "-s", TETHERING_SUBNET, "!", "-d", TETHERING_SUBNET,
     *tagged, "-j", "MASQUERADE"],
    [*command, op, TETHERING_INPUT_CHAIN, "-i", "wlan0", "-p", "udp", "--dport", "67", *tagged, "-j", "ACCEPT"],
    [*command, op, TETHERING_INPUT_CHAIN, "-i", "wlan0", "-p", "udp", "--dport", "53", *tagged, "-j", "ACCEPT"],
    [*command, op, TETHERING_INPUT_CHAIN, "-i", "wlan0", "-p", "tcp", "--dport", "53", *tagged, "-j", "ACCEPT"],
    [*command, op, TETHERING_FORWARD_CHAIN, "-i", "wlan0", "-s", TETHERING_SUBNET, *tagged, "-j", "ACCEPT"],
    [*command, op, TETHERING_FORWARD_CHAIN, "-o", "wlan0", "-d", TETHERING_SUBNET,
     "-m", "conntrack", "--ctstate", "ESTABLISHED,RELATED", *tagged, "-j", "ACCEPT"],
  ]
def _tethering_firewall_chains() -> list[tuple[list[str], str]]:
  command = ["sudo", _iptables_executable()]
  return [
    ([*command, "-t", "nat"], TETHERING_NAT_CHAIN),
    (command, TETHERING_INPUT_CHAIN),
    (command, TETHERING_FORWARD_CHAIN),
  ]


def _tethering_firewall_jumps(op: str) -> list[list[str]]:
  command = ["sudo", _iptables_executable()]
  tagged = ["-m", "comment", "--comment", TETHERING_NAT_COMMENT]
  return [
    [*command, "-t", "nat", op, "POSTROUTING", *tagged, "-j", TETHERING_NAT_CHAIN],
    [*command, op, "INPUT", *tagged, "-j", TETHERING_INPUT_CHAIN],
    [*command, op, "FORWARD", *tagged, "-j", TETHERING_FORWARD_CHAIN],
  ]


def _delete_firewall_jumps():
  for jump in _tethering_firewall_jumps("-D"):
    while subprocess.run(jump, capture_output=True, check=False).returncode == 0:
      pass


def _install_tethering_firewall_rules():
  for command, chain in _tethering_firewall_chains():
    subprocess.run([*command, "-N", chain], capture_output=True, check=False)
    subprocess.run([*command, "-F", chain], check=True)
  for rule in _tethering_firewall_rules("-A"):
    subprocess.run(rule, check=True)

  # Exactly one jump from each shared built-in chain into our owned chains.
  _delete_firewall_jumps()
  for jump in _tethering_firewall_jumps("-A"):
    subprocess.run(jump, check=True)




def _tethering_firewall_ready() -> bool:
  try:
    checks = [*_tethering_firewall_rules("-C"), *_tethering_firewall_jumps("-C")]
    return all(subprocess.run(rule, capture_output=True, check=False).returncode == 0
               for rule in checks)
  except OSError:
    cloudlog.exception("Failed to verify tethering firewall rules")
    return False


def _delete_tethering_firewall_rules():
  _delete_firewall_jumps()
  for command, chain in _tethering_firewall_chains():
    subprocess.run([*command, "-F", chain], capture_output=True, check=False)
    subprocess.run([*command, "-X", chain], capture_output=True, check=False)


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
    self._associated_ssid: str | None = None
    self._associated_epoch: int | None = None
    self._dhcp_adoption_ssid: str | None = None
    self._current_network_metered: MeteredType = MeteredType.UNKNOWN
    self._ipv4_forward: bool | None = None
    self._tethering_active = False
    self._tethering_psk = DEFAULT_TETHERING_PASSWORD
    self._dnsmasq_proc: subprocess.Popen | None = None
    self._station_operation: StationOperation | None = None
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
    # Serialize wlan0, wpa_supplicant, and DHCP lifecycle changes across STA and AP.
    self._radio_lock = threading.RLock()
    self._station_cleanup_pending = False
    self._tethering_epoch = 0
    self._tethering_transition_pending = False
    self._tethering_started = False
    self._tethering_password_epoch = 0
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
        persisted_password = store.get_tethering_password(self._tethering_ssid)
        if persisted_password is not None and is_valid_psk(persisted_password):
          self._tethering_psk = persisted_password
        else:
          if persisted_password is not None:
            cloudlog.warning("Ignoring invalid tethering password in NetworkManager profile")
          # The standalone file is migration input only; the keyfile remains the
          # durable source after a successful import.
          try:
            with open(TETHERING_PASSWORD_FILE) as f:
              raw = f.read()
            legacy_password = raw[:-1] if raw.endswith("\n") else raw
            if is_valid_psk(legacy_password) and store.set_tethering_password(self._tethering_ssid, legacy_password):
              self._tethering_psk = legacy_password
            else:
              self._tethering_psk = DEFAULT_TETHERING_PASSWORD
          except FileNotFoundError:
            self._tethering_psk = DEFAULT_TETHERING_PASSWORD
          except (OSError, UnicodeError):
            cloudlog.exception("Failed to migrate legacy tethering password")
            self._tethering_psk = DEFAULT_TETHERING_PASSWORD

        try:
          if not store.ensure_tethering_profile(self._tethering_ssid, self._tethering_psk):
            cloudlog.warning("Failed to create durable tethering profile")
        except Exception:
          cloudlog.exception("Failed to create durable tethering profile")

        with self._radio_lock:
          self._ensure_wpa_supplicant()

          # Load signal strength before rendering the connected network
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
    with self._radio_lock:
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
    with self._radio_lock:
      ctrl = self._ctrl
      if ctrl is None:
        raise OSError("wpa_supplicant ctrl not attached")
      try:
        return ctrl.request(cmd)
      except OSError:
        # Restart the monitor because recv may survive daemon death
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
        # Adopt a surviving hotspot before station cleanup
        if self._user_epoch != epoch:
          return
        if self._adopt_ap_state(ssid):
          return
        # Avoid treating an incomplete AP as a station connection
        self._wifi_state = WifiState(ssid=None, status=ConnectStatus.DISCONNECTED)
        return

      if wpa_state == "COMPLETED":
        connection_status = ConnectStatus.CONNECTED
      elif wpa_state in ("SCANNING", "AUTHENTICATING", "ASSOCIATING", "ASSOCIATED", "4WAY_HANDSHAKE", "GROUP_HANDSHAKE"):
        # Preserve mid-connect state for WRONG_KEY validation
        connection_status = ConnectStatus.CONNECTING
      else:
        connection_status = ConnectStatus.DISCONNECTED
        ssid = None

      if self._user_epoch != epoch:
        return

      if connection_status == ConnectStatus.CONNECTED and ssid is not None:
        self._handle_connected(ssid, expected_epoch=epoch, profile_uuid=status.get("id_str"))
      else:
        if connection_status == ConnectStatus.CONNECTING and self._last_connecting_at == 0.0:
          self._last_connecting_at = time.monotonic()
        elif connection_status == ConnectStatus.DISCONNECTED:
          self._dhcp_adoption_ssid = None
          self._clear_station_state()
          if self._user_epoch != epoch:
            return
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

  def _set_connecting(self, ssid: str | None, requested: bool = True,
                      kind: StationOperationKind = StationOperationKind.CONNECT,
                      operation_ssid: str | None = None):
    with self._state_lock:
      self._dhcp_adoption_ssid = None
      self._user_epoch += 1
      self._requested_ssid = ssid if requested else None
      self._network_not_found_epoch = None
      self._network_not_found_events = 0
      self._last_connecting_at = time.monotonic() if ssid is not None else 0.0
      self._last_scanning_recheck = 0.0
      self._associated_ssid = None
      self._associated_epoch = None
      self._station_operation = StationOperation(self._user_epoch, kind, operation_ssid if operation_ssid is not None else ssid)
      self._wifi_state = WifiState(ssid=ssid, status=ConnectStatus.DISCONNECTED if ssid is None else ConnectStatus.CONNECTING)

  def _clear_station_l3_state(self):
    with self._radio_lock:
      self._dhcp.stop()
      self._dhcp.clear_ipv6_state()
      self._ipv4_address = ""
      self._current_network_metered = MeteredType.UNKNOWN

  def _clear_station_state(self):
    with self._radio_lock:
      self._clear_station_l3_state()
      with self._state_lock:
        self._associated_ssid = None
        self._associated_epoch = None

  def _prepare_connection(self, epoch: int) -> bool:
    with self._radio_lock:
      with self._state_lock:
        if self._user_epoch != epoch:
          return False
        cleanup_pending = self._station_cleanup_pending
        self._station_cleanup_pending = False
      if cleanup_pending:
        self._clear_station_state()
      with self._state_lock:
        return self._user_epoch == epoch

  def _set_pending_connection(self, ssid: str, password: str, hidden: bool, security: SecurityType):
    profile_uuid = None
    if self._store is not None:
      entry = self._store.get(ssid)
      if isinstance(entry, dict):
        profile_uuid = entry.get("uuid")
    try:
      profile_uuid = str(uuid.UUID(profile_uuid)) if profile_uuid is not None else str(uuid.uuid4())
    except (AttributeError, TypeError, ValueError):
      profile_uuid = str(uuid.uuid4())

    with self._state_lock:
      epoch = self._user_epoch
      self._pending_connection = PendingConnection(
        ssid=ssid, password=password, hidden=hidden, security=security, epoch=epoch, profile_uuid=profile_uuid,
      )
      self._station_operation = StationOperation(
        epoch, StationOperationKind.CONNECT, ssid, profile_uuid=profile_uuid,
      )

  def _set_pending_network_id(self, net_id: str, epoch: int):
    with self._state_lock:
      pending = self._pending_connection
      if pending is None or pending.epoch != epoch or self._user_epoch != epoch:
        return
      self._pending_connection = PendingConnection(
        ssid=pending.ssid,
        password=pending.password,
        hidden=pending.hidden,
        security=pending.security,
        epoch=pending.epoch,
        profile_uuid=pending.profile_uuid,
        network_id=net_id,
      )
      self._station_operation = StationOperation(
        epoch, StationOperationKind.CONNECT, pending.ssid, pending.profile_uuid, net_id,
      )

  def _clear_pending_connection(self, ssid: str | None = None, epoch: int | None = None):
    with self._state_lock:
      pending = self._pending_connection
      if pending is None:
        return
      if epoch is not None and pending.epoch != epoch:
        return
      if ssid is None or pending.ssid == ssid:
        self._pending_connection = None

  def _restore_station_runtime(self, selected_id: str | None, previous_ids: list[str], removed_ids: list[str]):
    exact = True
    if selected_id is not None:
      try:
        self._remove_wpa_network_id(selected_id)
      except Exception:
        exact = False
        cloudlog.exception(f"Failed to remove selected runtime network {selected_id}")

    if exact and not removed_ids and previous_ids:
      try:
        self._select_network_ids(previous_ids)
        return
      except Exception:
        cloudlog.exception("Failed to restore previous runtime network selection")

    try:
      store = self._require_store()
      generate_wpa_conf(store)
      if self._ctrl is not None:
        response = self._request("RECONFIGURE").strip()
        if not response.startswith("OK"):
          raise RuntimeError(f"RECONFIGURE failed: {response}")
    except Exception:
      cloudlog.exception("Failed to restore runtime networks from durable profiles")

  def _persist_pending_connection(self, ssid: str | None):
    with self._state_lock:
      pending = self._pending_connection
      if pending is None or ssid is None:
        return

      if ssid != pending.ssid or pending.epoch != self._user_epoch:
        return

    # Retain credentials after transient persistence failures
    try:
      store = self._require_store()
      store.save_network(
        ssid,
        psk=pending.password,
        hidden=pending.hidden,
        security=pending.security,
        profile_uuid=pending.profile_uuid,
      )
      generate_wpa_conf(store)
    except Exception:
      cloudlog.exception("Failed to persist pending connection for %s", ssid)
      return
    with self._state_lock:
      if self._pending_connection is pending:
        self._pending_connection = None

  def _connected_transition_is_current(self, ssid: str, epoch: int) -> bool:
    with self._state_lock:
      return (
        self._user_epoch == epoch
        and self._associated_ssid == ssid
        and self._associated_epoch == epoch
      )

  def _wifi_default_route_ready(self) -> bool:
    try:
      result = subprocess.run(
        ["ip", "-4", "route", "show", "default", "dev", "wlan0"],
        capture_output=True,
        check=False,
        text=True,
        timeout=2,
      )
    except (OSError, subprocess.TimeoutExpired):
      return False
    if result.returncode != 0:
      return False

    routes = [line.split() for line in result.stdout.splitlines() if line.strip()]
    if len(routes) != 1:
      return False

    route = routes[0]
    try:
      via_index = route.index("via")
      metric_index = route.index("metric")
    except ValueError:
      return False
    dev_index = route.index("dev") if "dev" in route else None
    return (
      route[0] == "default"
      and via_index + 1 < len(route)
      and route[via_index + 1] not in ("dev", "metric")
      and (dev_index is None or (dev_index + 1 < len(route) and route[dev_index + 1] == "wlan0"))
      and metric_index + 1 < len(route)
      and route[metric_index + 1] == "600"
    )

  def _complete_station_connection(self, ssid: str, epoch: int) -> bool:
    with self._radio_lock, self._state_lock:
      if not self._ipv4_address or not self._connected_transition_is_current(ssid, epoch):
        return False
      if not self._wifi_default_route_ready():
        return False
      if self._wifi_state == WifiState(ssid, ConnectStatus.CONNECTED):
        return True
      self._requested_ssid = None
      self._last_connecting_at = 0.0
      self._last_scanning_recheck = 0.0
      self._network_not_found_epoch = None
      self._network_not_found_events = 0
      self._wifi_state = WifiState(ssid=ssid, status=ConnectStatus.CONNECTED)
      self._enqueue_callbacks(self._activated)
      return True

  def _enqueue_callbacks(self, cbs: list[Callable], *args):
    with self._callback_lock:
      for cb in cbs:
        self._callback_queue.append(lambda _cb=cb: _cb(*args))

  def _mark_networks_updated(self):
    # Coalesce scan callbacks to keep the undrained queue bounded
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
      snapshot = self.networks
      for cb in networks_cbs:
        cb(snapshot)

  def set_active(self, active: bool):
    self._active = active
    if active:
      self._init_wifi_state(block=False)
      self._update_networks(block=False)

  def _monitor_state(self):
    # Respawn an owned daemon whose control socket remains unreachable
    ATTACH_FAILURES_BEFORE_RESPAWN = 3
    attach_failures = 0
    while not self._exit:
      if self._ctrl is None:
        with self._radio_lock:
          # Avoid spawning STA while tethering is taking over wlan0
          if self._ctrl is None and not self._tethering_active:
            daemon_alive = wpa_supplicant_running(WPA_SUPPLICANT_CONF) or wpa_supplicant_running(WPA_AP_CONF)
            stale_daemon = daemon_alive and attach_failures >= ATTACH_FAILURES_BEFORE_RESPAWN
            if daemon_alive and not stale_daemon:
              ctrl = try_attach_ctrl()
              if ctrl is None:
                attach_failures += 1
              else:
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
        # Reattach after control-socket failure
        with self._radio_lock:
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
    with self._radio_lock:
      if not (tethering_dnsmasq_running() and _tethering_firewall_ready()):
        cloudlog.warning("AP services are incomplete; refusing adoption and tearing down orphan AP")
        self._stop_tethering()
        return False
      if self._ipv4_forward is not None:
        try:
          self._apply_ipv4_forward(self._ipv4_forward)
        except Exception:
          cloudlog.exception("Failed to enforce IPv4 forwarding policy while adopting AP")
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
        self._tethering_started = True
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

  def _handle_connected(self, ssid: str, expected_epoch: int | None = None,
                        profile_uuid: str | None = None):
    """Handle L2 association. CONNECTED and activation remain IP-ready states."""
    with self._radio_lock:
      with self._state_lock:
        if expected_epoch is not None and self._user_epoch != expected_epoch:
          return
        if self._requested_ssid is not None and self._requested_ssid != ssid:
          return
        transition_epoch = self._user_epoch
        previous_ssid = self._associated_ssid
        if previous_ssid is None and self._wifi_state.status == ConnectStatus.CONNECTED:
          previous_ssid = self._wifi_state.ssid
        adoption_ssid = self._dhcp_adoption_ssid
        adopt_dhcp = adoption_ssid == ssid
        if (
          (previous_ssid is not None and previous_ssid != ssid)
          or (adoption_ssid is not None and adoption_ssid != ssid)
        ):
          self._dhcp.clear_ipv6_state()
        self._dhcp_adoption_ssid = None
        already_associated = self._connected_transition_is_current(ssid, transition_epoch)
        already_connected = self._wifi_state == WifiState(ssid, ConnectStatus.CONNECTED)
        previous_operation = self._station_operation
        pending = self._pending_connection
        active_profile_uuid = profile_uuid or (pending.profile_uuid if pending is not None and pending.ssid == ssid else None)
        previous_profile_uuid = previous_operation.profile_uuid if previous_operation is not None else None
        profile_changed = (
          already_associated
          and active_profile_uuid is not None
          and active_profile_uuid != previous_profile_uuid
        )
        if not already_associated:
          self._associated_ssid = ssid
          self._associated_epoch = transition_epoch
          self._station_operation = StationOperation(
            transition_epoch,
            StationOperationKind.ASSOCIATED,
            ssid,
            profile_uuid=active_profile_uuid,
            runtime_network_id=pending.network_id if pending is not None and pending.ssid == ssid else None,
          )
          self._wifi_state = WifiState(ssid=ssid, status=ConnectStatus.CONNECTING)

      if profile_changed:
        self._clear_station_l3_state()
        with self._state_lock:
          if (
            not self._connected_transition_is_current(ssid, transition_epoch)
            or self._station_operation is not previous_operation
          ):
            return
          self._wifi_state = WifiState(ssid=ssid, status=ConnectStatus.CONNECTING)

      if not already_associated or profile_changed:
        try:
          ipv6_method = self._store.get_ipv6_method(ssid, active_profile_uuid) if self._store is not None else "auto"
          self._dhcp.set_ipv6_enabled(ipv6_method != "ignore")
        except Exception:
          cloudlog.exception("Failed to apply IPv6 policy for %s", ssid)
          if not already_associated:
            with self._state_lock:
              if self._connected_transition_is_current(ssid, transition_epoch):
                self._associated_ssid = None
                self._associated_epoch = None
                self._station_operation = previous_operation
          return

        if profile_changed:
          with self._state_lock:
            if (
              not self._connected_transition_is_current(ssid, transition_epoch)
              or self._station_operation is not previous_operation
            ):
              return
            self._station_operation = StationOperation(
              transition_epoch,
              StationOperationKind.ASSOCIATED,
              ssid,
              profile_uuid=active_profile_uuid,
              runtime_network_id=previous_operation.runtime_network_id if previous_operation is not None else None,
            )

      if already_connected and not profile_changed:
        # Retry persistence after transient filesystem failures.
        pending = self._pending_connection
        if pending is not None and pending.ssid == ssid:
          self._persist_pending_connection(ssid)
        return

      if already_associated and not profile_changed:
        self._persist_pending_connection(ssid)
        self._update_active_connection_info()
        self._complete_station_connection(ssid, transition_epoch)
        return

      if profile_changed or not adopt_dhcp or not self._dhcp.adopt():
        self._ipv4_address = ""
        self._dhcp.start()
      if not self._connected_transition_is_current(ssid, transition_epoch):
        return

      self._persist_pending_connection(ssid)
      if self._ctrl is not None and self._connected_transition_is_current(ssid, transition_epoch):
        try:
          # SELECT_NETWORK disables other profiles; re-enable them for roaming.
          self._request("ENABLE_NETWORK all")
        except Exception:
          cloudlog.exception("Failed to re-enable saved networks for auto-roam")
      if self._connected_transition_is_current(ssid, transition_epoch):
        self._poll_for_ip(ssid, transition_epoch)

  def _handle_event(self, event: str):
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
        self._handle_connected(ssid, expected_epoch=epoch, profile_uuid=status.get("id_str"))

    elif "CTRL-EVENT-DISCONNECTED" in event:
      with self._state_lock:
        epoch = self._user_epoch
        expected_state = self._wifi_state
      now = time.monotonic()

      with self._radio_lock, self._state_lock:
        if (
          self._tethering_active
          or self._user_epoch != epoch
          or self._wifi_state != expected_state
          or self._wifi_state.status in (ConnectStatus.CONNECTING, ConnectStatus.DISCONNECTED)
        ):
          return

        ssid = self._wifi_state.ssid
        if self._wifi_state.status == ConnectStatus.CONNECTED and ssid is not None:
          self._dhcp_adoption_ssid = ssid
          self._last_connecting_at = now
          self._last_scanning_recheck = 0.0
          self._wifi_state = WifiState(ssid=ssid, status=ConnectStatus.CONNECTING)
          return

        self._wifi_state = WifiState()
        self._clear_station_state()
        self._enqueue_callbacks(self._disconnected)

    elif "TEMP-DISABLED" in event and "reason=WRONG_KEY" in event:
      event_ssid = parse_event_ssid(event)
      event_network_id = parse_event_network_id(event)
      if event_ssid is not None:
        with self._radio_lock:
          with self._state_lock:
            current_ssid = self._wifi_state.ssid
            # The event SSID is authoritative for auto-connect
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

            failed_epoch = self._user_epoch
            failed_operation = self._station_operation
            failed_pending = pending

          # Debounce WRONG_KEY per profile, not per SSID
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

          # Try remaining profiles before requesting new credentials
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
                with self._state_lock:
                  if (
                    self._user_epoch != failed_epoch
                    or self._station_operation is not failed_operation
                    or (failed_pending is not None and self._pending_connection is not failed_pending)
                  ):
                    return
                  self._last_connecting_at = now
                  self._last_scanning_recheck = 0.0
                  self._network_not_found_epoch = None
                  self._network_not_found_events = 0
                return
              response = self._request("ENABLE_NETWORK all").strip()
              if not response.startswith("OK"):
                raise RuntimeError(f"ENABLE_NETWORK all failed: {response}")
            except Exception:
              cloudlog.exception("Failed to update saved networks after WRONG_KEY")

          with self._state_lock:
            if (
              self._user_epoch != failed_epoch
              or self._station_operation is not failed_operation
              or (failed_pending is not None and self._pending_connection is not failed_pending)
            ):
              return
            self._clear_pending_connection(event_ssid, epoch=failed_epoch)
            self._set_connecting(None, kind=StationOperationKind.AUTH_FAILURE, operation_ssid=event_ssid)
            auth_epoch = self._user_epoch
            auth_operation = self._station_operation
          # DISCONNECTED may arrive while CONNECTING and skip cleanup
          self._clear_station_state()
          with self._state_lock:
            if self._user_epoch != auth_epoch or self._station_operation is not auth_operation:
              return
            self._enqueue_callbacks(self._need_auth, event_ssid)
            self._enqueue_callbacks(self._disconnected)

    elif "CTRL-EVENT-NETWORK-NOT-FOUND" in event:
      now = time.monotonic()
      with self._state_lock:
        if self._wifi_state.status != ConnectStatus.CONNECTING:
          return
        # Reconciliation disambiguates delayed NETWORK-NOT-FOUND events
        if now - self._last_connecting_at >= CONNECTING_STALE_TIMEOUT_SECONDS:
          self._network_not_found_events += 1
          if self._network_not_found_events >= NETWORK_NOT_FOUND_EVENTS_REQUIRED:
            self._network_not_found_epoch = self._user_epoch

    elif "Trying to associate with" in event or "Associated with" in event:
      with self._state_lock:
        epoch = self._user_epoch
        expected_state = self._wifi_state
      if expected_state.status != ConnectStatus.DISCONNECTED:
        return

      ssid = None
      if self._ctrl:
        try:
          status = parse_status(self._request("STATUS"))
          ssid = status.get("ssid")
        except Exception:
          pass
      now = time.monotonic()
      with self._radio_lock, self._state_lock:
        if self._user_epoch != epoch or self._wifi_state != expected_state:
          return
        self._last_connecting_at = now
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
      associated = self._associated_ssid is not None
      self._request("SCAN TYPE=ONLY" if associated or self._wifi_state.status == ConnectStatus.CONNECTED else "SCAN")
    except Exception:
      cloudlog.exception("Failed to request scan")

  def _reconcile_tethering_state(self):
    now = time.monotonic()
    if now - self._last_connected_recheck < SCAN_PERIOD_SECONDS:
      return
    self._last_connected_recheck = now

    with self._radio_lock:
      if self._tethering_transition_pending or not self._tethering_active:
        return

      try:
        if self._ipv4_forward is not None:
          self._apply_ipv4_forward(self._ipv4_forward)
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
      # Ignore STATUS results superseded by newer user action
      if self._user_epoch != epoch:
        return
      # Re-adopt AP mode before station reconciliation
      if status.get("mode") == "AP":
        if self._adopt_ap_state(status.get("ssid")):
          return
        # Keep an incomplete AP disconnected so tethering can recover
        return
      if status.get("wpa_state") == "COMPLETED" and status.get("ssid"):
        self._handle_connected(status["ssid"], expected_epoch=epoch, profile_uuid=status.get("id_str"))
      return

    # Rate-limit recovery from missed DISCONNECTED events
    if current_state.status == ConnectStatus.CONNECTED:
      now = time.monotonic()
      if now - self._last_connected_recheck < SCAN_PERIOD_SECONDS:
        return
      self._last_connected_recheck = now
      with self._state_lock:
        epoch = self._user_epoch
        expected_operation = self._station_operation
        expected_association = (self._associated_ssid, self._associated_epoch)
      try:
        status = parse_status(self._request("STATUS"))
      except Exception:
        return
      # Ignore STATUS results superseded by newer user action
      if self._user_epoch != epoch:
        return
      wpa_state = status.get("wpa_state", "")
      status_ssid = status.get("ssid")
      if wpa_state == "COMPLETED" and status_ssid is not None and status_ssid == current_state.ssid:
        self._handle_connected(status_ssid, expected_epoch=epoch, profile_uuid=status.get("id_str"))
        return
      if wpa_state == "COMPLETED" and status_ssid:
        # Preserve the lease when adopting a roam missed by the monitor
        self._handle_connected(status_ssid, expected_epoch=epoch, profile_uuid=status.get("id_str"))
        return
      # Preserve the lease during transient roam and rekey states
      if wpa_state in ("SCANNING", "AUTHENTICATING", "ASSOCIATING", "ASSOCIATED",
                       "4WAY_HANDSHAKE", "GROUP_HANDSHAKE"):
        return
      with self._radio_lock:
        with self._state_lock:
          if (
            self._user_epoch != epoch
            or self._station_operation is not expected_operation
            or (self._associated_ssid, self._associated_epoch) != expected_association
          ):
            return

        try:
          latest_status = parse_status(self._request("STATUS"))
        except Exception:
          cloudlog.exception("Failed to confirm disconnected wifi state from STATUS")
          return
        latest_wpa_state = latest_status.get("wpa_state", "")
        latest_ssid = latest_status.get("ssid")
        if latest_wpa_state == "COMPLETED" and latest_ssid:
          self._handle_connected(
            latest_ssid, expected_epoch=epoch, profile_uuid=latest_status.get("id_str"),
          )
          return
        if latest_wpa_state in ("SCANNING", "AUTHENTICATING", "ASSOCIATING", "ASSOCIATED",
                                "4WAY_HANDSHAKE", "GROUP_HANDSHAKE"):
          return

        with self._state_lock:
          if (
            self._user_epoch != epoch
            or self._station_operation is not expected_operation
            or (self._associated_ssid, self._associated_epoch) != expected_association
          ):
            return
          self._wifi_state = WifiState()
          self._dhcp_adoption_ssid = None
          self._clear_station_state()
          self._enqueue_callbacks(self._disconnected)
      return

    if current_state.status != ConnectStatus.CONNECTING:
      return
    now = time.monotonic()
    if now - self._last_connecting_at < CONNECTING_STALE_TIMEOUT_SECONDS:
      return

    # Snapshot the operation before the blocking STATUS request
    with self._state_lock:
      epoch = self._user_epoch
      expected_operation = self._station_operation
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
      self._handle_connected(status_ssid, expected_epoch=epoch, profile_uuid=status.get("id_str"))
    elif wpa_state == "SCANNING" and self._network_not_found_epoch != epoch:
      # Hidden SSIDs may remain SCANNING beyond the stale timeout
      self._last_scanning_recheck = time.monotonic()
    elif wpa_state in ("DISCONNECTED", "INACTIVE", "SCANNING", "AUTHENTICATING", "ASSOCIATING",
                      "ASSOCIATED", "4WAY_HANDSHAKE", "GROUP_HANDSHAKE"):
      with self._radio_lock:
        with self._state_lock:
          if (
            self._user_epoch != epoch
            or self._station_operation is not expected_operation
          ):
            return

        try:
          latest_status = parse_status(self._request("STATUS"))
        except Exception:
          cloudlog.exception("Failed to confirm terminal wifi state from STATUS")
          return
        if latest_status.get("wpa_state") == "COMPLETED" and latest_status.get("ssid"):
          self._handle_connected(
            latest_status["ssid"], expected_epoch=epoch, profile_uuid=latest_status.get("id_str"),
          )
          return

        with self._state_lock:
          if self._user_epoch != epoch or self._station_operation is not expected_operation:
            return
          ssid = current_state.ssid
          pending = self._pending_connection
          pending_network_id = pending.network_id if (
            pending is not None
            and pending.epoch == epoch
            and pending.ssid == ssid
          ) else None

        self._restore_station_runtime(
          pending_network_id,
          [],
          [pending_network_id] if pending_network_id is not None else [],
        )

        with self._state_lock:
          if (
            self._user_epoch != epoch
            or self._station_operation is not expected_operation
            or self._pending_connection is not pending
          ):
            return
          self._clear_pending_connection(ssid, epoch=epoch)
          self._set_connecting(None, kind=StationOperationKind.TIMEOUT, operation_ssid=ssid)
        self._clear_station_state()
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

        # A successful empty scan clears stale networks
        self._networks = networks
        self._update_active_connection_info()
        self._mark_networks_updated()

    if block:
      worker()
    else:
      threading.Thread(target=worker, daemon=True).start()

  def _poll_for_ip(self, ssid: str | None = None, epoch: int | None = None):
    ssid = self._associated_ssid if ssid is None else ssid
    epoch = self._user_epoch if epoch is None else epoch

    def worker():
      for _ in range(50):  # 10 seconds max
        if ssid is None or not self._connected_transition_is_current(ssid, epoch):
          return
        self._update_active_connection_info()
        if self._ipv4_address and self._complete_station_connection(ssid, epoch):
          return
        time.sleep(0.2)
    threading.Thread(target=worker, daemon=True).start()

  def _update_active_connection_info(self):
    ipv4_address = ""
    metered = MeteredType.UNKNOWN
    profile_uuid = None

    with self._state_lock:
      station_epoch = self._user_epoch
      station_ssid = self._associated_ssid
      associated_epoch = self._associated_epoch
      station_operation = self._station_operation
      station_active = station_ssid is not None or self._wifi_state.status == ConnectStatus.CONNECTED

    if station_active:
      if self._ctrl:
        try:
          status = parse_status(self._request("STATUS"))
          ipv4_address = status.get("ip_address", "")
          profile_uuid = status.get("id_str")
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

      ssid = station_ssid or self._wifi_state.ssid
      if ssid and self._store is not None:
        metered = self._store.get_metered(ssid, profile_uuid)

    with self._state_lock:
      if (
        self._user_epoch != station_epoch
        or self._associated_ssid != station_ssid
        or self._associated_epoch != associated_epoch
        or self._station_operation is not station_operation
      ):
        return
      self._ipv4_address = ipv4_address
      self._current_network_metered = metered

  def connect_to_network(self, ssid: str, password: str, hidden: bool = False,
                         security: SecurityType | None = None):
    # Guard non-UI callers while tethering
    if self._tethering_active:
      cloudlog.warning(f"Ignoring connect to {ssid!r} while tethering is active")
      return
    if not is_valid_ssid(ssid):
      cloudlog.warning(f"Ignoring connect to invalid SSID {ssid!r}")
      return
    security = security if security is not None else (SecurityType.WPA if password else SecurityType.OPEN)
    if security not in (SecurityType.OPEN, SecurityType.WPA):
      cloudlog.warning(f"Ignoring connect to {ssid!r} with unsupported security")
      return
    if security == SecurityType.WPA and not is_valid_psk(password):
      cloudlog.warning(f"Ignoring connect to {ssid!r} with invalid passphrase")
      self._enqueue_callbacks(self._need_auth, ssid)
      return
    if security == SecurityType.OPEN and password:
      cloudlog.warning(f"Ignoring open-network connect to {ssid!r} with a passphrase")
      return
    with self._state_lock:
      self._station_cleanup_pending |= (
        self._associated_ssid is not None
        or self._wifi_state.status == ConnectStatus.CONNECTED
        or self._dhcp_adoption_ssid is not None
      )
      self._set_connecting(ssid)
      self._set_pending_connection(ssid, password, hidden, security)
      epoch = self._user_epoch

    def worker():
      with self._radio_lock:
        if not self._prepare_connection(epoch):
          return
        if self._ctrl is None:
          cloudlog.warning("No wpa_supplicant connection")
          # Ignore failures superseded by a newer connection attempt
          if self._user_epoch != epoch:
            return
          self._clear_pending_connection(ssid)
          # Reset inline because _init_wifi_state ignores a missing control socket
          self._set_connecting(None, operation_ssid=ssid)
          self._enqueue_callbacks(self._disconnected)
          return

        # Serialize the epoch check with runtime-network replacement
        if self._user_epoch != epoch:
          return
        existing_ids: list[str] = []
        removed_ids: list[str] = []
        net_id = None
        try:
          existing_ids = self._list_network_ids(ssid)
          with self._state_lock:
            pending = self._pending_connection
            profile_uuid = pending.profile_uuid if pending is not None and pending.epoch == epoch else None
          if profile_uuid is None:
            return
          net_id = self._add_and_select_network(
            ssid, password, hidden, profile_uuid=profile_uuid, security=security,
          )
          self._set_pending_network_id(net_id, epoch)
          if self._user_epoch != epoch:
            self._restore_station_runtime(net_id, existing_ids, removed_ids)
            return
          for existing_id in existing_ids:
            self._remove_wpa_network_id(existing_id)
            removed_ids.append(existing_id)
            if self._user_epoch != epoch:
              self._restore_station_runtime(net_id, existing_ids, removed_ids)
              return
        except Exception:
          cloudlog.exception(f"Failed to connect to {ssid}")
          if net_id is not None or removed_ids:
            self._restore_station_runtime(net_id, existing_ids, removed_ids)
          if self._user_epoch != epoch:
            return
          self._clear_pending_connection(ssid, epoch=epoch)
          self._set_connecting(None, operation_ssid=ssid)
          self._enqueue_callbacks(self._disconnected)

    threading.Thread(target=worker, daemon=True).start()

  def forget_connection(self, ssid: str, block: bool = False):
    with self._state_lock:
      forget_active = (
        self._wifi_state.ssid == ssid and self._wifi_state.status in (ConnectStatus.CONNECTING, ConnectStatus.CONNECTED)
        or self._associated_ssid == ssid
        or self._dhcp_adoption_ssid == ssid
      )
      cleanup_required = (
        self._wifi_state == WifiState(ssid, ConnectStatus.CONNECTED)
        or self._associated_ssid == ssid
        or self._dhcp_adoption_ssid == ssid
      )
      forget_epoch = self._user_epoch

    def transition():
      try:
        store = self._require_store()
      except RuntimeError:
        cloudlog.exception(f"forget_connection: manager not initialized for {ssid}")
        self._enqueue_callbacks(self._forget_failed, ssid)
        return

      existed = store.contains(ssid)
      removed = store.remove(ssid)
      if existed and not removed:
        # Keep runtime state when persistent removal fails
        cloudlog.warning(f"forget_connection: failed to remove {ssid} from disk; leaving runtime intact")
        self._enqueue_callbacks(self._forget_failed, ssid)
        return
      if not removed:
        cloudlog.warning(f"Trying to forget unknown connection: {ssid}")

      with self._radio_lock:
        try:
          generate_wpa_conf(store)
        except Exception:
          cloudlog.exception(f"Failed to regenerate configuration after forgetting {ssid}")

        self._clear_pending_connection(ssid, epoch=forget_epoch)
        with self._state_lock:
          owns_epoch = self._user_epoch == forget_epoch
          cleanup_epoch = None
          if forget_active and owns_epoch:
            self._station_cleanup_pending |= cleanup_required
            self._set_connecting(None, kind=StationOperationKind.FORGET, operation_ssid=ssid)
            cleanup_epoch = self._user_epoch
        owns_runtime = self._prepare_connection(cleanup_epoch) if cleanup_epoch is not None else owns_epoch

        try:
          if self._ctrl:
            with self._state_lock:
              preserve_selection = self._wifi_state.status == ConnectStatus.CONNECTING and self._wifi_state.ssid != ssid
            if forget_active and owns_runtime:
              self._request("DISCONNECT")
            self._remove_wpa_network(ssid)
            if not preserve_selection:
              self._request("ENABLE_NETWORK all")
            # Reassociate only when forgetting the active profile
            if forget_active and owns_runtime:
              self._request("REASSOCIATE")
        except Exception:
          cloudlog.exception(f"Failed to remove runtime connection after forgetting {ssid}")

      self._enqueue_callbacks(self._forgotten, ssid)

    def worker():
      with self._radio_lock:
        transition()

    if block:
      worker()
    else:
      threading.Thread(target=worker, daemon=True).start()

  def activate_connection(self, ssid: str, block: bool = False):
    if self._tethering_active:
      cloudlog.warning(f"Ignoring activate {ssid!r} while tethering is active")
      return
    with self._state_lock:
      self._station_cleanup_pending |= (
        self._associated_ssid is not None
        or self._wifi_state.status == ConnectStatus.CONNECTED
        or self._dhcp_adoption_ssid is not None
      )
      self._set_connecting(ssid, kind=StationOperationKind.ACTIVATE)
      self._clear_pending_connection()
      epoch = self._user_epoch

    def worker():
      with self._radio_lock:
        if not self._prepare_connection(epoch):
          return
        if self._ctrl is None:
          cloudlog.warning(f"No wpa_supplicant connection for activate {ssid}")
          # Ignore failures superseded by a newer connection attempt
          if self._user_epoch != epoch:
            return
          # Reset inline because _init_wifi_state ignores a missing control socket
          self._set_connecting(None, kind=StationOperationKind.ACTIVATE, operation_ssid=ssid)
          self._enqueue_callbacks(self._disconnected)
          return

        def reset_to_disconnected():
          if self._user_epoch != epoch:
            return
          self._restore_station_runtime(None, [], ["activation"])
          # Notify the UI when control-socket recovery fails
          self._set_connecting(None, kind=StationOperationKind.ACTIVATE, operation_ssid=ssid)
          self._enqueue_callbacks(self._disconnected)

        # Serialize the epoch check with saved-network activation
        if self._user_epoch != epoch:
          return
        try:
          ids = self._list_network_ids(ssid)
          if ids:
            self._select_network_ids(ids)
          else:
            profiles = [entry for profile_ssid, entry in self._require_store().get_profiles() if profile_ssid == ssid]
            if profiles:
              ids = [
                self._add_and_select_network(
                  ssid,
                  entry.get("psk", ""),
                  entry.get("hidden", False),
                  entry.get("priority", 0),
                  bssid=entry.get("bssid") or None,
                  profile_uuid=entry.get("uuid"),
                  security=entry.get("security"),
                )
                for entry in profiles
              ]
              if len(ids) > 1:
                self._select_network_ids(ids)
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
    commands = ["DISABLE_NETWORK all", *(f"ENABLE_NETWORK {net_id}" for net_id in net_ids), "REASSOCIATE"]
    for command in commands:
      resp = self._request(command).strip()
      if not resp.startswith("OK"):
        raise RuntimeError(f"{command} failed: {resp}")

  def _add_and_select_network(self, ssid: str, psk: str = "", hidden: bool = False, priority: int = 0,
                              bssid: str | None = None, profile_uuid: str | None = None,
                              security: SecurityType | None = None) -> str:
    security = security if security is not None else (SecurityType.WPA if psk else SecurityType.OPEN)
    if security not in (SecurityType.OPEN, SecurityType.WPA):
      raise ValueError(f"Unsupported security type: {security!r}")
    if security == SecurityType.WPA and not is_valid_psk(psk):
      raise ValueError("Invalid WPA passphrase")

    """Add a network and select it. Every SET_NETWORK is checked so a bad PSK/key_mgmt
    surfaces an immediate error instead of a delayed WRONG_KEY; orphans get REMOVE_NETWORK'd."""
    net_id = self._request("ADD_NETWORK").strip()
    if not net_id.isdigit():
      raise RuntimeError(f"ADD_NETWORK failed: {net_id}")

    try:
      self._wpa_set_network(net_id, "ssid", format_ssid_value(ssid))
      if security == SecurityType.WPA:
        self._wpa_set_network(net_id, "psk", format_psk_value(psk))
      else:
        self._wpa_set_network(net_id, "key_mgmt", "NONE")
      if hidden:
        self._wpa_set_network(net_id, "scan_ssid", "1")
      if bssid:
        self._wpa_set_network(net_id, "bssid", bssid)
      if profile_uuid:
        self._wpa_set_network(net_id, "id_str", f'"{sanitize_for_conf(profile_uuid)}"')
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
    # WPA PSKs use 8-63 UTF-8 bytes or 64 hexadecimal characters
    pw_bytes = len(password.encode("utf-8"))
    if not is_valid_psk(password):
      cloudlog.warning(f"set_tethering_password: rejecting invalid password (bytes={pw_bytes})")
      # Re-enable tethering controls after rejected input
      self._enqueue_callbacks(self._activated if self._tethering_active else self._disconnected)
      return
    self._tethering_password_epoch += 1
    epoch = self._tethering_password_epoch
    def transition():
      try:
        store = self._require_store()
        if not store.set_tethering_password(self._tethering_ssid, password):
          raise OSError("no durable tethering profile")
      except Exception:
        cloudlog.exception("Failed to persist tethering password; runtime state unchanged")
        self._enqueue_callbacks(self._activated if self._tethering_active else self._disconnected)
        return
      self._tethering_psk = password
      if self._tethering_active:
        try:
          # Keep the hotspot active during the password restart
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
      with self._radio_lock:
        if self._tethering_password_epoch == epoch:
          transition()
    threading.Thread(target=worker, daemon=True).start()

  def _apply_ipv4_forward(self, enabled: bool):
    value = "1" if enabled else "0"
    subprocess.run(["sudo", "sysctl", f"net.ipv4.ip_forward={value}"], check=True)
    actual = Path("/proc/sys/net/ipv4/ip_forward").read_text().strip()
    if actual != value:
      raise RuntimeError(f"Failed to set net.ipv4.ip_forward={value} (actual={actual!r})")

  def set_ipv4_forward(self, enabled: bool):
    with self._radio_lock:
      if self._ipv4_forward == enabled:
        return
      if self._tethering_active:
        self._apply_ipv4_forward(enabled)
      self._ipv4_forward = enabled

  def set_tethering_active(self, active: bool):
    # Report enable immediately and disable after station recovery
    self._tethering_epoch += 1
    epoch = self._tethering_epoch
    self._tethering_transition_pending = True
    if active:
      self._tethering_active = True
    def transition():
      if active:
        try:
          self._start_tethering()
        except Exception:
          cloudlog.exception("Failed to start tethering, rolling back")
          try:
            self._stop_tethering()
          except Exception:
            cloudlog.exception("Tethering rollback also failed")
            self._tethering_active = False
            self._wifi_state = WifiState()
            self._enqueue_callbacks(self._disconnected)
      else:
        if not self._tethering_started:
          self._tethering_active = False
          self._enqueue_callbacks(self._disconnected)
          return
        try:
          self._stop_tethering()
        except Exception:
          cloudlog.exception("Failed to stop tethering")
          # Clear UI state even if teardown fails
          self._tethering_active = False
          self._wifi_state = WifiState()
          self._ipv4_address = ""
          self._enqueue_callbacks(self._disconnected)

    def worker():
      with self._radio_lock:
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
    if self._ipv4_forward is None:
      raise RuntimeError("IPv4 forwarding policy is not initialized")

    self._tethering_active = True
    self._tethering_started = True
    self._set_connecting(self._tethering_ssid, requested=False)

    psk = self._tethering_psk

    if self._ctrl:
      self._ctrl.close()
      self._ctrl = None

    # Target only openpilot-owned daemons, including surviving AP instances
    self._monitor_epoch += 1
    stop_wpa_supplicant(WPA_SUPPLICANT_CONF)
    stop_wpa_supplicant(WPA_AP_CONF)
    self._dhcp.stop()
    prepare_wpa_runtime()
    time.sleep(0.5)
    self._apply_ipv4_forward(self._ipv4_forward)

    safe_tether_ssid = sanitize_for_conf(self._tethering_ssid)
    lines = [WPA_CTRL_INTERFACE, "ap_scan=2", "",
             "network={", f'  ssid="{safe_tether_ssid}"', "  mode=2",
             "  frequency=2437", "  key_mgmt=WPA-PSK", "  proto=RSN",
             "  pairwise=CCMP", "  group=CCMP", f'  psk={format_psk_value(psk)}', "}", ""]
    ap_conf = "\n".join(lines)
    with atomic_write(WPA_AP_CONF, overwrite=True) as f:
      f.write(ap_conf)

    subprocess.run([
      "sudo", "wpa_supplicant", "-B", "-i", "wlan0",
      "-c", WPA_AP_CONF, "-P", WPA_PID_FILE, "-D", "nl80211",
    ], check=False)
    time.sleep(1)

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
      "--no-daemon",
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
      start_new_session=True)
    time.sleep(0.2)
    if self._dnsmasq_proc.poll() is not None:
      rc = self._dnsmasq_proc.returncode
      self._dnsmasq_proc = None
      raise RuntimeError(f"dnsmasq exited during tethering bringup (rc={rc})")

    _install_tethering_firewall_rules()

    # Verify that our daemon owns wlan0 in AP mode
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
      self._apply_ipv4_forward(False)
    except (OSError, RuntimeError, subprocess.CalledProcessError):
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

    self._monitor_epoch += 1
    stop_wpa_supplicant(WPA_AP_CONF)
    time.sleep(0.5)

    subprocess.run(["sudo", "ip", "addr", "flush", "dev", "wlan0"], check=False)

    generate_wpa_conf(self._require_store())
    self._ensure_wpa_supplicant()

    self._tethering_active = False
    self._tethering_started = False
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
      # Network daemons outlive the UI and are adopted by the next controller
