import atexit
import configparser
import hashlib
import os
import re
import socket
import subprocess
import tempfile
import threading
import time
import urllib.parse
import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING

from openpilot.common.swaglog import cloudlog
from openpilot.common.utils import sudo_read

if TYPE_CHECKING:
  from openpilot.common.params import Params
else:
  try:
    from openpilot.common.params import Params
  except (ImportError, OSError):
    Params = None

WLAN = "wlan0"
WPA_CTRL_DIR = "/run/wpa_supplicant"
WPA_CTRL_PATH = f"{WPA_CTRL_DIR}/{WLAN}"
WPA_PID_PATH = f"{WPA_CTRL_DIR}/{WLAN}.pid"
WPA_CONF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wpa_supplicant.conf")
UDHCPC_PID_PATH = f"/run/udhcpc.{WLAN}.pid"
UDHCPC_SCRIPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "udhcpc.script")
DNSMASQ_PID_PATH = f"/run/dnsmasq.{WLAN}.pid"
PROFILE_DIRS = ("/data/etc/NetworkManager/system-connections", "/run/NetworkManager/system-connections")  # persistent first, netplan-generated second
NETPLAN_DIR = "/data/etc/netplan"

TETHERING_IP_ADDRESS = "192.168.43.1"
TETHERING_SUBNET = "192.168.43.0/24"
TETHERING_DHCP_RANGE = "192.168.43.2,192.168.43.254,24h"
TETHERING_FREQUENCY = 2437  # channel 6, NetworkManager's band=bg default
TETHERING_NAT_RULE = ["POSTROUTING", "-s", TETHERING_SUBNET, "!", "-d", TETHERING_SUBNET, "-j", "MASQUERADE",
                      "-m", "comment", "--comment", "openpilot-tethering"]
DEFAULT_TETHERING_PASSWORD = "swagswagcomma"
SCAN_PERIOD_SECONDS = 5
DHCP_TIMEOUT_SECONDS = 45  # NetworkManager ipv4.dhcp-timeout default
HANDOFF_TIMEOUT_SECONDS = 5
CTRL_TIMEOUT_SECONDS = 2
AP_TIMEOUT_SECONDS = 10


def normalize_ssid(ssid: str) -> str:
  return ssid.replace("’", "'")  # for iPhone hotspots


class SecurityType(IntEnum):
  OPEN = 0
  WPA = 1
  WPA2 = 2
  WPA3 = 3
  UNSUPPORTED = 4


class MeteredType(IntEnum):
  UNKNOWN = 0
  YES = 1
  NO = 2


@dataclass(frozen=True)
class Network:
  ssid: str
  strength: int
  security_type: SecurityType
  is_tethering: bool


class ConnectStatus(IntEnum):
  DISCONNECTED = 0
  CONNECTING = 1
  CONNECTED = 2


@dataclass(frozen=True)
class WifiState:
  ssid: str | None = None
  status: ConnectStatus = ConnectStatus.DISCONNECTED


class WpaCtrl:
  # wpa_supplicant control interface over unix datagram sockets
  def __init__(self, path: str):
    self._path = path
    self._lock = threading.Lock()
    self._sock = self._open()

  def _open(self) -> socket.socket:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    sock.bind(f"\0openpilot-wpa-{os.getpid()}-{time.monotonic_ns()}")
    sock.connect(self._path)
    sock.settimeout(CTRL_TIMEOUT_SECONDS)
    return sock

  def request(self, cmd: str) -> str:
    with self._lock:
      self._sock.send(cmd.encode())
      while True:
        reply = self._sock.recv(65536).decode("utf-8", "replace")
        if not reply.startswith("<"):
          return reply.rstrip("\n")

  def ok(self, cmd: str) -> bool:
    return self.request(cmd) == "OK"

  def attach(self) -> socket.socket:
    sock = self._open()
    sock.send(b"ATTACH")
    if sock.recv(64).rstrip(b"\n") != b"OK":
      sock.close()
      raise OSError("wpa_supplicant ATTACH failed")
    sock.settimeout(1)
    return sock

  def close(self):
    self._sock.close()


def decode_ssid(value: str) -> str:
  # wpa_supplicant printf_encode: printable ASCII as is, \\ \" \e \n \r \t, everything else as \xNN
  out = bytearray()
  i = 0
  while i < len(value):
    if value[i] == "\\" and i + 1 < len(value):
      esc = value[i + 1]
      if esc == "x" and i + 3 < len(value):
        out.append(int(value[i + 2:i + 4], 16))
        i += 4
        continue
      out.extend({"n": b"\n", "r": b"\r", "t": b"\t", "e": b"\x1b"}.get(esc, esc.encode()))
      i += 2
      continue
    out.extend(value[i].encode())
    i += 1
  return out.decode("utf-8", "replace")


def parse_status(raw: str) -> dict[str, str]:
  return dict(line.split("=", 1) for line in raw.splitlines() if "=" in line)


def parse_scan_results(raw: str) -> list[tuple[str, int, str]]:
  # bssid / frequency / signal level / flags / ssid
  results = []
  for line in raw.splitlines()[1:]:
    fields = line.split("\t")
    if len(fields) < 5:
      continue
    results.append((decode_ssid(fields[4]), int(fields[2]), fields[3]))
  return results


def dbm_to_percent(dbm: int) -> int:
  # NetworkManager nm-wifi-utils.c: -40 dBm is 100%, -100 dBm is 0%
  return 100 - int(100 * (-40 - max(-100, min(-40, dbm))) / 60)


def security_type_from_flags(flags: str) -> SecurityType:
  if "-PSK" in flags:
    return SecurityType.WPA
  if "WPA" not in flags and "WEP" not in flags:
    return SecurityType.OPEN
  return SecurityType.UNSUPPORTED


def wpa_psk(ssid: str, passphrase: str) -> str:
  # IEEE 802.11i PSK derivation; a keyfile may already hold the 64 hex character raw key
  if len(passphrase) == 64 and all(c in "0123456789abcdefABCDEF" for c in passphrase):
    return passphrase.lower()
  return hashlib.pbkdf2_hmac("sha1", passphrase.encode(), ssid.encode(), 4096, 32).hex()


def _sudo(*cmd: str, check: bool = True) -> subprocess.CompletedProcess:
  return subprocess.run(["sudo", *cmd], check=check, capture_output=True, text=True)


def _read_pid(pid_path: str) -> int | None:
  try:
    with open(pid_path) as f:
      return int(f.read().strip())
  except (OSError, ValueError):
    return None


def _pid_alive(pid_path: str) -> bool:
  pid = _read_pid(pid_path)
  if pid is None:
    return False
  try:
    os.kill(pid, 0)
  except ProcessLookupError:
    return False
  except PermissionError:
    pass  # root-owned daemon
  return True


@dataclass(frozen=True)
class Profile:
  path: str
  uuid: str
  ssid: str
  psk: str | None
  hidden: bool
  metered: MeteredType
  is_ap: bool


def _keyfile_ssid(value: str) -> str:
  # NetworkManager stores non-ASCII SSIDs as a byte;byte; list
  parts = value.split(";")
  if value.endswith(";") and len(parts) > 1 and all(p.isdigit() for p in parts[:-1]):
    return bytes(int(p) for p in parts[:-1]).decode("utf-8", "replace")
  return value


def _ssid_keyfile(ssid: str) -> str:
  if ssid.isascii() and ssid.isprintable():
    return ssid
  return "".join(f"{b};" for b in ssid.encode())


def read_profiles() -> list[Profile]:
  profiles = []
  for directory in PROFILE_DIRS:
    for path in sorted(Path(directory).glob("*.nmconnection")):
      cp = configparser.ConfigParser(interpolation=None, strict=False)
      try:
        cp.read_string(sudo_read(str(path)))
      except configparser.Error:
        cloudlog.warning(f"Unreadable connection profile {path}")
        continue
      if cp.get("connection", "type", fallback="") != "wifi":
        continue
      ssid = _keyfile_ssid(cp.get("wifi", "ssid", fallback=""))
      profile_uuid = cp.get("connection", "uuid", fallback="")
      if not ssid or not profile_uuid:
        cloudlog.warning(f"Wi-Fi profile without ssid or uuid {path}")
        continue
      metered = cp.getint("connection", "metered", fallback=0)
      profiles.append(Profile(path=str(path), uuid=profile_uuid, ssid=ssid, psk=cp.get("wifi-security", "psk", fallback=None),
                              hidden=cp.getboolean("wifi", "hidden", fallback=False),
                              metered=MeteredType(metered) if metered in (MeteredType.YES, MeteredType.NO) else MeteredType.UNKNOWN,
                              is_ap=cp.get("wifi", "mode", fallback="") == "ap"))
  return profiles


def remove_profile(profile: Profile) -> None:
  _sudo("rm", "-f", profile.path)
  if os.path.basename(profile.path).startswith("netplan-NM-"):
    _sudo("rm", "-f", os.path.join(NETPLAN_DIR, f"90-NM-{profile.uuid}.yaml"))


def write_profile(profile: Profile) -> Profile:
  # persistent keyfile in NetworkManager format, so a rollback keeps the network
  cp = configparser.ConfigParser(interpolation=None)
  cp["connection"] = {"id": "Hotspot" if profile.is_ap else f"openpilot connection {profile.ssid}", "uuid": profile.uuid, "type": "wifi",
                      "autoconnect-retries": "0"}
  cp["wifi"] = {"ssid": _ssid_keyfile(profile.ssid)}
  if profile.is_ap:
    cp["connection"].update({"interface-name": WLAN, "autoconnect": "false"})
    cp["wifi"].update({"band": "bg", "mode": "ap"})
    cp["wifi-security"] = {"group": "ccmp;", "key-mgmt": "wpa-psk", "pairwise": "ccmp;", "proto": "rsn;", "psk": profile.psk or ""}
    cp["ipv4"] = {"method": "shared", "address1": f"{TETHERING_IP_ADDRESS}/24,{TETHERING_IP_ADDRESS}", "never-default": "true"}
  else:
    if profile.metered != MeteredType.UNKNOWN:
      cp["connection"]["metered"] = str(int(profile.metered))
    cp["wifi"].update({"mode": "infrastructure", "hidden": "true" if profile.hidden else "false"})
    if profile.psk:
      cp["wifi-security"] = {"key-mgmt": "wpa-psk", "auth-alg": "open", "psk": profile.psk}
    cp["ipv4"] = {"method": "auto", "dns-priority": "600"}
  cp["ipv6"] = {"method": "ignore"}

  path = os.path.join(PROFILE_DIRS[0], f"{urllib.parse.quote(profile.ssid, safe='')}.nmconnection")
  with tempfile.NamedTemporaryFile("w", delete=False) as f:
    cp.write(f, space_around_delimiters=False)
  try:
    _sudo("install", "-m", "600", f.name, path)
  finally:
    os.unlink(f.name)
  for other in read_profiles():
    if other.uuid == profile.uuid and other.path != path:
      remove_profile(other)
  return replace(profile, path=path)


ASSOCIATING_STATES = ("AUTHENTICATING", "ASSOCIATING", "ASSOCIATED", "4WAY_HANDSHAKE", "GROUP_HANDSHAKE")


class WifiManager:
  def __init__(self):
    self._networks: list[Network] = []  # an unsorted list of available Networks. a Network can be comprised of multiple APs
    self._active = True  # used to not run when not in settings
    self._exit = threading.Event()
    self._ready = False
    self._lock = threading.RLock()

    self._ctrl: WpaCtrl | None = None
    self._events: socket.socket | None = None
    self._network_ids: dict[int, str] = {}  # wpa_supplicant network id -> ssid
    self._profiles: list[Profile] = []

    # State
    self._wifi_state = WifiState()
    self._selected: str | None = None  # ssid the user asked for, kept until the attempt ends
    self._pending: Profile | None = None  # new network, persisted once it has an address
    self._ipv4_address = ""
    self._current_network_metered = MeteredType.UNKNOWN
    self._ipv4_forward = False
    self._callback_queue: list[Callable] = []

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

    self._scan_thread = threading.Thread(target=self._network_scanner, daemon=True)
    self._monitor_thread = threading.Thread(target=self._monitor, daemon=True)
    self._init_thread = threading.Thread(target=self._initialize, daemon=True)
    self._init_thread.start()
    atexit.register(self.stop)

  def _initialize(self):
    try:
      self._start_supplicant()
      self._hotspot_profile()
    except Exception:
      cloudlog.exception("WifiManager failed to start wpa_supplicant")
    self._ready = True
    self._scan_thread.start()
    self._monitor_thread.start()
    cloudlog.debug("WifiManager initialized")

  def _start_supplicant(self):
    adopt = _pid_alive(WPA_PID_PATH)
    if not adopt:
      with self._lock:
        self._selected, self._pending = None, None
      _sudo("nmcli", "dev", "set", WLAN, "managed", "no", check=False)
      self._wait_for_handoff()
      subprocess.run(["sudo", "wpa_supplicant", "-B", "-i", WLAN, "-D", "nl80211", "-c", WPA_CONF_PATH, "-P", WPA_PID_PATH], check=True,
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    ctrl = self._connect_ctrl()
    events = ctrl.attach()
    with self._lock:
      old_ctrl, old_events = self._ctrl, self._events
      self._ctrl, self._events = ctrl, events
      self._profiles = read_profiles()
      self._network_ids = self._list_networks() if adopt else {}
      if not adopt:
        for profile in self._profiles:
          if not profile.is_ap:
            self._add_network(profile.ssid, profile.psk, profile.hidden)
        ctrl.ok("ENABLE_NETWORK all")
      for old in (old_events, old_ctrl):
        if old is not None:
          old.close()
    status = self._refresh_status()
    if status.get("mode") == "AP" and status.get("wpa_state") == "COMPLETED":
      self._ensure_tethering_services()
    elif status.get("mode") != "AP" and self._selected != self._tethering_ssid:
      self._start_dhcp()

  def _wait_for_handoff(self):
    # NetworkManager tears wlan0 down asynchronously before its control socket disappears
    deadline = time.monotonic() + HANDOFF_TIMEOUT_SECONDS
    while os.path.exists(WPA_CTRL_PATH) and time.monotonic() < deadline:
      time.sleep(0.2)
    if os.path.exists(WPA_CTRL_PATH):
      cloudlog.warning(f"{WLAN} was not released by NetworkManager")

  def _connect_ctrl(self) -> WpaCtrl:
    deadline = time.monotonic() + HANDOFF_TIMEOUT_SECONDS
    while True:
      try:
        ctrl = WpaCtrl(WPA_CTRL_PATH)
        if ctrl.request("PING") == "PONG":
          return ctrl
        ctrl.close()
      except OSError:
        pass
      if time.monotonic() > deadline:
        raise OSError(f"wpa_supplicant control socket {WPA_CTRL_PATH} not available")
      time.sleep(0.2)

  def _list_networks(self) -> dict[int, str]:
    ids = {}
    for line in self._ctrl.request("LIST_NETWORKS").splitlines()[1:]:
      fields = line.split("\t")
      if len(fields) >= 2 and fields[0].isdigit():
        ids[int(fields[0])] = decode_ssid(fields[1])
    return ids

  def _add_network(self, ssid: str, psk: str | None, hidden: bool) -> int:
    nid = int(self._ctrl.request("ADD_NETWORK"))
    settings = [f"ssid {ssid.encode().hex()}", f"psk {wpa_psk(ssid, psk)}" if psk else "key_mgmt NONE"]
    if hidden:
      settings.append("scan_ssid 1")
    for setting in settings:
      if not self._ctrl.ok(f"SET_NETWORK {nid} {setting}"):
        self._ctrl.ok(f"REMOVE_NETWORK {nid}")
        raise ValueError(f"wpa_supplicant rejected {setting.split()[0]} for {ssid}")
    self._network_ids[nid] = ssid
    return nid

  def _status(self) -> dict[str, str]:
    if self._ctrl is None:
      return {}
    try:
      return parse_status(self._ctrl.request("STATUS"))
    except OSError:
      cloudlog.warning("wpa_supplicant STATUS failed")
      return {}

  def _refresh_status(self):
    status = self._status()
    with self._lock:
      ssid = decode_ssid(status.get("ssid", ""))
      wpa_state = status.get("wpa_state", "")
      ipv4_address, metered = "", MeteredType.UNKNOWN
      if self._selected == self._tethering_ssid:
        wifi_state = WifiState(self._tethering_ssid, ConnectStatus.CONNECTING)
      elif self._selected is not None and ssid != self._selected:
        wifi_state = WifiState(self._selected, ConnectStatus.CONNECTING)
      elif wpa_state == "COMPLETED" and status.get("mode") == "AP":
        wifi_state, ipv4_address = WifiState(ssid, ConnectStatus.CONNECTED), TETHERING_IP_ADDRESS
      elif wpa_state == "COMPLETED" and status.get("ip_address"):
        wifi_state, ipv4_address = WifiState(ssid, ConnectStatus.CONNECTED), status["ip_address"]
        metered = next((p.metered for p in self._profiles if p.ssid == ssid), MeteredType.UNKNOWN)
      elif wpa_state == "COMPLETED" or wpa_state in ASSOCIATING_STATES:
        wifi_state = WifiState(ssid or self._selected, ConnectStatus.CONNECTING)
      elif self._selected is not None:
        wifi_state = WifiState(self._selected, ConnectStatus.CONNECTING)
      else:
        wifi_state = WifiState()
      if wifi_state.status == ConnectStatus.CONNECTED and wifi_state.ssid == self._selected:
        self._selected = None
      self._wifi_state, self._ipv4_address, self._current_network_metered = wifi_state, ipv4_address, metered
    return status

  def _start_dhcp(self):
    # one udhcpc for the life of the supplicant; a renew after each association fetches a lease for the new network
    pid = _read_pid(UDHCPC_PID_PATH)
    if pid is not None and _pid_alive(UDHCPC_PID_PATH):
      _sudo("kill", "-USR1", str(pid), check=False)
    else:
      subprocess.Popen(["sudo", "udhcpc", "-i", WLAN, "-f", "-R", "-s", UDHCPC_SCRIPT_PATH, "-p", UDHCPC_PID_PATH],
                       stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)

  def _stop_dhcp(self):
    pid = _read_pid(UDHCPC_PID_PATH)
    if pid is None or not _pid_alive(UDHCPC_PID_PATH):
      return
    _sudo("kill", str(pid), check=False)
    deadline = time.monotonic() + CTRL_TIMEOUT_SECONDS
    while _pid_alive(UDHCPC_PID_PATH) and time.monotonic() < deadline:
      time.sleep(0.1)
    if _pid_alive(UDHCPC_PID_PATH):
      raise RuntimeError("udhcpc did not stop")

  def add_callbacks(self, need_auth: Callable[[str], None] | None = None,
                    activated: Callable[[], None] | None = None,
                    forgotten: Callable[[str | None], None] | None = None,
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
    # Sort by connected/connecting, then known, then strength, then alphabetically. This is a pure UI ordering and should not affect underlying state.
    return sorted(self._networks, key=lambda n: (n.ssid != self._wifi_state.ssid, not self.is_connection_saved(n.ssid), -n.strength, n.ssid.lower()))

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

  def is_tethering_active(self) -> bool:
    # Check ssid, not connected_ssid, to also catch connecting state
    return self._wifi_state.ssid == self._tethering_ssid

  def is_connection_saved(self, ssid: str) -> bool:
    return any(p.ssid == ssid for p in self._profiles)

  def _enqueue_callbacks(self, cbs: list[Callable], *args):
    with self._lock:
      for cb in cbs:
        self._callback_queue.append(lambda _cb=cb: _cb(*args))

  def process_callbacks(self):
    # Call from UI thread to run any pending callbacks
    with self._lock:
      to_run, self._callback_queue = self._callback_queue, []
    for cb in to_run:
      cb()

  def set_active(self, active: bool):
    self._active = active

    # Update networks and WiFi state (to self-heal) immediately when activating for UI
    if active:
      threading.Thread(target=self._update_networks, daemon=True).start()

  def _network_scanner(self):
    while not self._exit.is_set():
      if self._active and self._ctrl is not None:
        try:
          self._ctrl.request("SCAN")
        except OSError:
          cloudlog.warning("wpa_supplicant SCAN failed")
      self._exit.wait(SCAN_PERIOD_SECONDS)

  def _update_networks(self):
    with self._lock:
      if self._exit.is_set() or not self._active or self._ctrl is None:
        return
      try:
        results = parse_scan_results(self._ctrl.request("SCAN_RESULTS"))
      except OSError:
        cloudlog.warning("wpa_supplicant SCAN_RESULTS failed")
        return

      best: dict[str, tuple[int, str]] = {}  # ssid -> strongest (dBm, flags)
      for ssid, dbm, flags in results:
        if ssid and (ssid not in best or dbm > best[ssid][0]):
          best[ssid] = (dbm, flags)

      self._refresh_status()
      self._networks = [Network(ssid, 100 if ssid == self._tethering_ssid else dbm_to_percent(dbm), security_type_from_flags(flags),
                                ssid == self._tethering_ssid)
                        for ssid, (dbm, flags) in best.items()]
      self._enqueue_callbacks(self._networks_updated, self.networks)  # sorted

  def _monitor(self):
    while not self._exit.is_set():
      events = self._events
      if events is None:
        self._check_daemons()
        self._exit.wait(0.5)
        continue
      try:
        data = events.recv(4096).decode("utf-8", "replace")
      except TimeoutError:
        self._check_daemons()
        continue
      except OSError:
        cloudlog.exception("wpa_supplicant event socket failed")
        with self._lock:
          if self._events is events:
            self._events = None
          events.close()
        continue
      event = re.sub(r"^<\d>", "", data).strip()
      try:
        self._handle_event(event)
      except Exception:
        cloudlog.exception(f"Failed to handle wpa_supplicant event: {event}")

  def _check_daemons(self):
    # runs once a second between events; a crashed supplicant or DHCP client is brought back without user action
    with self._lock:
      if self._exit.is_set():
        return
      try:
        alive = self._ctrl is not None and self._ctrl.request("PING") == "PONG"
      except OSError:
        alive = False
      if self._exit.is_set():
        return
      if not alive or self._events is None:
        cloudlog.warning("wpa_supplicant is not responding, restarting")
        try:
          self._start_supplicant()
        except Exception:
          cloudlog.exception("Failed to restart wpa_supplicant")
          self._refresh_status()
        return
      if self._wifi_state.status != ConnectStatus.DISCONNECTED and not self.is_tethering_active() and not _pid_alive(UDHCPC_PID_PATH):
        cloudlog.warning("udhcpc is not running, restarting")
        if self._exit.is_set():
          return
        try:
          self._start_dhcp()
        except Exception:
          cloudlog.exception("Failed to restart udhcpc")
          self._refresh_status()

  def _handle_event(self, event: str):
    if event.startswith("CTRL-EVENT-SCAN-RESULTS"):
      self._update_networks()
    elif event.startswith("CTRL-EVENT-CONNECTED"):
      self._on_associated()
    elif event.startswith("CTRL-EVENT-SSID-TEMP-DISABLED") and "reason=WRONG_KEY" in event.split():
      match = re.search(r"\bid=(\d+)", event)
      if match:
        self._on_wrong_key(int(match.group(1)))
    elif event.startswith("CTRL-EVENT-DISCONNECTED"):
      self._on_disconnected()

  def _on_disconnected(self):
    was_connected = self._wifi_state.status == ConnectStatus.CONNECTED
    self._refresh_status()
    if was_connected and self._wifi_state.status == ConnectStatus.DISCONNECTED:
      self._enqueue_callbacks(self._disconnected)

  def _on_associated(self):
    status = self._status()
    if status.get("mode") == "AP":
      return  # the hotspot is brought up by set_tethering_active
    ssid = decode_ssid(status.get("ssid", ""))
    with self._lock:
      if self._selected is not None and self._selected != ssid:
        return
    self._start_dhcp()
    if not self._ctrl.ok("ENABLE_NETWORK all"):  # SELECT_NETWORK disabled the other saved networks
      cloudlog.exception(f"wpa_supplicant rejected enabling networks after associating {ssid}")
      self._abandon(ssid)
      return

    deadline = time.monotonic() + DHCP_TIMEOUT_SECONDS
    while not self._exit.is_set() and time.monotonic() < deadline:
      with self._lock:
        if self._selected is not None and self._selected != ssid:
          return
      status = self._status()
      if status.get("wpa_state") != "COMPLETED" or decode_ssid(status.get("ssid", "")) != ssid:
        return  # association changed, the next event decides
      if status.get("ip_address"):
        with self._lock:
          if self._selected is not None and self._selected != ssid:
            return
          current_status = self._status()
          if (current_status.get("wpa_state") != "COMPLETED" or current_status.get("mode") != "station" or
              decode_ssid(current_status.get("ssid", "")) != ssid or current_status.get("id") != status.get("id") or
              (self._pending is not None and self._pending.ssid != ssid)):
            return
          pending, self._pending = self._pending, None
          if pending is not None and pending.ssid == ssid:
            write_profile(pending)
            self._profiles = read_profiles()
          self._refresh_status()
          self._enqueue_callbacks(self._activated)
        return
      self._exit.wait(0.5)

    with self._lock:
      if self._exit.is_set() or (self._selected is not None and self._selected != ssid):
        return
      current_status = self._status()
      if (current_status.get("wpa_state") != "COMPLETED" or current_status.get("mode") != "station" or
          decode_ssid(current_status.get("ssid", "")) != ssid or current_status.get("id") != status.get("id") or
          (self._pending is not None and self._pending.ssid != ssid)):
        return
      cloudlog.warning(f"No DHCP lease on {ssid}")
      if "id" in status:
        if not self._ctrl.ok(f"DISABLE_NETWORK {status['id']}"):
          cloudlog.exception(f"wpa_supplicant rejected disabling {ssid}")
          self._abandon(ssid)
          return
      self._selected, self._pending = None, None
      self._wifi_state, self._ipv4_address, self._current_network_metered = WifiState(), "", MeteredType.UNKNOWN
      self._enqueue_callbacks(self._disconnected)

  def _on_wrong_key(self, nid: int):
    ssid = ""
    try:
      with self._lock:
        ssid = self._network_ids.get(nid)
        if ssid is None or ssid != self._wifi_state.ssid:
          return
        # drop the network so the supplicant stops retrying and the UI is asked once; a saved profile is re-added on activation
        if not self._ctrl.ok(f"REMOVE_NETWORK {nid}"):
          raise ValueError(f"wpa_supplicant rejected removal of {ssid}")
        del self._network_ids[nid]
        self._selected, self._pending = None, None
        self._wifi_state, self._ipv4_address, self._current_network_metered = WifiState(), "", MeteredType.UNKNOWN
        if not self._ctrl.ok("ENABLE_NETWORK all"):
          raise ValueError("wpa_supplicant rejected enabling networks after wrong key")
        self._enqueue_callbacks(self._need_auth, ssid)
    except Exception:
      cloudlog.exception(f"Failed to remove wrong-key network {nid}")
      if ssid is not None:
        self._abandon(ssid)
      else:
        self._refresh_status()
      return

  def _abandon(self, ssid: str):
    with self._lock:
      if self._selected == ssid:
        self._selected, self._pending = None, None
    self._refresh_status()

  def _network_id(self, ssid: str) -> int | None:
    for nid, known in self._network_ids.items():
      if known == ssid:
        return nid
    profile = next((p for p in self._profiles if p.ssid == ssid and not p.is_ap), None)
    if profile is None:
      return None
    return self._add_network(profile.ssid, profile.psk, profile.hidden)

  def connect_to_network(self, ssid: str, password: str, hidden: bool = False):
    with self._lock:
      self._selected = ssid
      self._pending = Profile(path="", uuid=str(uuid.uuid4()), ssid=ssid, psk=password or None, hidden=hidden,
                              metered=MeteredType.UNKNOWN, is_ap=False)
      self._wifi_state = WifiState(ssid, ConnectStatus.CONNECTING)

    def worker():
      try:
        # Clear all connections that may already exist to the network we are connecting to
        self.forget_connection(ssid, block=True)
        with self._lock:
          if self._selected != ssid:
            return
          pending = self._pending
          if any(known == self._tethering_ssid for known in self._network_ids.values()):
            self._stop_tethering(False)
            self._selected, self._pending = ssid, pending
            self._wifi_state = WifiState(ssid, ConnectStatus.CONNECTING)
          nid = self._add_network(ssid, password or None, hidden)
          if not self._ctrl.ok(f"SELECT_NETWORK {nid}"):
            raise ValueError(f"wpa_supplicant rejected network {ssid}")
      except Exception:
        cloudlog.exception(f"Failed to connect to {ssid}")
        self._abandon(ssid)

    threading.Thread(target=worker, daemon=True).start()

  def activate_connection(self, ssid: str, block: bool = False):
    with self._lock:
      self._selected, self._pending = ssid, None
      self._wifi_state = WifiState(ssid, ConnectStatus.CONNECTING)

    def worker():
      try:
        with self._lock:
          if self._selected != ssid:
            return
          pending = self._pending
          if any(known == self._tethering_ssid for known in self._network_ids.values()):
            self._stop_tethering(False)
            self._selected, self._pending = ssid, pending
            self._wifi_state = WifiState(ssid, ConnectStatus.CONNECTING)
          nid = self._network_id(ssid)
          if nid is None:
            cloudlog.warning(f"Failed to activate connection for {ssid}: not saved")
            self._abandon(ssid)
            return
          if not self._ctrl.ok(f"SELECT_NETWORK {nid}"):
            raise ValueError(f"wpa_supplicant rejected network {ssid}")
      except Exception:
        cloudlog.exception(f"Failed to activate {ssid}")
        self._abandon(ssid)

    if block:
      worker()
    else:
      threading.Thread(target=worker, daemon=True).start()

  def forget_connection(self, ssid: str, block: bool = False):
    def worker():
      disconnected = False
      removed = False
      with self._lock:
        try:
          for nid, known in list(self._network_ids.items()):
            if known == ssid:
              if not self._ctrl.ok(f"REMOVE_NETWORK {nid}"):
                raise ValueError(f"wpa_supplicant rejected removal of {ssid}")
              del self._network_ids[nid]
              removed = True
          for profile in self._profiles:
            if profile.ssid == ssid:
              remove_profile(profile)
              removed = True
          self._profiles = read_profiles()
          if removed and self._wifi_state.ssid == ssid and self._wifi_state.status == ConnectStatus.CONNECTED:
            disconnected = self._wifi_state.status == ConnectStatus.CONNECTED
            self._selected, self._pending = None, None
            self._wifi_state, self._ipv4_address, self._current_network_metered = WifiState(), "", MeteredType.UNKNOWN
        except Exception:
          cloudlog.exception(f"Failed to forget {ssid}")
          self._refresh_status()
        if disconnected:
          self._enqueue_callbacks(self._disconnected)
        self._enqueue_callbacks(self._forgotten, ssid)

    if block:
      worker()
    else:
      threading.Thread(target=worker, daemon=True).start()

  def _hotspot_profile(self) -> Profile:
    with self._lock:
      profile = next((p for p in self._profiles if p.is_ap and p.ssid == self._tethering_ssid), None)
      if profile is None:
        profile = write_profile(Profile(path="", uuid=str(uuid.uuid4()), ssid=self._tethering_ssid, psk=DEFAULT_TETHERING_PASSWORD, hidden=False,
                                        metered=MeteredType.UNKNOWN, is_ap=True))
        self._profiles = read_profiles()
      return profile

  @property
  def tethering_password(self) -> str:
    return self._hotspot_profile().psk or ""

  def set_tethering_password(self, password: str):
    def worker():
      try:
        with self._lock:
          write_profile(replace(self._hotspot_profile(), psk=password))
          self._profiles = read_profiles()
          hotspot = self._hotspot_profile()
          if (self._selected not in (None, hotspot.ssid) or self._wifi_state.ssid != hotspot.ssid or
              not any(ssid == hotspot.ssid for ssid in self._network_ids.values())):
            return
          self._stop_tethering()
          self._selected, self._pending = hotspot.ssid, None
          self._wifi_state = WifiState(hotspot.ssid, ConnectStatus.CONNECTING)
        self._start_tethering()
      except Exception:
        cloudlog.exception("Failed to set tethering password")
        with self._lock:
          owns_hotspot = self._selected == self._tethering_ssid
        if owns_hotspot:
          self._abandon(self._tethering_ssid)

    threading.Thread(target=worker, daemon=True).start()

  def set_ipv4_forward(self, enabled: bool):
    self._ipv4_forward = enabled
    if self.is_tethering_active():
      _sudo("sysctl", f"net.ipv4.ip_forward={int(enabled)}", check=False)

  def set_tethering_active(self, active: bool):
    ap_ids: tuple[int, ...] = ()
    with self._lock:
      if active:
        self._selected, self._pending = self._tethering_ssid, None
        self._wifi_state = WifiState(self._tethering_ssid, ConnectStatus.CONNECTING)
      else:
        ap_ids = tuple(nid for nid, ssid in self._network_ids.items() if ssid == self._tethering_ssid)
        if self._selected == self._tethering_ssid:
          self._selected, self._pending = None, None

    def worker():
      try:
        if active:
          self._start_tethering()
        else:
          with self._lock:
            current_ap_ids = tuple(nid for nid, ssid in self._network_ids.items() if ssid == self._tethering_ssid)
            if (self._wifi_state.ssid == self._tethering_ssid and self._selected is None and
                ap_ids == current_ap_ids):
              self._stop_tethering()
      except Exception:
        cloudlog.exception(f"Failed to set tethering active={active}")
        with self._lock:
          owns_hotspot = active and self._selected == self._tethering_ssid
        if owns_hotspot:
          self._abandon(self._tethering_ssid)

    threading.Thread(target=worker, daemon=True).start()

  def _start_tethering(self):
    nid = None
    try:
      with self._lock:
        if self._exit.is_set() or self._selected != self._tethering_ssid:
          return
        hotspot = self._hotspot_profile()
        self._stop_dhcp()
        if self._exit.is_set() or self._selected != hotspot.ssid:
          return
        nid = int(self._ctrl.request("ADD_NETWORK"))
        self._network_ids[nid] = hotspot.ssid
        for setting in (f"ssid {hotspot.ssid.encode().hex()}", "mode 2", f"frequency {TETHERING_FREQUENCY}", "key_mgmt WPA-PSK", "proto RSN", "pairwise CCMP",
                        f"psk {wpa_psk(hotspot.ssid, hotspot.psk or '')}"):
          if not self._ctrl.ok(f"SET_NETWORK {nid} {setting}"):
            raise ValueError(f"wpa_supplicant rejected hotspot {setting.split()[0]}")
        if not self._ctrl.ok(f"SELECT_NETWORK {nid}"):
          raise ValueError("wpa_supplicant rejected hotspot selection")

      deadline = time.monotonic() + AP_TIMEOUT_SECONDS
      while not self._exit.is_set() and time.monotonic() < deadline:
        with self._lock:
          if self._selected != hotspot.ssid:
            if self._selected is None and self._wifi_state.ssid == hotspot.ssid:
              return
            if self._network_ids.get(nid) == hotspot.ssid:
              if not self._ctrl.ok(f"REMOVE_NETWORK {nid}"):
                raise ValueError(f"wpa_supplicant rejected hotspot removal {nid}")
              del self._network_ids[nid]
            return
        status = self._status()
        if (status.get("mode") == "AP" and status.get("wpa_state") == "COMPLETED" and decode_ssid(status.get("ssid", "")) == hotspot.ssid and
            status.get("id") == str(nid)):
          with self._lock:
            status = self._status()
            if self._selected != hotspot.ssid:
              if self._selected is not None or self._wifi_state.ssid != hotspot.ssid:
                if self._network_ids.get(nid) == hotspot.ssid:
                  if not self._ctrl.ok(f"REMOVE_NETWORK {nid}"):
                    raise ValueError(f"wpa_supplicant rejected hotspot removal {nid}")
                  del self._network_ids[nid]
              return
            if (status.get("mode") != "AP" or status.get("wpa_state") != "COMPLETED" or decode_ssid(status.get("ssid", "")) != hotspot.ssid or
                status.get("id") != str(nid)):
              continue
            self._ensure_tethering_services(True)
            self._selected = None
            self._wifi_state, self._ipv4_address, self._current_network_metered = (WifiState(hotspot.ssid, ConnectStatus.CONNECTED),
                                                                                    TETHERING_IP_ADDRESS, MeteredType.UNKNOWN)
            self._enqueue_callbacks(self._activated)
            return
        self._exit.wait(0.2)

      if self._exit.is_set():
        return
      timed_out = False
      with self._lock:
        if self._selected == hotspot.ssid and self._network_ids.get(nid) == hotspot.ssid:
          self._stop_tethering(False)
          self._refresh_status()
          timed_out = True
      if timed_out:
        raise TimeoutError("hotspot did not come up")
    except Exception:
      with self._lock:
        if self._selected == self._tethering_ssid and self._network_ids.get(nid) == self._tethering_ssid:
          self._stop_tethering(False)
          self._refresh_status()
      raise

  def _ensure_tethering_services(self, fresh: bool = False):
    if fresh:
      _sudo("ip", "addr", "flush", "dev", WLAN)
      _sudo("ip", "addr", "add", f"{TETHERING_IP_ADDRESS}/24", "dev", WLAN)
    else:
      _sudo("ip", "addr", "replace", f"{TETHERING_IP_ADDRESS}/24", "dev", WLAN)
    if not _pid_alive(DNSMASQ_PID_PATH):
      subprocess.run(["sudo", "dnsmasq", f"--interface={WLAN}", "--bind-interfaces", "--except-interface=lo", f"--dhcp-range={TETHERING_DHCP_RANGE}",
                      f"--pid-file={DNSMASQ_PID_PATH}"], check=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     start_new_session=True)
    if _sudo("iptables-legacy", "-t", "nat", "-C", *TETHERING_NAT_RULE, check=False).returncode != 0:
      _sudo("iptables-legacy", "-t", "nat", "-A", *TETHERING_NAT_RULE)
    _sudo("sysctl", f"net.ipv4.ip_forward={int(self._ipv4_forward)}")

  def _stop_tethering(self, notify: bool = True):
    try:
      with self._lock:
        pid = _read_pid(DNSMASQ_PID_PATH)
        if pid is not None and _pid_alive(DNSMASQ_PID_PATH):
          _sudo("kill", str(pid), check=False)
          deadline = time.monotonic() + CTRL_TIMEOUT_SECONDS
          while _pid_alive(DNSMASQ_PID_PATH) and time.monotonic() < deadline:
            time.sleep(0.1)
          if _pid_alive(DNSMASQ_PID_PATH):
            raise RuntimeError("dnsmasq did not stop")
        _sudo("iptables-legacy", "-t", "nat", "-D", *TETHERING_NAT_RULE, check=False)
        for network_id, known in list(self._network_ids.items()):
          if known == self._tethering_ssid:
            if not self._ctrl.ok(f"REMOVE_NETWORK {network_id}"):
              raise ValueError(f"wpa_supplicant rejected hotspot removal {network_id}")
            del self._network_ids[network_id]
        if not self._ctrl.ok("ENABLE_NETWORK all"):
          raise ValueError("wpa_supplicant rejected enabling networks after tethering")
        _sudo("ip", "addr", "flush", "dev", WLAN, check=False)
        self._selected, self._pending = None, None
        self._wifi_state, self._ipv4_address, self._current_network_metered = WifiState(), "", MeteredType.UNKNOWN
        self._start_dhcp()
        if notify:
          self._enqueue_callbacks(self._disconnected)
    except Exception:
      cloudlog.exception("Failed to stop tethering")
      with self._lock:
        self._selected, self._pending = None, None
      self._refresh_status()
      raise

  def set_current_network_metered(self, metered: MeteredType):
    with self._lock:
      ssid = None if self.is_tethering_active() else self.connected_ssid

    def worker():
      try:
        with self._lock:
          if ssid is None or self.is_tethering_active() or self.connected_ssid != ssid:
            cloudlog.warning("No active WiFi connection found")
            return
          profile = next((p for p in self._profiles if p.ssid == ssid and not p.is_ap), None)
          if profile is None:
            cloudlog.warning("No active WiFi connection found")
            return
          write_profile(replace(profile, metered=metered))
          self._profiles = read_profiles()
          self._current_network_metered = metered
      except Exception:
        cloudlog.exception("Failed to update metered setting")

    threading.Thread(target=worker, daemon=True).start()

  def __del__(self):
    self.stop()

  def stop(self):
    self._exit.set()
    for thread in (self._init_thread, self._scan_thread, self._monitor_thread):
      if thread is not threading.current_thread() and thread.is_alive():
        thread.join()
    with self._lock:
      for sock in (self._events, self._ctrl):
        if sock is not None:
          sock.close()
