"""
Wi-Fi for the UIs (settings, setup, updater).

AGNOS runs a classic Wi-Fi stack: wpa_supplicant owns the radio and connects to saved networks
on its own (boot, roaming, reconnects), dhcpcd does DHCP, routes and DNS, and tethering.service
runs the hotspot. Saved networks are wpa_supplicant's own (/data/etc/wpa_supplicant/wlan0.conf, which it
writes on SAVE_CONFIG), so this is only a client of its control socket, and networking doesn't depend on
this process running. Per network: priority, scan_ssid (hidden), and metered in id_str.

The hotspot is AGNOS's (tethering.service): this only sets its params (TetheringSsid, TetheringPassword,
TetheringShareInternet) and turns it on and off with TetheringEnabled. While it's on, AGNOS stops the client
wpa_supplicant and runs the hotspot in a wpa_supplicant of its own, on the same control socket. That one is only read
(STATUS, never a scan, which takes the hotspot off its channel), and the saved networks shown are the ones last read
from the client's.

One thread does all of it: it runs the UI's requests in order, follows wpa_supplicant's events and polls its
state. The newest connection request wins: it replaces the attempt in progress, and older requests that
haven't run yet are dropped.

One WifiManager runs at a time: AGNOS's comma.sh stops setup before it starts openpilot, launch_chffrplus.sh runs
the updater before openpilot, and the UI has one.
"""
import atexit
import functools
import queue
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from enum import IntEnum
from typing import TYPE_CHECKING

from openpilot.common.swaglog import cloudlog
from openpilot.system.ui.lib.wpa_ctrl import WpaCtrl, WpaCtrlError, decode_ssid, ssid_str

if TYPE_CHECKING:
  from openpilot.common.params import Params
else:
  try:
    from openpilot.common.params import Params
  except (ImportError, OSError):
    Params = None

DEFAULT_TETHERING_PASSWORD = "swagswagcomma"
SCAN_PERIOD_SECONDS = 5
CONNECT_TIMEOUT_SECONDS = 30  # for wpa_supplicant to join the network (DHCP isn't counted), or the hotspot to come up
TETHERING_STOP_TIMEOUT_SECONDS = 10
NO_IP_TIMEOUT_SECONDS = 60  # joined, but DHCP never answers: try another access point (NetworkManager gave up after 45 s)
POLL_SECONDS = 1.0

# wpa_supplicant states while it's joining a network
JOINING_STATES = ("AUTHENTICATING", "ASSOCIATING", "ASSOCIATED", "4WAY_HANDSHAKE", "GROUP_HANDSHAKE")


# id_str of a network being tried, until it works. Any other WifiManager's is left over (one runs at a time)
NEW_NETWORK_TAG = "openpilot-new"

# by its fields: a later wpa_supplicant may add some
WRONG_PASSWORD_EVENT = re.compile(r'CTRL-EVENT-SSID-TEMP-DISABLED id=(\d+) ssid="((?:[^\\"]|\\.)*)" (?:.* )?reason=(?:WRONG_KEY|AUTH_FAILED)(?: |$)')


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


# a saved network's metered setting is its id_str (HARDWARE.get_network_metered reads it from STATUS)
METERED_ID_STR = {MeteredType.YES: "metered", MeteredType.NO: "unmetered"}


def get_metered(id_str: str | None) -> MeteredType:
  return next((m for m, s in METERED_ID_STR.items() if s == id_str), MeteredType.UNKNOWN)


def get_security_type(flags: str) -> SecurityType:
  """From wpa_supplicant's BSS flags, e.g. '[WPA2-PSK-CCMP][WPS][ESS]'. Same classes NetworkManager's flags gave."""
  tokens = [t for t in flags.replace("]", "[").split("[") if t]
  # an open network in OWE transition mode ("Enhanced Open") is joined as open
  security = [t for t in tokens if t.startswith(("WPA", "RSN", "WEP", "OSEN", "OWE")) and t != "OWE-TRANS-OPEN"]
  if not security:
    return SecurityType.OPEN
  if any("PSK" in t for t in security) and not any("EAP" in t for t in security):
    return SecurityType.WPA
  cloudlog.warning(f"Unsupported network! flags: {flags}")
  return SecurityType.UNSUPPORTED


def wpa_psk(password: str) -> str | None:
  """wpa_supplicant's psk for a password: 64 hex digits are the key itself, anything else is a passphrase, between quotes
  as is (AGNOS's wpa_supplicant saves any passphrase so it reads back). None if it can't be one (8 to 63 bytes, no
  control characters)."""
  if re.fullmatch(r"[0-9a-fA-F]{64}", password):
    return password
  return f'"{password}"' if 8 <= len(password.encode()) <= 63 and not re.search(r"[\x00-\x1f\x7f]", password) else None


def dbm_to_strength(dbm: int) -> int:
  # NetworkManager's conversion: -100 dBm -> 0%, -40 dBm -> 100%
  return 100 - int(100 * abs(max(-100, min(-40, dbm)) + 40) / 60)


@dataclass(frozen=True)
class AccessPoint:
  ssid: bytes
  strength: int
  flags: str

  @classmethod
  def from_bss(cls, bss: dict[str, str]) -> "AccessPoint":
    return cls(
      ssid=decode_ssid(bss.get("ssid", "")),
      strength=dbm_to_strength(int(bss.get("level", "-100"))),
      flags=bss.get("flags", ""),
    )


@dataclass(frozen=True)
class Network:
  ssid: str
  strength: int
  security_type: SecurityType
  is_tethering: bool

  @classmethod
  def from_aps(cls, ssid: str, aps: list[AccessPoint]) -> "Network":
    # we only want to show the strongest AP for each Network/SSID
    strongest_ap = max(aps, key=lambda ap: ap.strength)
    return cls(ssid=ssid, strength=strongest_ap.strength, security_type=get_security_type(strongest_ap.flags), is_tethering=False)


class ConnectStatus(IntEnum):
  DISCONNECTED = 0
  CONNECTING = 1
  CONNECTED = 2


@dataclass(frozen=True)
class WifiState:
  ssid: str | None = None
  status: ConnectStatus = ConnectStatus.DISCONNECTED


@dataclass(frozen=True)
class SavedNetwork:
  net_id: int
  ssid: bytes
  metered: MeteredType


@dataclass(frozen=True)
class _Attempt:
  """A connection the user asked for: a network until wpa_supplicant has joined it, it fails or times out, or the
  hotspot until it's turned off, or isn't up in time."""
  ssid: str
  hotspot: bool = False
  new: bool = False  # a network added to try a password: saved once it's joined, removed if it isn't
  password: str = field(default="", repr=False)
  hidden: bool = False
  # the network on the current connection to wpa_supplicant, once it's been added or looked up. A restart renumbers
  # them, so it's done again then
  net_id: int | None = None
  raw_ssid: bytes = b""
  deadline: float = field(default_factory=lambda: time.monotonic() + CONNECT_TIMEOUT_SECONDS)


class WifiManager:
  def __init__(self):
    self._networks: list[Network] = []  # an unsorted list of available Networks. a Network can be comprised of multiple APs
    self._active = True  # scan and list networks, while a UI shows them
    self._exit = False

    self._ctrl = WpaCtrl()  # requests
    self._events = WpaCtrl()  # attached, receives events
    self._ap = False  # connected to the hotspot's wpa_supplicant, not the client's

    # State for the UI. A request shows its state at once, until the worker has run it (_state_lock)
    self._state_lock = threading.Lock()
    self._wifi_state: WifiState = WifiState()
    self._ipv4_address: str = ""
    self._current_network_metered: MeteredType = MeteredType.UNKNOWN
    self._tethering_active = False
    self._saved: dict[str, SavedNetwork] = {}  # ssid -> saved network, from wpa_supplicant

    # The worker's
    self._requests: queue.SimpleQueue[Callable[[], None]] = queue.SimpleQueue()
    self._latest_request = 0  # the newest connection request
    self._attempt: _Attempt | None = None
    self._saved_changed = False
    self._unsaved = False  # wpa_supplicant's last SAVE_CONFIG failed (e.g. /data is full): tried again until it works
    self._scanned_ssids: dict[str, bytes] = {}
    self._wpa_state = ""
    self._no_ip = ("", 0.0)  # joined access point without an IP, since
    self._ipv4_forward = False
    self._last_network_scan: float = 0.0

    self._params = Params() if Params is not None else None
    self._default_tethering_ssid = "weedle"
    if self._params is not None:
      dongle_id = self._params.get("DongleId")
      if dongle_id:
        self._default_tethering_ssid += "-" + dongle_id[:4]

    # Callbacks
    self._callback_queue: queue.SimpleQueue[Callable[[], None]] = queue.SimpleQueue()
    self._need_auth: list[Callable[[str], None]] = []
    self._activated: list[Callable[[], None]] = []
    self._forgotten: list[Callable[[str | None], None]] = []
    self._networks_updated: list[Callable[[list[Network]], None]] = []
    self._disconnected: list[Callable[[], None]] = []

    self._thread = threading.Thread(target=self._run, daemon=True)
    self._thread.start()
    atexit.register(self.stop)

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

  @property
  def tethering_password(self) -> str:
    return (self._params.get("TetheringPassword") if self._params is not None else None) or DEFAULT_TETHERING_PASSWORD

  def _tethering_ssid(self) -> str:
    # AGNOS imports an older hotspot's name into the param
    return (self._params.get("TetheringSsid") if self._params is not None else None) or self._default_tethering_ssid

  def _enqueue_callbacks(self, cbs: list[Callable], *args):
    for cb in cbs:
      self._callback_queue.put(functools.partial(cb, *args))

  def process_callbacks(self):
    # Call from UI thread to run any pending callbacks
    while True:
      try:
        cb = self._callback_queue.get_nowait()
      except queue.Empty:
        return
      cb()

  def set_active(self, active: bool):
    self._active = active

    # Update networks immediately when activating for UI
    if active:
      self._submit(self._update_networks)

  def is_connection_saved(self, ssid: str) -> bool:
    return ssid in self._saved

  # *** requests, run by the worker ***

  def _submit(self, request: Callable[[], None]) -> threading.Event:
    done = threading.Event()

    def run():
      try:
        request()
      except Exception:
        cloudlog.exception("Wi-Fi request failed")
      finally:
        done.set()

    self._requests.put(run)
    return done

  def _request_connection(self, attempt: _Attempt) -> threading.Event:
    """Shown as connecting at once. It replaces the attempt in progress, and the hotspot unless it's the hotspot."""
    def run(request: int):
      if request != self._latest_request:
        return  # a newer one replaced it
      self._end_attempt()
      if not attempt.hotspot:
        self._stop_tethering()
        if request != self._latest_request:
          return
      self._attempt = replace(attempt, deadline=time.monotonic() + CONNECT_TIMEOUT_SECONDS)
      if attempt.hotspot:
        self._start_tethering()
      else:
        self._start_attempt()

    with self._state_lock:
      self._latest_request += 1
      done = self._submit(functools.partial(run, self._latest_request))
      self._wifi_state, self._ipv4_address, self._tethering_active = WifiState(attempt.ssid, ConnectStatus.CONNECTING), "", attempt.hotspot
    return done

  def connect_to_network(self, ssid: str, password: str, hidden: bool = False):
    self._request_connection(_Attempt(ssid, new=True, password=password, hidden=hidden))

  def activate_connection(self, ssid: str):
    self._request_connection(_Attempt(ssid))

  def forget_connection(self, ssid: str):
    self._submit(lambda: self._forget(ssid))

  def is_tethering_active(self) -> bool:
    return self._tethering_active

  def set_tethering_password(self, password: str):
    params = self._params
    if params is None:
      return
    if not (8 <= len(password) <= 63 and password.isascii() and password.isprintable()):
      cloudlog.warning('Invalid tethering password')
      return
    self._submit(lambda: params.put("TetheringPassword", password, block=True))
    if self.is_tethering_active():
      self.set_tethering_active(True)  # restarts it with the new password

  def set_ipv4_forward(self, enabled: bool):
    self._ipv4_forward = enabled

  def set_tethering_active(self, active: bool):
    if self._params is None:
      return
    if active:
      self._request_connection(_Attempt(self._tethering_ssid(), hotspot=True))
    else:
      with self._state_lock:
        self._submit(self._stop_tethering)
        if self._tethering_active:
          self._wifi_state, self._ipv4_address, self._tethering_active = WifiState(), "", False

  def set_current_network_metered(self, metered: MeteredType):
    self._submit(lambda: self._set_metered(metered))

  # *** wpa_supplicant, from the worker ***

  def _list(self) -> list[tuple[int, bytes, str]]:
    """wpa_supplicant's networks: [(id, ssid, id_str)]"""
    return [(net_id, ssid, self._ctrl.get_network(net_id, "id_str") or "") for net_id, ssid, _ in self._ctrl.list_networks()]

  def _load_saved(self):
    if self._ap:
      return  # the client's are kept from the last time
    saved: dict[str, SavedNetwork] = {}
    for net_id, ssid, id_str in self._list():
      if id_str != NEW_NETWORK_TAG:  # not one being tried
        saved[ssid_str(ssid)] = SavedNetwork(net_id, ssid, get_metered(id_str))
    self._saved = saved

  def _raw_ssid(self, ssid: str) -> bytes:
    # the UI's SSIDs are text, which an SSID that isn't UTF-8 doesn't survive: use the bytes it came from
    saved = self._saved.get(ssid)
    return saved.ssid if saved is not None else self._scanned_ssids.get(ssid, ssid.encode())

  def _save(self):
    """Write wpa_supplicant's networks to its config, all enabled. Not while a network is being tried (it's the only one
    enabled, and the attempt saves when it ends)."""
    if self._attempt is not None and self._attempt.net_id is not None:
      return
    self._ctrl.request("ENABLE_NETWORK all")
    saved = self._ctrl.ok("SAVE_CONFIG")
    if not saved and not self._unsaved:
      cloudlog.error("wpa_supplicant failed to save its config, trying again")
    self._unsaved = not saved

  def _prefer(self, ssid: bytes):
    """Make ssid the preferred network, the only one at priority 1: when wpa_supplicant picks a network to join (at boot,
    or once it's lost one), it picks the highest priority one in range."""
    for net_id, net_ssid, _ in self._list():
      priority = "1" if net_ssid == ssid else "0"
      if self._ctrl.get_network(net_id, "priority") != priority:
        self._ctrl.request(f"SET_NETWORK {net_id} priority {priority}")

  def _select(self, net_id: int):
    # SELECT_NETWORK disables every other network, so wpa_supplicant tries just this one (and probes for it if it's
    # hidden), until the attempt ends and enables them again
    if not self._ctrl.ok(f"SELECT_NETWORK {net_id}"):
      raise WpaCtrlError(f"SELECT_NETWORK {net_id} failed")

  def _start_attempt(self):
    """Try the attempt's network: add it to try a password, or look up the saved one. If wpa_supplicant goes away
    meanwhile, or it's still the hotspot's, the attempt stays, and starts once the client's is back."""
    attempt = self._attempt
    assert attempt is not None and not attempt.hotspot
    if self._ap:
      return
    try:
      if attempt.new:
        psk = wpa_psk(attempt.password) if attempt.password else ""
        if psk is None:
          # it can't be right: ask again, like for a wrong password
          self._attempt = None
          self._enqueue_callbacks(self._need_auth, attempt.ssid)
          return
        net_id = self._add_network(attempt, psk)
      else:
        self._load_saved()
        net = self._saved.get(attempt.ssid)
        if net is None:
          cloudlog.warning(f"Failed to activate connection for {attempt.ssid}: not saved")
          self._attempt = None
          return
        net_id = net.net_id
        self._attempt = replace(attempt, net_id=net_id, raw_ssid=net.ssid)
      self._select(net_id)
    except (WpaCtrlError, ValueError):
      if self._ctrl.connected:
        cloudlog.exception(f"Failed to connect to {attempt.ssid}")
        self._end_attempt()

  def _add_network(self, attempt: _Attempt, psk: str) -> int:
    # a saved network with this SSID (e.g. with an old password) stays until this one works. The SSID's bytes are looked
    # up once, and kept: after a restart, the new wpa_supplicant may not have scanned it yet
    attempt = self._attempt = replace(attempt, raw_ssid=attempt.raw_ssid or self._raw_ssid(attempt.ssid))
    settings = {"id_str": f'"{NEW_NETWORK_TAG}"', "ssid": attempt.raw_ssid.hex()}
    if psk:
      # PMF when the AP has it: pmf=1 in AGNOS's globals
      settings.update({"key_mgmt": "WPA-PSK WPA-PSK-SHA256", "psk": psk})
    else:
      settings["key_mgmt"] = "NONE"
    # hidden, or it replaces a hidden saved network (e.g. its password changed)
    old = next((n for n in self._saved.values() if n.ssid == attempt.raw_ssid), None)
    if attempt.hidden or (old is not None and self._ctrl.get_network(old.net_id, "scan_ssid") == "1"):
      settings["scan_ssid"] = "1"
    net_id = int(self._ctrl.request("ADD_NETWORK"))
    self._attempt = replace(attempt, net_id=net_id)  # removed if this fails
    for key, value in settings.items():
      if not self._ctrl.ok(f"SET_NETWORK {net_id} {key} {value}"):
        raise WpaCtrlError(f"SET_NETWORK {key} failed")
    return net_id

  def _promote(self, attempt: _Attempt):
    """Its password works: it replaces the SSID's saved network, and keeps that one's metered setting (and hidden, which
    it was added with)."""
    old = next((n for n in self._saved.values() if n.ssid == attempt.raw_ssid), None)
    id_str = METERED_ID_STR.get(old.metered, "") if old is not None else ""
    if not self._ctrl.ok(f'SET_NETWORK {attempt.net_id} id_str "{id_str}"'):
      raise WpaCtrlError(f"SET_NETWORK {attempt.net_id} id_str failed")
    for net_id, ssid, _ in self._list():
      if ssid == attempt.raw_ssid and net_id != attempt.net_id:
        self._ctrl.request(f"REMOVE_NETWORK {net_id}")

  def _end_attempt(self):
    """Stop following the attempt: remove its network if it was added for it, and enable the others again."""
    attempt, self._attempt = self._attempt, None
    if attempt is None or attempt.net_id is None:
      return
    try:
      if attempt.new:
        self._ctrl.request(f"REMOVE_NETWORK {attempt.net_id}")
      self._save()  # with whatever changed meanwhile
    except WpaCtrlError:
      pass  # wpa_supplicant went away: attaching again does it

  def _forget(self, ssid: str):
    try:
      if self._ap:
        cloudlog.warning(f"Can't forget {ssid} while the hotspot is on")
        return
      raw_ssid = self._raw_ssid(ssid)
      if self._attempt is not None and not self._attempt.hotspot and self._attempt.ssid == ssid:
        self._end_attempt()
      for net_id, net_ssid, _ in self._list():  # every network with this SSID
        if net_ssid == raw_ssid:
          self._ctrl.request(f"REMOVE_NETWORK {net_id}")
      self._save()
      self._load_saved()
      self._update_state(self._ctrl.status())  # gone, before the UI hears it's forgotten
    finally:
      self._enqueue_callbacks(self._forgotten, ssid)

  def _set_metered(self, metered: MeteredType):
    # the saved network wpa_supplicant is on, which may be one of several with its SSID
    status = self._ctrl.status()
    if self._ap or status.get("wpa_state") != "COMPLETED" or status.get("id_str") == NEW_NETWORK_TAG:
      cloudlog.warning('No saved Wi-Fi network connected')
      return
    if self._ctrl.ok(f'SET_NETWORK {status.get("id")} id_str "{METERED_ID_STR.get(metered, "")}"'):
      self._save()
      self._load_saved()

  def _start_tethering(self):
    assert self._params is not None
    ssid = self._tethering_ssid()
    # tethering.service reads them all when TetheringEnabled is written; they're stored on first use
    self._params.put("TetheringSsid", ssid, block=True)
    self._params.put("TetheringPassword", self.tethering_password, block=True)
    self._params.put_bool("TetheringShareInternet", self._ipv4_forward, block=True)
    self._params.put_bool("TetheringEnabled", True, block=True)

  def _hotspot_off(self):
    if self._attempt is not None and self._attempt.hotspot:
      self._attempt = None
    if self._params is not None and self._params.get_bool("TetheringEnabled"):
      self._params.put_bool("TetheringEnabled", False, block=True)

  def _stop_tethering(self):
    """Turn the hotspot off, also one that's still starting, and wait until the client's wpa_supplicant is back, unless a
    newer connection request comes."""
    self._hotspot_off()
    request = self._latest_request
    deadline = time.monotonic() + TETHERING_STOP_TIMEOUT_SECONDS
    while time.monotonic() < deadline and request == self._latest_request and not self._exit:
      try:
        if not self._attached():
          self._attach()
        if not self._ap:
          return
        self._ctrl.status()  # fails once the hotspot's wpa_supplicant has exited
      except WpaCtrlError:
        self._detach()
      time.sleep(0.1)

  # *** worker ***

  def _run(self):
    while not self._exit:
      try:
        self._step()
      except Exception:  # nothing ends the worker
        cloudlog.exception("wifi worker error")
        self._sleep(POLL_SECONDS)

  def _step(self):
    if not self._attached():
      try:
        self._attach()  # before any request, which may only use network ids from this connection
      except WpaCtrlError:
        # wpa_supplicant isn't up yet, or is restarting
        self._detach()
        self._update_state({})
        self._sleep(POLL_SECONDS)
      return
    try:
      if not self._requests.empty():
        self._requests.get()()
      else:
        self._wait()
        if self._saved_changed:
          self._saved_changed = False
          self._load_saved()
        self._update_state(self._ctrl.status())
        if self._unsaved and not self._ap and self._attempt is None:
          self._unsaved = not self._ctrl.ok("SAVE_CONFIG")  # the save that failed already enabled them all
        if self._active and not self._ap and time.monotonic() - self._last_network_scan > SCAN_PERIOD_SECONDS:
          self._request_scan()
    except WpaCtrlError:
      self._detach()  # and attach again, at once

  def _sleep(self, seconds: float):
    end = time.monotonic() + seconds
    while not self._exit and time.monotonic() < end:
      time.sleep(0.1)

  def _wait(self):
    """Handle events until it's time to poll, or there's a request."""
    deadline = time.monotonic() + POLL_SECONDS
    while self._requests.empty() and not self._exit and (timeout := deadline - time.monotonic()) > 0:
      # in short steps, to pick up a request soon
      if (event := self._events.recv_event(min(timeout, 0.1))) is not None:
        self._handle_event(event)

  def _attached(self) -> bool:
    return self._ctrl.connected and self._events.connected

  def _detach(self):
    self._ctrl.close()
    self._events.close()
    self._ap = False

  def _attach(self):
    """Connect to wpa_supplicant. It may have restarted since the last connection, which renumbers its networks, so
    nothing from before is used: networks being tried are removed, the saved ones are read again, and an attempt in
    progress starts again. The hotspot's wpa_supplicant, with its one AP network, is left as it is."""
    if self._attempt is not None and not self._attempt.hotspot:
      self._attempt = replace(self._attempt, net_id=None)
    self._ctrl.connect()
    self._events.attach()

    self._ap = any(self._ctrl.get_network(net_id, "mode") == "2" for net_id, _, _ in self._ctrl.list_networks())
    if not self._ap:
      left = [net_id for net_id, _, id_str in self._list() if id_str == NEW_NETWORK_TAG]
      for net_id in left:
        self._ctrl.request(f"REMOVE_NETWORK {net_id}")
      # SELECT_NETWORK disabled the others while something was tried, and a save may have caught that
      if left:
        self._save()
      else:
        self._ctrl.request("ENABLE_NETWORK all")
      self._load_saved()
    self._update_networks()
    if self._attempt is not None and not self._attempt.hotspot:
      self._start_attempt()

  def _handle_event(self, event: str):
    if event.startswith("CTRL-EVENT-SCAN-RESULTS"):
      self._update_networks()

    elif event.startswith(("CTRL-EVENT-NETWORK-ADDED", "CTRL-EVENT-NETWORK-REMOVED")):
      self._saved_changed = True  # also by other clients, e.g. AGNOS's first-boot import of NetworkManager's networks

    # a wrong password? wpa_supplicant reports any handshake that ends early as one, also one it ended itself: it's
    # stopping (and answers nothing now), or the network was removed or another one picked. So it's one only if that
    # network (by id and SSID: ids are reused) is still disabled for it, and a saved network's only when nothing else
    # is being tried
    elif m := WRONG_PASSWORD_EVENT.match(event):
      net = (int(m.group(1)), decode_ssid(m.group(2)))
      if not any((i, s) == net and "[TEMP-DISABLED]" in flags for i, s, flags in self._ctrl.list_networks()):
        return
      attempt = self._attempt
      ssid: str | None = None
      if attempt is None:
        ssid = next((s for s, n in self._saved.items() if (n.net_id, n.ssid) == net), None)  # its password changed
      elif (attempt.net_id, attempt.raw_ssid) == net:
        ssid = attempt.ssid
        self._end_attempt()  # a new network is removed, a saved one stays (wpa_supplicant tries it again later)
      if ssid is not None:
        self._enqueue_callbacks(self._need_auth, ssid)

  def _update_state(self, status: dict[str, str]):
    """Follow the attempt, and publish the state. From STATUS, or {} without wpa_supplicant."""
    self._wpa_state = wpa_state = status.get("wpa_state", "")
    raw_ssid = decode_ssid(status["ssid"]) if "ssid" in status else None
    ssid = ssid_str(raw_ssid) if raw_ssid is not None else None
    ip = status.get("ip_address", "")

    attempt = self._attempt
    if attempt is None and self._ap and self._params is not None and self._params.get_bool("TetheringEnabled"):
      attempt = self._attempt = _Attempt(ssid or self._tethering_ssid(), hotspot=True)  # e.g. on before this started
    # the hotspot is up (without the attempt, it's been turned off and tethering.service is still stopping it).
    # If it goes down, e.g. to restart with a new password, it has as long again to come back
    hotspot_up = self._ap and status.get("mode") == "AP" and attempt is not None and attempt.hotspot
    if hotspot_up:
      self._attempt = attempt = replace(attempt, deadline=time.monotonic() + CONNECT_TIMEOUT_SECONDS)
    elif attempt is not None and attempt.net_id is not None and wpa_state == "COMPLETED" and status.get("id") == str(attempt.net_id):
      # joined: the password works, so a new network is saved now, also if DHCP is slow. The last network picked
      # that worked is the one to go back to
      if attempt.new:
        self._promote(attempt)
      self._prefer(attempt.raw_ssid)
      self._attempt = attempt = None
      self._save()
      self._load_saved()
    if attempt is not None and time.monotonic() > attempt.deadline:
      cloudlog.warning(f"Timed out connecting to {attempt.ssid}")
      if attempt.hotspot:
        self._hotspot_off()  # rather than it coming up later, unasked
      else:
        self._end_attempt()
      attempt = None

    tethering = attempt is not None and attempt.hotspot
    if attempt is not None:
      state = WifiState(attempt.ssid, ConnectStatus.CONNECTED if hotspot_up and wpa_state == "COMPLETED" else ConnectStatus.CONNECTING)
    elif not self._ap and (wpa_state == "COMPLETED" or wpa_state in JOINING_STATES):
      state = WifiState(ssid, ConnectStatus.CONNECTED if wpa_state == "COMPLETED" and ip else ConnectStatus.CONNECTING)
    else:
      state = WifiState()

    # joined, but no IP: wpa_supplicant is happy there, so make it skip that access point for a while. It tries another
    # one, or comes back to it if there's none
    bssid = status.get("bssid", "") if not self._ap and wpa_state == "COMPLETED" and not ip and attempt is None else ""
    if bssid != self._no_ip[0]:
      self._no_ip = (bssid, time.monotonic())
    elif bssid and time.monotonic() - self._no_ip[1] > NO_IP_TIMEOUT_SECONDS:
      cloudlog.warning(f"No IP from {ssid} ({bssid}), trying another access point")
      self._no_ip = ("", 0.0)
      self._ctrl.request(f"BSSID_IGNORE {bssid}")
      self._ctrl.request("REASSOCIATE")

    self._publish(state, ip if state.status == ConnectStatus.CONNECTED else "", get_metered(status.get("id_str")), tethering)

  def _publish(self, state: WifiState, ip: str = "", metered: MeteredType = MeteredType.UNKNOWN, tethering: bool = False):
    with self._state_lock:
      if not self._requests.empty():
        return  # the state a request set stands until it has run
      prev, self._wifi_state = self._wifi_state, state
      self._ipv4_address, self._current_network_metered, self._tethering_active = ip, metered, tethering

    if state.status == ConnectStatus.CONNECTED and (prev.status != ConnectStatus.CONNECTED or prev.ssid != state.ssid):
      self._enqueue_callbacks(self._activated)
    elif state.status == ConnectStatus.DISCONNECTED and prev.status != ConnectStatus.DISCONNECTED:
      self._enqueue_callbacks(self._disconnected)

  def _request_scan(self):
    # while connected, just scan: a plain SCAN also picks a network from the results, which can roam away. Otherwise it
    # must pick: a TYPE=ONLY scan takes the place of wpa_supplicant's own next scan, and nothing joins from its results
    reply = self._ctrl.request("SCAN TYPE=ONLY" if self._wpa_state == "COMPLETED" else "SCAN").strip()
    if reply not in ("OK", "FAIL-BUSY"):
      cloudlog.warning(f"Failed to request scan: {reply}")
    self._last_network_scan = time.monotonic()

  def _update_networks(self):
    if not self._active:
      return

    aps: dict[bytes, list[AccessPoint]] = {}
    for bss in self._ctrl.bss_list():
      try:
        ap = AccessPoint.from_bss(bss)
      except ValueError:
        cloudlog.exception(f"Failed to parse BSS {bss}")
        continue
      if ap.ssid and not ap.ssid.startswith(b"\x00"):  # not hidden
        aps.setdefault(ap.ssid, []).append(ap)

    self._scanned_ssids = {ssid_str(ssid): ssid for ssid in aps}
    networks = [Network.from_aps(ssid_str(ssid), ap_list) for ssid, ap_list in aps.items()]
    if self._tethering_active:
      # this device's own hotspot isn't in its scan results
      hotspot = self._tethering_ssid()
      networks = [n for n in networks if n.ssid != hotspot] + [Network(hotspot, 100, SecurityType.WPA, is_tethering=True)]
    self._networks = networks
    self._enqueue_callbacks(self._networks_updated, self.networks)  # sorted

  def __del__(self):
    self.stop()

  def stop(self):
    if not self._exit:
      self._exit = True
      if self._thread.is_alive():
        self._thread.join()
      self._end_attempt()  # nothing follows it any more
      self._detach()
