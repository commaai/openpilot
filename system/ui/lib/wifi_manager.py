import sys
from typing import Any
import atexit
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum

from collections import deque
from jeepney import DBusAddress, new_method_call
from jeepney.wrappers import Properties
from jeepney.bus_messages import message_bus, MatchRule
from jeepney.io.blocking import open_dbus_connection as open_dbus_connection_blocking
from jeepney.low_level import MessageType
from jeepney.io.threading import DBusRouter, open_dbus_connection as open_dbus_connection_threading

from openpilot.common.swaglog import cloudlog
from openpilot.system.ui.lib.networkmanager import (NM, NM_WIRELESS_IFACE, NM_802_11_AP_SEC_PAIR_WEP40,
                                                    NM_802_11_AP_SEC_PAIR_WEP104, NM_802_11_AP_SEC_GROUP_WEP40,
                                                    NM_802_11_AP_SEC_GROUP_WEP104, NM_802_11_AP_SEC_KEY_MGMT_PSK,
                                                    NM_802_11_AP_SEC_KEY_MGMT_802_1X, NM_802_11_AP_FLAGS_NONE,
                                                    NM_802_11_AP_FLAGS_PRIVACY, NM_802_11_AP_FLAGS_WPS,
                                                    NM_PATH, NM_IFACE, NM_ACCESS_POINT_IFACE, NM_SETTINGS_PATH,
                                                    NM_SETTINGS_IFACE, NM_CONNECTION_IFACE, NM_DEVICE_IFACE,
                                                    NM_DEVICE_TYPE_WIFI, NM_DEVICE_STATE_REASON_SUPPLICANT_DISCONNECT,
                                                    NM_DEVICE_STATE_REASON_NEW_ACTIVATION,
                                                    NMDeviceState)

TETHERING_IP_ADDRESS = "192.168.43.1"
DEFAULT_TETHERING_PASSWORD = "swagswagcomma"


class SecurityType(IntEnum):
  OPEN = 0
  WPA = 1
  WPA2 = 2
  WPA3 = 3
  UNSUPPORTED = 4


def get_security_type(flags: int, wpa_flags: int, rsn_flags: int) -> SecurityType:
  wpa_props = wpa_flags | rsn_flags

  # obtained by looking at flags of networks in the office as reported by an Android phone
  supports_wpa = (NM_802_11_AP_SEC_PAIR_WEP40 | NM_802_11_AP_SEC_PAIR_WEP104 | NM_802_11_AP_SEC_GROUP_WEP40 |
                  NM_802_11_AP_SEC_GROUP_WEP104 | NM_802_11_AP_SEC_KEY_MGMT_PSK)

  if (flags == NM_802_11_AP_FLAGS_NONE) or ((flags & NM_802_11_AP_FLAGS_WPS) and not (wpa_props & supports_wpa)):
    return SecurityType.OPEN
  elif (flags & NM_802_11_AP_FLAGS_PRIVACY) and (wpa_props & supports_wpa) and not (wpa_props & NM_802_11_AP_SEC_KEY_MGMT_802_1X):
    return SecurityType.WPA
  else:
    cloudlog.warning(f"Unsupported network! flags: {flags}, wpa_flags: {wpa_flags}, rsn_flags: {rsn_flags}")
    return SecurityType.UNSUPPORTED


@dataclass(frozen=True)
class Network:
  ssid: str
  strength: int
  is_connected: bool
  security_type: SecurityType
  is_saved: bool

  @classmethod
  def from_dbus(cls, ssid: str, aps: list["AccessPoint"], active_ap_path: str, is_saved: bool) -> "Network":
    # we only want to show the strongest AP for each Network/SSID
    strongest_ap = max(aps, key=lambda ap: ap.strength)

    is_connected = any(ap.ap_path == active_ap_path for ap in aps)  # TODO: just any is_connected aps!
    security_type = get_security_type(strongest_ap.flags, strongest_ap.wpa_flags, strongest_ap.rsn_flags)

    return cls(
      ssid=ssid,
      strength=strongest_ap.strength,
      is_connected=is_connected and is_saved,
      security_type=security_type,
      is_saved=is_saved,
    )


@dataclass(frozen=True)
class AccessPoint:
  ssid: str
  bssid: str
  strength: int
  is_connected: bool
  flags: int
  wpa_flags: int
  rsn_flags: int
  ap_path: str

  @classmethod
  def from_dbus(cls, ap_props: dict[str, tuple[str, Any]], ap_path: str, active_ap_path: str) -> "AccessPoint":
    ssid = bytes(ap_props['Ssid'][1]).decode("utf-8", "replace")
    bssid = str(ap_props['HwAddress'][1])
    strength = int(ap_props['Strength'][1])
    flags = int(ap_props['Flags'][1])
    wpa_flags = int(ap_props['WpaFlags'][1])
    rsn_flags = int(ap_props['RsnFlags'][1])

    return cls(
      ssid=ssid,
      bssid=bssid,
      strength=strength,
      is_connected=ap_path == active_ap_path,
      flags=flags,
      wpa_flags=wpa_flags,
      rsn_flags=rsn_flags,
      ap_path=ap_path,
    )


class WifiManager:
  def __init__(self):
    self._networks = []  # a network can be comprised of multiple APs
    self._active = True  # used to not run when not in settings
    self._exit = False

    # DBus connections
    self._router_main = DBusRouter(open_dbus_connection_threading(bus="SYSTEM"))  # used by scanner / general method calls
    self._conn_monitor = open_dbus_connection_blocking(bus="SYSTEM")  # used by state monitor thread
    self._nm = DBusAddress(NM_PATH, bus_name=NM, interface=NM_IFACE)

    # store wifi device path
    self._wifi_device: str | None = None

    # State
    self._connecting_to_ssid: str = ""

    # Callbacks
    # TODO: some of these are called from threads, either:
    # 1. make sure this is fine
    # 2. add callback event list that user can call from main thread to get callbacks safely
    self._need_auth: Callable[[str], None] | None = None
    self._activated: Callable[[], None] | None = None
    self._forgotten: Callable[[str], None] | None = None
    self._networks_updated: Callable[[list[Network]], None] | None = None
    self._disconnected: Callable[[], None] | None = None

    self._lock = threading.Lock()

    self._tmp_init()
    # sys.exit()

    self._scan_thread = threading.Thread(target=self._network_scanner, daemon=True)
    self._scan_thread.start()

    self._state_thread = threading.Thread(target=self._monitor_state, daemon=True)
    self._state_thread.start()

    atexit.register(self.stop)

  def _tmp_init(self):
    return

    def worker1():
      while 1:
        try:
          print('hi')
          conn_path = self._connection_by_ssid('unifi')
          print(conn_path)
          assert conn_path == '/org/freedesktop/NetworkManager/Settings/143', conn_path
        except Exception as e:
          print(e)
          raise e

    def worker2():
      while 1:
        try:
          print('hi')
          conn_path = self._connection_by_ssid('FBI Security Van')
          print(conn_path)
          assert conn_path == '/org/freedesktop/NetworkManager/Settings/141', conn_path
        except Exception as e:
          print(e)
          raise e

    threading.Thread(target=worker1, daemon=True).start()
    threading.Thread(target=worker2, daemon=True).start()
    time.sleep(10)

    t = time.monotonic()
    self.connect_to_network_old('...', '...')
    print('first', time.monotonic() - t)
    t = time.monotonic()
    self.connect_to_network('...', '...')
    print('second', time.monotonic() - t)

    return
    self._wait_for_wifi_device()
    self._update_networks_old()
    import copy
    nets = copy.deepcopy(self._networks)
    self._update_networks()
    print('networks match:', nets == self._networks)
    return
    t = time.monotonic()
    self.forget_connection('unifi', block=True)
    print('initial activation took', time.monotonic() - t)

    t = time.monotonic()
    self.forget_connection_jeepney('unifi', block=True)
    print('initial activation2 took', time.monotonic() - t)

    return
    print('conn by ssid')
    t = time.monotonic()
    print('got conn', self._connection_by_ssid("unifi"))
    print('took', time.monotonic() - t)
    t = time.monotonic()
    print('got conn2', self._connection_by_ssid_jeepney("unifi"))
    print('took2', time.monotonic() - t)

    t = time.monotonic()
    a = self._get_connections()
    print(time.monotonic() - t)

    time.sleep(1)

    t = time.monotonic()
    b = self._get_connections_jeepney()
    print(time.monotonic() - t)

    print(a == b)

  def set_callbacks(self, need_auth: Callable[[str], None],
                    activated: Callable[[], None] | None,
                    forgotten: Callable[[str], None],
                    networks_updated: Callable[[list[Network]], None],
                    disconnected: Callable[[], None]):
    self._need_auth = need_auth
    self._activated = activated
    self._forgotten = forgotten
    self._networks_updated = networks_updated
    self._disconnected = disconnected

  def set_active(self, active: bool):
    print('SETTING ACTIVE', active)
    self._active = active
    # fast refresh when going active
    if active:
      threading.Thread(target=self._update_networks, daemon=True).start()

  def _monitor_state(self):
    device_path: str = self._wait_for_wifi_device()

    rule = MatchRule(
      type="signal",
      interface=NM_DEVICE_IFACE,
      member="StateChanged",
      path=device_path,
    )

    # Filter for StateChanged signal
    self._conn_monitor.send_and_get_reply(message_bus.AddMatch(rule))

    with self._conn_monitor.filter(rule, queue=deque(maxlen=10)) as q:  # TODO: not sure what to choose for this
      while not self._exit:
        # TODO: now that we have a nice poller we can run always?
        # TODO: actually nah since it affects UI currently? or not?
        if not self._active:
          time.sleep(1)
          continue

        # Block until a matching signal arrives
        try:
          msg = self._conn_monitor.recv_until_filtered(q, timeout=1)
        except TimeoutError:
          continue

        print('msg.body', msg.body)
        new_state, previous_state, change_reason = msg.body

        print(f"------------ WiFi device state change: {new_state}, change reason: {change_reason}")
        # BAD PASSWORD
        if new_state == NMDeviceState.NEED_AUTH and change_reason == NM_DEVICE_STATE_REASON_SUPPLICANT_DISCONNECT and len(self._connecting_to_ssid):
          print('------ NEED AUTH - SUPPLICANT DISCONNECT')
          # TODO: this didn't show wrong password dialog but we were here
          print('forgetting connection to', self._connecting_to_ssid)
          self.forget_connection(self._connecting_to_ssid, block=True)
          print('forgot successful')
          print('need auth is none?', self._need_auth is None)
          if self._need_auth is not None:
            print('CALLING NEED AUTH CALLBACK')
            self._need_auth(self._connecting_to_ssid)
            print('done calling')
          self._connecting_to_ssid = ""
        elif new_state == NMDeviceState.ACTIVATED:
          print('------ ACTIVATED')
          if self._activated is not None:
            self._update_networks()
            print('CALLING ACTIVATED CALLBACK1')
            self._activated()
          self._connecting_to_ssid = ""
        elif new_state == NMDeviceState.DISCONNECTED and change_reason != NM_DEVICE_STATE_REASON_NEW_ACTIVATION:
          print('------ DISCONNECTED')
          self._connecting_to_ssid = ""
          if self._disconnected is not None:
            self._disconnected()

  def _network_scanner(self):
    self._wait_for_wifi_device()

    while not self._exit:
      if self._active:
        print('we;re acti!!!!!!!!!!!!')
        # Scan for networks every 5 seconds
        # TODO: should watch when scan is complete (PropertiesChanged), but this is more than good enough for now
        self._update_networks()
        self._request_scan()

      time.sleep(5)

  def _wait_for_wifi_device(self) -> str:
    with self._lock:
      device_path: str | None = None
      while not self._exit:
        device_path = self._get_wifi_device()
        if device_path is not None:
          break
        time.sleep(1)
      return device_path

  def _get_wifi_device(self) -> str | None:
    if self._wifi_device is not None:
      return self._wifi_device

    t = time.monotonic()
    device_paths = self._router_main.send_and_get_reply(new_method_call(self._nm, 'GetDevices')).body[0]

    for device_path in device_paths:
      dev_addr = DBusAddress(device_path, bus_name=NM, interface=NM_DEVICE_IFACE)
      dev_type = self._router_main.send_and_get_reply(Properties(dev_addr).get('DeviceType')).body[0]

      if dev_type[1] == NM_DEVICE_TYPE_WIFI:
        self._wifi_device = device_path
        break

    print(f"Got wifi device in {time.monotonic() - t}s: {self._wifi_device}")
    return self._wifi_device

  def _get_connections(self) -> list[str]:
    settings_addr = DBusAddress(NM_SETTINGS_PATH, bus_name=NM, interface=NM_SETTINGS_IFACE)
    return self._router_main.send_and_get_reply(new_method_call(settings_addr, 'ListConnections')).body[0]

  def _connection_by_ssid(self, ssid: str, known_connections: list[str] | None = None) -> str | None:
    for conn_path in known_connections or self._get_connections():
      conn_addr = DBusAddress(conn_path, bus_name=NM, interface=NM_CONNECTION_IFACE)
      reply = self._router_main.send_and_get_reply(new_method_call(conn_addr, "GetSettings"))

      # ignore connections removed during iteration (need auth, etc.)
      if reply.header.message_type == MessageType.error:
        cloudlog.warning(f"Failed to get connection properties for {conn_path}")
        continue

      settings = reply.body[0]
      if "802-11-wireless" in settings and settings['802-11-wireless']['ssid'][1].decode("utf-8", "replace") == ssid:
        return conn_path
    return None

  def connect_to_network(self, ssid: str, password: str):
    def worker():
      t = time.monotonic()

      # Clear all connections that may already exist to the network we are connecting
      self._connecting_to_ssid = ssid
      self.forget_connection(ssid, block=True)

      is_hidden = False

      connection = {
        'connection': {
          'type': ('s', '802-11-wireless'),
          'uuid': ('s', str(uuid.uuid4())),
          'id': ('s', f'openpilot connection {ssid}'),
          'autoconnect-retries': ('i', 0),
        },
        '802-11-wireless': {
          'ssid': ('ay', ssid.encode("utf-8")),
          'hidden': ('b', is_hidden),
          'mode': ('s', 'infrastructure'),
        },
        'ipv4': {
          'method': ('s', 'auto'),
          'dns-priority': ('i', 600),
        },
        'ipv6': {'method': ('s', 'ignore')},
      }

      if password:
        connection['802-11-wireless-security'] = {
          'key-mgmt': ('s', 'wpa-psk'),
          'auth-alg': ('s', 'open'),
          'psk': ('s', password),
        }

      settings_addr = DBusAddress(NM_SETTINGS_PATH, bus_name=NM, interface=NM_SETTINGS_IFACE)

      conn_path = self._router_main.send_and_get_reply(new_method_call(settings_addr, 'AddConnection', 'a{sa{sv}}', (connection,)))

      print('Added connection', conn_path)

      print(f'Connecting to network took {time.monotonic() - t}s')

      self.activate_connection(ssid, block=True)

    threading.Thread(target=worker, daemon=True).start()

  def forget_connection(self, ssid: str, block: bool = False):
    def worker():
      t = time.monotonic()
      conn_path = self._connection_by_ssid(ssid)
      print(f'Finding connection by SSID took {time.monotonic() - t}s: {conn_path}')
      if conn_path is not None:
        conn_addr = DBusAddress(conn_path, bus_name=NM, interface=NM_CONNECTION_IFACE)
        self._router_main.send_and_get_reply(new_method_call(conn_addr, 'Delete'))

        print(f'Forgetting connection took {time.monotonic() - t}s')
        if self._forgotten is not None:
          self._update_networks()
          self._forgotten(ssid)

    # TODO: make a helper when it makes sense
    if block:
      worker()
    else:
      threading.Thread(target=worker, daemon=True).start()

  def activate_connection(self, ssid: str, block: bool = False):
    def worker():
      t = time.monotonic()
      conn_path = self._connection_by_ssid(ssid)
      if conn_path is not None:
        self._connecting_to_ssid = ssid
        if self._wifi_device is None:
          cloudlog.warning("No WiFi device found")
          return

        print(f'Activating connection to {ssid}')
        self._router_main.send_and_get_reply(new_method_call(self._nm, 'ActivateConnection', 'ooo',
                                                             (conn_path, self._wifi_device, "/")))
        print(f"Activated connection in {time.monotonic() - t}s")

    # TODO: make a helper when it makes sense
    if block:
      worker()
    else:
      threading.Thread(target=worker, daemon=True).start()

  def _request_scan(self):
    if self._wifi_device is None:
      cloudlog.warning("No WiFi device found")
      return

    wifi_addr = DBusAddress(self._wifi_device, bus_name=NM, interface=NM_WIRELESS_IFACE)
    reply = self._router_main.send_and_get_reply(new_method_call(wifi_addr, 'RequestScan', 'a{sv}', ({},)))
    print('Requested scan')

    if reply.header.message_type == MessageType.error:
      cloudlog.warning(f"Failed to request scan: {reply}")
      return

  def _update_networks(self):
    print('UPDATING NETWORKS!!!!')

    if self._wifi_device is None:
      cloudlog.warning("No WiFi device found")
      return

    # returns '/' if no active AP
    wifi_addr = DBusAddress(self._wifi_device, NM, interface=NM_WIRELESS_IFACE)
    active_ap_path = self._router_main.send_and_get_reply(Properties(wifi_addr).get('ActiveAccessPoint')).body[0][1]
    ap_paths = self._router_main.send_and_get_reply(new_method_call(wifi_addr, 'GetAllAccessPoints')).body[0]

    aps: dict[str, list[AccessPoint]] = {}

    for ap_path in ap_paths:
      ap_addr = DBusAddress(ap_path, NM, interface=NM_ACCESS_POINT_IFACE)
      ap_props = self._router_main.send_and_get_reply(Properties(ap_addr).get_all())

      # some APs have been seen dropping off during iteration
      if ap_props.header.message_type == MessageType.error:
        cloudlog.warning(f"Failed to get AP properties for {ap_path}")
        continue

      try:
        ap = AccessPoint.from_dbus(ap_props.body[0], ap_path, active_ap_path)
        if ap.ssid == "":
          continue

        if ap.ssid not in aps:
          aps[ap.ssid] = []

        aps[ap.ssid].append(ap)
      except Exception:
        # catch all for parsing errors
        cloudlog.exception(f"Failed to parse AP properties for {ap_path}")

    known_connections = self._get_connections()
    networks = [Network.from_dbus(ssid, ap_list, active_ap_path, self._connection_by_ssid(ssid, known_connections) is not None)
                for ssid, ap_list in aps.items()]
    networks.sort(key=lambda n: (-n.is_connected, -n.strength, n.ssid.lower()))
    self._networks = networks

    if self._networks_updated is not None:
      self._networks_updated(self._networks)

  def __del__(self):
    self.stop()

  def stop(self):
    self._exit = True
    self._scan_thread.join()
    self._state_thread.join()

    self._router_main.close()
    self._router_main.conn.close()
    self._conn_monitor.close()
