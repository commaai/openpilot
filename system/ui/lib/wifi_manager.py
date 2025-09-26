import atexit
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

from jeepney import DBusAddress, new_method_call
from jeepney.bus_messages import MatchRule, message_bus
from jeepney.io.blocking import open_dbus_connection as open_dbus_connection_blocking
from jeepney.io.threading import DBusRouter, open_dbus_connection as open_dbus_connection_threading
from jeepney.low_level import MessageType
from jeepney.wrappers import Properties

from openpilot.common.swaglog import cloudlog
from openpilot.common.params import Params
from openpilot.system.ui.lib.networkmanager import (NM, NM_WIRELESS_IFACE, NM_802_11_AP_SEC_PAIR_WEP40,
                                                    NM_802_11_AP_SEC_PAIR_WEP104, NM_802_11_AP_SEC_GROUP_WEP40,
                                                    NM_802_11_AP_SEC_GROUP_WEP104, NM_802_11_AP_SEC_KEY_MGMT_PSK,
                                                    NM_802_11_AP_SEC_KEY_MGMT_802_1X, NM_802_11_AP_FLAGS_NONE,
                                                    NM_802_11_AP_FLAGS_PRIVACY, NM_802_11_AP_FLAGS_WPS,
                                                    NM_PATH, NM_IFACE, NM_ACCESS_POINT_IFACE, NM_SETTINGS_PATH,
                                                    NM_SETTINGS_IFACE, NM_CONNECTION_IFACE, NM_DEVICE_IFACE,
                                                    NM_DEVICE_TYPE_WIFI, NM_DEVICE_TYPE_MODEM, NM_DEVICE_STATE_REASON_SUPPLICANT_DISCONNECT,
                                                    NM_DEVICE_STATE_REASON_NEW_ACTIVATION, NM_ACTIVE_CONNECTION_IFACE,
                                                    NM_IP4_CONFIG_IFACE, NMDeviceState)

TETHERING_IP_ADDRESS = "192.168.43.1"
DEFAULT_TETHERING_PASSWORD = "swagswagcomma"
SIGNAL_QUEUE_SIZE = 10
SCAN_PERIOD_SECONDS = 10


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
  def from_dbus(cls, ssid: str, aps: list["AccessPoint"], is_saved: bool) -> "Network":
    # we only want to show the strongest AP for each Network/SSID
    strongest_ap = max(aps, key=lambda ap: ap.strength)
    is_connected = any(ap.is_connected for ap in aps)
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
    self._networks: list[Network] = []  # a network can be comprised of multiple APs
    self._active = True  # used to not run when not in settings
    self._exit = False

    # DBus connections
    try:
      self._router_main = DBusRouter(open_dbus_connection_threading(bus="SYSTEM"))  # used by scanner / general method calls
      self._conn_monitor = open_dbus_connection_blocking(bus="SYSTEM")  # used by state monitor thread
      self._nm = DBusAddress(NM_PATH, bus_name=NM, interface=NM_IFACE)
    except FileNotFoundError:
      cloudlog.exception("Failed to connect to system D-Bus")
      self._exit = True

    # Store wifi device path
    self._wifi_device: str | None = None

    # State
    self._connecting_to_ssid: str = ""
    self._ipv4_address: str = ""
    self._current_network_metered: MeteredType = MeteredType.UNKNOWN
    self._tethering_password: str = ""
    self._last_network_update: float = 0.0
    self._callback_queue: list[Callable] = []

    self._tethering_ssid = "weedle"
    dongle_id = Params().get("DongleId")
    if dongle_id:
      self._tethering_ssid += "-" + dongle_id[:4]

    # Callbacks
    self._need_auth: list[Callable[[str], None]] = []
    self._activated: list[Callable[[], None]] = []
    self._forgotten: list[Callable[[], None]] = []
    self._networks_updated: list[Callable[[list[Network]], None]] = []
    self._disconnected: list[Callable[[], None]] = []

    self._lock = threading.Lock()
    self._scan_thread = threading.Thread(target=self._network_scanner, daemon=True)
    self._state_thread = threading.Thread(target=self._monitor_state, daemon=True)
    self._initialize()
    atexit.register(self.stop)

  def _initialize(self):
    def worker():
      self._wait_for_wifi_device()

      self._scan_thread.start()
      self._state_thread.start()

      if self._tethering_ssid not in self._get_connections():
        self._add_tethering_connection()

      self._tethering_password = self._get_tethering_password()
      cloudlog.debug("WifiManager initialized")

    threading.Thread(target=worker, daemon=True).start()

  def set_callbacks(self, need_auth: Callable[[str], None] | None = None,
                    activated: Callable[[], None] | None = None,
                    forgotten: Callable[[], None] | None = None,
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
  def ipv4_address(self) -> str:
    return self._ipv4_address

  @property
  def current_network_metered(self) -> MeteredType:
    return self._current_network_metered

  @property
  def tethering_password(self) -> str:
    return self._tethering_password

  def _enqueue_callbacks(self, cbs: list[Callable], *args):
    for cb in cbs:
      self._callback_queue.append(lambda _cb=cb: _cb(*args))

  def process_callbacks(self):
    # Call from UI thread to run any pending callbacks
    to_run, self._callback_queue = self._callback_queue, []
    for cb in to_run:
      cb()

  def set_active(self, active: bool):
    self._active = active

    # Scan immediately if we haven't scanned in a while
    if active and time.monotonic() - self._last_network_update > SCAN_PERIOD_SECONDS / 2:
      self._last_network_update = 0.0

  def _monitor_state(self):
    rule = MatchRule(
      type="signal",
      interface=NM_DEVICE_IFACE,
      member="StateChanged",
      path=self._wifi_device,
    )

    # Filter for StateChanged signal
    self._conn_monitor.send_and_get_reply(message_bus.AddMatch(rule))

    with self._conn_monitor.filter(rule, bufsize=SIGNAL_QUEUE_SIZE) as q:
      while not self._exit:
        if not self._active:
          time.sleep(1)
          continue

        # Block until a matching signal arrives
        try:
          msg = self._conn_monitor.recv_until_filtered(q, timeout=1)
        except TimeoutError:
          continue

        new_state, previous_state, change_reason = msg.body

        # BAD PASSWORD
        if new_state == NMDeviceState.NEED_AUTH and change_reason == NM_DEVICE_STATE_REASON_SUPPLICANT_DISCONNECT and len(self._connecting_to_ssid):
          self.forget_connection(self._connecting_to_ssid, block=True)
          self._enqueue_callbacks(self._need_auth, self._connecting_to_ssid)
          self._connecting_to_ssid = ""

        elif new_state == NMDeviceState.ACTIVATED:
          if len(self._activated):
            self._update_networks()
          self._enqueue_callbacks(self._activated)
          self._connecting_to_ssid = ""

        elif new_state == NMDeviceState.DISCONNECTED and change_reason != NM_DEVICE_STATE_REASON_NEW_ACTIVATION:
          self._connecting_to_ssid = ""
          self._enqueue_callbacks(self._forgotten)

  def _network_scanner(self):
    while not self._exit:
      if self._active:
        if time.monotonic() - self._last_network_update > SCAN_PERIOD_SECONDS:
          # Scan for networks every 10 seconds
          # TODO: should update when scan is complete (PropertiesChanged), but this is more than good enough for now
          self._update_networks()
          self._request_scan()
          self._last_network_update = time.monotonic()
      time.sleep(1 / 2.)

  def _wait_for_wifi_device(self):
    while not self._exit:
      # Cache and wait for a WiFi adapter path
      device_path = self._get_adapter(NM_DEVICE_TYPE_WIFI)
      if device_path is not None:
        self._wifi_device = device_path
        break
      time.sleep(1)

  def _get_adapter(self, adapter_type: int) -> str | None:
    """Return the first NetworkManager device path matching adapter_type, or None.

    Mirrors the behavior of the C++ getAdapter helper.
    """
    try:
      device_paths = self._router_main.send_and_get_reply(new_method_call(self._nm, 'GetDevices')).body[0]
      for device_path in device_paths:
        dev_addr = DBusAddress(device_path, bus_name=NM, interface=NM_DEVICE_IFACE)
        dev_type = self._router_main.send_and_get_reply(Properties(dev_addr).get('DeviceType')).body[0][1]
        if dev_type == adapter_type:
          return device_path
    except Exception as e:
      cloudlog.exception(f"Error getting adapter type {adapter_type}: {e}")
    return None


  def _get_connections(self) -> dict[str, str]:
    settings_addr = DBusAddress(NM_SETTINGS_PATH, bus_name=NM, interface=NM_SETTINGS_IFACE)
    known_connections = self._router_main.send_and_get_reply(new_method_call(settings_addr, 'ListConnections')).body[0]

    conns: dict[str, str] = {}
    for conn_path in known_connections:
      settings = self._get_connection_settings(conn_path)

      if len(settings) == 0:
        cloudlog.warning(f'Failed to get connection settings for {conn_path}')
        continue

      if "802-11-wireless" in settings:
        ssid = settings['802-11-wireless']['ssid'][1].decode("utf-8", "replace")
        if ssid != "":
          conns[ssid] = conn_path
    return conns

  def _get_active_connections(self):
    return self._router_main.send_and_get_reply(Properties(self._nm).get('ActiveConnections')).body[0][1]

  def _get_connection_settings(self, conn_path: str) -> dict:
    conn_addr = DBusAddress(conn_path, bus_name=NM, interface=NM_CONNECTION_IFACE)
    reply = self._router_main.send_and_get_reply(new_method_call(conn_addr, 'GetSettings'))
    if reply.header.message_type == MessageType.error:
      cloudlog.warning(f'Failed to get connection settings: {reply}')
      return {}
    return dict(reply.body[0])

  def _add_tethering_connection(self):
    connection = {
      'connection': {
        'type': ('s', '802-11-wireless'),
        'uuid': ('s', str(uuid.uuid4())),
        'id': ('s', 'Hotspot'),
        'autoconnect-retries': ('i', 0),
        'interface-name': ('s', 'wlan0'),
        'autoconnect': ('b', False),
      },
      '802-11-wireless': {
        'band': ('s', 'bg'),
        'mode': ('s', 'ap'),
        'ssid': ('ay', self._tethering_ssid.encode("utf-8")),
      },
      '802-11-wireless-security': {
        'group': ('as', ['ccmp']),
        'key-mgmt': ('s', 'wpa-psk'),
        'pairwise': ('as', ['ccmp']),
        'proto': ('as', ['rsn']),
        'psk': ('s', DEFAULT_TETHERING_PASSWORD),
      },
      'ipv4': {
        'method': ('s', 'shared'),
        'address-data': ('aa{sv}', [[
          ('address', ('s', TETHERING_IP_ADDRESS)),
          ('prefix', ('u', 24)),
        ]]),
        'gateway': ('s', TETHERING_IP_ADDRESS),
        'never-default': ('b', True),
      },
      'ipv6': {'method': ('s', 'ignore')},
    }

    settings_addr = DBusAddress(NM_SETTINGS_PATH, bus_name=NM, interface=NM_SETTINGS_IFACE)
    self._router_main.send_and_get_reply(new_method_call(settings_addr, 'AddConnection', 'a{sa{sv}}', (connection,)))

  def connect_to_network(self, ssid: str, password: str, hidden: bool = False):
    def worker():
      # Clear all connections that may already exist to the network we are connecting to
      self._connecting_to_ssid = ssid
      self.forget_connection(ssid, block=True)

      connection = {
        'connection': {
          'type': ('s', '802-11-wireless'),
          'uuid': ('s', str(uuid.uuid4())),
          'id': ('s', f'openpilot connection {ssid}'),
          'autoconnect-retries': ('i', 0),
        },
        '802-11-wireless': {
          'ssid': ('ay', ssid.encode("utf-8")),
          'hidden': ('b', hidden),
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
      self._router_main.send_and_get_reply(new_method_call(settings_addr, 'AddConnection', 'a{sa{sv}}', (connection,)))
      self.activate_connection(ssid, block=True)

    threading.Thread(target=worker, daemon=True).start()

  def forget_connection(self, ssid: str, block: bool = False):
    def worker():
      conn_path = self._get_connections().get(ssid, None)
      if conn_path is not None:
        conn_addr = DBusAddress(conn_path, bus_name=NM, interface=NM_CONNECTION_IFACE)
        self._router_main.send_and_get_reply(new_method_call(conn_addr, 'Delete'))

        if len(self._forgotten):
          self._update_networks()
        self._enqueue_callbacks(self._forgotten)

    if block:
      worker()
    else:
      threading.Thread(target=worker, daemon=True).start()

  def activate_connection(self, ssid: str, block: bool = False):
    def worker():
      conn_path = self._get_connections().get(ssid, None)
      if conn_path is not None:
        if self._wifi_device is None:
          cloudlog.warning("No WiFi device found")
          return

        self._connecting_to_ssid = ssid
        self._router_main.send(new_method_call(self._nm, 'ActivateConnection', 'ooo',
                                               (conn_path, self._wifi_device, "/")))

    if block:
      worker()
    else:
      threading.Thread(target=worker, daemon=True).start()

  def _deactivate_connection(self, ssid: str):
    for conn_path in self._get_active_connections():
      conn_addr = DBusAddress(conn_path, bus_name=NM, interface=NM_ACTIVE_CONNECTION_IFACE)
      specific_obj_path = self._router_main.send_and_get_reply(Properties(conn_addr).get('SpecificObject')).body[0][1]

      if specific_obj_path != "/":
        ap_addr = DBusAddress(specific_obj_path, bus_name=NM, interface=NM_ACCESS_POINT_IFACE)
        ap_ssid = bytes(self._router_main.send_and_get_reply(Properties(ap_addr).get('Ssid')).body[0][1]).decode("utf-8", "replace")

        if ap_ssid == ssid:
          self._deactivate_connection_by_path(conn_path)
          return

  def is_tethering_active(self) -> bool:
    for network in self._networks:
      if network.is_connected:
        return bool(network.ssid == self._tethering_ssid)
    return False

  def set_tethering_password(self, password: str):
    def worker():
      conn_path = self._get_connections().get(self._tethering_ssid, None)
      if conn_path is None:
        cloudlog.warning('No tethering connection found')
        return

      settings = self._get_connection_settings(conn_path)
      if len(settings) == 0:
        cloudlog.warning(f'Failed to get tethering settings for {conn_path}')
        return

      settings['802-11-wireless-security']['psk'] = ('s', password)

      conn_addr = DBusAddress(conn_path, bus_name=NM, interface=NM_CONNECTION_IFACE)
      reply = self._router_main.send_and_get_reply(new_method_call(conn_addr, 'Update', 'a{sa{sv}}', (settings,)))
      if reply.header.message_type == MessageType.error:
        cloudlog.warning(f'Failed to update tethering settings: {reply}')
        return

      self._tethering_password = password
      if self.is_tethering_active():
        self.activate_connection(self._tethering_ssid, block=True)

    threading.Thread(target=worker, daemon=True).start()

  def _get_tethering_password(self) -> str:
    conn_path = self._get_connections().get(self._tethering_ssid, None)
    if conn_path is None:
      cloudlog.warning('No tethering connection found')
      return ''

    reply = self._router_main.send_and_get_reply(new_method_call(
      DBusAddress(conn_path, bus_name=NM, interface=NM_CONNECTION_IFACE),
      'GetSecrets', 's', ('802-11-wireless-security',)
    ))

    if reply.header.message_type == MessageType.error:
      cloudlog.warning(f'Failed to get tethering password: {reply}')
      return ''

    secrets = reply.body[0]
    if '802-11-wireless-security' not in secrets:
      return ''

    return str(secrets['802-11-wireless-security'].get('psk', ('s', ''))[1])

  def set_tethering_active(self, active: bool):
    def worker():
      if active:
        self.activate_connection(self._tethering_ssid, block=True)
      else:
        self._deactivate_connection(self._tethering_ssid)

    threading.Thread(target=worker, daemon=True).start()

  def _update_current_network_metered(self) -> None:
    if self._wifi_device is None:
      cloudlog.warning("No WiFi device found")
      return

    self._current_network_metered = MeteredType.UNKNOWN
    for active_conn in self._get_active_connections():
      conn_addr = DBusAddress(active_conn, bus_name=NM, interface=NM_ACTIVE_CONNECTION_IFACE)
      conn_type = self._router_main.send_and_get_reply(Properties(conn_addr).get('Type')).body[0][1]

      if conn_type == '802-11-wireless':
        conn_path = self._router_main.send_and_get_reply(Properties(conn_addr).get('Connection')).body[0][1]
        if conn_path == "/":
          continue

        settings = self._get_connection_settings(conn_path)

        if len(settings) == 0:
          cloudlog.warning(f'Failed to get connection settings for {conn_path}')
          continue

        metered_prop = settings['connection'].get('metered', ('i', 0))[1]
        if metered_prop == MeteredType.YES:
          self._current_network_metered = MeteredType.YES
        elif metered_prop == MeteredType.NO:
          self._current_network_metered = MeteredType.NO
        return

  def set_current_network_metered(self, metered: MeteredType):
    def worker():
      for active_conn in self._get_active_connections():
        conn_addr = DBusAddress(active_conn, bus_name=NM, interface=NM_ACTIVE_CONNECTION_IFACE)
        conn_type = self._router_main.send_and_get_reply(Properties(conn_addr).get('Type')).body[0][1]

        if conn_type == '802-11-wireless' and not self.is_tethering_active():
          conn_path = self._router_main.send_and_get_reply(Properties(conn_addr).get('Connection')).body[0][1]
          if conn_path == "/":
            continue

          settings = self._get_connection_settings(conn_path)

          if len(settings) == 0:
            cloudlog.warning(f'Failed to get connection settings for {conn_path}')
            return

          settings['connection']['metered'] = ('i', int(metered))

          conn_addr = DBusAddress(conn_path, bus_name=NM, interface=NM_CONNECTION_IFACE)
          reply = self._router_main.send_and_get_reply(new_method_call(conn_addr, 'Update', 'a{sa{sv}}', (settings,)))
          if reply.header.message_type == MessageType.error:
            cloudlog.warning(f'Failed to update tethering settings: {reply}')
            return

    threading.Thread(target=worker, daemon=True).start()

  def _request_scan(self):
    if self._wifi_device is None:
      cloudlog.warning("No WiFi device found")
      return

    wifi_addr = DBusAddress(self._wifi_device, bus_name=NM, interface=NM_WIRELESS_IFACE)
    reply = self._router_main.send_and_get_reply(new_method_call(wifi_addr, 'RequestScan', 'a{sv}', ({},)))

    if reply.header.message_type == MessageType.error:
      cloudlog.warning(f"Failed to request scan: {reply}")

  def _update_networks(self):
    with self._lock:
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
      networks = [Network.from_dbus(ssid, ap_list, ssid in known_connections) for ssid, ap_list in aps.items()]
      networks.sort(key=lambda n: (-n.is_connected, -n.strength, n.ssid.lower()))
      self._networks = networks

      self._update_ipv4_address()
      self._update_current_network_metered()

      self._enqueue_callbacks(self._networks_updated, self._networks)

  def _update_ipv4_address(self):
    if self._wifi_device is None:
      cloudlog.warning("No WiFi device found")
      return

    self._ipv4_address = ""

    for conn_path in self._get_active_connections():
      conn_addr = DBusAddress(conn_path, bus_name=NM, interface=NM_ACTIVE_CONNECTION_IFACE)
      conn_type = self._router_main.send_and_get_reply(Properties(conn_addr).get('Type')).body[0][1]
      if conn_type == '802-11-wireless':
        ip4config_path = self._router_main.send_and_get_reply(Properties(conn_addr).get('Ip4Config')).body[0][1]

        if ip4config_path != "/":
          ip4config_addr = DBusAddress(ip4config_path, bus_name=NM, interface=NM_IP4_CONFIG_IFACE)
          address_data = self._router_main.send_and_get_reply(Properties(ip4config_addr).get('AddressData')).body[0][1]

          for entry in address_data:
            if 'address' in entry:
              self._ipv4_address = entry['address'][1]
              return

  def __del__(self):
    self.stop()

  def update_gsm_settings(self, roaming: bool, apn: str, metered: bool):
    """Update GSM settings for cellular connection"""
    def worker():
      try:
        lte_connection_path = self._get_lte_connection_path()
        if not lte_connection_path:
          cloudlog.warning("No LTE connection found")
          return

        settings = self._get_connection_settings(lte_connection_path)

        print('settings', settings)

        if len(settings) == 0:
          cloudlog.warning(f"Failed to get connection settings for {lte_connection_path}")
          return

        # Ensure dicts exist
        if 'gsm' not in settings:
          settings['gsm'] = {}
        if 'connection' not in settings:
          settings['connection'] = {}

        changes = False
        auto_config = apn == ""

        if settings['gsm'].get('auto-config', ('b', False))[1] != auto_config:
          cloudlog.info(f'Changing gsm.auto-config to {auto_config}')
          settings['gsm']['auto-config'] = ('b', auto_config)
          changes = True

        if settings['gsm'].get('apn', ('s', ''))[1] != apn:
          cloudlog.info(f'Changing gsm.apn to {apn}')
          settings['gsm']['apn'] = ('s', apn)
          changes = True

        if settings['gsm'].get('home-only', ('b', False))[1] == roaming:
          cloudlog.info(f'Changing gsm.home-only to {not roaming}')
          settings['gsm']['home-only'] = ('b', not roaming)
          changes = True

        # Unknown means NetworkManager decides
        metered_int = int(MeteredType.UNKNOWN if metered else MeteredType.NO)
        if settings['connection'].get('metered', ('i', 0))[1] != metered_int:
          cloudlog.info(f'Changing connection.metered to {metered_int}')
          settings['connection']['metered'] = ('i', metered_int)
          changes = True

        print('new settings', settings)

        if changes:
          # Update the connection settings (temporary update)
          conn_addr = DBusAddress(lte_connection_path, bus_name=NM, interface=NM_CONNECTION_IFACE)
          reply = self._router_main.send_and_get_reply(new_method_call(conn_addr, 'UpdateUnsaved', 'a{sa{sv}}', (settings,)))

          if reply.header.message_type == MessageType.error:
            cloudlog.warning(f"Failed to update GSM settings: {reply}")
            return

          # Try to activate the connection directly first
          # If it's already active, NetworkManager should handle it gracefully
          self._activate_modem_connection(lte_connection_path)

      except Exception as e:
        cloudlog.exception(f"Error updating GSM settings: {e}")

    threading.Thread(target=worker, daemon=True).start()

  def _get_lte_connection_path(self) -> str | None:
    """Find the LTE connection path by looking for connection with id='lte'"""
    try:
      settings_addr = DBusAddress(NM_SETTINGS_PATH, bus_name=NM, interface=NM_SETTINGS_IFACE)
      known_connections = self._router_main.send_and_get_reply(
        new_method_call(settings_addr, 'ListConnections')
      ).body[0]

      for conn_path in known_connections:
        settings = self._get_connection_settings(conn_path)
        if settings and settings.get('connection', {}).get('id', ['s', ''])[1] == 'lte':
          return conn_path

      return None
    except Exception as e:
      cloudlog.exception(f"Error finding LTE connection: {e}")
      return None

  def _find_active_connection_by_path(self, connection_path: str) -> str | None:
    """Find the active connection path for a given connection path"""
    try:
      active_connections = self._get_active_connections()
      for active_conn in active_connections:
        conn_addr = DBusAddress(active_conn, bus_name=NM, interface=NM_ACTIVE_CONNECTION_IFACE)
        conn_path = self._router_main.send_and_get_reply(
          Properties(conn_addr).get('Connection')
        ).body[0][1]

        if conn_path == connection_path:
          return active_conn
      return None
    except Exception as e:
      cloudlog.exception(f"Error finding active connection: {e}")
      return None

  def _deactivate_connection_by_path(self, connection_path: str):
    self._router_main.send_and_get_reply(new_method_call(self._nm, 'DeactivateConnection', 'o', (connection_path,)))

  def _activate_modem_connection(self, connection_path: str):
    try:
      modem_device = self._get_adapter(NM_DEVICE_TYPE_MODEM)
      # Activate connection with modem device (matches C++ asyncCall)
      if modem_device and connection_path:
        self._router_main.send_and_get_reply(new_method_call(self._nm, 'ActivateConnection', 'ooo', (connection_path, modem_device, "/")))
    except Exception as e:
      cloudlog.exception(f"Error activating modem connection: {e}")

  def stop(self):
    if not self._exit:
      self._exit = True
      self._scan_thread.join()
      self._state_thread.join()

      self._router_main.close()
      self._router_main.conn.close()
      self._conn_monitor.close()
