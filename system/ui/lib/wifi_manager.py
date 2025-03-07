import asyncio
from dbus_next.aio import MessageBus
from dbus_next import BusType, Variant, Message
from dbus_next.errors import DBusError
from dbus_next.constants import MessageType
from enum import IntEnum
import uuid
from dataclasses import dataclass
from openpilot.common.swaglog import cloudlog

NM = "org.freedesktop.NetworkManager"
NM_PATH = '/org/freedesktop/NetworkManager'
NM_IFACE = 'org.freedesktop.NetworkManager'
NM_SETTINGS_PATH = '/org/freedesktop/NetworkManager/Settings'
NM_SETTINGS_IFACE = 'org.freedesktop.NetworkManager.Settings'
NM_CONNECTION_IFACE = 'org.freedesktop.NetworkManager.Settings.Connection'
NM_WIRELESS_IFACE = 'org.freedesktop.NetworkManager.Device.Wireless'
NM_PROPERTIES_IFACE = 'org.freedesktop.DBus.Properties'
NM_DEVICE_IFACE = "org.freedesktop.NetworkManager.Device"


# NetworkManager device states
class NMDeviceState(IntEnum):
  DISCONNECTED = 30
  PREPARE = 40
  NEED_AUTH = 60
  IP_CONFIG = 70
  ACTIVATED = 100


class SecurityType(IntEnum):
  OPEN = 0
  WPA = 1
  WPA2 = 2
  UNSUPPORTED = 3


@dataclass
class NetworkInfo:
  ssid: str
  strength: int
  is_connected: bool
  security_type: SecurityType
  path: str
  bssid: str
  # saved_path: str


class WifiManager:
  def __init__(self):
    self.networks: list[NetworkInfo] = []
    self.bus: MessageBus | None = None
    self.device_path: str | None = None
    self.device_proxy = None
    self.saved_connections: dict[str, str] = dict()
    self.active_ap_path: str = ''
    self.scan_task: asyncio.Task | None = None
    self.running: bool = True

  def is_saved(self, ssid: str) -> bool:
    return ssid in self.saved_connections

  async def connect(self):
    """Connect to the DBus system bus."""
    try:
      self.bus = await MessageBus(bus_type=BusType.SYSTEM).connect()
      if not await self._find_wifi_device():
        raise ValueError("No Wi-Fi device found")
      await self._setup_signals(self.device_path)

      self.active_ap_path = await self.get_active_access_point()
      self.saved_connections = await self._get_saved_connections()
      self.scan_task = asyncio.create_task(self._periodic_scan())
    except DBusError as e:
      cloudlog.error(f"Failed to connect to DBus: {e}")
      raise
    except Exception as e:
      cloudlog.error(f"Unexpected error during connect: {e}")
      raise

  async def shutdown(self) -> None:
    self.running = False
    if self.scan_task:
      self.scan_task.cancel()
      await self.scan_task
    if self.bus:
      await self.bus.disconnect()

  async def request_scan(self):
    try:
      interface = self.device_proxy.get_interface(NM_WIRELESS_IFACE)
      await interface.call_request_scan({})
    except DBusError as e:
      cloudlog.warning(f"Scan request failed: {e}")

  async def get_active_access_point(self):
    try:
      props_iface = self.device_proxy.get_interface(NM_PROPERTIES_IFACE)
      ap_path = await props_iface.call_get(NM_WIRELESS_IFACE, 'ActiveAccessPoint')
      return ap_path.value
    except DBusError as e:
      cloudlog.error(f"Error fetching active access point: {e}")
      return ''

  async def forgot_connection(self, ssid: str) -> bool:
    path = self.saved_connections.get(ssid)
    if not path:
      return False

    try:
      nm_iface = await self._get_interface(NM, path, NM_CONNECTION_IFACE)
      await nm_iface.call_delete()
      self.saved_connections.pop(ssid)
      return True
    except DBusError as e:
      cloudlog.error(f"Failed to delete connection for SSID: {ssid}. Error: {e}")
      return False
    except Exception as e:
      cloudlog.error(f"Unexpected error while deleting connection for SSID: {ssid}: {e}")
      return False

  async def activate_connection(self, ssid: str) -> None:
    connection_path = self.saved_connections.get(ssid)
    if connection_path:
      cloudlog.info('activate connection:', connection_path)
      introspection = await self.bus.introspect(NM, NM_PATH)
      proxy = self.bus.get_proxy_object(NM, NM_PATH, introspection)
      interface = proxy.get_interface(NM_IFACE)

      await interface.call_activate_connection(connection_path, self.device_path, '/')

  async def connect_to_network(self, ssid: str, password: str = None, is_hidden: bool = False):
    """Connect to a selected WiFi network."""
    try:
      settings_iface = await self._get_interface(NM, NM_SETTINGS_PATH, NM_SETTINGS_IFACE)
      connection = {
        'connection': {
          'type': Variant('s', '802-11-wireless'),
          'uuid': Variant('s', str(uuid.uuid4())),
          'id': Variant('s', ssid),
          'autoconnect-retries': Variant('i', 0),
        },
        '802-11-wireless': {
          'ssid': Variant('ay', ssid.encode('utf-8')),
          'hidden': Variant('b', is_hidden),
          'mode': Variant('s', 'infrastructure'),
        },
        'ipv4': {'method': Variant('s', 'auto')},
        'ipv6': {'method': Variant('s', 'ignore')},
      }

      # if bssid:
      #   connection['802-11-wireless']['bssid'] = Variant('ay', bssid.encode('utf-8'))

      if password:
        connection['802-11-wireless-security'] = {
          'key-mgmt': Variant('s', 'wpa-psk'),
          'auth-alg': Variant('s', 'open'),
          'psk': Variant('s', password),
        }

      await settings_iface.call_add_connection(connection)

      for network in self.networks:
        network.is_connected = True if network.ssid == ssid else False

    except DBusError as e:
      cloudlog.error(f"Error connecting to network: {e}")

  async def _find_wifi_device(self) -> bool:
    nm_iface = await self._get_interface(NM, NM_PATH, NM_IFACE)
    devices = await nm_iface.get_devices()

    for device_path in devices:
      device = await self.bus.introspect(NM, device_path)
      device_proxy = self.bus.get_proxy_object(NM, device_path, device)
      device_interface = device_proxy.get_interface(NM_DEVICE_IFACE)
      device_type = await device_interface.get_device_type()
      if device_type == 2:  # WiFi device
        self.device_path = device_path
        self.device_proxy = device_proxy
        return True

    return False

  async def _periodic_scan(self):
    while self.running:
      try:
        await self.request_scan()
        await self._get_available_networks()
        await asyncio.sleep(30)
      except asyncio.CancelledError:
        break
      except DBusError as e:
        cloudlog.error(f"Scan failed: {e}")
        await asyncio.sleep(5)

  async def _setup_signals(self, device_path: str) -> None:
    rules = [
      f"type='signal',interface='{NM_PROPERTIES_IFACE}',member='PropertiesChanged',path='{device_path}'",
      f"type='signal',interface='{NM_DEVICE_IFACE}',member='StateChanged',path='{device_path}'",
      f"type='signal',interface='{NM_SETTINGS_IFACE}',member='NewConnection',path='{NM_SETTINGS_PATH}'",
      f"type='signal',interface='{NM_SETTINGS_IFACE}',member='ConnectionRemoved',path='{NM_SETTINGS_PATH}'",
    ]
    for rule in rules:
      await self._add_match_rule(rule)

    # Set up signal handlers
    self.device_proxy.get_interface(NM_PROPERTIES_IFACE).on_properties_changed(
      self._on_properties_changed
    )
    self.device_proxy.get_interface(NM_DEVICE_IFACE).on_state_changed(self._on_state_changed)

    settings_iface = await self._get_interface(NM, NM_SETTINGS_PATH, NM_SETTINGS_IFACE)
    settings_iface.on_new_connection(self._on_new_connection)
    settings_iface.on_connection_removed(self._on_connection_removed)

  def _on_properties_changed(self, interface: str, changed: dict, invalidated: list):
    # print("property changed", interface, changed, invalidated)
    if 'LastScan' in changed:
      asyncio.create_task(self._get_available_networks())
    elif "ActiveAccessPoint" in changed:
      self.active_ap_path = changed["ActiveAccessPoint"].value
      asyncio.create_task(self._get_available_networks())

  def _on_state_changed(self, new_state: int, old_state: int, reason: int):
    print(f"State changed: {old_state} -> {new_state}, reason: {reason}")
    if new_state == NMDeviceState.ACTIVATED:
      asyncio.create_task(self._update_connection_status())
    elif new_state in (NMDeviceState.DISCONNECTED, NMDeviceState.NEED_AUTH):
      for network in self.networks:
        network.is_connected = False

  def _on_new_connection(self, path: str) -> None:
    """Callback for NewConnection signal."""
    print(f"New connection added: {path}")
    asyncio.create_task(self._add_saved_connection(path))

  def _on_connection_removed(self, path: str) -> None:
    """Callback for ConnectionRemoved signal."""
    print(f"Connection removed: {path}")
    for ssid, p in self.saved_connections.items():
      if path == p:
        del self.saved_connections[ssid]
        break

  async def _add_saved_connection(self, path: str) -> None:
    """Add a new saved connection to the dictionary."""
    try:
      settings = await self._get_connection_settings(path)
      if ssid := self._extract_ssid(settings):
        self.saved_connections[ssid] = path
    except DBusError as e:
      cloudlog.error(f"Failed to add connection {path}: {e}")

  def _extract_ssid(self, settings: dict) -> str | None:
    """Extract SSID from connection settings."""
    ssid_variant = settings.get('802-11-wireless', {}).get('ssid', Variant('ay', b'')).value
    return ''.join(chr(b) for b in ssid_variant) if ssid_variant else None

  async def _update_connection_status(self):
    self.active_ap_path = await self.get_active_access_point()
    await self._get_available_networks()

  async def _add_match_rule(self, rule):
    """ "Add a match rule on the bus."""
    reply = await self.bus.call(
      Message(
        message_type=MessageType.METHOD_CALL,
        destination='org.freedesktop.DBus',
        interface="org.freedesktop.DBus",
        path='/org/freedesktop/DBus',
        member='AddMatch',
        signature='s',
        body=[rule],
      )
    )

    assert reply.message_type == MessageType.METHOD_RETURN
    return reply

  async def _get_available_networks(self):
    """Get a list of available networks via NetworkManager."""
    networks = []
    wifi_iface = self.device_proxy.get_interface(NM_WIRELESS_IFACE)
    access_points = await wifi_iface.get_access_points()

    for ap_path in access_points:
      try:
        props_iface = await self._get_interface(NM, ap_path, NM_PROPERTIES_IFACE)
        properties = await props_iface.call_get_all('org.freedesktop.NetworkManager.AccessPoint')
        ssid_variant = properties['Ssid'].value
        ssid = ''.join(chr(byte) for byte in ssid_variant)
        if not ssid:
          continue

        bssid = properties.get('HwAddress', Variant('s', '')).value
        print(bssid)
        flags = properties['Flags'].value
        wpa_flags = properties['WpaFlags'].value
        rsn_flags = properties['RsnFlags'].value

        networks.append(
          NetworkInfo(
            ssid=ssid,
            strength=properties['Strength'].value,
            security_type=self._get_security_type(flags, wpa_flags, rsn_flags),
            path=ap_path,
            bssid=bssid,
            is_connected=self.active_ap_path == ap_path,
          )
        )
      except DBusError as e:
        cloudlog.error(f"Error fetching networks: {e}")
      except Exception as e:
        cloudlog.error({e})

    self.networks = sorted(
      networks,
      key=lambda network: (
        not network.is_connected,
        -network.strength,  # Higher signal strength first
        network.ssid.lower(),
      ),
    )

  async def _get_connection_settings(self, path):
    """Fetch connection settings for a specific connection path."""
    connection_proxy = await self.bus.introspect(NM, path)
    connection = self.bus.get_proxy_object(NM, path, connection_proxy)
    settings = connection.get_interface(NM_CONNECTION_IFACE)
    return await settings.call_get_settings()

  async def _process_chunk(self, paths_chunk):
    """Process a chunk of connection paths."""
    tasks = [self._get_connection_settings(path) for path in paths_chunk]
    results = await asyncio.gather(*tasks)
    return results

  async def _get_saved_connections(self):
    settings_iface = await self._get_interface(NM, NM_SETTINGS_PATH, NM_SETTINGS_IFACE)
    connection_paths = await settings_iface.call_list_connections()

    saved_ssids: dict[str, str] = {}
    batch_size = 120
    for i in range(0, len(connection_paths), batch_size):
      chunk = connection_paths[i : i + batch_size]
      results = await self._process_chunk(chunk)

      # Loop through the results and filter Wi-Fi connections
      for path, config in zip(chunk, results, strict=True):
        if '802-11-wireless' in config:
          saved_ssids[self._extract_ssid(config)] = path

    return saved_ssids

  async def _get_interface(self, bus_name: str, path: str, name: str):
    introspection = await self.bus.introspect(bus_name, path)
    proxy = self.bus.get_proxy_object(bus_name, path, introspection)
    return proxy.get_interface(name)

  def _get_security_type(self, flags, wpa_flags, rsn_flags):
    """Helper function to determine the security type of a network."""
    if flags == 0:
      return SecurityType.OPEN
    if wpa_flags:
      return SecurityType.WPA
    if rsn_flags:
      return SecurityType.WPA2
    else:
      return SecurityType.UNSUPPORTED

  async def _get_interface(self, bus_name: str, path: str, name: str):
    introspection = await self.bus.introspect(bus_name, path)
    proxy = self.bus.get_proxy_object(bus_name, path, introspection)
    return proxy.get_interface(name)
