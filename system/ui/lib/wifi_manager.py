import asyncio
import concurrent.futures
import copy
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
from typing import TypeVar

from dbus_next.aio import MessageBus
from dbus_next import BusType, Variant, Message
from dbus_next.errors import DBusError
from dbus_next.constants import MessageType

try:
  from openpilot.common.params import Params
except ImportError:
  # Params/Cythonized modules are not available in zipapp
  Params = None
from openpilot.common.swaglog import cloudlog

T = TypeVar("T")

# NetworkManager constants
NM = "org.freedesktop.NetworkManager"
NM_PATH = '/org/freedesktop/NetworkManager'
NM_IFACE = 'org.freedesktop.NetworkManager'
NM_SETTINGS_PATH = '/org/freedesktop/NetworkManager/Settings'
NM_SETTINGS_IFACE = 'org.freedesktop.NetworkManager.Settings'
NM_CONNECTION_IFACE = 'org.freedesktop.NetworkManager.Settings.Connection'
NM_WIRELESS_IFACE = 'org.freedesktop.NetworkManager.Device.Wireless'
NM_PROPERTIES_IFACE = 'org.freedesktop.DBus.Properties'
NM_DEVICE_IFACE = "org.freedesktop.NetworkManager.Device"

NM_DEVICE_STATE_REASON_SUPPLICANT_DISCONNECT = 8

TETHERING_IP_ADDRESS = "192.168.43.1"
DEFAULT_TETHERING_PASSWORD = "12345678"


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
  WPA3 = 3
  UNSUPPORTED = 4


@dataclass
class NetworkInfo:
  ssid: str
  strength: int
  is_connected: bool
  security_type: SecurityType
  path: str
  bssid: str
  is_saved: bool = False
  # saved_path: str


@dataclass
class WifiManagerCallbacks:
  need_auth: Callable[[str], None] | None = None
  activated: Callable[[], None] | None = None
  forgotten: Callable[[str], None] | None = None
  networks_updated: Callable[[list[NetworkInfo]], None] | None = None
  connection_failed: Callable[[str, str], None] | None = None  # Added for error feedback


class WifiManager:
  def __init__(self, callbacks):
    self.callbacks: WifiManagerCallbacks = callbacks
    self.networks: list[NetworkInfo] = []
    self.bus: MessageBus = None
    self.device_path: str = ""
    self.device_proxy = None
    self.saved_connections: dict[str, str] = {}
    self.active_ap_path: str = ""
    self.scan_task: asyncio.Task | None = None
    # Set tethering ssid as "weedle" + first 4 characters of a dongle id
    self._tethering_ssid = "weedle"
    if Params is not None:
      dongle_id = Params().get("DongleId", encoding="utf-8")
      if dongle_id:
        self._tethering_ssid += "-" + dongle_id[:4]
    self.running: bool = True
    self._current_connection_ssid: str | None = None

  async def connect(self) -> None:
    """Connect to the DBus system bus."""
    try:
      self.bus = await MessageBus(bus_type=BusType.SYSTEM).connect()
      if not await self._find_wifi_device():
        raise ValueError("No Wi-Fi device found")

      await self._setup_signals(self.device_path)
      self.active_ap_path = await self.get_active_access_point()
      await self.add_tethering_connection(self._tethering_ssid, DEFAULT_TETHERING_PASSWORD)
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
      try:
        await self.scan_task
      except asyncio.CancelledError:
        pass
    if self.bus:
      self.bus.disconnect()

  async def _request_scan(self) -> None:
    try:
      interface = self.device_proxy.get_interface(NM_WIRELESS_IFACE)
      await interface.call_request_scan({})
    except DBusError as e:
      cloudlog.warning(f"Scan request failed: {str(e)}")

  async def get_active_access_point(self):
    try:
      props_iface = self.device_proxy.get_interface(NM_PROPERTIES_IFACE)
      ap_path = await props_iface.call_get(NM_WIRELESS_IFACE, 'ActiveAccessPoint')
      return ap_path.value
    except DBusError as e:
      cloudlog.error(f"Error fetching active access point: {str(e)}")
      return ''

  async def forget_connection(self, ssid: str) -> bool:
    path = self.saved_connections.get(ssid)
    if not path:
      return False

    try:
      nm_iface = await self._get_interface(NM, path, NM_CONNECTION_IFACE)
      await nm_iface.call_delete()

      if self._current_connection_ssid == ssid:
        self._current_connection_ssid = None

      if ssid in self.saved_connections:
        del self.saved_connections[ssid]

      for network in self.networks:
        if network.ssid == ssid:
          network.is_saved = False
          network.is_connected = False
          break

      # Notify UI of forgotten connection
      if self.callbacks.networks_updated:
        self.callbacks.networks_updated(copy.deepcopy(self.networks))

      return True
    except DBusError as e:
      cloudlog.error(f"Failed to delete connection for SSID: {ssid}. Error: {e}")
      return False

  async def activate_connection(self, ssid: str) -> bool:
    connection_path = self.saved_connections.get(ssid)
    if not connection_path:
      return False
    try:
      nm_iface = await self._get_interface(NM, NM_PATH, NM_IFACE)
      await nm_iface.call_activate_connection(connection_path, self.device_path, "/")
      return True
    except DBusError as e:
      cloudlog.error(f"Failed to activate connection {ssid}: {str(e)}")
      return False

  async def connect_to_network(self, ssid: str, password: str = None, bssid: str = None, is_hidden: bool = False) -> None:
    """Connect to a selected Wi-Fi network."""
    try:
      self._current_connection_ssid = ssid

      if ssid in self.saved_connections:
        # Forget old connection if new password provided
        if password:
          await self.forget_connection(ssid)
          await asyncio.sleep(0.2)  # NetworkManager delay
        else:
          # Just activate existing connection
          await self.activate_connection(ssid)
          return

      connection = {
        'connection': {
          'type': Variant('s', '802-11-wireless'),
          'uuid': Variant('s', str(uuid.uuid4())),
          'id': Variant('s', f'openpilot connection {ssid}'),
          'autoconnect-retries': Variant('i', 0),
        },
        '802-11-wireless': {
          'ssid': Variant('ay', ssid.encode('utf-8')),
          'hidden': Variant('b', is_hidden),
          'mode': Variant('s', 'infrastructure'),
        },
        'ipv4': {
          'method': Variant('s', 'auto'),
          'dns-priority': Variant('i', 600),
        },
        'ipv6': {'method': Variant('s', 'ignore')},
      }

      if bssid:
        connection['802-11-wireless']['bssid'] = Variant('ay', bssid.encode('utf-8'))

      if password:
        connection['802-11-wireless-security'] = {
          'key-mgmt': Variant('s', 'wpa-psk'),
          'auth-alg': Variant('s', 'open'),
          'psk': Variant('s', password),
        }

      nm_iface = await self._get_interface(NM, NM_PATH, NM_IFACE)
      await nm_iface.call_add_and_activate_connection(connection, self.device_path, "/")
    except Exception as e:
      self._current_connection_ssid = None
      cloudlog.error(f"Error connecting to network: {e}")
      # Notify UI of failure
      if self.callbacks.connection_failed:
        self.callbacks.connection_failed(ssid, str(e))

  def is_saved(self, ssid: str) -> bool:
    return ssid in self.saved_connections

  async def _find_wifi_device(self) -> bool:
    nm_iface = await self._get_interface(NM, NM_PATH, NM_IFACE)
    devices = await nm_iface.get_devices()

    for device_path in devices:
      device = await self.bus.introspect(NM, device_path)
      device_proxy = self.bus.get_proxy_object(NM, device_path, device)
      device_interface = device_proxy.get_interface(NM_DEVICE_IFACE)
      device_type = await device_interface.get_device_type()  # type: ignore[attr-defined]
      if device_type == 2:  # Wi-Fi device
        self.device_path = device_path
        self.device_proxy = device_proxy
        return True

    return False

  async def add_tethering_connection(self, ssid: str, password: str = "12345678") -> bool:
    """Create a WiFi tethering connection."""
    if len(password) < 8:
      print("Tethering password must be at least 8 characters")
      return False

    try:
      # First, check if a hotspot connection already exists
      settings_iface = await self._get_interface(NM, NM_SETTINGS_PATH, NM_SETTINGS_IFACE)
      connection_paths = await settings_iface.call_list_connections()

      # Look for an existing hotspot connection
      for path in connection_paths:
        try:
          settings = await self._get_connection_settings(path)
          conn_type = settings.get('connection', {}).get('type', Variant('s', '')).value
          wifi_mode = settings.get('802-11-wireless', {}).get('mode', Variant('s', '')).value

          if conn_type == '802-11-wireless' and wifi_mode == 'ap':
            # Extract the SSID to check
            connection_ssid = self._extract_ssid(settings)
            if connection_ssid == ssid:
              return True
        except DBusError:
          continue

      connection = {
        'connection': {
          'id': Variant('s', 'Hotspot'),
          'uuid': Variant('s', str(uuid.uuid4())),
          'type': Variant('s', '802-11-wireless'),
          'interface-name': Variant('s', 'wlan0'),
          'autoconnect': Variant('b', False),
        },
        '802-11-wireless': {
          'band': Variant('s', 'bg'),
          'mode': Variant('s', 'ap'),
          'ssid': Variant('ay', ssid.encode('utf-8')),
        },
        '802-11-wireless-security': {
          'group': Variant('as', ['ccmp']),
          'key-mgmt': Variant('s', 'wpa-psk'),
          'pairwise': Variant('as', ['ccmp']),
          'proto': Variant('as', ['rsn']),
          'psk': Variant('s', password),
        },
        'ipv4': {
          'method': Variant('s', 'shared'),
          'address-data': Variant('aa{sv}', [{'address': Variant('s', TETHERING_IP_ADDRESS), 'prefix': Variant('u', 24)}]),
          'gateway': Variant('s', TETHERING_IP_ADDRESS),
          'never-default': Variant('b', True),
        },
        'ipv6': {
          'method': Variant('s', 'ignore'),
        },
      }

      settings_iface = await self._get_interface(NM, NM_SETTINGS_PATH, NM_SETTINGS_IFACE)
      new_connection = await settings_iface.call_add_connection(connection)
      print(f"Added tethering connection with path: {new_connection}")
      return True
    except DBusError as e:
      print(f"Failed to add tethering connection: {e}")
      return False
    except Exception as e:
      print(f"Unexpected error adding tethering connection: {e}")
      return False

  async def get_tethering_password(self) -> str:
    """Get the current tethering password."""
    try:
      hotspot_path = self.saved_connections.get(self._tethering_ssid)
      if hotspot_path:
        conn_iface = await self._get_interface(NM, hotspot_path, NM_CONNECTION_IFACE)
        secrets = await conn_iface.call_get_secrets('802-11-wireless-security')
        if secrets and '802-11-wireless-security' in secrets:
          psk = secrets.get('802-11-wireless-security', {}).get('psk', Variant('s', '')).value
          return str(psk) if psk is not None else ""
      return ""
    except DBusError as e:
      print(f"Failed to get tethering password: {e}")
      return ""
    except Exception as e:
      print(f"Unexpected error getting tethering password: {e}")
      return ""

  async def set_tethering_password(self, password: str) -> bool:
    """Set the tethering password."""
    if len(password) < 8:
      cloudlog.error("Tethering password must be at least 8 characters")
      return False

    try:
      hotspot_path = self.saved_connections.get(self._tethering_ssid)
      if not hotspot_path:
        print("No hotspot connection found")
        return False

      # Update the connection settings with new password
      settings = await self._get_connection_settings(hotspot_path)
      if '802-11-wireless-security' not in settings:
        settings['802-11-wireless-security'] = {}
      settings['802-11-wireless-security']['psk'] = Variant('s', password)

      # Apply changes
      conn_iface = await self._get_interface(NM, hotspot_path, NM_CONNECTION_IFACE)
      await conn_iface.call_update(settings)

      # Check if connection is active and restart if needed
      is_active = False
      nm_iface = await self._get_interface(NM, NM_PATH, NM_IFACE)
      active_connections = await nm_iface.get_active_connections()

      for conn_path in active_connections:
        props_iface = await self._get_interface(NM, conn_path, NM_PROPERTIES_IFACE)
        conn_id_path = await props_iface.call_get('org.freedesktop.NetworkManager.Connection.Active', 'Connection')
        if conn_id_path.value == hotspot_path:
          is_active = True
          await nm_iface.call_deactivate_connection(conn_path)
          break

      if is_active:
        await nm_iface.call_activate_connection(hotspot_path, self.device_path, "/")

      print("Tethering password updated successfully")
      return True
    except DBusError as e:
      print(f"Failed to set tethering password: {e}")
      return False
    except Exception as e:
      print(f"Unexpected error setting tethering password: {e}")
      return False

  async def is_tethering_active(self) -> bool:
    """Check if tethering is active for the specified SSID."""
    try:
      hotspot_path = self.saved_connections.get(self._tethering_ssid)
      if not hotspot_path:
        return False

      nm_iface = await self._get_interface(NM, NM_PATH, NM_IFACE)
      active_connections = await nm_iface.get_active_connections()

      for conn_path in active_connections:
        props_iface = await self._get_interface(NM, conn_path, NM_PROPERTIES_IFACE)
        conn_id_path = await props_iface.call_get('org.freedesktop.NetworkManager.Connection.Active', 'Connection')

        if conn_id_path.value == hotspot_path:
          return True

      return False
    except Exception:
      return False

  async def _periodic_scan(self):
    while self.running:
      try:
        await self._request_scan()
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
    self.device_proxy.get_interface(NM_PROPERTIES_IFACE).on_properties_changed(self._on_properties_changed)
    self.device_proxy.get_interface(NM_DEVICE_IFACE).on_state_changed(self._on_state_changed)

    settings_iface = await self._get_interface(NM, NM_SETTINGS_PATH, NM_SETTINGS_IFACE)
    settings_iface.on_new_connection(self._on_new_connection)
    settings_iface.on_connection_removed(self._on_connection_removed)

  def _on_properties_changed(self, interface: str, changed: dict, invalidated: list):
    # print("property changed", interface, changed, invalidated)
    if 'LastScan' in changed:
      asyncio.create_task(self._refresh_networks())
    elif interface == NM_WIRELESS_IFACE and "ActiveAccessPoint" in changed:
      new_ap_path = changed["ActiveAccessPoint"].value
      if self.active_ap_path != new_ap_path:
        self.active_ap_path = new_ap_path
        asyncio.create_task(self._refresh_networks())

  def _on_state_changed(self, new_state: int, old_state: int, reason: int):
    print("State changed", new_state, old_state, reason)
    if new_state == NMDeviceState.ACTIVATED:
      if self.callbacks.activated:
        self.callbacks.activated()
      asyncio.create_task(self._refresh_networks())
      self._current_connection_ssid = None
    elif new_state in (NMDeviceState.DISCONNECTED, NMDeviceState.NEED_AUTH):
      for network in self.networks:
        network.is_connected = False

      if new_state == NMDeviceState.NEED_AUTH and reason == NM_DEVICE_STATE_REASON_SUPPLICANT_DISCONNECT and self.callbacks.need_auth:
        if self._current_connection_ssid:
          self.callbacks.need_auth(self._current_connection_ssid)
        else:
          # Try to find the network from active_ap_path
          for network in self.networks:
            if network.path == self.active_ap_path:
              self.callbacks.need_auth(network.ssid)
              break
          else:
            # Couldn't identify the network that needs auth
            cloudlog.error("Network needs authentication but couldn't identify which one")

  def _on_new_connection(self, path: str) -> None:
    """Callback for NewConnection signal."""
    asyncio.create_task(self._add_saved_connection(path))

  def _on_connection_removed(self, path: str) -> None:
    """Callback for ConnectionRemoved signal."""
    for ssid, p in list(self.saved_connections.items()):
      if path == p:
        del self.saved_connections[ssid]

        if self.callbacks.forgotten:
          self.callbacks.forgotten(ssid)

        # Update network list to reflect the removed saved connection
        asyncio.create_task(self._refresh_networks())
        break

  async def _add_saved_connection(self, path: str) -> None:
    """Add a new saved connection to the dictionary."""
    try:
      settings = await self._get_connection_settings(path)
      if ssid := self._extract_ssid(settings):
        self.saved_connections[ssid] = path
        await self._refresh_networks()
    except DBusError as e:
      cloudlog.error(f"Failed to add connection {path}: {e}")

  def _extract_ssid(self, settings: dict) -> str | None:
    """Extract SSID from connection settings."""
    ssid_variant = settings.get('802-11-wireless', {}).get('ssid', Variant('ay', b'')).value
    return ''.join(chr(b) for b in ssid_variant) if ssid_variant else None

  async def _add_match_rule(self, rule):
    """Add a match rule on the bus."""
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

  async def _refresh_networks(self):
    """Get a list of available networks via NetworkManager."""
    wifi_iface = self.device_proxy.get_interface(NM_WIRELESS_IFACE)
    access_points = await wifi_iface.get_access_points()
    self.active_ap_path = await self.get_active_access_point()
    network_dict = {}
    for ap_path in access_points:
      try:
        props_iface = await self._get_interface(NM, ap_path, NM_PROPERTIES_IFACE)
        properties = await props_iface.call_get_all('org.freedesktop.NetworkManager.AccessPoint')
        ssid_variant = properties['Ssid'].value
        ssid = ''.join(chr(byte) for byte in ssid_variant)
        if not ssid:
          continue

        bssid = properties.get('HwAddress', Variant('s', '')).value
        strength = properties['Strength'].value
        flags = properties['Flags'].value
        wpa_flags = properties['WpaFlags'].value
        rsn_flags = properties['RsnFlags'].value
        existing_network = network_dict.get(ssid)
        if not existing_network or ((not existing_network.bssid and bssid) or (existing_network.strength < strength)):
          network_dict[ssid] = NetworkInfo(
            ssid=ssid,
            strength=strength,
            security_type=self._get_security_type(flags, wpa_flags, rsn_flags),
            path=ap_path,
            bssid=bssid,
            is_connected=self.active_ap_path == ap_path and self._current_connection_ssid != ssid,
            is_saved=ssid in self.saved_connections
          )

      except DBusError as e:
        cloudlog.error(f"Error fetching networks: {e}")
      except Exception as e:
        cloudlog.error({e})

    self.networks = sorted(
      network_dict.values(),
      key=lambda network: (
        not network.is_connected,
        -network.strength,  # Higher signal strength first
        network.ssid.lower(),
      ),
    )

    if self.callbacks.networks_updated:
      self.callbacks.networks_updated(copy.deepcopy(self.networks))

  async def _get_connection_settings(self, path):
    """Fetch connection settings for a specific connection path."""
    try:
      settings = await self._get_interface(NM, path, NM_CONNECTION_IFACE)
      return await settings.call_get_settings()
    except DBusError as e:
      cloudlog.error(f"Failed to get settings for {path}: {str(e)}")
      return {}

  async def _process_chunk(self, paths_chunk):
    """Process a chunk of connection paths."""
    tasks = [self._get_connection_settings(path) for path in paths_chunk]
    return await asyncio.gather(*tasks, return_exceptions=True)

  async def _get_saved_connections(self) -> dict[str, str]:
    try:
      settings_iface = await self._get_interface(NM, NM_SETTINGS_PATH, NM_SETTINGS_IFACE)
      connection_paths = await settings_iface.call_list_connections()
      saved_ssids: dict[str, str] = {}
      batch_size = 20
      for i in range(0, len(connection_paths), batch_size):
        chunk = connection_paths[i : i + batch_size]
        results = await self._process_chunk(chunk)
        for path, config in zip(chunk, results, strict=True):
          if isinstance(config, dict) and '802-11-wireless' in config:
            if ssid := self._extract_ssid(config):
              saved_ssids[ssid] = path
      return saved_ssids
    except DBusError as e:
      cloudlog.error(f"Error fetching saved connections: {str(e)}")
      return {}

  async def _get_interface(self, bus_name: str, path: str, name: str):
    introspection = await self.bus.introspect(bus_name, path)
    proxy = self.bus.get_proxy_object(bus_name, path, introspection)
    return proxy.get_interface(name)

  def _get_security_type(self, flags: int, wpa_flags: int, rsn_flags: int) -> SecurityType:
    """Determine the security type based on flags."""
    if flags == 0 and not (wpa_flags or rsn_flags):
      return SecurityType.OPEN
    if rsn_flags & 0x200:  # SAE (WPA3 Personal)
      # TODO: support WPA3
      return SecurityType.UNSUPPORTED
    if rsn_flags:  # RSN indicates WPA2 or higher
      return SecurityType.WPA2
    if wpa_flags:  # WPA flags indicate WPA
      return SecurityType.WPA
    return SecurityType.UNSUPPORTED


class WifiManagerWrapper:
  def __init__(self):
    self._manager: WifiManager | None = None
    self._callbacks: WifiManagerCallbacks = WifiManagerCallbacks()

    self._thread = threading.Thread(target=self._run, daemon=True)
    self._loop: asyncio.EventLoop | None = None
    self._running = False

  def set_callbacks(self, callbacks: WifiManagerCallbacks):
    self._callbacks = callbacks

  def start(self) -> None:
    if not self._running:
      self._thread.start()
      while self._thread is not None and not self._running:
        time.sleep(0.1)

  def _run(self):
    self._loop = asyncio.new_event_loop()
    asyncio.set_event_loop(self._loop)

    try:
      self._manager = WifiManager(self._callbacks)
      self._running = True
      self._loop.run_forever()
    except Exception as e:
      cloudlog.error(f"Error in WifiManagerWrapper thread: {e}")
    finally:
      if self._loop.is_running():
        self._loop.stop()
      self._running = False

  def shutdown(self) -> None:
    if self._running:
      if self._manager is not None and self._loop:
        shutdown_future = asyncio.run_coroutine_threadsafe(self._manager.shutdown(), self._loop)
        shutdown_future.result(timeout=3.0)

      if self._loop and self._loop.is_running():
        self._loop.call_soon_threadsafe(self._loop.stop)
      if self._thread and self._thread.is_alive():
        self._thread.join(timeout=2.0)
      self._running = False

  def is_saved(self, ssid: str) -> bool:
    """Check if a network is saved."""
    return self._run_coroutine_sync(lambda manager: manager.is_saved(ssid), default=False)

  def connect(self):
    """Connect to DBus and start Wi-Fi scanning."""
    if not self._manager:
      return
    self._run_coroutine(self._manager.connect())

  def forget_connection(self, ssid: str):
    """Forget a saved Wi-Fi connection."""
    if not self._manager:
      return
    self._run_coroutine(self._manager.forget_connection(ssid))

  def activate_connection(self, ssid: str):
    """Activate an existing Wi-Fi connection."""
    if not self._manager:
      return
    self._run_coroutine(self._manager.activate_connection(ssid))

  def connect_to_network(self, ssid: str, password: str = None, bssid: str = None, is_hidden: bool = False):
    """Connect to a Wi-Fi network."""
    if not self._manager:
      return
    self._run_coroutine(self._manager.connect_to_network(ssid, password, bssid, is_hidden))

  def _run_coroutine(self, coro):
    """Run a coroutine in the async thread."""
    if not self._running or not self._loop:
      cloudlog.error("WifiManager thread is not running")
      return
    asyncio.run_coroutine_threadsafe(coro, self._loop)

  def _run_coroutine_sync(self, func: Callable[[WifiManager], T], default: T) -> T:
    """Run a function synchronously in the async thread."""
    if not self._running or not self._loop or not self._manager:
      return default
    future = concurrent.futures.Future[T]()

    def wrapper(manager: WifiManager) -> None:
      try:
        future.set_result(func(manager))
      except Exception as e:
        future.set_exception(e)

    try:
      self._loop.call_soon_threadsafe(wrapper, self._manager)
      return future.result(timeout=1.0)
    except Exception as e:
      cloudlog.error(f"WifiManagerWrapper property access failed: {e}")
      return default
