import asyncio
from dbus_next.aio import MessageBus
from dbus_next import BusType, Variant, Message
from dbus_next.errors import DBusError
from dbus_next.constants import MessageType
from enum import IntEnum
import uuid
from dataclasses import dataclass

NM = "org.freedesktop.NetworkManager"
NM_DBUS_PATH = '/org/freedesktop/NetworkManager'
NM_DBUS_INTERFACE = 'org.freedesktop.NetworkManager'
NM_DBUS_PATH_SETTINGS = '/org/freedesktop/NetworkManager/Settings'
NM_DBUS_INTERFACE_SETTINGS = 'org.freedesktop.NetworkManager.Settings'
NM_DBUS_INTERFACE_SETTINGS_CONNECTION = 'org.freedesktop.NetworkManager.Settings.Connection'
NM_DBUS_INTERFACE_DEVICE_WIRELESS = 'org.freedesktop.NetworkManager.Device.Wireless'
NM_DBUS_INTERFACE_PROPERTIES = 'org.freedesktop.DBus.Properties'

NM_DEVICE_STATE_NEED_AUTH = 60


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
  is_saved: bool


class WifiManager:
  def __init__(self):
    self.networks = []
    self.connected_network = None
    self.bus = None
    self.device_path = None
    self.device_proxy = None
    self.saved_connections = dict()
    self.active_ap_path = ''

  async def connect(self):
    """Connect to the DBus system bus."""
    try:
      self.bus = await MessageBus(bus_type=BusType.SYSTEM).connect()
      nm_interface = await self.get_interface(NM, '/org/freedesktop/NetworkManager', NM)

      # Get the list of available devices (WiFi devices)
      devices = await nm_interface.get_devices()

      for device_path in devices:
        # Introspect each device and check if it's a WiFi device (DeviceType == 2)
        device = await self.bus.introspect(NM, device_path)
        device_proxy = self.bus.get_proxy_object(NM, device_path, device)
        device_interface = device_proxy.get_interface('org.freedesktop.NetworkManager.Device')
        device_type = await device_interface.get_device_type()

        if device_type == 2:  # WiFi device
          self.device_proxy = device_proxy
          self.device_path = device_path
          rule = f"type='signal',interface='org.freedesktop.DBus.Properties',member='PropertiesChanged',path='{device_path}'"
          # await self.bus.call_add_match(rule)
          await self._dbus_add_match(rule)
          rule = f"type='signal',interface='org.freedesktop.NetworkManager.Device',member='StateChanged',path='{device_path}'"
          await self._dbus_add_match(rule)
          rule = "type='signal',interface='org.freedesktop.NetworkManager.Settings',member='NewConnection',path='/org/freedesktop/NetworkManager/Settings'"
          await self._dbus_add_match(rule)
          break

      self.active_ap_path = await self.get_active_access_point()
      self.saved_connections = await self.get_saved_ssids()

    except DBusError as e:
      print(f"Failed to connect to DBus: {e}")

  async def request_scan(self):
    interface = self.device_proxy.get_interface(NM_DBUS_INTERFACE_DEVICE_WIRELESS)
    await interface.call_request_scan({})

  async def get_active_access_point(self):
    properties_interface = self.device_proxy.get_interface(NM_DBUS_INTERFACE_PROPERTIES)
    response = await properties_interface.call_get(NM_DBUS_INTERFACE_DEVICE_WIRELESS, 'ActiveAccessPoint')
    return response.value

  async def _dbus_add_match(self, body):
    """ "Add a match rule on the bus."""
    reply = await self.bus.call(
      Message(
        message_type=MessageType.METHOD_CALL,
        destination='org.freedesktop.DBus',
        interface="org.freedesktop.DBus",
        path='/org/freedesktop/DBus',
        member='AddMatch',
        signature='s',
        body=[body],
      )
    )

    assert reply.message_type == MessageType.METHOD_RETURN
    return reply

  async def get_available_networks(self):
    """Get a list of available networks via NetworkManager."""
    networks = []
    try:
      wifi_interface = self.device_proxy.get_interface(NM_DBUS_INTERFACE_DEVICE_WIRELESS)
      access_points = await wifi_interface.get_access_points()

      for ap_path in access_points:
        ap = await self.bus.introspect(NM, ap_path)
        ap_proxy = self.bus.get_proxy_object(NM, ap_path, ap)
        properties_interface = ap_proxy.get_interface(NM_DBUS_INTERFACE_PROPERTIES)
        properties = await properties_interface.call_get_all('org.freedesktop.NetworkManager.AccessPoint')

        ssid_variant = properties['Ssid'].value
        ssid = ''.join([chr(byte) for byte in ssid_variant])
        if not ssid:
          continue

        flags = properties['Flags'].value
        wpa_flags = properties['WpaFlags'].value
        rsn_flags = properties['RsnFlags'].value

        networks.append(
          NetworkInfo(
            ssid=ssid,
            strength=properties['Strength'].value,
            security_type=self._get_security_type(flags, wpa_flags, rsn_flags),
            path=ap_path,
            is_connected=self.active_ap_path == ap_path,
            is_saved=ssid in self.saved_connections,
          )
        )

    except DBusError as e:
      print(f"Error fetching networks: {e}")
    except Exception as e:
      print({e})
    finally:
      self.networks = sorted(
        networks,
        key=lambda network: (
          not network.is_connected,
          -network.strength,  # Higher signal strength first
          network.ssid.lower(),
        ),
      )

  async def get_connection_settings(self, path):
    """Fetch connection settings for a specific connection path."""
    connection_proxy = await self.bus.introspect(NM, path)
    connection = self.bus.get_proxy_object(NM, path, connection_proxy)
    settings = connection.get_interface(NM_DBUS_INTERFACE_SETTINGS_CONNECTION)
    return await settings.call_get_settings()

  async def process_chunk(self, paths_chunk):
    """Process a chunk of connection paths."""
    tasks = [self.get_connection_settings(path) for path in paths_chunk]
    results = await asyncio.gather(*tasks)
    return results

  async def get_saved_ssids(self):
    # Get the NetworkManager object
    introspection = await self.bus.introspect(NM, NM_DBUS_PATH_SETTINGS)
    network_manager = self.bus.get_proxy_object(NM, NM_DBUS_PATH_SETTINGS, introspection)
    settings_interface = network_manager.get_interface(NM_DBUS_INTERFACE_SETTINGS)
    connection_paths = await settings_interface.call_list_connections()

    saved_ssids = dict()
    batch_size = 120
    for i in range(0, len(connection_paths), batch_size):
      chunk = connection_paths[i : i + batch_size]
      results = await self.process_chunk(chunk)

      # Loop through the results and filter Wi-Fi connections
      for path, config in zip(chunk, results, strict=True):
        if '802-11-wireless' in config:
          ssid_variant = config['802-11-wireless']['ssid']
          ssid = ''.join(chr(b) for b in ssid_variant.value)
          saved_ssids[ssid] = path

    return saved_ssids

  async def get_interface(self, bus_name: str, path: str, name: str):
    introspection = await self.bus.introspect(bus_name, path)
    proxy_object = self.bus.get_proxy_object(bus_name, path, introspection)
    return proxy_object.get_interface(name)

  def _get_security_type(self, flags, wpa_flags, rsn_flags):
    """Helper function to determine the security type of a network."""
    if flags == 0:
      return SecurityType.OPEN
    elif wpa_flags != 0:
      return SecurityType.WPA
    elif rsn_flags != 0:
      return SecurityType.WPA2
    else:
      return SecurityType.UNSUPPORTED

  async def activate_connection(self, ssid: str):
    connection_path = self.saved_connections.get(ssid)
    if connection_path:
      print('activate connection:', connection_path)
      introspection = await self.bus.introspect(NM, NM_DBUS_PATH)
      proxy = self.bus.get_proxy_object(NM, NM_DBUS_PATH, introspection)
      interface = proxy.get_interface(NM_DBUS_INTERFACE)

      await interface.call_activate_connection(connection_path, self.device_path, '/')

  async def connect_to_network(self, ssid: str, password: str = None, is_hidden: bool = False):
    """Connect to a selected WiFi network."""
    try:
      print('connect_to_network', ssid, password)
      network_manager = await self.bus.introspect(NM, '/org/freedesktop/NetworkManager/Settings')
      proxy_object = self.bus.get_proxy_object(NM, '/org/freedesktop/NetworkManager/Settings', network_manager)
      settings_interface = proxy_object.get_interface('org.freedesktop.NetworkManager.Settings')

      # Create a connection dictionary
      connection = {
        'connection': {
          'type': Variant('s', '802-11-wireless'),
          'uuid': Variant('s', str(uuid.uuid4())),
          'id': Variant('s', ssid),
          'autoconnect-retries': Variant('i', 0),
        },
        '802-11-wireless': {'ssid': Variant('ay', ssid.encode('utf-8')), 'hidden': Variant('b', is_hidden), 'mode': Variant('s', 'infrastructure')},
        'ipv4': {'method': Variant('s', 'auto')},
        'ipv6': {'method': Variant('s', 'ignore')},
      }

      if password:
        connection['802-11-wireless-security'] = {
          'key-mgmt': Variant('s', 'wpa-psk'),
          'auth-alg': Variant('s', 'open'),
          'psk': Variant('s', password),
        }

      await settings_interface.call_add_connection(connection)

      self.connected_network = ssid

    except DBusError as e:
      print(f"Error connecting to network: {e}")

  async def forgot_connection(self, ssid: str):
    path = self.saved_connections.get(ssid)
    if path:
      connection_proxy = await self.bus.introspect(NM, path)
      connection = self.bus.get_proxy_object(NM, path, connection_proxy)
      settings = connection.get_interface(NM_DBUS_INTERFACE_SETTINGS_CONNECTION)
      await settings.call_delete()
