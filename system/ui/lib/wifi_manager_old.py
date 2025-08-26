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

from openpilot.system.ui.lib.networkmanager import (NM, NM_PATH, NM_IFACE, NM_SETTINGS_PATH, NM_SETTINGS_IFACE,
                                                    NM_CONNECTION_IFACE, NM_WIRELESS_IFACE, NM_PROPERTIES_IFACE,
                                                    NM_DEVICE_IFACE, NM_DEVICE_STATE_REASON_SUPPLICANT_DISCONNECT,
                                                    NMDeviceState)

try:
  from openpilot.common.params import Params
except ImportError:
  # Params/Cythonized modules are not available in zipapp
  Params = None
from openpilot.common.swaglog import cloudlog

T = TypeVar("T")

TETHERING_IP_ADDRESS = "192.168.43.1"
DEFAULT_TETHERING_PASSWORD = "swagswagcomma"


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
      dongle_id = Params().get("DongleId")
      if dongle_id:
        self._tethering_ssid += "-" + dongle_id[:4]
    self.running: bool = True
    self._current_connection_ssid: str | None = None

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

  def _extract_ssid(self, settings: dict) -> str | None:
    """Extract SSID from connection settings."""
    ssid_variant = settings.get('802-11-wireless', {}).get('ssid', Variant('ay', b'')).value
    return bytes(ssid_variant).decode('utf-8') if ssid_variant else None

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
