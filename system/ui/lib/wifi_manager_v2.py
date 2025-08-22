import asyncio
import concurrent.futures
import copy
from collections import defaultdict
import dbus
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

NM_DEVICE_TYPE_WIFI = 2

TETHERING_IP_ADDRESS = "192.168.43.1"
DEFAULT_TETHERING_PASSWORD = "swagswagcomma"


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


@dataclass(frozen=True)
class Network:
  ssid: str
  strength: int
  is_connected: bool
  security_type: SecurityType  # TODO
  is_saved: bool = False  # TODO


@dataclass(frozen=True)
class AccessPoint:
  ssid: str
  bssid: str
  strength: int
  is_connected: bool
  flags: int
  wpa_flags: int
  rsn_flags: int
  ap_path: dbus.ObjectPath

  @classmethod
  def from_dbus(cls, ap_props: dbus.Interface, ap_path: dbus.ObjectPath, active_ap_path: dbus.ObjectPath) -> "AccessPoint":
    ssid = bytes(ap_props.Get("org.freedesktop.NetworkManager.AccessPoint", "Ssid")).decode("utf-8", "replace")
    bssid = str(ap_props.Get("org.freedesktop.NetworkManager.AccessPoint", "HwAddress"))
    strength = int(ap_props.Get("org.freedesktop.NetworkManager.AccessPoint", "Strength"))
    flags = int(ap_props.Get("org.freedesktop.NetworkManager.AccessPoint", "Flags"))
    wpa_flags = int(ap_props.Get("org.freedesktop.NetworkManager.AccessPoint", "WpaFlags"))
    rsn_flags = int(ap_props.Get("org.freedesktop.NetworkManager.AccessPoint", "RsnFlags"))
    is_connected = ap_path == active_ap_path

    return cls(
      ssid=ssid,
      bssid=bssid,
      strength=strength,
      is_connected=is_connected,
      flags=flags,
      wpa_flags=wpa_flags,
      rsn_flags=rsn_flags,
      ap_path=ap_path,
    )


@dataclass
class WifiManagerCallbacks:
  need_auth: Callable[[str], None] | None = None
  activated: Callable[[], None] | None = None
  forgotten: Callable[[str], None] | None = None
  networks_updated: Callable[[list[Network]], None] | None = None
  connection_failed: Callable[[str, str], None] | None = None  # Added for error feedback


class WifiManager:
  def __init__(self):
    self._networks = []  # a network can be comprised of multiple APs
    self._active = True  # used to not run this cycle when not in settings
    self._running = True

    # DBus and NetworkManager setup
    self._bus = dbus.SystemBus()
    self._nm = dbus.Interface(self._bus.get_object(NM, NM_PATH), NM_IFACE)
    self._props = dbus.Interface(self._bus.get_object(NM, NM_PATH), NM_PROPERTIES_IFACE)

    self._thread = threading.Thread(target=self._run, daemon=True)
    self._thread.start()

  def stop(self):
    self._running = False
    self._thread.join()

  def _run(self):
    while True:
      if self._active:
        self._update_networks()

      if not self._running:
        break

      time.sleep(1)

  def set_active(self, active: bool):
    self._active = active

  def _get_wifi_device(self) -> dbus.ObjectPath | None:
    # TODO: cache if slow
    t = time.monotonic()
    device_paths = self._nm.GetDevices()
    # print(f'DEVICE PATHS: {device_paths}')

    wifi_device = None
    for device_path in device_paths:
      dev_props = dbus.Interface(self._bus.get_object(NM, device_path), NM_PROPERTIES_IFACE)
      dev_type = dev_props.Get(NM_DEVICE_IFACE, "DeviceType")

      if dev_type == NM_DEVICE_TYPE_WIFI:
        wifi_device = device_path
        break

    print(f"Got wifi device in {time.monotonic() - t}s: {wifi_device}")
    return wifi_device

  def _update_networks(self):
    # TODO: only run this function on scan complete!
    print('UPDATING NETWORKS!!!!')

    device_path = self._get_wifi_device()
    if device_path is None:
      cloudlog.warning("No WiFi device found")
      return

    wifi_iface = dbus.Interface(self._bus.get_object(NM, device_path), NM_WIRELESS_IFACE)
    dev_props = dbus.Interface(self._bus.get_object(NM, device_path), NM_PROPERTIES_IFACE)
    active_ap_path = dev_props.Get(NM_WIRELESS_IFACE, "ActiveAccessPoint")

    aps: dict[str, list[AccessPoint]] = {}

    for ap_path in wifi_iface.GetAllAccessPoints():
      ap_props = dbus.Interface(self._bus.get_object(NM, ap_path), NM_PROPERTIES_IFACE)

      ap = AccessPoint.from_dbus(ap_props, ap_path, active_ap_path)
      if ap.ssid not in aps:
        aps[ap.ssid] = []

      aps[ap.ssid].append(ap)

    networks = []
    for ssid, ap_list in aps.items():
      # we only want to show the strongest AP for each SSID
      strongest_ap = max(ap_list, key=lambda ap: ap.strength)

      is_connected = any(ap.ap_path == active_ap_path for ap in ap_list)

      networks.append(Network(
        ssid=ssid,
        strength=strongest_ap.strength,
        is_connected=is_connected,
      ))

    # TODO: lock this? i don't think so since this is atomic replace right?!!?!
    self._networks = networks

  def get_networks(self):
    return self._networks
