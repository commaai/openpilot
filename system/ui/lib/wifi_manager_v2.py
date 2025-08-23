import atexit
import copy
import dbus
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum

from openpilot.common.swaglog import cloudlog
from openpilot.system.ui.lib.networkmanager import *

try:
  from openpilot.common.params import Params
except ImportError:
  # Params/Cythonized modules are not available in zipapp
  Params = None

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
  supports_wpa = NM_802_11_AP_SEC_PAIR_WEP40 | NM_802_11_AP_SEC_PAIR_WEP104 | NM_802_11_AP_SEC_GROUP_WEP40 | NM_802_11_AP_SEC_GROUP_WEP104 | NM_802_11_AP_SEC_KEY_MGMT_PSK;

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
  def from_dbus(cls, ssid: str, aps: list["AccessPoint"], active_ap_path: dbus.ObjectPath, is_saved: bool) -> "Network":
    # we only want to show the strongest AP for each Network/SSID
    strongest_ap = max(aps, key=lambda ap: ap.strength)

    is_connected = any(ap.ap_path == active_ap_path for ap in aps)  # TODO: just any is_connected aps!
    security_type = get_security_type(strongest_ap.flags, strongest_ap.wpa_flags, strongest_ap.rsn_flags)

    return cls(
      ssid=ssid,
      strength=strongest_ap.strength,
      is_connected=is_connected,
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
  ap_path: dbus.ObjectPath

  @classmethod
  def from_dbus(cls, ap_props: dbus.Interface, ap_path: dbus.ObjectPath, active_ap_path: dbus.ObjectPath) -> "AccessPoint":
    ssid = bytes(ap_props.Get(NM_ACCESS_POINT_IFACE, "Ssid")).decode("utf-8", "replace")
    bssid = str(ap_props.Get(NM_ACCESS_POINT_IFACE, "HwAddress"))
    strength = int(ap_props.Get(NM_ACCESS_POINT_IFACE, "Strength"))
    flags = int(ap_props.Get(NM_ACCESS_POINT_IFACE, "Flags"))
    wpa_flags = int(ap_props.Get(NM_ACCESS_POINT_IFACE, "WpaFlags"))
    rsn_flags = int(ap_props.Get(NM_ACCESS_POINT_IFACE, "RsnFlags"))

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
    self._active = True  # used to not run this cycle when not in settings
    self._running = True

    # DBus and NetworkManager setup
    self._bus = dbus.SystemBus()
    self._nm = dbus.Interface(self._bus.get_object(NM, NM_PATH), NM_IFACE)
    self._props = dbus.Interface(self._bus.get_object(NM, NM_PATH), NM_PROPERTIES_IFACE)

    self._bus2 = dbus.SystemBus(private=True)

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

    self._thread = threading.Thread(target=self._run, daemon=True)
    self._thread.start()

    self._state_thread = threading.Thread(target=self._monitor_state, daemon=True)
    self._state_thread.start()

    atexit.register(self.stop)

  def __del__(self):
    self.stop()

  def stop(self):
    self._running = False
    self._thread.join()
    self._state_thread.join()
    self._bus.close()
    self._bus2.close()

  def _monitor_state(self):
    prev_state = -1

    device_path = self._get_wifi_device()
    props_dev = dbus.Interface(self._bus2.get_object(NM, device_path), NM_PROPERTIES_IFACE)
    _props = dbus.Interface(self._bus2.get_object(NM, NM_PATH), NM_PROPERTIES_IFACE)

    while self._running:
      if self._active:
        dev_state = int(props_dev.Get(NM_DEVICE_IFACE, "State"))
        state_reason = props_dev.Get(NM_DEVICE_IFACE, "StateReason")  # (u state, u reason)
        reason = int(state_reason[1]) if isinstance(state_reason, (list, tuple)) and len(state_reason) == 2 else 0

        if dev_state != prev_state:
          print(f"    WiFi device state change: {dev_state}, reason: {reason}")
          if dev_state == NMDeviceState.NEED_AUTH and reason == NM_DEVICE_STATE_REASON_SUPPLICANT_DISCONNECT and self._connecting_to_ssid:
            print('------ NEED AUTH - SUPPLICANT DISCONNECT')
            self.forget_connection(self._connecting_to_ssid, block=True)
            if self._need_auth is not None:
              self._need_auth(self._connecting_to_ssid)
            self._connecting_to_ssid = ""
          elif dev_state == NMDeviceState.ACTIVATED:
            print('------ ACTIVATED')
            if self._activated is not None:
              self._activated()
            self._connecting_to_ssid = ""
          elif dev_state == NMDeviceState.DISCONNECTED:
            print('------ DISCONNECTED')
            self._connecting_to_ssid = ""
            if self._disconnected is not None:
              self._disconnected()

          print()

        if self._connecting_to_ssid:
          print('    CONNECTING', self._connecting_to_ssid)

        prev_state = dev_state

      time.sleep(1 / 2.)

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

  def _run(self):
    while self._running:
      if self._active:
        self._update_networks()

      time.sleep(3)

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

    # print(f"Got wifi device in {time.monotonic() - t}s: {wifi_device}")
    return wifi_device

  def connect_to_network(self, ssid: str, password: str | None):
    def worker():
      t = time.monotonic()

      # Clear all connections that may already exist to the network we are connecting
      self._connecting_to_ssid = ssid
      self.forget_connection(ssid, block=True)

      is_hidden = False

      connection = {
        'connection': {
          'type': '802-11-wireless',
          'uuid': str(uuid.uuid4()),
          'id': f'openpilot connection {ssid}',
          'autoconnect-retries': 0,
        },
        '802-11-wireless': {
          'ssid': dbus.ByteArray(ssid.encode("utf-8")),
          'hidden': is_hidden,
          'mode': 'infrastructure',
        },
        'ipv4': {
          'method': 'auto',
          'dns-priority': 600,
        },
        'ipv6': {'method': 'ignore'},
      }

      if password is not None:
        connection['802-11-wireless-security'] = {
          'key-mgmt': 'wpa-psk',
          'auth-alg': 'open',
          'psk': password,
        }

      settings = dbus.Interface(
        self._bus.get_object(NM, NM_SETTINGS_PATH),
        NM_SETTINGS_IFACE
      )

      conn_path = settings.AddConnection(connection)

      print('Added connection', conn_path)

      print(f'Connecting to network took {time.monotonic() - t}s')

      self.activate_connection(ssid)

    threading.Thread(target=worker, daemon=True).start()

  def _get_connections(self) -> list[dbus.ObjectPath]:
    settings_iface = dbus.Interface(self._bus.get_object(NM, NM_SETTINGS_PATH), NM_SETTINGS_IFACE)
    return settings_iface.ListConnections()

  def _connection_by_ssid(self, ssid: str, known_connections: list[dbus.ObjectPath] | None = None) -> dbus.ObjectPath | None:
    for conn_path in known_connections or self._get_connections():
      conn_props = dbus.Interface(self._bus.get_object(NM, conn_path), NM_CONNECTION_IFACE)
      settings = conn_props.GetSettings()
      if "802-11-wireless" in settings and bytes(settings["802-11-wireless"]["ssid"]).decode("utf-8", "replace") == ssid:
        return conn_path
    return None

  def forget_connection(self, ssid: str, block: bool = False):
    def worker():
      t = time.monotonic()
      conn_path = self._connection_by_ssid(ssid)
      print(f'Finding connection by SSID took {time.monotonic() - t}s: {conn_path}')
      if conn_path is not None:
        conn_iface = dbus.Interface(self._bus.get_object(NM, conn_path), NM_CONNECTION_IFACE)
        conn_iface.Delete()
        print(f'Forgetting connection took {time.monotonic() - t}s')
        if self._forgotten is not None:
          self._forgotten(ssid)

    # TODO: make a helper when it makes sense
    if block:
      worker()
    else:
      threading.Thread(target=worker, daemon=True).start()

  def activate_connection(self, ssid: str):
    t = time.monotonic()
    conn_path = self._connection_by_ssid(ssid)
    if conn_path is not None:
      self._connecting_to_ssid = ssid
      device_path = self._get_wifi_device()
      if device_path is None:
        cloudlog.warning("No WiFi device found")
        return

      print(f'Activating connection to {ssid}')
      self._nm.ActivateConnection(conn_path, device_path, dbus.ObjectPath("/"))
      print(f"Activated connection in {time.monotonic() - t}s")
      # FIXME: deadlock issue with ui
      # if self._activated is not None:
      #   self._activated()

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

      try:
        ap = AccessPoint.from_dbus(ap_props, ap_path, active_ap_path)
        if ap.ssid == "":
          continue

        if ap.ssid not in aps:
          aps[ap.ssid] = []

        aps[ap.ssid].append(ap)
      except dbus.exceptions.DBusException:
        # some APs have been seen dropping off during iteration
        cloudlog.exception(f"Failed to get AP properties for {ap_path}")

    known_connections = self._get_connections()
    self._networks = [Network.from_dbus(ssid, ap_list, active_ap_path, self._connection_by_ssid(ssid, known_connections) is not None)
                      for ssid, ap_list in aps.items()]
    if self._networks_updated is not None:
      self._networks_updated(self._networks)

  def get_networks(self):
    return self._networks
