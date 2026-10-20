import atexit
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
import queue
import threading
import time

from openpilot.common.wifi import request
from openpilot.common.swaglog import cloudlog

TETHERING_IP_ADDRESS = '192.168.43.1'
DEFAULT_TETHERING_PASSWORD = 'swagswagcomma'
SCAN_PERIOD_SECONDS = 5


def normalize_ssid(ssid: str) -> str:
  return ssid.replace('’', "'")


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


class ConnectStatus(IntEnum):
  DISCONNECTED = 0
  CONNECTING = 1
  CONNECTED = 2


@dataclass(frozen=True)
class Network:
  ssid: str
  strength: int
  security_type: SecurityType
  is_tethering: bool


@dataclass(frozen=True)
class WifiState:
  ssid: str | None = None
  status: ConnectStatus = ConnectStatus.DISCONNECTED


class WifiManager:
  """UI client for AGNOS wifi.py. One worker serializes mutations and snapshots."""
  def __init__(self):
    self._networks: list[Network] = []
    self._connections: set[str] = set()
    self._wifi_state = WifiState()
    self._ipv4_address = ''
    self._current_network_metered = MeteredType.UNKNOWN
    self._tethering_password = ''
    self._hotspot = False
    self._ipv4_forward = False
    self._active = True
    self._epoch = 0
    self._last_error = 0
    self._callbacks: dict[str, list[Callable]] = {name: [] for name in ('need_auth', 'activated', 'forgotten', 'networks_updated', 'disconnected')}
    self._callback_queue: queue.SimpleQueue = queue.SimpleQueue()
    self._commands: queue.Queue = queue.Queue()
    self._exit = threading.Event()
    self._thread = threading.Thread(target=self._run, daemon=True)
    self._thread.start()
    atexit.register(self.stop)

  def add_callbacks(self, need_auth=None, activated=None, forgotten=None, networks_updated=None, disconnected=None):
    for name, callback in [('need_auth', need_auth), ('activated', activated), ('forgotten', forgotten),
                           ('networks_updated', networks_updated), ('disconnected', disconnected)]:
      if callback is not None:
        self._callbacks[name].append(callback)

  def _emit(self, name, *args):
    for callback in self._callbacks[name]:
      self._callback_queue.put(lambda cb=callback: cb(*args))

  def process_callbacks(self):
    while not self._callback_queue.empty():
      self._callback_queue.get()()

  @property
  def networks(self):
    return sorted(self._networks, key=lambda n: (n.ssid != self._wifi_state.ssid, n.ssid not in self._connections, -n.strength, n.ssid.lower()))

  @property
  def wifi_state(self):
    return self._wifi_state

  @property
  def ipv4_address(self):
    return self._ipv4_address

  @property
  def current_network_metered(self):
    return self._current_network_metered

  @property
  def connecting_to_ssid(self):
    return self._wifi_state.ssid if self._wifi_state.status == ConnectStatus.CONNECTING else None

  @property
  def connected_ssid(self):
    return self._wifi_state.ssid if self._wifi_state.status == ConnectStatus.CONNECTED else None

  @property
  def tethering_password(self):
    return self._tethering_password

  def is_connection_saved(self, ssid):
    return ssid in self._connections

  def is_tethering_active(self):
    return self._hotspot

  def set_active(self, active):
    self._active = active

  def _send(self, command, block=False, **args):
    self._epoch += 1
    done = threading.Event()
    self._commands.put((command, args, done))
    if block:
      done.wait(5)

  def connect_to_network(self, ssid, password, hidden=False):
    self._wifi_state = WifiState(ssid, ConnectStatus.CONNECTING)
    self._send('connect', ssid=ssid, password=password, hidden=hidden)

  def activate_connection(self, ssid, block=False):
    self._wifi_state = WifiState(ssid, ConnectStatus.CONNECTING)
    self._send('activate', block=block, ssid=ssid)

  def forget_connection(self, ssid, block=False):
    self._send('forget', block=block, ssid=ssid)

  def set_tethering_password(self, password):
    self._send('hotspot_config', password=password)

  def set_ipv4_forward(self, enabled):
    self._ipv4_forward = enabled

  def set_tethering_active(self, active):
    self._send('hotspot', active=active, forward=self._ipv4_forward)

  def set_current_network_metered(self, metered):
    self._send('metered', metered=int(metered))

  def _apply(self, state, epoch):
    if epoch != self._epoch or not self._commands.empty():
      return  # A newer UI action takes precedence over an in-flight snapshot.
    previous = self._wifi_state
    self._wifi_state = WifiState(state.get('ssid'), ConnectStatus(state.get('status', 0)))
    self._networks = [Network(**n) for n in state.get('networks', [])]
    self._connections = set(state.get('saved', []))
    self._ipv4_address = state.get('ipv4_address', '')
    self._current_network_metered = MeteredType(state.get('metered', 0))
    self._tethering_password = state.get('hotspot_password', '')
    self._hotspot = state.get('hotspot', False)
    error = state.get('error_id', 0)
    if error != self._last_error and state.get('auth_error'):
      self._emit('need_auth', state['auth_error'])
    self._last_error = error
    if self._wifi_state != previous:
      if self._wifi_state.status == ConnectStatus.CONNECTED:
        self._emit('activated')
      elif self._wifi_state.status == ConnectStatus.DISCONNECTED:
        self._emit('disconnected')
    self._emit('networks_updated', self.networks)

  def _run(self):
    last_scan = 0
    while not self._exit.is_set():
      try:
        while not self._commands.empty():
          command, args, done = self._commands.get()
          try:
            request(command, **args)
            if command == 'forget':
              self._emit('forgotten', args['ssid'])
          finally:
            done.set()
        epoch = self._epoch
        self._apply(request('status'), epoch)
        if self._active and time.monotonic() - last_scan > SCAN_PERIOD_SECONDS:
          last_scan = time.monotonic()
          request('scan')
      except (OSError, ValueError):
        # A service restart or unavailable adapter must not kill the UI worker.
        cloudlog.debug('Wi-Fi service unavailable')
      self._exit.wait(.5)

  def stop(self):
    self._exit.set()
    if self._thread is not threading.current_thread():
      self._thread.join(timeout=4)
