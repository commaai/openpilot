import cereal.messaging as messaging
import threading
import time

from cereal import log

DEVICE_STATE_UPDATE_INTERVAL = 1.0  # in seconds

class NetworkStatus:
  def __init__(self):
    self._metered: bool = False
    self._network_type: log.DeviceState.NetworkType = log.DeviceState.NetworkType.none
    self._sm = messaging.SubMaster(['deviceState'])
    self._lock = threading.Lock()

  def update(self) -> None:
    elapsed_time = time.monotonic() - self._sm.recv_time['deviceState']
    if elapsed_time >= DEVICE_STATE_UPDATE_INTERVAL:
        self._sm.update(0)
        with self._lock:
          self._metered = self._sm['deviceState'].networkMetered
          self._network_type = self._sm['deviceState'].networkType.raw

  def get_status(self) -> tuple[bool, log.DeviceState.NetworkType]:
    with self._lock:
      return self._metered, self._network_type
