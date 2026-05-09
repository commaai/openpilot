"""Shared pytest fixtures for wifi_manager tests."""
import threading
import time

import pytest

from openpilot.system.ui.lib.wifi_manager import (
  CONNECTING_STALE_TIMEOUT_SECONDS,
  WifiManager,
  WifiState,
)


@pytest.fixture
def wm(mocker):
  """WifiManager stub with mocked dependencies for state-machine tests."""
  mocker.patch.object(WifiManager, "_initialize")
  wm = WifiManager.__new__(WifiManager)
  wm._exit = True
  wm._ctrl = mocker.MagicMock()
  wm._dhcp = mocker.MagicMock()
  wm._store = mocker.MagicMock()
  wm._store.get_metered.return_value = 0
  wm._tethering_active = False
  wm._wifi_state = WifiState()
  wm._user_epoch = 0
  wm._callback_queue = []
  wm._callback_lock = threading.Lock()
  wm._connect_lock = threading.Lock()
  wm._networks_updated_pending = False
  wm._need_auth = []
  wm._activated = []
  wm._disconnected = []
  wm._networks_updated = []
  wm._forgotten = []
  wm._networks = []
  wm._ipv4_address = ""
  wm._current_network_metered = 0
  wm._pending_connection = None
  wm._last_connecting_at = time.monotonic() - CONNECTING_STALE_TIMEOUT_SECONDS - 1
  wm._last_connected_recheck = 0.0
  wm._last_wrong_key_dispatch = {}
  wm._monitor_epoch = 0
  wm._update_active_connection_info = mocker.MagicMock()
  wm._poll_for_ip = mocker.MagicMock()
  wm._ctrl.request.return_value = "wpa_state=COMPLETED\nssid=TestNet\n"
  return wm
