import unittest
from unittest.mock import patch

from openpilot.system.ui.lib.wifi_manager import WifiManager, ConnectStatus, WifiState


class TestWifiManager(unittest.TestCase):
  def setUp(self):
    with patch('threading.Thread.start'), patch('atexit.register'):
      self.wm = WifiManager()

  def test_stale_snapshot_does_not_override_user(self):
    self.wm._epoch = 2
    self.wm._wifi_state = WifiState('B', ConnectStatus.CONNECTING)
    self.wm._apply({'ssid': 'A', 'status': 2}, epoch=1)
    self.assertEqual(self.wm.connecting_to_ssid, 'B')

  def test_callbacks_run_on_ui_thread_once(self):
    called = []
    self.wm.add_callbacks(need_auth=called.append)
    state = {'auth_error': 'B', 'error_id': 1}
    self.wm._apply(state, 0)
    self.wm._apply(state, 0)
    self.assertEqual(called, [])
    self.wm.process_callbacks()
    self.assertEqual(called, ['B'])

  def test_newer_queued_action_blocks_poll(self):
    self.wm._wifi_state = WifiState('B', ConnectStatus.CONNECTING)
    self.wm._commands.put(('connect', {}, None))
    self.wm._apply({'ssid': 'A', 'status': 2}, 0)
    self.assertEqual(self.wm.connecting_to_ssid, 'B')

  def test_sort_saved_before_stronger_unknown(self):
    self.wm._apply({'saved': ['weak'], 'networks': [
      {'ssid': 'strong', 'strength': 90, 'security_type': 1, 'is_tethering': False},
      {'ssid': 'weak', 'strength': 20, 'security_type': 1, 'is_tethering': False},
    ]}, 0)
    self.assertEqual([n.ssid for n in self.wm.networks], ['weak', 'strong'])


if __name__ == '__main__':
  unittest.main()
