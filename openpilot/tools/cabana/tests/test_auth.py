from concurrent.futures import ThreadPoolExecutor
import unittest
import socket
import tempfile
from urllib.parse import parse_qs, urlparse
from urllib.request import urlopen
from unittest.mock import patch, MagicMock

from openpilot.tools.lib.auth import login_for_cabana
from openpilot.tools.lib.auth_config import get_token, set_token


class TestCabanaAuth(unittest.TestCase):
  def run_login(self, query, response=None, validation_error=None):
    callbacks = []
    pool = ThreadPoolExecutor(max_workers=1)
    self.addCleanup(pool.shutdown, wait=True)

    def browser(url, new):
      state = parse_qs(urlparse(url).query)['state'][0]
      port = state.rsplit(':', 1)[1]

      def callback():
        # Browsers may preconnect and request a favicon before the OAuth callback.
        with socket.create_connection(('localhost', port), timeout=2):
          with urlopen(f'http://localhost:{port}/favicon.ico', timeout=2) as reply:
            assert reply.status == 204
          with urlopen(f'http://localhost:{port}/auth?{query}', timeout=2) as reply:
            assert b'return to Cabana' in reply.read()
      callbacks.append(pool.submit(callback))
      callbacks[-1].result(timeout=3)  # Also cover launchers that wait for the browser.
      return True

    api = MagicMock()
    api.post.return_value = response or {'access_token': 'test-token'}
    api.get.side_effect = validation_error
    with patch('openpilot.tools.lib.auth.webbrowser.open', side_effect=browser), \
         patch('openpilot.tools.lib.auth.CommaApi', return_value=api), \
         patch('openpilot.tools.lib.auth.set_token') as save:
      result = login_for_cabana('google', timeout=2)
      for future in callbacks:
        future.result(timeout=5)
      return result, api, save

  def test_success_validates_and_saves_token(self):
    result, api, save = self.run_login('code=test-code&provider=g')
    self.assertEqual(result, {'success': True})
    api.post.assert_called_once_with('v2/auth/', data={'code': ['test-code'], 'provider': ['g']}, timeout=30)
    api.get.assert_called_once_with('v1/me', timeout=30)
    save.assert_called_once_with('test-token')

  def test_provider_denial(self):
    result, api, save = self.run_login('error=access_denied')
    assert 'declined' in result['error']
    api.post.assert_not_called()
    save.assert_not_called()

  def test_missing_provider(self):
    result, api, save = self.run_login('code=test-code')
    assert 'Invalid' in result['error']
    api.post.assert_not_called()
    save.assert_not_called()

  def test_missing_token(self):
    result, _, save = self.run_login('code=test-code&provider=g', {'unexpected': True})
    assert 'access token' in result['error']
    save.assert_not_called()

  def test_invalid_token_is_not_saved(self):
    result, _, save = self.run_login('code=test-code&provider=g', validation_error=RuntimeError('invalid token'))
    assert 'Could not complete' in result['error']
    save.assert_not_called()

  def test_browser_failure(self):
    with patch('openpilot.tools.lib.auth.webbrowser.open', return_value=False):
      assert 'browser' in login_for_cabana('github')['error']

  def test_timeout(self):
    with patch('openpilot.tools.lib.auth.webbrowser.open', return_value=True):
      assert 'timed out' in login_for_cabana('apple', timeout=0)['error']

  def test_failed_save_preserves_login(self):
    with tempfile.TemporaryDirectory() as directory, patch('openpilot.tools.lib.auth_config.Paths.config_root', return_value=directory):
      set_token('existing-token')
      with patch('openpilot.tools.lib.auth_config.json.dump', side_effect=OSError('disk full')):
        with self.assertRaises(OSError):
          set_token('new-token')
      assert get_token() == 'existing-token'
