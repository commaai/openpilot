import threading
import unittest
from urllib.parse import parse_qs, urlparse
from urllib.request import urlopen
from unittest.mock import patch, MagicMock

from openpilot.tools.lib.auth import login_for_cabana


class TestCabanaAuth(unittest.TestCase):
  def run_login(self, query, response=None, validation_error=None):
    callbacks = []

    def browser(url, new):
      state = parse_qs(urlparse(url).query)['state'][0]
      port = state.rsplit(':', 1)[1]

      def callback():
        with urlopen(f'http://localhost:{port}/auth?{query}', timeout=5) as reply:
          self.assertIn(b'return to Cabana', reply.read())
      thread = threading.Thread(target=callback)
      callbacks.append(thread)
      thread.start()
      return True

    api = MagicMock()
    api.post.return_value = response or {'access_token': 'test-token'}
    api.get.side_effect = validation_error
    with patch('openpilot.tools.lib.auth.webbrowser.open', side_effect=browser), \
         patch('openpilot.tools.lib.auth.CommaApi', return_value=api), \
         patch('openpilot.tools.lib.auth.set_token') as save:
      result = login_for_cabana('google', timeout=2)
      for thread in callbacks:
        thread.join(timeout=5)
      return result, api, save

  def test_success_validates_and_saves_token(self):
    result, api, save = self.run_login('code=test-code&provider=google')
    self.assertEqual(result, {'success': True})
    api.post.assert_called_once_with('v2/auth/', data={'code': ['test-code'], 'provider': ['google']}, timeout=30)
    api.get.assert_called_once_with('v1/me', timeout=30)
    save.assert_called_once_with('test-token')

  def test_provider_denial(self):
    result, api, save = self.run_login('error=access_denied')
    self.assertIn('declined', result['error'])
    api.post.assert_not_called()
    save.assert_not_called()

  def test_missing_provider(self):
    result, api, save = self.run_login('code=test-code')
    self.assertIn('Invalid', result['error'])
    api.post.assert_not_called()
    save.assert_not_called()

  def test_missing_token(self):
    result, _, save = self.run_login('code=test-code&provider=google', {'unexpected': True})
    self.assertIn('access token', result['error'])
    save.assert_not_called()

  def test_invalid_token_is_not_saved(self):
    result, _, save = self.run_login('code=test-code&provider=google', validation_error=RuntimeError('invalid token'))
    self.assertIn('Could not complete', result['error'])
    save.assert_not_called()

  def test_browser_failure(self):
    with patch('openpilot.tools.lib.auth.webbrowser.open', return_value=False):
      self.assertIn('browser', login_for_cabana('github')['error'])

  def test_timeout(self):
    with patch('openpilot.tools.lib.auth.webbrowser.open', return_value=True):
      self.assertIn('timed out', login_for_cabana('apple', timeout=0)['error'])
