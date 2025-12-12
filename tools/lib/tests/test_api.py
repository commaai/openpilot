
import unittest
from unittest.mock import patch, MagicMock
from tools.lib.api import CallbackRetry

class TestCallbackRetry(unittest.TestCase):
  @patch("tools.lib.api.cloudlog")
  def test_increment_logs_warning(self, mock_cloudlog):
    retry = CallbackRetry(total=1)

    # Simulate a retry increment
    retry.increment(method="GET", url="http://test.com")

    # Verify cloudlog.warning was called with specific message
    mock_cloudlog.warning.assert_called_with("[API Failure] Retrying GET http://test.com")

  @patch("tools.lib.api.cloudlog")
  def test_increment_no_url(self, mock_cloudlog):
    retry = CallbackRetry(total=1)

    # Simulate a retry increment without URL (shouldn't log)
    retry.increment(method="GET")

    # Verify cloudlog.warning was NOT called
    mock_cloudlog.warning.assert_not_called()

class TestCommaApi(unittest.TestCase):
  def setUp(self):
    from tools.lib.api import CommaApi
    self.api = CommaApi()

  def test_init_token(self):
    from tools.lib.api import CommaApi
    api = CommaApi(token="test_token")
    self.assertEqual(api.session.headers['Authorization'], 'JWT test_token')

  @patch("tools.lib.api.requests.Session.request")
  def test_request_success(self, mock_request):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"key": "value"}
    mock_resp.status_code = 200
    mock_request.return_value.__enter__.return_value = mock_resp

    resp = self.api.request("GET", "test_endpoint")
    self.assertEqual(resp, {"key": "value"})

  @patch("tools.lib.api.requests.Session.request")
  def test_request_unauthorized(self, mock_request):
    from tools.lib.api import UnauthorizedError
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"error": "unauthorized"}
    mock_resp.status_code = 401
    mock_request.return_value.__enter__.return_value = mock_resp

    with self.assertRaises(UnauthorizedError):
      self.api.request("GET", "test_endpoint")

  @patch("tools.lib.api.requests.Session.request")
  def test_request_api_error(self, mock_request):
    from tools.lib.api import APIError
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"error": "server error", "description": "details"}
    mock_resp.status_code = 500
    mock_request.return_value.__enter__.return_value = mock_resp

    with self.assertRaises(APIError) as cm:
      self.api.request("GET", "test_endpoint")
    self.assertEqual(cm.exception.status_code, 500)
    self.assertIn("details", str(cm.exception))

  @patch("tools.lib.api.CommaApi.request")
  def test_get_post(self, mock_request):
    self.api.get("endpoint", param="1")
    mock_request.assert_called_with("GET", "endpoint", param="1")

    self.api.post("endpoint", data="2")
    mock_request.assert_called_with("POST", "endpoint", data="2")

if __name__ == "__main__":
  unittest.main()
