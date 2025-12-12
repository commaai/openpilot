import pytest
from openpilot.tools.lib.api import LogCallbackRetry, CommaApi, UnauthorizedError, APIError

class TestLogCallbackRetry:
  def test_increment_logs_warning(self, mocker):
    mock_cloudlog = mocker.patch("openpilot.tools.lib.api.cloudlog")
    retry = LogCallbackRetry(total=1)

    retry.increment(method="GET", url="http://test.com")

    mock_cloudlog.warning.assert_called_with("[API Failure] Retrying GET http://test.com")

  def test_increment_no_url(self, mocker):
    mock_cloudlog = mocker.patch("openpilot.tools.lib.api.cloudlog")
    retry = LogCallbackRetry(total=1)

    retry.increment(method="GET")

    mock_cloudlog.warning.assert_not_called()

  def test_retry_configuration(self):
    from openpilot.tools.lib.api import LogCallbackRetry
    from requests.adapters import HTTPAdapter

    api = CommaApi(token="test_token")
    adapter = api.session.adapters['https://']
    assert isinstance(adapter, HTTPAdapter)
    assert isinstance(adapter.max_retries, LogCallbackRetry)
    assert adapter.max_retries.total == 5


class TestCommaApi:
  @pytest.fixture(autouse=True)
  def setup(self):
    self.api = CommaApi()

  def test_init_token(self):
    api = CommaApi(token="test_token")
    assert api.session.headers['Authorization'] == 'JWT test_token'

  def test_request_success(self, mocker):
    mock_resp = mocker.MagicMock()
    mock_resp.json.return_value = {"key": "value"}
    mock_resp.status_code = 200
    mock_request = mocker.patch("openpilot.tools.lib.api.requests.Session.request")
    mock_request.return_value.__enter__.return_value = mock_resp

    resp = self.api.request("GET", "test_endpoint")
    assert resp == {"key": "value"}

  def test_request_unauthorized(self, mocker):
    mock_resp = mocker.MagicMock()
    mock_resp.json.return_value = {"error": "unauthorized"}
    mock_resp.status_code = 401
    mock_request = mocker.patch("openpilot.tools.lib.api.requests.Session.request")
    mock_request.return_value.__enter__.return_value = mock_resp

    with pytest.raises(UnauthorizedError):
      self.api.request("GET", "test_endpoint")

  def test_request_api_error(self, mocker):
    mock_resp = mocker.MagicMock()
    mock_resp.json.return_value = {"error": "server error", "description": "details"}
    mock_resp.status_code = 500
    mock_request = mocker.patch("openpilot.tools.lib.api.requests.Session.request")
    mock_request.return_value.__enter__.return_value = mock_resp

    with pytest.raises(APIError) as cm:
      self.api.request("GET", "test_endpoint")
    assert cm.value.status_code == 500
    assert "details" in str(cm.value)

  def test_get_post(self, mocker):
    mock_request = mocker.patch("openpilot.tools.lib.api.CommaApi.request")
    self.api.get("endpoint", param="1")
    mock_request.assert_called_with("GET", "endpoint", param="1")

    self.api.post("endpoint", data="2")
    mock_request.assert_called_with("POST", "endpoint", data="2")
