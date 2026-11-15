import io
import json
import unittest
from contextlib import redirect_stdout
from http import HTTPStatus
from unittest.mock import patch

from requests import Response

from openpilot.tools.lib.api import APIError, CommaApi, UnauthorizedError
from openpilot.tools.lib.file_downloader import api_call


def make_response(status: int, body: str) -> Response:
  response = Response()
  response.status_code = status
  response.reason = HTTPStatus(status).phrase
  response.raw = io.BytesIO(body.encode())
  return response


class TestCommaApi(unittest.TestCase):
  def test_success(self):
    api = CommaApi()
    with patch.object(api.session, "request", return_value=make_response(200, '[{"dongle_id": "1234"}]')):
      self.assertEqual(api.get("v1/me/devices"), [{"dongle_id": "1234"}])

  def test_http_errors(self):
    cases = [
      (401, "", UnauthorizedError, "op auth"),
      (401, '{"error": 1}', UnauthorizedError, "op auth"),
      (403, '{"error": 1}', APIError, "Access denied"),
      (404, "Not Found", APIError, "404: Not Found"),
      (429, '{"description": "Slow down"}', APIError, "429: Slow down"),
      (503, "<html>Unavailable</html>", APIError, "503: Service Unavailable"),
      (200, "invalid JSON", APIError, "invalid JSON"),
      (200, '{"error": "API error"}', APIError, "API error"),
    ]
    for status, body, error_type, message in cases:
      with self.subTest(status=status, body=body):
        api = CommaApi()
        with patch.object(api.session, "request", return_value=make_response(status, body)), self.assertRaisesRegex(error_type, message) as cm:
          api.get("v1/me/devices")
        self.assertIs(type(cm.exception), error_type)
        self.assertEqual(cm.exception.status_code, status)

  def test_downloader_error_response(self):
    for status, error in ((401, "unauthorized"), (403, "forbidden"), (404, "not_found")):
      with self.subTest(status=status), patch("openpilot.tools.lib.auth_config.get_token", return_value=None), \
           patch("requests.Session.request", return_value=make_response(status, "")), redirect_stdout(io.StringIO()) as stdout:
        api_call(lambda api: api.get("v1/me/devices"))
        self.assertEqual(json.loads(stdout.getvalue()), {"error": error})
