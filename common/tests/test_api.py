import pytest

from openpilot.common import api as api_module
from openpilot.common.api import APIError, CommaApi, UnauthorizedError, api_get


class FakeResponse:
  def __init__(self, payload, status_code=200):
    self.payload = payload
    self.status_code = status_code

  def __enter__(self):
    return self

  def __exit__(self, *args):
    pass

  def json(self):
    return self.payload


class FakeSession:
  def __init__(self, response):
    self.headers = {}
    self.response = response
    self.requests = []
    self.mounts = []

  def mount(self, prefix, adapter):
    self.mounts.append((prefix, adapter))

  def request(self, method, url, **kwargs):
    self.requests.append((method, url, kwargs))
    return self.response


def test_api_get_returns_raw_response(monkeypatch):
  monkeypatch.setattr(api_module, "API_HOST", "https://api.example")
  monkeypatch.setattr(api_module, "get_version", lambda: "test")
  response = object()
  session = FakeSession(response)

  assert api_get("/v1/me", access_token="token", session=session, timeout=5, foo="bar") is response
  assert session.requests == [(
    "GET",
    "https://api.example/v1/me",
    {"timeout": 5, "headers": {"Authorization": "JWT token", "User-Agent": "openpilot-test"}, "params": {"foo": "bar"}},
  )]


def test_comma_api_returns_json(monkeypatch):
  monkeypatch.setattr(api_module, "API_HOST", "https://api.example")
  session = FakeSession(FakeResponse({"ok": True}))

  assert CommaApi("token", session=session).get("/v1/me", timeout=5) == {"ok": True}
  assert session.headers == {"User-Agent": "OpenpilotTools", "Authorization": "JWT token"}
  assert session.requests == [("GET", "https://api.example/v1/me", {"timeout": 5})]


def test_comma_api_raises_api_error():
  session = FakeSession(FakeResponse({"error": "missing", "description": "not found"}, status_code=404))

  with pytest.raises(APIError) as exc:
    CommaApi(session=session).get("v1/route/missing")

  assert str(exc.value) == "404:not found"
  assert exc.value.status_code == 404


@pytest.mark.parametrize("status_code", [401, 403])
def test_comma_api_raises_unauthorized(status_code):
  session = FakeSession(FakeResponse({"error": "auth"}, status_code=status_code))

  with pytest.raises(UnauthorizedError) as exc:
    CommaApi(session=session).get("v1/me")

  assert str(exc.value) == "Unauthorized. Authenticate with tools/lib/auth.py"
  assert exc.value.status_code == status_code
