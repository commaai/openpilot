"""Small HTTP client with the subset of the requests API used by openpilot."""

import http.client
import json as _json
import socket
import ssl
import time
from collections.abc import Sequence
from http.cookiejar import CookieJar
from types import SimpleNamespace
from typing import cast
from urllib.parse import urlencode, urljoin, urlsplit, urlunsplit
from urllib.request import Request as CookieRequest

import urllib3
from urllib3.connection import HTTPConnection
from urllib3.exceptions import (
  ConnectTimeoutError,
  HTTPError as _Urllib3Error,
  NewConnectionError,
  ReadTimeoutError,
  SSLError as _Urllib3SSLError,
)


class RequestException(Exception):
  pass


class ConnectionError(RequestException):  # noqa: A001
  pass


class Timeout(RequestException):
  pass


class ConnectTimeout(Timeout, ConnectionError):
  pass


class SSLError(ConnectionError):
  pass


class HTTPError(RequestException):
  def __init__(self, message: str = "", response: "Response | None" = None):
    self.response = response
    super().__init__(message or (f"{response.status_code} {response.reason}" if response else "HTTP error"))


def _translate_error(error: BaseException, url: str) -> RequestException:
  reason = error
  if isinstance(reason, (ConnectTimeoutError, NewConnectionError)):
    return ConnectTimeout(f"timed out connecting to {url}") if isinstance(reason, ConnectTimeoutError) else ConnectionError(
      f"failed to connect to {url}: {reason}")
  if isinstance(reason, ReadTimeoutError) or isinstance(reason, (TimeoutError, socket.timeout)):
    return Timeout(f"request to {url} timed out")
  if isinstance(reason, _Urllib3SSLError) or isinstance(reason, ssl.SSLError):
    return SSLError(f"TLS request to {url} failed: {reason}")
  return ConnectionError(f"request to {url} failed: {reason}")


class Response:
  def __init__(self, raw: urllib3.HTTPResponse, request, stream: bool) -> None:
    self.status_code = raw.status
    self.reason = raw.reason
    self.headers = raw.headers
    self.request = request
    self.raw = raw
    self._stream = stream
    self._content: bytes | None = None if stream else raw.data
    self._text: str | None = None

  @property
  def ok(self) -> bool:
    return self.status_code < 400

  @property
  def content(self) -> bytes:
    if self._content is None:
      try:
        self._content = self.raw.read(decode_content=True)
      except (OSError, http.client.HTTPException, _Urllib3Error) as e:
        self.close()
        raise _translate_error(e, self.request.url) from e
    return self._content

  @property
  def text(self) -> str:
    if self._text is None:
      self._text = self.content.decode("utf-8", errors="replace")
    return self._text

  def json(self):
    return _json.loads(self.content)

  def raise_for_status(self) -> None:
    if not self.ok:
      raise HTTPError(response=self)

  def iter_content(self, chunk_size: int = 1):
    try:
      while chunk := self.raw.read(chunk_size, decode_content=True):
        yield chunk
    except (OSError, http.client.HTTPException, _Urllib3Error) as e:
      self.close()
      raise _translate_error(e, self.request.url) from e

  def close(self) -> None:
    self.raw.close()
    self.raw.release_conn()

  def __enter__(self) -> "Response":
    return self

  def __exit__(self, exc_type, exc_value, traceback) -> None:
    self.close()


class _CookieResponse:
  def __init__(self, headers) -> None:
    self.headers = headers

  def info(self):
    return self.headers


def _add_params(url: str, params) -> str:
  if not params:
    return url
  split = urlsplit(url)
  query = params if isinstance(params, str) else urlencode(
    [(key, value) for key, value in params.items() if value is not None], doseq=True)
  return urlunsplit((split.scheme, split.netloc, split.path, f"{split.query}&{query}" if split.query else query, split.fragment))


def _prepare_body(data, json, headers: dict[str, str]):
  if json is not None:
    if not any(key.lower() == "content-type" for key in headers):
      headers["Content-Type"] = "application/json"
    return _json.dumps(json).encode()
  if data is None:
    return None
  if isinstance(data, str):
    return data.encode()
  if isinstance(data, (bytes, bytearray, memoryview)) or hasattr(data, "read"):
    return data
  if isinstance(data, dict) or isinstance(data, (list, tuple)):
    if not any(key.lower() == "content-type" for key in headers):
      headers["Content-Type"] = "application/x-www-form-urlencoded"
    pairs = data.items() if isinstance(data, dict) else data
    return urlencode([(key, value) for key, value in pairs if value is not None], doseq=True).encode()
  return data


class Session:
  def __init__(self, *, retries: int = 0, retry_statuses=(), backoff_factor: float = 0,
               socket_options: Sequence[tuple[int, int, int]] = (), persist_cookies: bool = True) -> None:
    self.headers: dict[str, str] = {}
    self.retries = retries
    self.retry_statuses = set(retry_statuses)
    self.backoff_factor = backoff_factor
    self.socket_options = [*HTTPConnection.default_socket_options, *socket_options]
    self.persist_cookies = persist_cookies
    self.cookies = CookieJar()
    self._pools: dict[object, urllib3.PoolManager] = {}

  def _pool(self, verify) -> urllib3.PoolManager:
    key = verify
    if key not in self._pools:
      cert_reqs = "CERT_REQUIRED"
      ca_certs = None
      if verify is False:
        cert_reqs = "CERT_NONE"
      elif verify is not True and verify is not None:
        ca_certs = str(verify)
      self._pools[key] = urllib3.PoolManager(socket_options=self.socket_options, cert_reqs=cert_reqs, ca_certs=ca_certs)
    return self._pools[key]

  def request(self, method: str, url: str, *, params=None, data=None, json=None,
              headers: dict[str, str] | None = None, timeout=None,
              stream: bool = False, verify=True) -> Response:
    """Send an HTTP request.

    File-like bodies must include an explicit Content-Length header. Seekable
    bodies are rewound to their initial position when a retry or redirect
    requires replaying them.
    """
    method = str(method).upper()
    url = _add_params(url, params)
    request_headers = dict(self.headers)
    for key, value in (headers or {}).items():
      request_headers = {old_key: old_value for old_key, old_value in request_headers.items() if old_key.lower() != key.lower()}
      request_headers[key] = value
    body = _prepare_body(data, json, request_headers)
    if isinstance(body, (bytes, bytearray, memoryview)) and not any(key.lower() == "content-length" for key in request_headers):
      request_headers["Content-Length"] = str(len(body))

    retryable = method in {"DELETE", "GET", "HEAD", "OPTIONS", "PUT", "TRACE"}
    body_pos = body.tell() if hasattr(body, "tell") else None
    explicit_cookie = any(key.lower() == "cookie" for key in request_headers)

    for attempt in range(self.retries + 1):
      try:
        current_url, current_method, current_body = url, method, body
        current_headers = dict(request_headers)
        for redirect_count in range(6):
          if self.persist_cookies and not explicit_cookie:
            current_headers = {key: value for key, value in current_headers.items() if key.lower() != "cookie"}
          cookie_request = CookieRequest(current_url, headers=current_headers, method=current_method)
          if self.persist_cookies:
            self.cookies.add_cookie_header(cookie_request)
          current_headers = dict(cookie_request.header_items())
          raw = self._pool(verify).request(current_method, current_url, body=current_body, headers=current_headers, timeout=timeout,
                                           preload_content=not stream, decode_content=True, redirect=False, retries=False)
          if self.persist_cookies:
            self.cookies.extract_cookies(_CookieResponse(raw.headers), cookie_request)  # ty: ignore[invalid-argument-type]

          location = raw.headers.get("Location")
          if current_method == "HEAD" or raw.status not in {301, 302, 303, 307, 308} or not location:
            break
          if redirect_count == 5:
            raw.close()
            raise ConnectionError(f"too many redirects for {url}")

          next_url = urljoin(current_url, location)
          old_origin = (urlsplit(current_url).scheme, urlsplit(current_url).hostname, urlsplit(current_url).port)
          new_origin = (urlsplit(next_url).scheme, urlsplit(next_url).hostname, urlsplit(next_url).port)
          if old_origin != new_origin:
            current_headers = {key: value for key, value in current_headers.items()
                               if key.lower() not in {"authorization", "cookie", "proxy-authorization"}}
            explicit_cookie = False

          # Match requests.SessionRedirectMixin.rebuild_method:
          # 302 converts all non-HEAD methods, while 301 only converts POST.
          if ((raw.status == 303 and current_method != "HEAD") or
              (raw.status == 302 and current_method != "HEAD") or
              (raw.status == 301 and current_method == "POST")):
            current_method, current_body = "GET", None
            current_headers = {key: value for key, value in current_headers.items()
                               if key.lower() not in {"content-length", "content-type", "transfer-encoding"}}
          elif hasattr(current_body, "seek"):
            current_body.seek(body_pos)
            if hasattr(current_body, "total_read"):
              current_body.total_read = 0

          raw.close()
          raw.release_conn()
          current_url = next_url
      except (OSError, http.client.HTTPException, _Urllib3Error) as e:
        if attempt == self.retries or not retryable:
          raise _translate_error(e, url) from e
      else:
        prepared_request = SimpleNamespace(method=current_method, url=current_url, headers=current_headers)
        response = Response(cast(urllib3.HTTPResponse, raw), prepared_request, stream)
        if response.status_code not in self.retry_statuses or attempt == self.retries or not retryable:
          return response
        response.close()
      # urllib3 Retry has no delay after the first failure, then sleeps
      # backoff_factor * 2, * 4, etc.
      if self.backoff_factor and attempt:
        time.sleep(self.backoff_factor * (2 ** attempt))
      if hasattr(body, "seek"):
        body.seek(body_pos)
    raise AssertionError("unreachable")

  def get(self, url, **kwargs) -> Response:
    return self.request("GET", url, **kwargs)

  def post(self, url, **kwargs) -> Response:
    return self.request("POST", url, **kwargs)

  def put(self, url, **kwargs) -> Response:
    return self.request("PUT", url, **kwargs)

  def head(self, url, **kwargs) -> Response:
    return self.request("HEAD", url, **kwargs)

  def close(self) -> None:
    for pool in self._pools.values():
      pool.clear()


# Shared for connection pooling, but stateless across calls by design.
_DEFAULT_SESSION = Session(persist_cookies=False)


def request(method: str, url: str, **kwargs) -> Response:
  return _DEFAULT_SESSION.request(method, url, **kwargs)


def get(url, **kwargs) -> Response:
  return request("GET", url, **kwargs)


def post(url, **kwargs) -> Response:
  return request("POST", url, **kwargs)


def put(url, **kwargs) -> Response:
  return request("PUT", url, **kwargs)


def head(url, **kwargs) -> Response:
  return request("HEAD", url, **kwargs)
