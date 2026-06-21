import os
from datetime import UTC, datetime, timedelta

import jwt
import requests
from requests.adapters import HTTPAdapter, Retry

from openpilot.common.hardware.hw import Paths
from openpilot.common.version import get_version

API_HOST = os.getenv('API_HOST', 'https://api.commadotai.com')
TOOLS_USER_AGENT = 'OpenpilotTools'

# name: jwt signature algorithm
KEYS = {"id_rsa": "RS256",
        "id_ecdsa": "ES256"}


class Api:
  def __init__(self, dongle_id=None, token=None, session=None, user_agent=None, retry=False):
    self.dongle_id = dongle_id
    self.token = token
    self.user_agent = user_agent or default_user_agent()
    self.session = session
    self.jwt_algorithm, self.private_key, _ = get_key_pair() if dongle_id is not None else (None, None, None)

    if retry and session is None:
      self.session = requests.Session()
      retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
      self.session.mount('https://', HTTPAdapter(max_retries=retries))

  @classmethod
  def from_user_token(cls, token=None, session=None):
    return cls(token=token, session=session, user_agent=TOOLS_USER_AGENT, retry=True)

  def get(self, *args, **kwargs):
    return self.request('GET', *args, **kwargs)

  def post(self, *args, **kwargs):
    return self.request('POST', *args, **kwargs)

  def request(self, method, endpoint, timeout=None, access_token=None, session=None, **params):
    return self._send(method, endpoint, timeout=timeout, access_token=access_token, session=session, params=params)

  def get_json(self, *args, **kwargs):
    return self.request_json('GET', *args, **kwargs)

  def post_json(self, *args, **kwargs):
    return self.request_json('POST', *args, **kwargs)

  def request_json(self, method, endpoint, **kwargs):
    with self._send(method, endpoint, **kwargs) as resp:
      return parse_api_response(resp)

  def _send(self, method, endpoint, timeout=None, access_token=None, session=None, params=None, **kwargs):
    session = self.session if session is None else session
    access_token = self.token if access_token is None else access_token
    return request_api(method, endpoint, session=session, timeout=timeout,
                       access_token=access_token, user_agent=self.user_agent, params=params, **kwargs)

  def get_token(self, payload_extra=None, expiry_hours=1):
    if self.dongle_id is None:
      raise ValueError("dongle_id is required to generate an API token")
    if self.private_key is None:
      raise ValueError("private key is required to generate an API token")

    now = datetime.now(UTC).replace(tzinfo=None)
    payload = {
      'identity': self.dongle_id,
      'nbf': now,
      'iat': now,
      'exp': now + timedelta(hours=expiry_hours)
    }
    if payload_extra is not None:
      payload.update(payload_extra)
    token = jwt.encode(payload, self.private_key, algorithm=self.jwt_algorithm)
    if isinstance(token, bytes):
      token = token.decode('utf8')
    return token


class _APIException(Exception):
  def __init__(self, message, status_code=None):
    super().__init__(message)
    self.status_code = status_code


class APIError(_APIException):
  pass


class UnauthorizedError(_APIException):
  pass


def _api_url(endpoint):
  return API_HOST + "/" + endpoint.lstrip("/")


def default_user_agent():
  return "openpilot-" + get_version()


def api_headers(access_token=None, user_agent=None):
  headers = {'User-Agent': user_agent or default_user_agent()}
  if access_token is not None:
    headers['Authorization'] = "JWT " + access_token
  return headers


def request_api(method, endpoint, session=None, timeout=None, access_token=None, user_agent=None, params=None, headers=None, **kwargs):
  request_headers = api_headers(access_token, user_agent)
  if headers is not None:
    request_headers.update(headers)
  req = requests if session is None else session
  return req.request(method, _api_url(endpoint), timeout=timeout, headers=request_headers, params=params, **kwargs)


def parse_api_response(resp):
  resp_json = resp.json()
  if isinstance(resp_json, dict) and resp_json.get('error'):
    if resp.status_code in [401, 403]:
      raise UnauthorizedError('Unauthorized. Authenticate with tools/lib/auth.py', resp.status_code)

    raise APIError(str(resp.status_code) + ":" + resp_json.get('description', str(resp_json['error'])), resp.status_code)
  return resp_json


def api_get(endpoint, method='GET', timeout=None, access_token=None, session=None, **params):
  return request_api(method, endpoint, session=session, timeout=timeout,
                     access_token=access_token, params=params)


def get_key_pair() -> tuple[str, str, str] | tuple[None, None, None]:
  for key in KEYS:
    if os.path.isfile(Paths.persist_root() + f'/comma/{key}') and os.path.isfile(Paths.persist_root() + f'/comma/{key}.pub'):
      with open(Paths.persist_root() + f'/comma/{key}') as private, open(Paths.persist_root() + f'/comma/{key}.pub') as public:
        return KEYS[key], private.read(), public.read()
  return None, None, None
