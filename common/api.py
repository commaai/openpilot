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
  def __init__(self, dongle_id, session=None):
    self.dongle_id = dongle_id
    self.session = session
    self.jwt_algorithm, self.private_key, _ = get_key_pair()

  def get(self, *args, **kwargs):
    return self.request('GET', *args, **kwargs)

  def post(self, *args, **kwargs):
    return self.request('POST', *args, **kwargs)

  def request(self, method, endpoint, timeout=None, access_token=None, session=None, **params):
    session = self.session if session is None else session
    return api_get(endpoint, method=method, timeout=timeout, access_token=access_token,
                   session=session, **params)

  def get_token(self, payload_extra=None, expiry_hours=1):
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


class CommaApi:
  def __init__(self, token=None, session=None):
    self.session = session or requests.Session()
    self.session.headers['User-Agent'] = TOOLS_USER_AGENT
    if token:
      self.session.headers['Authorization'] = 'JWT ' + token

    if session is None:
      retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
      self.session.mount('https://', HTTPAdapter(max_retries=retries))

  def request(self, method, endpoint, **kwargs):
    with _request(method, endpoint, session=self.session, **kwargs) as resp:
      resp_json = resp.json()
      if isinstance(resp_json, dict) and resp_json.get('error'):
        if resp.status_code in [401, 403]:
          raise UnauthorizedError('Unauthorized. Authenticate with tools/lib/auth.py', resp.status_code)

        raise APIError(str(resp.status_code) + ":" + resp_json.get('description', str(resp_json['error'])), resp.status_code)
      return resp_json

  def get(self, endpoint, **kwargs):
    return self.request('GET', endpoint, **kwargs)

  def post(self, endpoint, **kwargs):
    return self.request('POST', endpoint, **kwargs)


class APIError(Exception):
  def __init__(self, message, status_code=None):
    super().__init__(message)
    self.status_code = status_code


class UnauthorizedError(APIError):
  pass


def _api_url(endpoint):
  return API_HOST + "/" + endpoint.lstrip("/")


def _request(method, endpoint, session=None, **kwargs):
  req = requests if session is None else session
  return req.request(method, _api_url(endpoint), **kwargs)


def api_get(endpoint, method='GET', timeout=None, access_token=None, session=None, **params):
  headers = {}
  if access_token is not None:
    headers['Authorization'] = "JWT " + access_token

  headers['User-Agent'] = "openpilot-" + get_version()

  return _request(method, endpoint, session=session, timeout=timeout, headers=headers, params=params)


def get_key_pair() -> tuple[str, str, str] | tuple[None, None, None]:
  for key in KEYS:
    if os.path.isfile(Paths.persist_root() + f'/comma/{key}') and os.path.isfile(Paths.persist_root() + f'/comma/{key}.pub'):
      with open(Paths.persist_root() + f'/comma/{key}') as private, open(Paths.persist_root() + f'/comma/{key}.pub') as public:
        return KEYS[key], private.read(), public.read()
  return None, None, None
