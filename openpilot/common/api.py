import os
from datetime import UTC, datetime, timedelta

import jwt
import requests
from requests.adapters import HTTPAdapter, Retry

from openpilot.common.hardware.hw import Paths
from openpilot.common.version import get_version


API_HOST = os.getenv('API_HOST', 'https://api.commadotai.com')

# name: jwt signature algorithm
KEYS = {"id_rsa": "RS256",
        "id_ecdsa": "ES256"}


class APIError(Exception):
  def __init__(self, message, status_code=None):
    super().__init__(message)
    self.status_code = status_code


class UnauthorizedError(APIError):
  pass


class Api:
  def __init__(self, dongle_id):
    self.dongle_id = dongle_id
    self.jwt_algorithm, self.private_key, _ = get_key_pair()

  def get(self, *args, **kwargs):
    return self.request('GET', *args, **kwargs)

  def post(self, *args, **kwargs):
    return self.request('POST', *args, **kwargs)

  def request(self, method, endpoint, timeout=None, access_token=None, **kwargs):
    return api_get(endpoint, method=method, timeout=timeout, access_token=access_token, **kwargs)

  def get_token(self, payload_extra=None, expiry_hours=1):
    now = datetime.now(UTC).replace(tzinfo=None)
    payload = {
      'identity': self.dongle_id,
      'nbf': now,
      'iat': now,
      'exp': now + timedelta(hours=expiry_hours),
    }
    if payload_extra is not None:
      payload.update(payload_extra)
    token = jwt.encode(payload, self.private_key, algorithm=self.jwt_algorithm)
    if isinstance(token, bytes):
      token = token.decode('utf8')
    return token


def api_get(endpoint, method='GET', timeout=None, access_token=None, session=None, **kwargs):
  headers = dict(kwargs.pop('headers', None) or {})
  headers.setdefault('User-Agent', "openpilot-" + get_version())
  if access_token is not None:
    headers.setdefault('Authorization', "JWT " + access_token)
  return _request(method, endpoint, session=session, timeout=timeout, headers=headers, **kwargs)


def api_get_json(endpoint, access_token=None, **kwargs):
  return _api_request_json('GET', endpoint, access_token=access_token, **kwargs)


def api_post_json(endpoint, access_token=None, **kwargs):
  return _api_request_json('POST', endpoint, access_token=access_token, **kwargs)


def _api_request_json(method, endpoint, access_token=None, **kwargs):
  with requests.Session() as session:
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))

    with api_get(endpoint, method=method, access_token=access_token, session=session, **kwargs) as response:
      response_json = response.json()
      api_error = response_json.get('error') if isinstance(response_json, dict) else None
      if api_error:
        if response.status_code in (401, 403):
          raise UnauthorizedError("Unauthorized", response.status_code)

        description = response_json.get('description', str(api_error))
        raise APIError(f"{response.status_code}:{description}", response.status_code)
      return response_json


def _request(method, endpoint, session=None, **kwargs):
  requester = requests if session is None else session
  url = f"{API_HOST.rstrip('/')}/{endpoint.lstrip('/')}"
  return requester.request(method, url, **kwargs)


def get_key_pair() -> tuple[str, str, str] | tuple[None, None, None]:
  for key in KEYS:
    if os.path.isfile(Paths.persist_root() + f'/comma/{key}') and os.path.isfile(Paths.persist_root() + f'/comma/{key}.pub'):
      with open(Paths.persist_root() + f'/comma/{key}') as private, open(Paths.persist_root() + f'/comma/{key}.pub') as public:
        return KEYS[key], private.read(), public.read()
  return None, None, None
