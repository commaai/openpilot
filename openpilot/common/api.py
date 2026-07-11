import jwt
import os
import requests
from requests.adapters import HTTPAdapter, Retry
from datetime import datetime, timedelta, UTC
from openpilot.common.hardware.hw import Paths
from openpilot.common.version import get_version

API_HOST = os.getenv('API_HOST', 'https://api.commadotai.com')

# name: jwt signature algorithm
KEYS = {"id_rsa": "RS256",
        "id_ecdsa": "ES256"}


class Api:
  def __init__(self, dongle_id):
    self.dongle_id = dongle_id
    self.jwt_algorithm, self.private_key, _ = get_key_pair()

  def get(self, *args, **kwargs):
    return self.request('GET', *args, **kwargs)

  def post(self, *args, **kwargs):
    return self.request('POST', *args, **kwargs)

  def request(self, method, endpoint, timeout=None, access_token=None, **params):
    return api_get(endpoint, method=method, timeout=timeout, access_token=access_token, **params)

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


def api_get(endpoint, method='GET', timeout=None, access_token=None, session=None, **params):
  headers = {}
  if access_token is not None:
    headers['Authorization'] = "JWT " + access_token

  headers['User-Agent'] = "openpilot-" + get_version()

  # TODO: add session to Api
  req = requests if session is None else session
  return req.request(method, API_HOST + "/" + endpoint, timeout=timeout, headers=headers, params=params)


def get_key_pair() -> tuple[str, str, str] | tuple[None, None, None]:
  for key in KEYS:
    if os.path.isfile(Paths.persist_root() + f'/comma/{key}') and os.path.isfile(Paths.persist_root() + f'/comma/{key}.pub'):
      with open(Paths.persist_root() + f'/comma/{key}') as private, open(Paths.persist_root() + f'/comma/{key}.pub') as public:
        return KEYS[key], private.read(), public.read()
  return None, None, None


class CommaApi:
  def __init__(self, token=None):
    self.session = requests.Session()
    self.session.headers['User-agent'] = 'OpenpilotTools'
    if token:
      self.session.headers['Authorization'] = 'JWT ' + token

    retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    self.session.mount('https://', HTTPAdapter(max_retries=retries))

  def request(self, method, endpoint, **kwargs):
    with self.session.request(method, API_HOST + '/' + endpoint, **kwargs) as resp:
      resp_json = resp.json()
      if isinstance(resp_json, dict) and resp_json.get('error'):
        if resp.status_code in [401, 403]:
          raise UnauthorizedError('Unauthorized. Authenticate with openpilot/tools/lib/auth.py')

        e = APIError(str(resp.status_code) + ":" + resp_json.get('description', str(resp_json['error'])))
        e.status_code = resp.status_code
        raise e
      return resp_json

  def get(self, endpoint, **kwargs):
    return self.request('GET', endpoint, **kwargs)

  def post(self, endpoint, **kwargs):
    return self.request('POST', endpoint, **kwargs)


class APIError(Exception):
  pass


class UnauthorizedError(Exception):
  pass
