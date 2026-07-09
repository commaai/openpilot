import jwt
import os
import requests
from datetime import datetime, timedelta, UTC
from requests.adapters import HTTPAdapter, Retry
from openpilot.common.hardware.hw import Paths
from openpilot.common.version import get_version

API_HOST = os.getenv('API_HOST', 'https://api.commadotai.com')

# name: jwt signature algorithm
KEYS = {"id_rsa": "RS256",
        "id_ecdsa": "ES256"}


class APIError(Exception):
  def __init__(self, message: str, status_code: int | None = None):
    super().__init__(message)
    self.status_code = status_code


class UnauthorizedError(APIError):
  pass


def _default_user_agent() -> str:
  return "openpilot-" + get_version()


def _create_session(user_agent: str | None = None, token: str | None = None) -> requests.Session:
  session = requests.Session()
  session.headers['User-Agent'] = user_agent or _default_user_agent()
  if token:
    session.headers['Authorization'] = 'JWT ' + token

  retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
  session.mount('https://', HTTPAdapter(max_retries=retries))
  return session


def _api_url(endpoint: str) -> str:
  return API_HOST + "/" + endpoint.lstrip("/")


def _parse_json_response(resp: requests.Response):
  """Parse a JSON response, raising APIError/UnauthorizedError on API errors."""
  resp_json = resp.json()
  if isinstance(resp_json, dict) and resp_json.get('error'):
    if resp.status_code in (401, 403):
      raise UnauthorizedError('Unauthorized. Authenticate with openpilot/tools/lib/auth.py',
                              status_code=resp.status_code)

    raise APIError(str(resp.status_code) + ":" + resp_json.get('description', str(resp_json['error'])),
                   status_code=resp.status_code)
  return resp_json


class Api:
  """Client for the comma API.

  Device usage (dongle-authenticated JWT)::

    api = Api(dongle_id)
    token = api.get_token()
    resp = api.get("v1.4/.../upload_url/", access_token=token, path=key)

  Tools usage (user JWT from auth.py)::

    api = Api(token=get_token())
    data = api.get_json("v1/me/devices")
  """

  def __init__(self, dongle_id: str | None = None, token: str | None = None,
               user_agent: str | None = None, session: requests.Session | None = None):
    self.dongle_id = dongle_id
    self.session = session if session is not None else _create_session(user_agent=user_agent, token=token)
    if token and 'Authorization' not in self.session.headers:
      self.session.headers['Authorization'] = 'JWT ' + token
    if user_agent:
      self.session.headers['User-Agent'] = user_agent

    if dongle_id is not None:
      self.jwt_algorithm, self.private_key, _ = get_key_pair()
    else:
      self.jwt_algorithm, self.private_key = None, None

  def get(self, *args, **kwargs):
    return self.request('GET', *args, **kwargs)

  def post(self, *args, **kwargs):
    return self.request('POST', *args, **kwargs)

  def request(self, method, endpoint, timeout=None, access_token=None, **params):
    """Low-level request with query params. Returns a raw ``requests.Response``."""
    return api_get(endpoint, method=method, timeout=timeout, access_token=access_token,
                   session=self.session, **params)

  def get_json(self, endpoint, **kwargs):
    """GET request that returns parsed JSON and raises on API errors."""
    return self.request_json('GET', endpoint, **kwargs)

  def post_json(self, endpoint, **kwargs):
    """POST request that returns parsed JSON and raises on API errors."""
    return self.request_json('POST', endpoint, **kwargs)

  def request_json(self, method, endpoint, **kwargs):
    """Request with full ``requests`` kwargs. Returns parsed JSON; raises APIError on errors."""
    with self.session.request(method, _api_url(endpoint), **kwargs) as resp:
      return _parse_json_response(resp)

  def get_token(self, payload_extra=None, expiry_hours=1):
    if self.dongle_id is None or self.private_key is None:
      raise RuntimeError("get_token requires a dongle_id and device key pair")

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
  """One-shot API request. Returns a raw ``requests.Response``.

  Extra keyword arguments are sent as query parameters.
  """
  headers = {}
  if access_token is not None:
    headers['Authorization'] = "JWT " + access_token

  headers['User-Agent'] = _default_user_agent()

  req = requests if session is None else session
  return req.request(method, _api_url(endpoint), timeout=timeout, headers=headers, params=params)


def get_key_pair() -> tuple[str, str, str] | tuple[None, None, None]:
  for key in KEYS:
    if os.path.isfile(Paths.persist_root() + f'/comma/{key}') and os.path.isfile(Paths.persist_root() + f'/comma/{key}.pub'):
      with open(Paths.persist_root() + f'/comma/{key}') as private, open(Paths.persist_root() + f'/comma/{key}.pub') as public:
        return KEYS[key], private.read(), public.read()
  return None, None, None
