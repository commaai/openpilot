import jwt
import os
import requests
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
  base_dir = os.path.realpath(os.path.join(Paths.persist_root(), 'comma'))
  for key in KEYS:
    private_path = os.path.realpath(os.path.join(base_dir, key))
    public_path = os.path.realpath(os.path.join(base_dir, key + '.pub'))
    if not private_path.startswith(base_dir + os.sep) and private_path != base_dir:
      continue
    if not public_path.startswith(base_dir + os.sep) and public_path != base_dir:
      continue
    if os.path.isfile(private_path) and os.path.isfile(public_path):
      with open(private_path) as private, open(public_path) as public:
        return KEYS[key], private.read(), public.read()
  return None, None, None
