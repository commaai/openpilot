import jwt
import os
import requests
from datetime import datetime, timedelta
from openpilot.system.hardware.hw import Paths
from openpilot.system.version import get_version

API_HOST = os.getenv('API_HOST', 'https://api.commadotai.com')

class Api():
  def __init__(self, dongle_id):
    self.dongle_id = dongle_id
    with open(Paths.persist_root()+'/comma/id_rsa') as f:
      self.private_key = f.read()

  def get(self, *args, **kwargs):
    return self.request('GET', *args, **kwargs)

  def post(self, *args, **kwargs):
    return self.request('POST', *args, **kwargs)

  def request(self, method, endpoint, timeout=None, access_token=None, **params):
    return api_get(endpoint, method=method, timeout=timeout, access_token=access_token, **params)

  def get_token(self, expiry_hours=1):
    now = datetime.utcnow()
    payload = {
      'identity': self.dongle_id,
      'nbf': now,
      'iat': now,
      'exp': now + timedelta(hours=expiry_hours)
    }
    token = jwt.encode(payload, self.private_key, algorithm='RS256')
    if isinstance(token, bytes):
      token = token.decode('utf8')
    return token


def api_get(endpoint, method='GET', timeout=None, access_token=None, **params):
  headers = {}
  if access_token is not None:
    headers['Authorization'] = "JWT " + access_token

  headers['User-Agent'] = "openpilot-" + get_version()

  return requests.request(method, API_HOST + "/" + endpoint, timeout=timeout, headers=headers, params=params)
